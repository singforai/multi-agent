import torch
import torch.nn as nn 

from algorithms.utils.rnn import RNNLayer
from algorithms.utils.act import ACTLayer

from algorithms.utils.mast_utils import *
from algorithms.utils.util import init, check
from utils.util import get_shape_from_obs_space



class Encoder(nn.Module):
    def __init__(self, args, config):
        super(Encoder, self).__init__()        
        action_space = config["action_space"]
        
        self._gain: bool = args.gain
        self.num_head: int = config["num_head"]
        self.num_agents: int = config["num_agents"]
        self.inputs_dim: int = config["inputs_dim"]
        self.hidden_size: int = config["hidden_size"]
        self._recurrent_N: int = config["_recurrent_N"]
        self._use_orthogonal: bool = config["_use_orthogonal"]
        
        self.embed_layer = nn.Sequential(
            nn.Linear(self.inputs_dim, self.hidden_size),
        )
        self.rnn_layer = RNNLayer(
            inputs_dim = self.hidden_size,
            outputs_dim = self.hidden_size,
            recurrent_N = self._recurrent_N,
            use_orthogonal = self._use_orthogonal 
        )
        
        self.encoder_block = nn.Sequential(
            SetAttentionBlock(
                d = self.hidden_size,
                h = self.num_head,
                rff = RFF(self.hidden_size),
            ),
        )
        self.before_act_layer = nn.Sequential(
            SetAttentionBlock(
                d = self.hidden_size,
                h = self.num_head,
                rff = RFF(self.hidden_size),
            ),
        )
        
        self.act_layer = ACTLayer(
            action_space = action_space,
            inputs_dim = self.hidden_size,
            use_orthogonal = self._use_orthogonal,
            gain = self._gain
        )

    def forward(self, obs, rnn_states, masks, available_actions=None, deterministic=False):
        embedding_features = self.embed_layer(obs)
        embedding_features, rnn_states = self.rnn_layer(embedding_features, rnn_states, masks)
        embedding_features = embedding_features.reshape(-1, self.num_agents, self.hidden_size)
        block_ouput = self.encoder_block(embedding_features)
        action_input = self.before_act_layer(block_ouput)
        actions, action_log_probs = self.act_layer(
            action_input.reshape(-1, self.hidden_size),
            available_actions, 
            deterministic
        )
        
        return actions, action_log_probs, rnn_states, block_ouput

class Decoder(nn.Module):
    def __init__(self, args, config):
        super(Decoder, self).__init__()
        self.use_pma_block: bool = args.use_pma_block
        self._use_orthogonal: bool = config["_use_orthogonal"]
        
        self.num_seed_vector: int = args.n_seed_vector
        self.num_head: int = config["num_head"]
        self.hidden_size = config["hidden_size"]
        self.num_agents: int = config["num_agents"]
        self._recurrent_N: int = config["_recurrent_N"]
        
        
        if self.use_pma_block:
            self.decoder_block = nn.Sequential(
                PoolingMultiheadAttention(
                    d = self.hidden_size,
                    k = self.num_seed_vector, 
                    h = self.num_head, 
                    rff = RFF(self.hidden_size)
                )
            )
            self.v_net = nn.Sequential(
                nn.Linear(
                self.hidden_size * self.num_seed_vector, self.hidden_size//2
                ),
                nn.Linear(self.hidden_size//2, 1)
            )
        else:
            self.decoder_block = nn.Sequential(
                SetAttentionBlock(
                    d = self.hidden_size,
                    h = self.num_head,
                    rff = RFF(self.hidden_size),
                ),
            )
            self.v_net = nn.Sequential(
                nn.Linear(
                self.hidden_size, self.hidden_size
                ),
                nn.Linear(self.hidden_size, 1)
            )

    def forward(self, block_output, rnn_states, masks):

        output = self.decoder_block(block_output)
        if self.use_pma_block:
            output = output.reshape(-1, self.num_seed_vector * self.hidden_size)
            values = self.v_net(output)
            values = values.unsqueeze(1).repeat(1, self.num_agents, 1).reshape(-1, 1)
        else:
            output = self.v_net(output)
            values = output.reshape(-1, 1)

        return values, rnn_states
    
class MultiAgentSetTransformer(nn.Module):
    def __init__(self, args, obs_space, action_space, num_agents, device=torch.device("cpu")):
        super(MultiAgentSetTransformer, self).__init__()
        self.device = device
        
        self._gain: bool = args.gain
        self._use_orthogonal: bool = args.use_orthogonal
        self._use_policy_active_masks: bool = args.use_policy_active_masks
        
        self.num_agents: int = num_agents
        self.hidden_size: int = args.hidden_size # attention_dim size
        self._recurrent_N: int = args.recurrent_N
        
        self.tpdv = dict(dtype=torch.float32, device=device)
        
        obs_shape = get_shape_from_obs_space(obs_space)
        if type(obs_shape[-1]) == list:
            obs_shape = obs_shape[:1]
        self.obs_shape = (*obs_shape,)[0]
        #self.act_space = action_space.n
        
        self.num_head = args.n_head
        
        encoder_config = {
            "num_agents" : self.num_agents,
            "inputs_dim" : self.obs_shape,
            "hidden_size" : self.hidden_size,
            "num_head" : self.num_head,
            "_recurrent_N" : self._recurrent_N,
            "action_space" : action_space,
            "_use_orthogonal": self._use_orthogonal ,
            "_gain" : self._gain
        }
        decoder_config = {
            "num_agents" : self.num_agents,
            "hidden_size" : self.hidden_size,
            "num_head" : self.num_head,
            "_recurrent_N" : self._recurrent_N,
            "_use_orthogonal": self._use_orthogonal
        }
        self.encoder = Encoder(
            args = args,
            config = encoder_config
        )
        self.decoder = Decoder(
            args = args,
            config = decoder_config
        )
        self.to(self.device)
    
    def get_actions(self, obs, rnn_states,rnn_states_critic, masks, available_actions=None, deterministic=False):
        """
        Arguements
            - obs               => encoder : block_output
            - rnn_states        => encoder : rnn_states
            - rnn_states_critic => decoder : rnn_states_critic(Not changes)
            - masks             => encoder
            - available_actions => encoder
            - deterministic     => encoder
        params
            - block_output => encoder : actions, action_log_probs
                           => decoder : values
        """
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        rnn_states_critic = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
        actions, action_log_probs, rnn_states, block_output = self.encoder(
            obs,
            rnn_states,
            masks,
            available_actions,
            deterministic
        )

        values, rnn_states_critic = self.decoder(
            block_output,
            rnn_states_critic,
            masks
        )
        
        return values, actions, action_log_probs, rnn_states, rnn_states_critic

    def get_values(self, obs, rnn_states, rnn_states_critic, masks, available_actions=None, deterministic=False):
        """
        Arguements
            - obs               => encoder : block_output
            - rnn_states        => encoder : rnn_states
            - rnn_states_critic => decoder : rnn_states_critic(Not changes)
            - masks             => encoder
            - available_actions => encoder
            - deterministic     => encoder
        params
            - block_output => encoder : _ (useless)
                           => decoder : values
        """
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        rnn_states_critic = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        _, _, _, block_output = self.encoder(
            obs,
            rnn_states,
            masks,
            available_actions,
            deterministic
        )
        
        values, _ = self.decoder(
            block_output,
            rnn_states_critic,
            masks
        )
        return values

    def evaluate_actions(self, obs, rnn_states, rnn_states_critic, action, masks, critic_masks, available_actions=None, active_masks=None):
        """
        Arguements
            - obs               => embed_layer : embedding_features
            - rnn_states        => rnn_layer   : rnn_states
            - rnn_states_critic => decoder     : rnn_states_critic(Not changes)
            - action            => act_layer  
            - masks             => rnn_layer   : 
            - critic_masks      => Not use    
            - available_actions => act_layer   
            - active_masks      => act_layer   
        params
            - embedding_features => encoder_block : block_output
            - block_output       => before_act_layer : action_input
                                 => decoder : values
            - action_input       => act_layer 
        """
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        rnn_states_critic = check(rnn_states_critic).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
        
        embedding_features = self.encoder.embed_layer(obs)
        embedding_features, _ = self.encoder.rnn_layer(
            embedding_features, 
            rnn_states, 
            masks
        )
        embedding_features = embedding_features.reshape(-1, self.num_agents, self.hidden_size)
        block_output = self.encoder.encoder_block(embedding_features)
        action_input = self.encoder.before_act_layer(block_output)
        
        action_log_probs, dist_entropy = self.encoder.act_layer.evaluate_actions(
            action_input.reshape(-1, self.hidden_size),
            action,
            available_actions, 
            active_masks = active_masks if self._use_policy_active_masks else None
        )
        values, _ = self.decoder(
            block_output,
            rnn_states_critic,
            masks
        )
        
            
        return values, action_log_probs, dist_entropy
    
    def act(self, obs, rnn_states, masks, available_actions, deterministic):
        """
        Arguements
            - obs               => encoder 
            - rnn_states        => encoder 
            - masks             => encoder
            - available_actions => encoder
            - deterministic     => encoder
        """
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
            
        actions, _, rnn_states, _ = self.encoder(
            obs,
            rnn_states,
            masks,
            available_actions,
            deterministic
        )
        
        return actions, rnn_states