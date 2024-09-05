import torch
import torch.nn as nn 

from algorithms.utils.rnn import RNNLayer
from algorithms.utils.act import ACTLayer

from algorithms.utils.mast_utils import *
from algorithms.utils.util import init, check
from utils.util import get_shape_from_obs_space

class FeatureEncoder(nn.Module):
    def __init__(self, args, config):
        super(FeatureEncoder, self).__init__()
        """
        입력 데이터를 전처리한 뒤 actor와 critic에 각각 보내기 위한 데이터 전처리 클래스 
        """
        
        self._recurrent_N: int = config["_recurrent_N"]  
        self._use_orthogonal: bool = config["_use_orthogonal"]

        self.inputs_dim: int = config["inputs_dim"]
        self.num_agents: int = config["num_agents"]
        self.hidden_size: int = config["hidden_size"]

        self.embed_layer = nn.Sequential(
            nn.Linear(self.inputs_dim, self.hidden_size),
            nn.ReLU(inplace= True),
        )
        self.rnn_layer = RNNLayer(
            inputs_dim = self.hidden_size,
            outputs_dim = self.hidden_size,
            recurrent_N = self._recurrent_N,
            use_orthogonal = self._use_orthogonal 
        )
    
    def forward(self, obs, rnn_states, masks):
        feature = self.embed_layer(obs)
        embedding_features, rnn_states = self.rnn_layer(feature, rnn_states, masks)
        embedding_features = embedding_features.reshape(
            -1, self.num_agents, self.hidden_size
        )
        return embedding_features, rnn_states
        

class Encoder(nn.Module):
    def __init__(self, args, config):
        super(Encoder, self).__init__()
        """
        FeatureEncoder에서 임베딩된 데이터를 SAB block에 2번 통과시켜 self-attention 수행 후, ACTlayer를 통해 action sampling 
        """
        
        self._gain: bool = args.gain
        self._use_orthogonal: bool = config["_use_orthogonal"]
        
        self._recurrent_N: int = config["_recurrent_N"]
        self.num_agents: int = config["num_agents"]
        self.inputs_dim: int = config["inputs_dim"]
        self.hidden_size: int = config["hidden_size"]
        self.num_head: int = config["num_head"]
        action_space = config["action_space"]
    
        self.feature_encoder = FeatureEncoder(
            args,
            config
        )
        
        self.encoder_block = nn.Sequential(
            SetAttentionBlock(
                d = self.hidden_size,
                h = self.num_head,
                rff = RFF(self.hidden_size),
            ),
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

    def forward(self, input, rnn_states, masks, available_actions=None, deterministic=False):
        feature, rnn_states= self.feature_encoder(input, rnn_states, masks)
        encoder_output = self.encoder_block(feature)
        actions, action_log_probs = self.act_layer(
            encoder_output.reshape(-1, self.hidden_size),
            available_actions, 
            deterministic
        )
        
        return actions, action_log_probs, rnn_states

class Decoder(nn.Module):
    def __init__(self, args, config):
        super(Decoder, self).__init__()
        """
        FeatureEncoder에서 임베딩된 데이터를 PMA block과 SAB block에 통과시킨 뒤 MLP layer를 통해 v값 출력
        """
        self._use_orthogonal: bool = config["_use_orthogonal"]
        
        self.num_seed_vector: int = 4
        self.num_head: int = config["num_head"]
        self.hidden_size = config["hidden_size"]
        self.num_agents: int = config["num_agents"]
        self._recurrent_N: int = config["_recurrent_N"]
        
        self.feature_encoder = FeatureEncoder(
            args,
            config
        )

        self.decoder_block = nn.Sequential(
            PoolingMultiheadAttention(
                d = self.hidden_size,
                k = self.num_seed_vector, 
                h = self.num_head, 
                rff = RFF(self.hidden_size)
            ),
            SetAttentionBlock(
                d = self.hidden_size,
                h = self.num_head,
                rff = RFF(self.hidden_size),
            )
        )
        self.reduction_net = nn.Linear(
            self.hidden_size * self.num_seed_vector, 
            self.hidden_size
        )
        
        # self.v2_net = RNNLayer(
        #     inputs_dim = self.hidden_size,
        #     outputs_dim = self.hidden_size,
        #     recurrent_N = self._recurrent_N,
        #     use_orthogonal = self._use_orthogonal 
        # )
        self.v_net = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.Linear(self.hidden_size // 2, 1),
        )
        
    def forward(self, input, rnn_states, masks):
        feature, rnn_states= self.feature_encoder(input, rnn_states, masks)
        pma_output = self.decoder_block(feature)
        pma_output = pma_output.reshape(-1, self.num_seed_vector * self.hidden_size)
        x = self.reduction_net(pma_output)
        values = self.v_net(x)
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
        
        self.num_head = 4
        
        config = {
            "num_agents" : self.num_agents,
            "inputs_dim" : self.obs_shape,
            "hidden_size" : self.hidden_size,
            "num_head" : self.num_head,
            "_recurrent_N" : self._recurrent_N,
            "action_space" : action_space,
            "_use_orthogonal": self._use_orthogonal ,
            "_gain" : self._gain
        }
        
        self.encoder = Encoder(
            args = args,
            config = config
        )
        self.decoder = Decoder(
            args = args,
            config = config
        )
        self.to(self.device)
    
    def get_actions(self, obs, rnn_states,rnn_states_critic, masks, available_actions=None, deterministic=False):

        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        rnn_states_critic = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
            
        actions, action_log_probs, rnn_states = self.encoder(
            obs,
            rnn_states, 
            masks,
            available_actions,
            deterministic
        )

        values, rnn_states_critic = self.decoder(
            obs,
            rnn_states_critic,
            masks
        )
        
        return values, actions, action_log_probs, rnn_states, rnn_states_critic

    def get_values(self, obs, rnn_states, rnn_states_critic, masks, available_actions=None, deterministic=False):

        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        rnn_states_critic = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        values, _ = self.decoder(
            obs,
            rnn_states_critic,
            masks
        )
        return values

    def evaluate_actions(self, obs, rnn_states, rnn_states_critic, action, masks, critic_masks, available_actions=None, active_masks=None):
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        rnn_states_critic = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
        
        feature, _ = self.encoder.feature_encoder(
            obs,
            rnn_states,
            masks
        )
        encoder_output = self.encoder.encoder_block(feature)
        action_log_probs, dist_entropy = self.encoder.act_layer.evaluate_actions(
            encoder_output.reshape(-1, self.hidden_size),
            action,
            available_actions, 
            active_masks = active_masks if self._use_policy_active_masks else None
        )
        
        values, _ = self.decoder(
            obs,
            rnn_states_critic,
            masks
        )
        
            
        return values, action_log_probs, dist_entropy
    
    def act(self, obs, rnn_states, masks, available_actions, deterministic):
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)
        
        actions, _, rnn_states = self.encoder(
            obs,
            rnn_states,
            masks,
            available_actions,
            deterministic
        )
        
        return actions, rnn_states