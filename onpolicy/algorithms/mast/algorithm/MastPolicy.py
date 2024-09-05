import torch

from utils.util import update_linear_schedule
from utils.util import get_shape_from_obs_space
from algorithms.mast.algorithm.setransformer import  MultiAgentSetTransformer

class MastPolicy:
    def __init__(self, args, obs_space, cent_obs_space, act_space, num_agents, device=torch.device("cpu")):
        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay

        self.obs_space = obs_space
        self.share_obs_space = cent_obs_space
        self.act_space = act_space
        self.num_agents = num_agents

        self.obs_dim = get_shape_from_obs_space(obs_space)[0]

        self.model = MultiAgentSetTransformer(
            args, 
            self.obs_space, 
            self.act_space, 
            self.num_agents,
            device = self.device
        )

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr, 
            eps=self.opti_eps,
            weight_decay=self.weight_decay
        )

    def lr_decay(self, episode, episodes):
        update_linear_schedule(self.optimizer, episode, episodes, self.lr)
        
    def get_actions(self, cent_obs, obs, rnn_states, rnn_states_critic, masks, available_actions=None, deterministic=False):
        
        values, actions, action_log_probs, rnn_states, rnn_states_critic = self.model.get_actions(
            obs,
            rnn_states,
            rnn_states_critic,
            masks,
            available_actions,
            deterministic,
        )
        
        return values, actions, action_log_probs, rnn_states, rnn_states_critic
    
    def get_values(self, cent_obs, obs, rnn_states, rnn_states_critic, masks):
        values = self.model.get_values(
            obs,
            rnn_states,
            rnn_states_critic,
            masks,
        )
        
        return values
    
    def evaluate_actions(self, obs, rnn_states, rnn_states_critic,
                          action, masks, critic_masks_batch, available_actions, active_masks):
        
        values, action_log_probs, dist_entropy = self.model.evaluate_actions(
            obs = obs, 
            rnn_states = rnn_states,
            rnn_states_critic = rnn_states_critic, 
            action = action, 
            masks = masks, 
            critic_masks = critic_masks_batch,
            available_actions=available_actions, 
            active_masks=active_masks
        )
        return values, action_log_probs, dist_entropy
    
    def act(self, obs, rnn_states, masks, available_actions=None, deterministic=False):
        actions, rnn_states = self.model.act(
            obs,
            rnn_states,
            masks,
            available_actions,
            deterministic,
        )
        
        return actions, rnn_states 