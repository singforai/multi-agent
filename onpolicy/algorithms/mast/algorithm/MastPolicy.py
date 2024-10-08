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
        """
        If use_linear_lr_decay is True, the learning rate decreases linearly

        Arguement:
            - episode           | int
            - episodes          | int
        
        """
        update_linear_schedule(self.optimizer, episode, episodes, self.lr)
        
    def get_actions(self, cent_obs, obs, rnn_states, rnn_states_critic, masks, available_actions=None, deterministic=False):
        """
        A function to sample the next action during the sampling process

        Arguement:
            - cent_obs          | np.ndarray (n_rollout_threads * num_agents, share_obs_dim)
            - obs               | np.ndarray (n_rollout_threads * num_agents, obs_dim)
            - rnn_states        | np.ndarray (n_rollout_threads * num_agents, recurrent_N, hidden_size)
            - rnn_states_critic | np.ndarray (n_rollout_threads * num_agents, recurrent_N, hidden_size)
            - masks             | np.ndarray (n_rollout_threads * num_agents, 1)
            - available_actions | np.ndarray (n_rollout_threads * num_agents, action_space)
            - deterministic     | bool

        return:
            - values            | tensor (n_rollout_threads * num_agents , 1)
            - actions           | tensor (n_rollout_threads * num_agents , 1)
            - action_log_probs  | tensor (n_rollout_threads * num_agents , 1)
            - rnn_states        | tensor (n_rollout_threads * num_agents , recurrent_N, hidden_size)
            - rnn_states_critic | tensor (n_rollout_threads * num_agents , recurrent_N, hidden_size)
        """
        
        values, actions, action_log_probs, rnn_states, rnn_states_critic = self.model.get_actions(
            obs,
            rnn_states,
            rnn_states_critic,
            masks,
            available_actions,
            deterministic,
        )
        return values, actions, action_log_probs, rnn_states, rnn_states_critic
    
    def get_values(self, cent_obs, obs, rnn_states, rnn_states_critic, masks, available_actions=None):
        """
        A function to predict the values of the last step stored in the buffer 
        for calculating the GAE return
        
        Arguement:
            - cent_obs          | np.ndarray (n_rollout_threads * num_agents, share_obs_dim)
            - obs               | np.ndarray (n_rollout_threads * num_agents, obs_dim)
            - rnn_states        | np.ndarray (n_rollout_threads * num_agents, recurrent_N, hidden_size)
            - rnn_states_critic | np.ndarray (n_rollout_threads * num_agents, recurrent_N, hidden_size)
            - masks             | np.ndarray (n_rollout_threads * num_agents, 1)

        return:
            - values            | tensor (n_rollout_threads * num_agents , 1)
        """
        values = self.model.get_values(
            obs,
            rnn_states,
            rnn_states_critic,
            masks,
        )
        return values
    
    def evaluate_actions(self, obs, rnn_states, rnn_states_critic,
                          action, masks, critic_masks_batch, available_actions, active_masks):
        """
        A function to calculate the importance weight and value loss between the updated network 
        from training and the network used for sampling
        
        Arguement:
            - obs               | np.ndarray (data_chunk_length * mini_batch_size * num_agents ,  obs_dim)
            - rnn_states        | np.ndarray (mini_batch_size * num_agents , recurrent_N , hidden_size)
            - rnn_states_critic | np.ndarray (mini_batch_size * num_agents , recurrent_N , hidden_size)
            - action            | np.ndarray (data_chunk_length * mini_batch_size * num_agents ,  1)
            - masks             | np.ndarray (data_chunk_length * mini_batch_size * num_agents ,  1)
            - critic_masks_batch| np.ndarray (data_chunk_length * mini_batch_size * num_agents ,  1)
            - available_actions | np.ndarray (data_chunk_length * mini_batch_size * num_agents ,  action_space)
            - active_masks      | np.ndarray (data_chunk_length * mini_batch_size * num_agents ,  1)
            
        return:
            - values            | tensor (data_chunk_length * mini_batch_size * num_agents ,  1)
            - action_log_probs  | tensor (data_chunk_length * mini_batch_size * num_agents ,  1)
            - dist_entropy      | tensor float
        """
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
    
    def act(self,cent_obs, obs, rnn_states, masks, available_actions=None, deterministic=False):
        """
        A function for an agent to sample actions during the evaluation process
        
        Arguement:
            - obs               | np.ndarray (n_rollout_threads * num_agents , obs_dim)
            - rnn_states        | np.ndarray (n_rollout_threads * num_agents , recurrent_N , hidden_size)
            - masks             | np.ndarray (n_rollout_threads * num_agents , 1)
            - available_actions | np.ndarray (n_rollout_threads * num_agents , action_space)
            - deterministic     | bool
        
        return:
            - actions           | tensor (n_rollout_threads * num_agents , 1)
            - rnn_states        | tensor (n_rollout_threads * num_agents , recurrent_N , hidden_size)
        """
        actions, rnn_states = self.model.act(
            obs,
            rnn_states,
            masks,
            available_actions,
            deterministic,
        )
        return actions, rnn_states 

    def save(self, save_dir):
        torch.save(self.model.state_dict(), str(save_dir) + "/mast.pt")

    def restore(self, model_dir):
        state_dict = torch.load(str(model_dir) + '/mast.pt', map_location=torch.device('cpu'))
        self.model.load_state_dict(state_dict)