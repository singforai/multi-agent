import wandb
import torch

import numpy as np
from gym import spaces
from utils.shared_buffer import SharedReplayBuffer

def _t2n(x):
    """Convert torch tensor to a numpy array."""
    return x.detach().cpu().numpy()

class Runner(object):
    """
    Base class for training recurrent policies.
    :param config: (dict) Config dictionary containing parameters for training.
    """
    def __init__(self, config):

        self.all_args = config['all_args']
        self.envs = config['envs']
        self.eval_envs = config['eval_envs']
        self.device = config['device']
        self.num_agents = config['num_agents']
        if config.__contains__("render_envs"):
            self.render_envs = config['render_envs']       

        # parameters
        self.use_eval: bool = self.all_args.use_eval
        self.use_wandb: bool = self.all_args.use_wandb
        self.use_render: bool = self.all_args.use_render
        self.use_centralized_V: bool = self.all_args.use_centralized_V
        self.use_linear_lr_decay: bool = self.all_args.use_linear_lr_decay
        self.use_obs_instead_of_state: bool = self.all_args.use_obs_instead_of_state
        
        self.env_name: str = self.all_args.env_name
        self.algorithm_name: str = self.all_args.algorithm_name
        self.experiment_name: str = self.all_args.experiment_name


        self.hidden_size: int = self.all_args.hidden_size
        self.recurrent_N: int = self.all_args.recurrent_N
        self.log_interval: int = self.all_args.log_interval
        self.save_interval: int = self.all_args.save_interval
        self.eval_interval: int = self.all_args.eval_interval
        self.eval_episodes: int = self.all_args.eval_episodes
        self.num_env_steps: int = self.all_args.num_env_steps
        self.episode_length: int = self.all_args.episode_length
        self.n_rollout_threads: int = self.all_args.n_rollout_threads
        self.n_eval_rollout_threads: int = self.all_args.n_eval_rollout_threads
        self.n_render_rollout_threads: int = self.all_args.n_render_rollout_threads

        # dir
        self.model_dir = self.all_args.model_dir

        if self.use_wandb:
            self.save_dir = str(wandb.run.dir)
            self.run_dir = str(wandb.run.dir)

        if self.algorithm_name == "mat" or self.algorithm_name == "mat_dec":
            from algorithms.mat.mat_trainer import MATTrainer as TrainAlgo
            from algorithms.mat.algorithm.transformer_policy import TransformerPolicy as Policy
        elif self.algorithm_name == "tizero":
            from algorithms.tizero.tizero import TiZero as TrainAlgo
            from algorithms.tizero.algorithm.TiZeroPolicy import TiZeroPolicy as Policy
        elif self.algorithm_name == "mast":
            from algorithms.mast.mast import Mast as TrainAlgo
            from algorithms.mast.algorithm.MastPolicy import MastPolicy as Policy
            
        else:
            from algorithms.r_mappo.r_mappo import R_MAPPO as TrainAlgo
            from algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy
        if self.env_name == "football":
            low = np.full((330,), -np.inf)
            high = np.full((330,), np.inf)
            observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

            low = np.full((220,), -np.inf)
            high = np.full((220,), np.inf)
            share_observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        else:
            observation_space = self.envs.observation_space[0]
            share_observation_space = self.envs.share_observation_space[0] if self.use_centralized_V else self.envs.observation_space[0]
        
        # policy network
        if self.algorithm_name == "mat" or self.algorithm_name == "mat_dec":
            self.policy = Policy(self.all_args, 
                observation_space, 
                share_observation_space, 
                self.envs.action_space[0], 
                self.num_agents, 
                device = self.device
            )
        elif self.algorithm_name == "mast":
            self.policy = Policy(
                args = self.all_args, 
                obs_space = observation_space, 
                cent_obs_space = observation_space, 
                act_space = self.envs.action_space[0], 
                num_agents = self.num_agents, 
                device = self.device
            )
        else:
            self.policy = Policy(
                self.all_args, 
                observation_space, 
                share_observation_space, 
                self.envs.action_space[0], 
                device = self.device
            )

        if self.model_dir is not None:
            self.restore(self.model_dir)

        # algorithm
        if self.algorithm_name == "mat" or self.algorithm_name == "mat_dec":
            self.trainer = TrainAlgo(
                self.all_args, 
                self.policy, 
                self.num_agents, 
                device = self.device
            )
        elif self.algorithm_name == "mast":
            self.trainer = TrainAlgo(
                self.all_args, 
                self.policy, 
                self.num_agents, 
                device = self.device
            )
        else:
            self.trainer = TrainAlgo(
                self.all_args, 
                self.policy, 
                device = self.device
            )
        
        # buffer
        self.buffer = SharedReplayBuffer(
            self.all_args,
            self.num_agents,
            observation_space,
            share_observation_space,
            self.envs.action_space[0]
        )

    def run(self):
        """Collect training data, perform training updates, and evaluate policy."""
        raise NotImplementedError

    def warmup(self):
        """Collect warmup pre-training data."""
        raise NotImplementedError

    def collect(self, step):
        """Collect rollouts for training."""
        raise NotImplementedError

    def insert(self, data):
        """
        Insert data into buffer.
        :param data: (Tuple) data to insert into training buffer.
        """
        raise NotImplementedError
    
    @torch.no_grad()
    def compute(self):
        """Calculate returns for the collected data."""
        self.trainer.prep_rollout()
        if self.algorithm_name == "mat" or self.algorithm_name == "mat_dec":
            next_values = self.trainer.policy.get_values(np.concatenate(self.buffer.share_obs[-1]),
                                                        np.concatenate(self.buffer.obs[-1]),
                                                        np.concatenate(self.buffer.rnn_states_critic[-1]),
                                                        np.concatenate(self.buffer.masks[-1]))
        elif self.algorithm_name == "mast":
            next_values = self.trainer.policy.get_values(np.concatenate(self.buffer.share_obs[-1]),
                                                        np.concatenate(self.buffer.obs[-1]),
                                                        np.concatenate(self.buffer.rnn_states[-1]), 
                                                        np.concatenate(self.buffer.rnn_states_critic[-1]),
                                                        np.concatenate(self.buffer.masks[-1]))
        else:
            next_values = self.trainer.policy.get_values(np.concatenate(self.buffer.share_obs[-1]),
                                                        np.concatenate(self.buffer.rnn_states_critic[-1]),
                                                        np.concatenate(self.buffer.masks[-1]))
        next_values = np.array(np.split(_t2n(next_values), self.n_rollout_threads))
        self.buffer.compute_returns(next_values, self.trainer.value_normalizer)
    
    def train(self):
        """Train policies with data in buffer. """
        self.trainer.prep_training()
        train_infos = self.trainer.train(self.buffer)   
        self.buffer.after_update()   
        return train_infos

    def save(self, episode=0):
        """Save policy's actor and critic networks."""
        if self.algorithm_name == "mat" or self.algorithm_name == "mat_dec":
            self.policy.save(self.save_dir, episode)
        else:
            policy_actor = self.trainer.policy.actor
            torch.save(policy_actor.state_dict(), str(self.save_dir) + "/actor.pt")
            policy_critic = self.trainer.policy.critic
            torch.save(policy_critic.state_dict(), str(self.save_dir) + "/critic.pt")

    def restore(self, model_dir):
        """Restore policy's networks from a saved model."""
        if self.algorithm_name == "mat" or self.algorithm_name == "mat_dec":
            self.policy.restore(model_dir)
        else:
            policy_actor_state_dict = torch.load(str(self.model_dir) + '/actor.pt')
            self.policy.actor.load_state_dict(policy_actor_state_dict)
            if not self.all_args.use_render:
                policy_critic_state_dict = torch.load(str(self.model_dir) + '/critic.pt')
                self.policy.critic.load_state_dict(policy_critic_state_dict)

    def log_train(self, train_infos, total_num_steps):
        """
        Log training info.
        :param train_infos: (dict) information about training update.
        :param total_num_steps: (int) total number of training env steps.
        """
        for k, v in train_infos.items():
            if self.use_wandb:
                wandb.log({k: v}, step=total_num_steps)

    def log_env(self, env_infos, total_num_steps):
        """
        Log env info.
        :param env_infos: (dict) information about env state.
        :param total_num_steps: (int) total number of training env steps.
        """
        for k, v in env_infos.items():
            if len(v)>0:
                if self.use_wandb:
                    wandb.log({k: np.mean(v)}, step=total_num_steps)