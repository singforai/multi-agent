from collections import defaultdict, deque
from itertools import chain
import os
import time
import json
import shutil
import imageio
import numpy as np
import torch
import wandb

from utils.util import update_linear_schedule
from runner.shared.base_runner import Runner

def _t2n(x):
    return x.detach().cpu().numpy()

class FootballRunner(Runner):
    def __init__(self, config):
        super(FootballRunner, self).__init__(config)
        self.env_infos = defaultdict(list)

    def warmup(self, file_path):
        
        # reset env
        self.buffer.sampling_demo(file_path)
        obs, share_obs, ava = self.envs.reset()
        if not self.use_centralized_V:
            share_obs = obs

        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()
        self.buffer.available_actions[0] = ava.copy()
    
    def run(self, file_path):
        
        backward_progress = 0.0
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        train_episode_rewards = [0 for _ in range(self.n_rollout_threads)]
        done_episodes_rewards = []
        train_episode_scores = [0 for _ in range(self.n_rollout_threads)]
        done_episodes_scores = []
        
        self.result = []
        episode = 0
        total_num_steps = 0
        
        outofbound_rewards = []
        passing_rewards = []
        ball_pos_rewards = []
        """
        backward learning: backward sampling이 initial state에 도달할 때까지 실행
        """
        while episodes > episode or backward_progress > 0.9:
            self.warmup(file_path = file_path)
            
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)
                
            for step in range(self.episode_length):
                
                values, actions, action_log_probs, rnn_states, rnn_states_critic = self.collect(step)
                obs, share_obs, rewards, dones, infos, available_actions, oob_rewards, pass_rewards, ball_rewards\
                    = self.envs.step(actions)
                dones_env = np.all(dones, axis=1)
                reward_env = np.mean(rewards, axis=1).flatten()
                train_episode_rewards += reward_env
                outofbound_rewards.append(np.mean(oob_rewards.reshape(-1)))
                passing_rewards.append(np.mean(pass_rewards.reshape(-1)))
                ball_pos_rewards.append(np.mean(ball_rewards.reshape(-1)))
                
                score_env = [t_info[0]["score_reward"] for t_info in infos]
                train_episode_scores += np.array(score_env)
                for t in range(self.n_rollout_threads):
                    if dones_env[t]:
                        done_episodes_rewards.append(train_episode_rewards[t])
                        train_episode_rewards[t] = 0
                        done_episodes_scores.append(train_episode_scores[t])
                        self.result.append(train_episode_scores[t])
                        train_episode_scores[t] = 0

                data = obs, share_obs, rewards, dones, infos, available_actions, \
                       values, actions, action_log_probs, \
                       rnn_states, rnn_states_critic
                self.insert(data)

            episode += 1
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            self.compute()
            train_infos = self.train()
            
            backward_progress_rate, mean_win_rate, dis2goal = self.buffer.update_progress(
                win_rate = [0.0 if x < 0 else x for x in self.result]
            )
            train_infos["dis2goal"] = dis2goal
            train_infos["win_rate"] = mean_win_rate
            train_infos["backward_progress"] = backward_progress_rate
            train_infos["outofbound_rewards"] = np.mean(outofbound_rewards)
            train_infos["passing_rewards"] = np.mean(passing_rewards)
            train_infos["ball_pos_rewards"] = np.mean(ball_pos_rewards)
            self.result = []
                
            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}.\n"
                        .format(self.all_args.scenario_name,
                                self.algorithm_name,
                                self.experiment_name,
                                episode,
                                episodes,
                                total_num_steps,
                                self.num_env_steps))

                self.log_train(train_infos, total_num_steps)

                if len(done_episodes_rewards) > 0:
                    aver_episode_rewards = np.mean(done_episodes_rewards)
                    aver_episode_scores = np.mean(done_episodes_scores)
                    done_episodes_rewards = []
                    done_episodes_scores = []
                    print("some episodes done, average rewards: {}, scores: {}"
                          .format(aver_episode_rewards, aver_episode_scores))

                    if self.use_wandb:
                        wandb.log({"train_avg_rewards": aver_episode_rewards}, step=total_num_steps)
                        wandb.log({"train_avg_score": aver_episode_scores}, step=total_num_steps)


            if (episode % self.save_interval) == 0 and self.save_model and episode != 0:
                self.save(episode = episode)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.reverse_eval(total_num_steps, episode)
                


    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        # [n_envs, n_agents, ...] -> [n_envs*n_agents, ...]
        values, actions, action_log_probs, rnn_states, rnn_states_critic = self.trainer.policy.get_actions(
            np.concatenate(self.buffer.share_obs[step]),
            np.concatenate(self.buffer.obs[step]),
            np.concatenate(self.buffer.rnn_states[step]),
            np.concatenate(self.buffer.rnn_states_critic[step]),
            np.concatenate(self.buffer.masks[step]),
            np.concatenate(self.buffer.available_actions[step])
        )

        # [n_envs*n_agents, ...] -> [n_envs, n_agents, ...]
        values = np.array(np.split(_t2n(values), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(actions), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_probs), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))
        return values, actions, action_log_probs, rnn_states, rnn_states_critic

    def insert(self, data):
        obs, share_obs, rewards, dones, infos, available_actions, \
        values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        dones_env = np.all(dones, axis=1)
        
        rnn_states[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, *self.buffer.rnn_states_critic.shape[3:]), dtype=np.float32)

        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        active_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        active_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        active_masks[dones_env == True] = np.ones(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        # bad_masks = np.array([[[0.0] if info[agent_id]['bad_transition'] else [1.0] for agent_id in range(self.num_agents)] for info in infos])

        if not self.use_centralized_V:
            share_obs = obs

        self.buffer.insert(
            share_obs = share_obs, 
            obs = obs, 
            rnn_states_actor = rnn_states, 
            rnn_states_critic = rnn_states_critic,
            actions = actions, 
            action_log_probs = action_log_probs, 
            value_preds = values, 
            rewards = rewards, 
            masks = masks, 
            bad_masks = None, 
            active_masks = active_masks,
            available_actions = available_actions
        )

    def log_env(self, env_infos, total_num_steps):
        for k, v in env_infos.items():
            if len(v) > 0:
                if self.use_wandb:
                    wandb.log({k: np.mean(v)}, step=total_num_steps)

    def supervisor(self, win_rate, episode):
        with open(self.file_path, 'r') as json_file:
            data = json.load(json_file)
        if np.mean(win_rate) >= 0.6:
            data["level_stack"] += 1
            if data["level_stack"] >= 10:
                data["difficulty_level"] += 1
                self.save(episode = episode)
                data["level_stack"] = 0
        with open(self.file_path, 'w') as json_file:
            json.dump(data, json_file, indent=4)
            
        return data["difficulty_level"], data["level_stack"]

    @torch.no_grad()
    def reverse_eval(self, total_num_steps, episode):
        eval_episode = 0
        eval_episode_rewards = []
        one_episode_rewards = [0 for _ in range(self.all_args.eval_episodes)]
        eval_episode_scores = []
        one_episode_scores = [0 for _ in range(self.all_args.eval_episodes)]

        eval_obs, eval_share_obs, ava = self.eval_envs.reset()
        eval_rnn_states = np.zeros((self.all_args.eval_episodes, self.num_agents, self.recurrent_N,
                                    self.hidden_size), dtype=np.float32)
        eval_masks = np.ones((self.all_args.eval_episodes, self.num_agents, 1), dtype=np.float32)

        while True:
            self.trainer.prep_rollout()
            eval_actions, eval_rnn_states = \
                self.trainer.policy.act(np.concatenate(eval_share_obs),
                                        np.concatenate(eval_obs),
                                        np.concatenate(eval_rnn_states),
                                        np.concatenate(eval_masks),
                                        np.concatenate(ava),
                                        deterministic=False)
                
            eval_actions = np.array(np.split(_t2n(eval_actions), self.all_args.eval_episodes))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.all_args.eval_episodes))

            # Obser reward and next obs
            eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, ava, _, _, _ \
                = self.eval_envs.step(eval_actions)
            eval_rewards = np.mean(eval_rewards, axis=1).flatten()
            one_episode_rewards += eval_rewards

            eval_scores = [t_info[0]["score_reward"] for t_info in eval_infos]
            one_episode_scores += np.array(eval_scores)

            eval_dones_env = np.all(eval_dones, axis=1)
            eval_rnn_states[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents,
                                                                self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.all_args.eval_episodes, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, 1),
                                                          dtype=np.float32)

            for eval_i in range(self.all_args.eval_episodes):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    eval_episode_rewards.append(one_episode_rewards[eval_i])
                    one_episode_rewards[eval_i] = 0

                    eval_episode_scores.append(one_episode_scores[eval_i])
                    one_episode_scores[eval_i] = 0

            if eval_episode >= self.all_args.eval_episodes:
                key_average = '/eval_average_episode_rewards'
                key_max = '/eval_max_episode_rewards'
                key_scores = '/eval_average_episode_scores'
                eval_env_infos = {key_average: eval_episode_rewards,
                                  key_max: [np.max(eval_episode_rewards)],
                                  key_scores: eval_episode_scores}
                
                if "curriculum_learning" in self.all_args.scenario_name:
                    difficulty_level, level_stack = self.supervisor(win_rate= eval_episode_scores, episode = episode)
                    eval_env_infos["difficulty_level"] = [difficulty_level for _ in range(self.all_args.eval_episodes)]
                    eval_env_infos["level_stack"] = [level_stack for _ in range(self.all_args.eval_episodes)] 

                self.log_env(eval_env_infos, total_num_steps)

                print("eval average episode rewards: {}, scores: {}."
                      .format(np.mean(eval_episode_rewards), np.mean(eval_episode_scores)))
                break