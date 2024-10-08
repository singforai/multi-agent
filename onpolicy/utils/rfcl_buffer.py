import os 

import h5py
import json
import numpy as np
from utils.shared_buffer import SharedReplayBuffer

class RFCL_Buffer(SharedReplayBuffer):
    def __init__(self, args, num_agents, obs_space, cent_obs_space, act_space):
        super().__init__(args, num_agents, obs_space, cent_obs_space, act_space)
        np.set_printoptions(threshold=np.inf)
        
        win_data_only = True
        self.num_demo_data = 5 # ****************
        self.geometric_prob = 0.5
        self.reverse_step_size = 2
        self.backward_progress = 0.0
        
        self.level_up_condition = 0.8 # ****************

        """
        가공된 데이터를 일정 개수만큼 가져오는 코드가 필요함, 그리고 가져온 개수만큼의 curriculum level을 초기화.
        """
        home_directory = os.path.expanduser("~")
        file_path = os.path.join(home_directory, 'level', 'gfootball_demo_level10.h5')
        with h5py.File(file_path, 'r') as dataset:
            state_data = dataset["state"][:]
            label = dataset["rewards"][:]
            ava_data = dataset["available_actions"][:]
            loc_data = dataset["location"][:]

        if win_data_only:
            win_index = np.where(label == 1)[0]            
            state_data = state_data[win_index]
            ava_data = ava_data[win_index]
            loc_data = loc_data[win_index]
            label = label[win_index]
            
        # 승리 에피소드만 활용할 것인지 결정 
        if len(state_data) < self.num_demo_data:
            raise NotImplementedError(f"We need at least {self.num_demo_data} pieces of episode data, U have only {len(state_data)} data..")
        
        # 임의의 개수의 demo를 가지는 데이터셋 생성
        demo_data_index = np.random.choice(len(state_data), size=self.num_demo_data, replace=False)
        self.demo_state_set = state_data[demo_data_index]
        self.demo_reward_set = label[demo_data_index]
        self.demo_ava_set = ava_data[demo_data_index]
        self.demo_loc_set = loc_data[demo_data_index]
        
        self.sampling_step = []
        for idx, demo_data in enumerate(self.demo_state_set):
            zero_states = np.where(np.all(demo_data == 0, axis=(1, 2)))[0]
            if len(zero_states) > 0:
                chosen_step = zero_states[0] - self.reverse_step_size
            else:
                chosen_step = demo_data.shape[0]
            self.sampling_step.append(chosen_step)
            
        self.demo_total_steps = np.array(self.sampling_step)
        self.progresses_step = np.zeros(self.num_demo_data)
    
    def ball_goal_dis(self):
        
        dis2goal = []
        steps =  np.array(self.demo_total_steps - self.progresses_step, dtype = int)
        for idx, data in enumerate(self.demo_loc_set):
            dis2goal.append(np.sqrt((data[steps[idx]][22][0] - 1) ** 2 + (data[steps[idx]][22][1] - 0) ** 2))
        return dis2goal

    def cal_progress_rate(self):
        return self.progresses_step/self.demo_total_steps

    def update_sampling_prob(self):
        x_margin = -np.array(self.ball_goal_dis())
        self.demo_sampling_prob = np.exp(x_margin) / np.sum(np.exp(x_margin)) 

    def geometric_noisesr(self):
        offset_value = np.random.geometric(self.geometric_prob) - 1
        return offset_value

    def update_progress(self, win_rate):
        if np.mean(win_rate) >= self.level_up_condition:
            self.update_demo_step(demo_index=self.demo_index)
        dis2goal = self.ball_goal_dis()
        backward_progress_rates = self.cal_progress_rate()
        _backward_progress_rate = np.mean(backward_progress_rates)
        print("="*20)
        print(f"demo_idx: {self.demo_index}")
        print(f"backward_progress: {backward_progress_rates}")
        print(f"result: {win_rate}")
        print(f"dis2goal: {dis2goal}")
        print("="*20)
        return _backward_progress_rate, np.mean(win_rate) , np.mean(dis2goal)

    def update_demo_step(self, demo_index):
        self.progresses_step[demo_index] = \
            min(
                self.progresses_step[demo_index] + self.reverse_step_size, \
                    self.demo_total_steps[demo_index]
        )
        
    def sampling_demo(self, file_path):
        if np.mean(self.ball_goal_dis()) < 1.0: 
            self.update_sampling_prob()
            self.demo_index = np.random.choice(self.num_demo_data, p=self.demo_sampling_prob)
            demo_init_step = self.demo_total_steps[self.demo_index] \
                - (self.progresses_step[self.demo_index] + self.geometric_noisesr())
            noisy_init_state = int(max(demo_init_step, 0))
            player_location = self.demo_loc_set[self.demo_index][noisy_init_state]
            # observation = self.demo_state_set[self.demo_index][noisy_init_state]
            # available_action = self.demo_ava_set[self.demo_index][noisy_init_state]
    
            with open(file_path, 'r') as json_file:
                data = json.load(json_file)
            data = {
                "left_team": player_location[:11].tolist(),
                "right_team": player_location[11: 22].tolist(),
                "ball": player_location[22].tolist()
                
            }
            with open(file_path, 'w') as json_file:
                json.dump(data, json_file, indent=4)
                
        else:
            with open(file_path, 'r') as json_file:
                data = json.load(json_file)
            player_location = {
                "left_team": [
                    [-1.000000, 0.000000],
                    [0.000000,  0.020000], 
                    [0.000000, -0.020000], 
                    [-0.422000, -0.19576], 
                    [-0.500000, -0.06356], 
                    [-0.500000, 0.063559], 
                    [-0.422000, 0.195760], 
                    [-0.184212, -0.10568], 
                    [-0.267574, 0.000000], 
                    [-0.184212, 0.105680], 
                    [-0.010000, -0.21610], 
                ],
                "right_team": [
                    [-1.000000, 0.000000],
                    [-0.050000, 0.000000],
                    [-0.010000, 0.216102],
                    [-0.422000, -0.19576],
                    [-0.500000, -0.06356],
                    [-0.500000, 0.063559],
                    [-0.422000, 0.195760],
                    [-0.184212, -0.10568],
                    [-0.267574, 0.000000],
                    [-0.184212, 0.105680],
                    [-0.010000, -0.21610],
                ],
                "ball" : [0.000000, 0.00000]
            }
            with open(file_path, 'w') as json_file:
                json.dump(data, json_file, indent=4)
        