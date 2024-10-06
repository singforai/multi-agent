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
        
        self.progresses = np.zeros(self.num_demo_data)
        
        self.sampling_step = []
        for demo_data in self.demo_state_set:
            zero_states = np.where(np.all(demo_data == 0, axis=(1, 2)))[0]
            if len(zero_states) > 0:
                self.sampling_step.append(zero_states[0] - self.reverse_step_size)
            else:
                self.sampling_step.append(demo_data.shape[0])
        self.update_sampling_prob()
        
        self.level_up_rate = self.reverse_step_size / np.array(self.sampling_step)
        
        
    def update_sampling_prob(self):
        self.demo_sampling_prob = np.exp(-self.progresses) / np.sum(np.exp(-self.progresses)) 
        pass

    def geometric_noisesr(self):
        offset_value = np.random.geometric(self.geometric_prob) - 1
        return offset_value

    def update_progress(self, win_rate):
        update = False
        if np.mean(win_rate) >= self.level_up_condition:
            self.sampling_step[self.demo_index] = max(self.sampling_step[self.demo_index] - self.reverse_step_size, 0)
            self.progresses[self.demo_index] = min(self.progresses[self.demo_index] + self.level_up_rate[self.demo_index], 100)
            self.backward_progress = np.mean(self.progresses)
            self.update_sampling_prob()
            update = True
        print(f"backward_progress: {self.progresses}")
        print(f"result: {win_rate}")
        return update, self.backward_progress, np.mean(win_rate)
        
    def sampling_demo(self, file_path):

        self.demo_index = np.random.choice(len(self.demo_sampling_prob), p=self.demo_sampling_prob)
        noisy_init_state = max(self.sampling_step[self.demo_index] - self.geometric_noisesr(), 0)
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