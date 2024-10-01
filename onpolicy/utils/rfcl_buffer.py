import h5py

import numpy as np
from utils.shared_buffer import SharedReplayBuffer

class RFCL_Buffer(SharedReplayBuffer):
    def __init__(self, args, num_agents, obs_space, cent_obs_space, act_space):
        super().__init__(args, num_agents, obs_space, cent_obs_space, act_space)
        np.set_printoptions(threshold=np.inf)
        
        win_data_only = True
        self.num_demo_data = 2
        
        self.reverse_step_size = 4
        self.init_state_noiser = "geometric_noiser"
        
        """
        가공된 데이터를 일정 개수만큼 가져오는 코드가 필요함, 그리고 가져온 개수만큼의 curriculum level을 초기화.
        """
        with h5py.File("./gfootball_demo_level10.h5", 'r') as dataset:
            data = dataset["data"][:]
            label = dataset["rewards"][:]
            ava_data = dataset["available_actions"][:]

        if win_data_only:
            win_index = np.where(label == 1)[0]            
            data = data[win_index]
            ava_data = ava_data[win_index]
            label = label[win_index]
            
        # 승리 에피소드만 활용할 것인지 결정 
        if len(data) < self.num_demo_data:
            raise NotImplementedError(f"We need at least {self.num_demo_data} pieces of episode data, U have only {len(data)}.")
        
        # 임의의 개수의 demo를 가지는 데이터셋 생성
        demo_data_index = np.random.choice(len(data), size=self.num_demo_data, replace=False)
        self.demo_dataset = data[demo_data_index]
        self.demo_rewardset = label[demo_data_index]
        self.demo_avaset = ava_data[demo_data_index]
        
        self.progresses = np.zeros(self.num_demo_data)
        
        self.sampling_step = []
        for demo_data in self.demo_dataset:
            zero_states = np.where(np.all(demo_data == 0, axis=(1, 2)))[0]
            if len(zero_states) > 0:
                self.sampling_step.append(max(zero_states[0] - 100, 0))
            else:
                self.sampling_step.append(demo_data.shape[0])
        
    def update_sampling_prob(self):
        self.demo_sampling_prob = np.exp(-self.progresses) / np.sum(np.exp(-self.progresses)) 
        pass

    def geometric_noisesr(self):
        offset_value = np.random.geometric(0.4) - 1
        return offset_value
    
    def update_progress(self):
        pass 
    
    def init_rollout(self, idx, step, obs, ava):
        self.share_obs[step][idx] = obs
        self.obs[step][idx] = obs
        self.rnn_states[step][idx] = 0
        self.rnn_states_critic[step][idx] = 0
        self.available_actions[step][idx] = ava
        
    def sampling_demo(self, step, dones_env):
        num_done_episodes = np.sum(dones_env)
        self.update_sampling_prob()
        demo_index_list = np.random.choice(len(self.demo_sampling_prob), num_done_episodes, p=self.demo_sampling_prob)
        done_stack= 0
        
        for idx, done in enumerate(dones_env):
            if np.all(done):
                demo_index = demo_index_list[done_stack]
                noisy_init_state = self.sampling_step[demo_index] - self.geometric_noisesr()
            
                self.init_rollout(
                    idx = idx, 
                    step = step, 
                    obs = self.demo_dataset[demo_index][noisy_init_state],
                    ava = self.demo_avaset[demo_index][noisy_init_state]
                )
                done_stack += 1
                
    
