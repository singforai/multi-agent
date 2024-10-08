import numpy as np
import torch
class Rewarder:
    def __init__(self) -> None:
        self.player_last_hold_ball = -1
    def calc_reward(self, rew, prev_obs, obs):
        
        if obs["ball_owned_team"] == 0:
            self.player_last_hold_ball = obs["ball_owned_player"]
            
        ball_r = ball_position_reward(obs)
        poss_r = possession_reward(prev_obs, obs)
        oob_r = oob_reward(obs)
        pass_r = passing_reward(prev_obs, obs)
        reward = (
            + preprocess_score(rew)
            + ball_r
            + yellow_reward(prev_obs, obs)
            + oob_r
            + pass_r
            + poss_r
        )
        return reward, oob_r, pass_r, ball_r + poss_r
    
def possession_reward(prev_obs, obs):
    if obs["game_mode"] == 0 and prev_obs["game_mode"] == 0:
        if prev_obs["ball_owned_team"] == 0 and obs["ball_owned_team"] == 1:
            return -0.1
        if prev_obs["ball_owned_team"] == 1 and obs["ball_owned_team"] == 0:
            return +0.3
    return 0
    
def preprocess_score(rew_signal):
    if rew_signal > 0:
        return 5.0 * rew_signal
    else:
        return rew_signal

def oob_reward(obs):
    if obs["game_mode"] == 0:
        oob_player= 0
        for x_pos, y_pos in obs["left_team"][1:]:
            if x_pos <= -1 or x_pos >= 0.9 or y_pos <= -0.42 or y_pos >= 0.42:
                oob_player += 1
    else:
        oob_player= 0
    return - 0.02 * oob_player

def passing_reward(prev_obs, obs):
    if prev_obs["game_mode"] == 0 and obs["game_mode"] == 0:
        if prev_obs["ball_owned_team"] == 0 and obs["ball_owned_team"] == 0:
            if prev_obs["ball_owned_player"] != obs["ball_owned_player"]:
                if prev_obs["ball"][0] <= obs["ball"][0]:
                    reward = 0.2
                else:
                    reward = 0.05
            else:
                reward = 0.0
        else:
            reward = 0.0
    else:
        reward = 0.0
    return reward

def yellow_reward(prev_obs, obs):
    left_yellow = np.sum(obs["left_team_yellow_card"]) - np.sum(
        prev_obs["left_team_yellow_card"]
    )
    right_yellow = np.sum(obs["right_team_yellow_card"]) - np.sum(
        prev_obs["right_team_yellow_card"]
    )
    yellow_r = right_yellow - left_yellow
    return 0.05 * yellow_r

def ball_position_reward(obs):
    ball_x, ball_y, ball_z = obs["ball"]
    MIDDLE_X, PENALTY_X, END_X = 0.2, 0.64, 1.0
    PENALTY_Y, END_Y = 0.27, 0.42
    ball_position_r = 0.0
    if (-END_X <= ball_x and ball_x < -PENALTY_X) and (
        -PENALTY_Y < ball_y and ball_y < PENALTY_Y
    ):
        ball_position_r = -1.0
    elif (-END_X <= ball_x and ball_x < -MIDDLE_X) and (
        -END_Y < ball_y and ball_y < END_Y
    ):
        ball_position_r = -0.5
    elif (-MIDDLE_X <= ball_x and ball_x <= MIDDLE_X) and (
        -END_Y < ball_y and ball_y < END_Y
    ):
        ball_position_r = 0.0
    elif (PENALTY_X < ball_x and ball_x <= END_X) and (
        -PENALTY_Y < ball_y and ball_y < PENALTY_Y
    ):
        ball_position_r = 1.0
    elif (MIDDLE_X < ball_x and ball_x <= END_X) and (
        -END_Y < ball_y and ball_y < END_Y
    ):
        ball_position_r = 0.5
    else:
        ball_position_r = 0.0
    return 0.002 * ball_position_r