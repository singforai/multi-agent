import numpy as np
import torch
class Rewarder:
    def __init__(self) -> None:
        self.player_last_hold_ball = -1
    def calc_reward(self, rew, prev_obs, obs):
        if obs["ball_owned_team"] == 0:
            self.player_last_hold_ball = obs["ball_owned_player"]
        reward = (
            + win_reward(obs)
            + preprocess_score(rew)
            + ball_position_reward(obs, self.player_last_hold_ball)
            + yellow_reward(prev_obs, obs)
            + oob_reward(obs)
            # + passing_reward(prev_obs, obs)
        )
        return reward
def preprocess_score(rew_signal):
    if rew_signal > 0:
        return 5.0 * rew_signal
    else:
        return rew_signal
def win_reward(obs):
    win_reward = 0.0
    # print(f"steps left: {obs['steps_left']}")
    if obs["steps_left"] == 0:
        # print("STEPS LEFT == 0!")
        [my_score, opponent_score] = obs["score"]
        if my_score > opponent_score:
            win_reward = my_score - opponent_score
    return 5.0 * win_reward
def oob_reward(obs):
    if obs["game_mode"] == 0:
        left_team_position_x = obs["left_team"][1:][:, 0]
        left_team_position_y = obs["left_team"][1:][:, 1]
        out_of_range_count = np.sum((left_team_position_x < -1.0) | (left_team_position_x > 1.0)) + np.sum((left_team_position_y < -0.42) | (left_team_position_y > 0.42))
    else:
        out_of_range_count = 0
    return - 0.003 * out_of_range_count
def passing_reward(prev_obs, obs):
    if prev_obs["ball_owned_team"] == 0 and obs["ball_owned_team"] == 0:
        if prev_obs["ball_owned_player"] != obs["ball_owned_player"]:
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
def ball_position_reward(obs, player_last_hold_ball):
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