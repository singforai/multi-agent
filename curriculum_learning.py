# coding=utf-8
# Copyright 2019 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import json 

import numpy as np

from . import *

def build_scenario(builder):
    current_file_path = __file__
    time_code = os.path.basename(current_file_path).replace(".py", "").split('_')[-1]
    
    file_path = f"./level/level_{time_code}.json"
    
    if os.path.exists(file_path):
        with open(file_path, "r") as json_file:
            level = json.load(json_file)
    else:
        raise ValueError(f"Error: [level_{time_code}.json] does not exist.")
    
    difficulty_level = min(10, level["difficulty_level"])
    builder.config().end_episode_on_score = True
    builder.config().game_duration = level["game_length"]
    builder.config().left_team_difficulty = 1.0
    builder.config().right_team_difficulty = 1.0
    builder.config().deterministic = False

    first_team = Team.e_Left
    second_team = Team.e_Right

    builder.SetTeam(first_team)
    builder.AddPlayer(-1.000000, 0.000000, e_PlayerRole_GK, controllable=False)

    right_margin = 1 - difficulty_level * 0.1
    left_margin = right_margin - 0.5
    x_pos = list(np.random.random(10) * 0.5 + left_margin)
    y_pos = np.random.normal(loc=0.0, scale=0.1, size=10)
    y_pos = list(np.clip(y_pos, -0.3, 0.3))
    # print(x_pos)
    # print(y_pos)
    player_has_ball = np.random.randint(10)
    builder.SetBallPosition(x_pos[player_has_ball], y_pos[player_has_ball])

    builder.AddPlayer(x_pos[0], y_pos[0], e_PlayerRole_RM)
    builder.AddPlayer(x_pos[1], y_pos[1], e_PlayerRole_CF)
    builder.AddPlayer(x_pos[2], y_pos[2], e_PlayerRole_LB)
    builder.AddPlayer(x_pos[3], y_pos[3], e_PlayerRole_CB)
    builder.AddPlayer(x_pos[4], y_pos[4], e_PlayerRole_CB)
    builder.AddPlayer(x_pos[5], y_pos[5], e_PlayerRole_RB)
    builder.AddPlayer(x_pos[6], y_pos[6], e_PlayerRole_CM)
    builder.AddPlayer(x_pos[7], y_pos[7], e_PlayerRole_CM)
    builder.AddPlayer(x_pos[8], y_pos[8], e_PlayerRole_CM)
    builder.AddPlayer(x_pos[9], y_pos[9], e_PlayerRole_LM)

    builder.SetTeam(second_team)
    builder.AddPlayer(-1.000000, 0.000000, e_PlayerRole_GK, controllable=False)

    builder.AddPlayer(-0.9000000, 0.100000, e_PlayerRole_RM)
    builder.AddPlayer(-0.900000, -0.100000, e_PlayerRole_CF)

    builder.AddPlayer(x_pos[2], y_pos[2], e_PlayerRole_LB)
    builder.AddPlayer(x_pos[3], y_pos[3], e_PlayerRole_CB)
    builder.AddPlayer(x_pos[4], y_pos[4], e_PlayerRole_CB)
    builder.AddPlayer(x_pos[5], y_pos[5], e_PlayerRole_RB)
    builder.AddPlayer(x_pos[6], y_pos[6], e_PlayerRole_CM)
    builder.AddPlayer(x_pos[7], y_pos[7], e_PlayerRole_CM)
    builder.AddPlayer(x_pos[8], y_pos[8], e_PlayerRole_CM)
    builder.AddPlayer(x_pos[9], y_pos[9], e_PlayerRole_LM)
