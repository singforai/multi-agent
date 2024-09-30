#!/bin/sh
env="football"
map="academy_counterattack_easy"
algo="mast" 
num_env_steps=10000000

parallel -j 5 "python ../../train_grf.py --env_name ${env} --algorithm_name ${algo} --experiment_name mast_academy_counterattack_easy\
 --scenario_name ${map} --seed {1} --num_env_steps ${num_env_steps} --n_head 4 --use_wandb --group_name ${algo} --save_model  --num_gpu 0 --num_agents 10" ::: 1 2 3 4 5