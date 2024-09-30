#!/bin/sh
env="football"
map="academy_3_vs_1_with_keeper"
algo="mast" 
num_env_steps=5000000

parallel -j 5 "python ../../train_grf.py --env_name ${env} --algorithm_name ${algo} --experiment_name mast_academy_3_vs_1_with_keeper\
 --scenario_name ${map} --seed {1} --num_env_steps ${num_env_steps} --n_head 4 --use_wandb --group_name ${algo}+"origin" --save_model  --num_gpu 1 --num_agents 3" ::: 1 2 3 4 5