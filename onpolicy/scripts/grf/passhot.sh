#!/bin/sh
env="football"
map="academy_pass_and_shoot_with_keeper"
algo="mast" 
num_env_steps=5000000

parallel -j 5 "python ../../train_grf.py --env_name ${env} --algorithm_name ${algo} --experiment_name mast_academy_pass_and_shoot_with_keeper\
 --scenario_name ${map} --seed {1} --num_env_steps ${num_env_steps} --n_head 4 --use_wandb --group_name ${algo} --save_model  --num_gpu 1 --num_agents 2\
 --n_rollout_threads 20 --n_eval_rollout_threads 20 --eval_episodes 20 --eval_interval 10 --ppo_epoch 10 --clip_param 0.05 --episode_length 200" ::: 433 244 324 678 563
