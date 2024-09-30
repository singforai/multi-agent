#!/bin/sh
env="smac"
map="25m"
algo="mast" 
num_env_steps=2000000

for seed in 1 2 3 4 5;
do
    CUDA_VISIBLE_DEVICES=3 python ../../train_smac.py --env_name ${env} --algorithm_name ${algo} --experiment_name "mast_25m"\
    --map_name ${map} --seed ${seed} --num_env_steps ${num_env_steps} --n_head 4 --use_wandb --group_name mast_005 --save_model --ppo_epoch 15 --clip_param 0.05
done