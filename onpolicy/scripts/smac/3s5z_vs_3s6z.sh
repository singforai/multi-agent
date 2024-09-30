#!/bin/sh
env="smac"
map="3s5z_vs_3s6z"
algo="mast" 
num_env_steps=20000000

for seed in 1 2 3 4 5;
do
    CUDA_VISIBLE_DEVICES=3 python ../../train_smac.py --env_name ${env} --algorithm_name ${algo} --experiment_name "mast_3s5z_vs_3s6z"\
    --map_name ${map} --seed ${seed} --num_env_steps ${num_env_steps} --n_head 4 --use_wandb --group_name ${algo} --save_model
done