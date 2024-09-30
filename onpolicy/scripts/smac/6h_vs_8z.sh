#!/bin/sh
env="smac"
map="6h_vs_8z"
algo="mast" 
num_env_steps=10000000

for seed in 1 2 3 4 5;
do
    CUDA_VISIBLE_DEVICES=1 python ../../train_smac.py --env_name ${env} --algorithm_name ${algo} --experiment_name "mast_6h_vs_8z"\
    --map_name ${map} --seed ${seed} --num_env_steps ${num_env_steps} --n_head 4 --use_wandb --group_name ${algo} --save_model
done