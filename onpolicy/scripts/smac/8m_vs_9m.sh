#!/bin/sh
env="smac"
map="8m_vs_9m"
algo="mast" 
num_env_steps=5000000

for seed in 433 244 324 678 563;
do
    CUDA_VISIBLE_DEVICES=1 python ../../train_smac.py --env_name ${env} --algorithm_name ${algo} --experiment_name "mast_8m_vs_9m"\
    --map_name ${map} --seed ${seed} --num_env_steps ${num_env_steps} --n_head 4 --use_wandb --group_name mast_00515 --save_model --ppo_epoch 15 --clip_param 0.05
done