#!/bin/bash

python ../../train_grf.py --use_wandb\
    --num_env_steps 100000000 \
    --n_rollout_threads 10 \
    --n_eval_rollout_threads 10 \
    --eval_interval 15 \
    --eval_episodes 10 \
    --clip_param 0.2 \
    --ppo_epoch 5 \
    --episode_length 500 \
    --algorithm_name mast \
    --n_head 4 \
    --save_model \
    --save_interval 300 \
    --experiment_name backward-final \
    --num_gpu 1 \
    --group_name rfcl \