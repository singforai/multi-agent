import argparse


def get_config():

    parser = argparse.ArgumentParser(
        description='MARL', formatter_class=argparse.RawDescriptionHelpFormatter)
    
    function = parser.add_argument
    
    function(
        "--use_rfcl",
        action="store_false",
        default=True,
    )

    function(
        "--use_pma_block",
        action="store_false",
        default=True,
        help="whether to use pwa block for central state-value",
    )
    function(
        "--use_additional_obs",
        action="store_false",
        default=True,
        help="whether to use additional obs in football env",
    )
    
    #Tizero algorithms 
    function(
        "--use_joint_action_loss",
        action="store_true",
        default=False,
        help="whether to use joint action loss for mast",
    )
    
    function("--num_gpu", type=int, default=0, help="사용할 gpu number")
    function(
        "--group_name",
        type=str,
        default="test",
        help="[for wandb usage], to specify user's name for simply collecting training data.",
    )
    
    # prepare parameters
    function("--algorithm_name", type=str,
                        default='rmappo', choices=["rmappo", "mappo", "happo", "hatrpo", "mat", "mat_dec", "mast", "tizero"])

    function("--experiment_name", type=str, default="check", help="an identifier to distinguish different experiment.")
    function("--seed", type=int, default=1, help="Random seed for numpy/torch")
    function("--cuda", action='store_false', default=True, help="by default True, will use GPU to train; or else will use CPU;")
    function("--cuda_deterministic",
                        action='store_false', default=True, help="by default, make sure random seed effective. if set, bypass such function.")
    function("--n_torch_threads", type=int,
                        default=16, help="Number of torch threads for training")
    function("--n_rollout_threads", type=int, default=32,
                        help="Number of parallel envs for training rollouts")
    function("--n_eval_rollout_threads", type=int, default=32,
                        help="Number of parallel envs for evaluating rollouts")
    function("--n_render_rollout_threads", type=int, default=1,
                        help="Number of parallel envs for rendering rollouts")
    function("--num_env_steps", type=int, default=10e6,
                        help='Number of environment steps to train (default: 10e6)')
    function("--user_name", type=str, default='singfor7012', help="[for wandb usage], to specify user's name for simply collecting training data.")
    function("--use_wandb", action='store_true', default=False, help="[for wandb usage], by default True, will log date to wandb server")

    # env parameters
    function("--env_name", type=str, default='smac', choices = ["smac", "football"], help="specify the name of environment")
    function("--use_obs_instead_of_state", action='store_true',
                        default=False, help="Whether to use global state or concatenated obs")

    # replay buffer parameters
    function("--episode_length", type=int,
                        default=100, help="Max length for any episode")
    # network parameters
    function("--share_policy", action='store_false',
                        default=True, help='Whether agent share the same policy')
    function("--use_centralized_V", action='store_false',
                        default=True, help="Whether to use centralized V function")
    function("--stacked_frames", type=int, default=1,
                        help="Dimension of hidden layers for actor/critic networks")
    function("--use_stacked_frames", action='store_true',
                        default=False, help="Whether to use stacked_frames")
    function("--hidden_size", type=int, default=64,
                        help="Dimension of hidden layers for actor/critic networks") 
    function("--layer_N", type=int, default=1,
                        help="Number of layers for actor/critic networks")
    function("--use_ReLU", action='store_false',
                        default=True, help="Whether to use ReLU")
    function("--use_popart", action='store_true', default=False, help="by default False, use PopArt to normalize rewards.")
    function("--use_valuenorm", action='store_false', default=True, help="by default True, use running mean and std to normalize rewards.")
    function("--use_feature_normalization", action='store_true',
                        default=False, help="Whether to apply layernorm to the inputs")
    function("--use_orthogonal", action='store_false', default=True,
                        help="Whether to use Orthogonal initialization for weights and 0 initialization for biases")
    function("--gain", type=float, default=0.01,
                        help="The gain # of last action layer")

    # recurrent parameters
    function("--use_naive_recurrent_policy", action='store_true',
                        default=False, help='Whether to use a naive recurrent policy')
    function("--use_recurrent_policy", action='store_false',
                        default=True, help='use a recurrent policy')
    function("--recurrent_N", type=int, default=1, help="The number of recurrent layers.")
    function("--data_chunk_length", type=int, default=10,
                        help="Time length of chunks used to train a recurrent_policy")

    # optimizer parameters
    function("--lr", type=float, default=5e-4,
                        help='learning rate (default: 5e-4)')
    function("--critic_lr", type=float, default=5e-4,
                        help='critic learning rate (default: 5e-4)')
    function("--opti_eps", type=float, default=1e-5,
                        help='RMSprop optimizer epsilon (default: 1e-5)')
    function("--weight_decay", type=float, default=0)

    # trpo parameters
    function("--kl_threshold", type=float, 
                        default=0.01, help='the threshold of kl-divergence (default: 0.01)')
    function("--ls_step", type=int, 
                        default=10, help='number of line search (default: 10)')
    function("--accept_ratio", type=float, 
                        default=0.5, help='accept ratio of loss improve (default: 0.5)')

    # ppo parameters
    function("--ppo_epoch", type=int, default=15,
                        help='number of ppo epochs (default: 15)')
    function("--use_clipped_value_loss",
                        action='store_false', default=True, help="by default, clip loss value. If set, do not clip loss value.")
    function("--clip_param", type=float, default=0.2,
                        help='ppo clip parameter (default: 0.2)')
    function("--num_mini_batch", type=int, default=1,
                        help='number of batches for ppo (default: 1)')
    function("--entropy_coef", type=float, default=0.01,
                        help='entropy term coefficient (default: 0.01)')
    function("--value_loss_coef", type=float,
                        default=1, help='value loss coefficient (default: 0.5)')
    function("--use_max_grad_norm",
                        action='store_false', default=True, help="by default, use max norm of gradients. If set, do not use.")
    function("--max_grad_norm", type=float, default=10.0,
                        help='max norm of gradients (default: 0.5)')
    function("--use_gae", action='store_false',
                        default=True, help='use generalized advantage estimation')
    function("--gamma", type=float, default=0.99,
                        help='discount factor for rewards (default: 0.99)')
    function("--gae_lambda", type=float, default=0.95,
                        help='gae lambda parameter (default: 0.95)')
    function("--use_proper_time_limits", action='store_true',
                        default=False, help='compute returns taking into account time limits')
    function("--use_huber_loss", action='store_false', default=True, help="by default, use huber loss. If set, do not use huber loss.")
    function("--use_value_active_masks",
                        action='store_false', default=True, help="by default True, whether to mask useless data in value loss.")
    function("--use_policy_active_masks",
                        action='store_false', default=True, help="by default True, whether to mask useless data in policy loss.")
    function("--huber_delta", type=float, default=10.0, help=" coefficience of huber loss.")

    # run parameters
    function("--use_linear_lr_decay", action='store_true',
                        default=False, help='use a linear schedule on the learning rate')
    # save parameters
    function("--save_model", action = "store_true", default=False, help="time duration between contiunous twice models saving.")
    function("--save_interval", type=int, default=500, help="time duration between contiunous twice models saving.")

    # log parameters
    function("--log_interval", type=int, default=1, help="time duration between contiunous twice log printing.")

    # eval parameters
    function("--use_eval", action='store_false', default=True, help="by default, do not start evaluation. If set`, start evaluation alongside with training.")
    function("--eval_interval", type=int, default=10, help="time duration between contiunous twice evaluation progress.")
    function("--eval_episodes", type=int, default=32, help="number of episodes of a single evaluation.")

    # render parameters
    function("--save_gifs", action='store_true', default=False, help="by default, do not save render video. If set, save video.")
    function("--use_render", action='store_true', default=False, help="by default, do not render the env during training. If set, start render. Note: something, the environment has internal render process which is not controlled by this hyperparam.")
    function("--render_episodes", type=int, default=1, help="the number of episodes to render a given env")
    function("--video_dir", type=str, default="./video", help="set the path to save video.")
    function("--ifi", type=float, default=0.1, help="the play interval of each rendered image in saved video.")

    # pretrained parameters
    function("--model_dir", type=str, default=None, help="by default None. set the path to pretrained model.")
     
    # add for transformer
    function("--encode_state", action='store_true', default=False)
    function("--n_block", type=int, default=1)
    function("--n_embd", type=int, default=64)
    function("--n_head", type=int, default=1)
    function("--dec_actor", action='store_true', default=False)
    function("--share_actor", action='store_true', default=False)
    function("--n_seed_vector", type=int, default= 1)
    

    # add for online multi-task
    function("--train_maps", type=str, nargs='+', default=None)
    function("--eval_maps", type=str, nargs='+', default=None)
    
    return parser

