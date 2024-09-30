import os 
import sys
import torch 
import wandb 
import socket
import warnings
import setproctitle
from pathlib import Path
from config import get_config
from utils.util import fix_seed
from utils.hyper_setting import hyper_check

from envs.env_wrappers import ShareSubprocVecEnv, ShareDummyVecEnv

from runner.shared.smac_runner import SMACRunner as Runner

def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "smac":
                from envs.starcraft2.StarCraft2_Env import StarCraft2Env
                env = StarCraft2Env(all_args)
            else:
                print("Can not support the " + all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env

        return init_env

    if all_args.n_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])
    
def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "smac":
                from envs.starcraft2.StarCraft2_Env import StarCraft2Env
                env = StarCraft2Env(all_args)
            else:
                print("Can not support the " + all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env

        return init_env

    if all_args.n_eval_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])

def parse_args(args, parser):
    function = parser.add_argument
    function('--map_name', type=str, default='3m',
            choices = [
                "3m",
                "8m", 
                "1c3s5z",
                "MMM", 
                "2c_vs_64zg", 
                "3s_vs_5z",
                "3s5z",
                "5m_vs_6m",
                "8m_vs_9m",
                "10m_vs_11m",
                "25m",
                 "27m_vs_30m",
                 "MMM2",
                 "6h_vs_8z",
                 "3s5z_vs_3s6z"
            ],
                help="Which smac map to run on")
    function("--difficulty_level", type=str,
                        default="7", help="difficulty level of map")
    function("--add_move_state", action='store_true', default=False)
    function("--add_local_obs", action='store_true', default=False)
    function("--add_distance_state", action='store_true', default=False)
    function("--add_enemy_action_state", action='store_true', default=False)
    function("--add_agent_id", action='store_true', default=False)
    function("--add_visible_state", action='store_true', default=False)
    function("--add_xy_state", action='store_true', default=False)
    function("--use_state_agent", action='store_false', default=True)
    function("--use_mustalive", action='store_false', default=True)
    function("--add_center_xy", action='store_false', default=True)
    all_args = parser.parse_known_args(args)[0]

    return all_args

def main(args):
    
    parser = get_config()
    all_args = parse_args(args = args, parser = parser)
    
    if all_args.env_name != "smac":
        print(f"We changed the all_args.env_name from {all_args.env_name} to smac!")
        all_args.env_name = "smac"
    
    all_args, num_agents = hyper_check(all_args = all_args)
    
    fix_seed(seed = all_args.seed)

    if all_args.cuda and torch.cuda.is_available():
        print("Device type: GPU")
        device = torch.device(f"cuda:{all_args.num_gpu}")
        torch.set_num_threads(all_args.n_torch_threads)
    else:
        print("Device type: CPU")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_torch_threads)

    run_dir = (
        Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/results")
        / all_args.env_name
        / all_args.map_name
        / all_args.algorithm_name
        / (all_args.experiment_name + "_" + str(all_args.seed))
    )
    
    if not run_dir.exists():
        os.makedirs(str(run_dir))
        
    setproctitle.setproctitle(
        "-".join([all_args.env_name, all_args.map_name, all_args.algorithm_name, all_args.experiment_name])
        + "@"
        + all_args.user_name
        + str(all_args.seed)
    )
    
    if all_args.use_wandb:
        run = wandb.init(
            config=all_args,
            project= all_args.env_name + "_" + all_args.map_name,  
            notes=socket.gethostname(),
            entity=all_args.user_name,
            name="-".join([all_args.algorithm_name, all_args.experiment_name, "seed" + str(all_args.seed)]),
            group=all_args.group_name,
            dir=str(run_dir),
            job_type="training",
        )
        
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="gym")
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None
        
    config = {
        "all_args": all_args, 
        "envs": envs, 
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device, 
        "run_dir": run_dir
    }
    
    runner = Runner(config)
    runner.run()

    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    if all_args.use_wandb:
        run.finish()
    
if __name__ == "__main__":
    main(args=sys.argv[1:])