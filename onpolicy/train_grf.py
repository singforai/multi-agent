import os 
import sys
import torch 
import wandb 

import warnings
import setproctitle
from pathlib import Path
from config import get_config
from utils.util import fix_seed
from utils.hyper_setting import hyper_check

from envs.football.Football_Env import FootballEnv
from envs.env_wrappers import SubprocVecEnv, DummyVecEnv
from runner.shared.football_runner import FootballRunner as Runner

def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "football":
                env = FootballEnv(all_args)
 
            else:
                print("Can not support the " + 
                      all_args.env_name + " environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env
    if all_args.n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(
            all_args.n_rollout_threads)])

def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "football":
                env = FootballEnv(all_args)
            else:
                print("Can not support the " +
                      all_args.env_name + " environment.")
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env
        return init_env
    if all_args.n_eval_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(
            all_args.n_eval_rollout_threads)])

def parse_args(args, parser):
    function = parser.add_argument
    function(
        "--init_level_dir",
        type=str, 
        default='level/init_level.json', 
        help="directory for setting init level"
    ),
    function(
        "--level_dir",
        type=str, 
        default='level/level.json', 
        help="directory for level change"
    ),
    function(
        "--game_length", 
        type=int,
        default=500, 
        help="Max length for any game"
    )
    function(
        "--scenario_name",
        type=str,
        default="curriculum_learning",
        choices=[
            "curriculum_learning",
            "curriculum_learning_tizero"
        ],
        help="which scenario to run on.",
    )
    function("--num_agents", type=int, default=10, help="number of controlled players. (exclude goalkeeper)")
    function(
        "--representation",
        type=str,
        default="simple115v2",
        choices=["simple115v2", "extracted", "pixels_gray", "pixels"],
        help="representation used to build the observation.",
    )
    function("--rewards", type=str, default="scoring", help="comma separated list of rewards to be added.")
    function("--smm_width", type=int, default=96, help="width of super minimap.")
    function("--smm_height", type=int, default=72, help="height of super minimap.")
    function(
        "--zero_feature", action="store_true", default=False, help="by default False. If True, replace -1 by 0"
    )
    function(
        "--eval_deterministic",
        action="store_false",
        default=True,
        help="by default True. If False, sample action according to probability",
    )
    function(
        "--share_reward",
        action="store_false",
        default=True,
        help="by default true. If false, use different reward for each agent.",
    )
    all_args = parser.parse_known_args(args)[0]

    return all_args

def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    level_dir = all_args.level_dir
    if os.path.exists(level_dir):
        os.remove(level_dir)
        
    if all_args.env_name != "football":
        print(f"We changed the all_args.env_name from {all_args.env_name} to football!")
        all_args.env_name = "football"
        
    all_args, _ = hyper_check(all_args = all_args)    
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
        / all_args.scenario_name
        / all_args.algorithm_name
        / all_args.experiment_name
    )
    if not run_dir.exists():
        os.makedirs(str(run_dir))
    setproctitle.setproctitle(
        "-".join([all_args.env_name, all_args.scenario_name, all_args.algorithm_name, all_args.experiment_name])
        + "@"
        + all_args.user_name
    )
    
    if all_args.use_wandb:
        run = wandb.init(
            config=all_args,
            project=all_args.env_name,
            entity=all_args.user_name,
            notes=all_args.notes,
            name="-".join([all_args.algorithm_name, all_args.experiment_name, "seed" + str(all_args.seed)]),
            group=all_args.group_name,
            dir=str(run_dir),
            job_type="training",
            reinit=True,
        )
        
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="gym")
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None
        
    config = {
        "all_args": all_args, 
        "envs": envs, 
        "eval_envs": eval_envs, 
        "num_agents": all_args.num_agents,
        "device": device, 
        "run_dir": run_dir
    }
    
    runner = Runner(config)
    runner.run()

    if all_args.use_wandb:
        run.finish()
    
if __name__ == "__main__":
    main(args=sys.argv[1:])