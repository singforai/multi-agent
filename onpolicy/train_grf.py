import os 
import sys
import torch 
import wandb 


import warnings
import setproctitle
from pathlib import Path
from config import get_config
from utils.util import fix_seed, generate_file
from utils.hyper_setting import hyper_check



from envs.football.football_env import FootballEnv
from envs.env_wrappers import ShareSubprocVecEnv, ShareDummyVecEnv




def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "football":
                env_args = {"scenario": all_args.scenario_name,
                            "n_agent": all_args.num_agents,
                            "reward": all_args.rewards,
                            "use_render": all_args.use_render}
                env = FootballEnv(env_args = env_args)
 
            else:
                print("Can not support the " + 
                      all_args.env_name + " environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env
    if all_args.n_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(
            all_args.n_rollout_threads)])

def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "football":
                env_args = {"scenario": all_args.scenario_name,
                            "n_agent": all_args.num_agents,
                            "reward": all_args.rewards,
                            "use_render": False}
                env = FootballEnv(env_args = env_args)
            else:
                print("Can not support the " +
                      all_args.env_name + " environment.")
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env
        return init_env
    if all_args.n_eval_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(
            all_args.n_eval_rollout_threads)])
        
def make_render_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "football":
                env_args = {"scenario": all_args.scenario_name,
                            "n_agent": all_args.num_agents,
                            "reward": all_args.rewards,
                            "use_render": all_args.use_render,
                            "video_dir" : all_args.video_dir}
                env = FootballEnv(env_args = env_args)
            else:
                print("Can not support the " +
                      all_args.env_name + " environment.")
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env
        return init_env
    if all_args.n_render_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(
            all_args.n_eval_rollout_threads)])

def parse_args(args, parser):
    function = parser.add_argument
    function("--game_length", type=int, default=2999, help="Max length for any game")
    function("--scenario_name", type=str, default="11_vs_11_hard_stochastic",
        choices=[
            "curriculum_learning",
            "academy_3_vs_1_with_keeper",
            "academy_pass_and_shoot_with_keeper",
            "academy_counterattack_easy",
            "academy_corner",
            "11_vs_11_hard_stochastic",
            "sampling",
        ],
    )
    function("--num_agents", type=int, default = 10, help="number of controlled players. (exclude goalkeeper)")
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
            project=all_args.env_name + "_" + all_args.scenario_name,
            entity=all_args.user_name,
            name="-".join([all_args.algorithm_name, all_args.experiment_name, "seed" + str(all_args.seed)]),
            group=all_args.group_name,
            dir=str(run_dir),
            job_type="training",
            reinit=True,
        )
    if all_args.scenario_name == "curriculum_learning":
        all_args.scenario_name, file_path = generate_file(all_args)
    else:
        file_path = None
     
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="gym")
    
    if not all_args.use_render:
        envs = make_train_env(all_args)
        eval_envs = make_eval_env(all_args) if all_args.use_eval else None
    else:
        envs = make_render_env(all_args) 
        eval_envs = envs
    num_agents = envs.n_agents
        
    config = {
        "all_args": all_args, 
        "envs": envs, 
        "eval_envs": eval_envs, 
        "num_agents": num_agents,
        "device": device, 
        "run_dir": run_dir
    }

    
    if all_args.use_rfcl:
        from runner.shared.football_runner_v2 import FootballRunner as Runner
        runner = Runner(config)
        runner.run(file_path)
    else:
        from runner.shared.football_runner import FootballRunner as Runner
        runner = Runner(config)
        runner.run(file_path)
    
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    if all_args.use_wandb:
        run.finish()
    
if __name__ == "__main__":
    main(args=sys.argv[1:])
