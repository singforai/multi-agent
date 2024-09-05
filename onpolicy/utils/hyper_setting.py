
def hyper_check(all_args):
    print(f"Algorithm: {all_args.algorithm_name}")
    if all_args.algorithm_name == "rmappo":
        all_args.use_recurrent_policy = True
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "mappo":
        all_args.use_recurrent_policy = False 
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "mat":
        all_args.use_recurrent_policy = False 
        all_args.use_naive_recurrent_policy = False
        all_args.dec_actor = False
        all_args.share_actor = False
    elif all_args.algorithm_name == "mat_dec":
        all_args.use_recurrent_policy = False 
        all_args.use_naive_recurrent_policy = False
        all_args.dec_actor = True
        all_args.share_actor = True  
    elif all_args.algorithm_name == "tizero":
        if all_args.env_name != "football":
            print("Tizero only can use in GRF environment!!")
            raise NotImplementedError
        all_args.use_joint_action_loss = True
        all_args.use_recurrent_policy = True 
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "mast":
        all_args.use_joint_action_loss = True
    else:
        raise NotImplementedError
    
    if all_args.env_name == "smac":
        from envs.starcraft2.smac_maps import get_map_params
        num_agents = get_map_params(all_args.map_name)["n_agents"]
    elif all_args.env_name == "football":
        num_agents = all_args.num_agents
    else: 
        raise NotImplementedError
    
    return all_args, num_agents
    