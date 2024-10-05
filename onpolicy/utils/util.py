import os 
import math
import time 
import json
import torch 
import shutil
import atexit
import numpy as np



def check(input):
    if type(input) == np.ndarray:
        return torch.from_numpy(input)
        
def get_gard_norm(it):
    sum_grad = 0
    for x in it:
        if x.grad is None:
            continue
        sum_grad += x.grad.norm() ** 2
    return math.sqrt(sum_grad)

def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def huber_loss(e, d):
    a = (abs(e) <= d).float()
    b = (abs(e) > d).float()
    return a*e**2/2 + b*d*(abs(e)-d/2)

def mse_loss(e):
    return e**2/2

def get_shape_from_obs_space(obs_space):
    if obs_space.__class__.__name__ == 'Box':
        obs_shape = obs_space.shape
    elif obs_space.__class__.__name__ == 'list':
        obs_shape = obs_space
    else:
        raise NotImplementedError
    return obs_shape

def get_shape_from_act_space(act_space):
    if act_space.__class__.__name__ == 'Discrete':
        act_shape = 1
    elif act_space.__class__.__name__ == "MultiDiscrete":
        act_shape = act_space.shape
    elif act_space.__class__.__name__ == "Box":
        act_shape = act_space.shape[0]
    elif act_space.__class__.__name__ == "MultiBinary":
        act_shape = act_space.shape[0]
    else:  # agar
        act_shape = act_space[0].shape[0] + 1  
    return act_shape


def tile_images(img_nhwc):
    """
    Tile N images into one big PxQ image
    (P,Q) are chosen to be as close as possible, and if N
    is square, then P=Q.
    input: img_nhwc, list or array of images, ndim=4 once turned into array
        n = batch index, h = height, w = width, c = channel
    returns:
        bigim_HWc, ndarray with ndim=3
    """
    img_nhwc = np.asarray(img_nhwc)
    N, h, w, c = img_nhwc.shape
    H = int(np.ceil(np.sqrt(N)))
    W = int(np.ceil(float(N)/H))
    img_nhwc = np.array(list(img_nhwc) + [img_nhwc[0]*0 for _ in range(N, H*W)])
    img_HWhwc = img_nhwc.reshape(H, W, h, w, c)
    img_HhWwc = img_HWhwc.transpose(0, 2, 1, 3, 4)
    img_Hh_Ww_c = img_HhWwc.reshape(H*h, W*w, c)
    return img_Hh_Ww_c

def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
def generate_subfile(all_args):

    time_code = int(time.time())
    print(f"Time code: {time_code}")
    
    if all_args.scenario_name == "curriculum_learning":
        from gfootball.scenarios import curriculum_learning
        x = str(curriculum_learning.__file__).replace('.py', '')
        scenario_file = curriculum_learning

        data = {
            'difficulty_level': 1,
            'level_stack': 0,
            "game_length": all_args.game_length
        }
        
    if all_args.use_rfcl:
        from gfootball.scenarios import rfcl
        x= str(rfcl.__file__).replace('.py', '')
        scenario_file = rfcl
        init_loc = [
                [-1.000000, 0.000000],
                [0.000000,  0.020000],
                [0.000000, -0.020000],
                [-0.422000, -0.19576],
                [-0.500000, -0.06356],
                [-0.500000, 0.063559],
                [-0.422000, 0.195760],
                [-0.184212, -0.10568],
                [-0.267574, 0.000000],
                [-0.184212, 0.105680],
                [-0.010000, -0.21610]
        ]
        data = {
            "left_team": init_loc,
            "right_team": init_loc,
            "ball": [0,0]
        }
    
    new_file_dir = f"{x}_{time_code}.py"
    json_name = f'level_{time_code}.json'
    directory = os.path.join(os.path.expanduser('~'), 'level')

    if not os.path.exists(directory):
        os.makedirs(directory)
    json_path = os.path.join(directory, json_name)
    
    shutil.copy(scenario_file.__file__, new_file_dir)
    
    def del_new_file():
        if os.path.exists(new_file_dir):
                os.remove(new_file_dir)
        if os.path.exists(json_path):
                os.remove(json_path)
                
    atexit.register(del_new_file)

    with open(json_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)
    return new_file_dir.split("/")[-1].replace(".py", ""), json_path