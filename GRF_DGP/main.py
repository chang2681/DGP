# -*- coding: utf-8 -*-
import gym
import torch
import numpy as np
from env.gym_foo.gym_foo.envs.pac_men import CustomEnv

from env.wrapper_grf_3vs2 import make_football_env_3vs1
#from env.wrapper_grf_3_vs_2_full import make_football_env_3vs3_full
from env.wrapper_grf_3vs3 import make_football_env_3vs3
from env.wrapper_grf_3vs4_BackBall import make_football_env_3vs4_BackBall
from env.wrapper_grf_4vs3 import make_football_env_4vs3
from runner import Runner
from common.arguments import get_common_args, get_mixer_args
import random

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(8)


def get_env(args):
    print(args.env)
    if 'pacmen' in args.env:
        env = CustomEnv(4, args.seed,args.mode)
        env.seed(args.seed)

    elif args.env == '3_vs_2':
        # google research football
        env = make_football_env_3vs1(args.seed,dense_reward=False)
    #
    # elif args.env == '3_vs_3_full':
    #     # google research football
    #     env = make_football_env_3vs3_full(args.seed, dense_reward=False)
    elif args.env == '3_vs_3':
        env = make_football_env_3vs3(args.seed, dense_reward=False)
    elif args.env == '3_vs_4_BackBall':
        env = make_football_env_3vs4_BackBall(args.seed,dense_reward=False)
    elif args.env == '4_vs_3':
        env = make_football_env_4vs3(args.seed,dense_reward=False)


    return env


if __name__ == '__main__':
    args = get_common_args()
    args = get_mixer_args(args)
    args.GPU = "cuda:" + (str(random.randint(0, 5)))
    setup_seed(args.seed)
    env = get_env(args)
    print('--------------', args.env, '------------')
    print('label:', args.label,args.GPU,'seed:',args.seed)


    env_info = env.get_env_info()
    args.n_actions = env_info["n_actions"]
    args.n_agents = env_info["n_agents"]
    args.state_shape = env_info["state_shape"]
    args.obs_shape = env_info["obs_shape"]
    args.episode_limit = env_info["episode_limit"]
    args.n_enemy= env_info['n_enemy']

    args.reuse_network = False

    runner = Runner(env, args)
    runner.run(0)
    env.close()
