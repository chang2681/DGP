# -*- coding: utf-8 -*-
import gym
import torch
import numpy as np
from env.wrapper_grf_2vs3 import make_football_env_2vs3

from env.wrapper_grf_3vs2 import make_football_env_3vs1
from env.wrapper_grf_3_vs_2_full import make_football_env_3vs3_full
from env.wrapper_grf_4_vs_2_full import make_football_env_4vs2_full
from env.wrapper_grf_4vs4 import make_football_env_4vs4
from env.wrapper_grf_3vs3 import make_football_env_3vs3
from env.wrapper_grf_3vs4_BackBall import make_football_env_3vs4_BackBall
from env.gym_foo.gym_foo.envs.pac_men import CustomEnv

from env.wrapper_grf_4vs3 import make_football_env_4vs3
from env.wrapper_grf_4vs5 import make_football_env_4vs5
from env.wrapper_grf_11vs4 import make_football_env_11vs4
from env.wrapper_grf_5vs3 import make_football_env_5vs3
from env.wrapper_grf_5vs5 import make_football_env_5vs5
from runner import Runner
from common.arguments import get_common_args, get_mixer_args

def setup_seed(seed):
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
    elif args.env == '2_vs_3':
            # google research football
        env = make_football_env_2vs3(dense_reward=False,seed=1256,render=True)
    elif args.env == '3_vs_2':
        # google research football
        env = make_football_env_3vs1(dense_reward=False,seed=1256,render=True)

    elif args.env == '3_vs_2_full':
        # google research football
        env = make_football_env_3vs3_full(dense_reward=False, seed=1256, render=True)
    elif args.env == '3_vs_3':
        # google research football
        env = make_football_env_3vs3(dense_reward=False,seed=1256,render=True)
    elif args.env == '3_vs_4_BackBall':
        # google research football
        env = make_football_env_3vs4_BackBall(dense_reward=False,seed=1256,render=True)
    elif args.env == '4_vs_2_full':
        # google research football
        env = make_football_env_4vs2_full(dense_reward=False,seed=1256,render=True)

    elif args.env == '4_vs_4':
        # google research football
        env = make_football_env_4vs4(dense_reward=False,seed=1256,render=True)

    elif args.env == '4_vs_3':
        # google research football
        env = make_football_env_4vs3(dense_reward=False,seed=1256,render=True)

    elif args.env == '4_vs_5':
        # google research football
        env = make_football_env_4vs5(dense_reward=False,seed=1256,render=True)

    elif args.env == '11_vs_4':
        # google research football
        env = make_football_env_11vs4(dense_reward=False,seed=1256,render=True)
    elif args.env == '5_vs_3':
        # google research football
        env = make_football_env_5vs3(dense_reward=False,seed=1256,render=True)
    elif args.env == '5_vs_5':
        # google research football
        env = make_football_env_5vs5(dense_reward=False,seed=1256,render=True)

    return env


if __name__ == '__main__':
    args = get_common_args()
    args = get_mixer_args(args)

    # args.env = 'pacmen'
    args.env = '3_vs_3'
    #args.env = '4_vs_3'
    args.seed = 1257
    if args.env=='3_vs_4_BackBall':
        args.model_step =380  #383
        args.label = '.5.5.06_env810_.5_addid'
    elif args.env=='4_vs_3':
        args.seed = 1257
        args.model_step =154  #383
        args.label = '.5.5.08_count_attack'
    elif args.env=='3_vs_3':
        args.seed = 1256
        args.model_step =380  #383
        args.label='.5.5.06_dif1_pos.4dif.5_re'
    args.render = True
    args.cuda = False
    #args.label='.5.5.08_count_attack'
    #args.label='.5.5.06_'
   #args.label='.5.5.02_dif1_pos.6_envmatrix'
    #args.label='.5.5.06_pos.6dif1'
    #args.label='.5.5.02_dif1_pos.5dif.8'
    #args.label='.5.5.06_dif1_pos.4dif.5_re'
    #args.other_label='2'
    print('args.model_step:',args.model_step)

    import random
    if 'pacmen' in args.env:
        args.QPLEX_mixer = 'dmaq'
        #np.savetxt('obs_2_green3.txt',o.swapaxes(0,1).reshape(4,-1))
    if 'CDS' in args.alg:
        args.beta1 = .5
        args.beta2 = 2.
        args.beta = .05
        args.local_q_L1loss=True
        args.lf_q= 1
    elif 'orginal_qplex' in args.label:
        args.beta = 0
        args.lf_q = 5


    env = get_env(args)

    env_info = env.get_env_info()
    args.n_actions = env_info["n_actions"]
    args.n_agents = env_info["n_agents"]
    args.state_shape = env_info["state_shape"]
    args.obs_shape = env_info["obs_shape"]
    args.episode_limit = env_info["episode_limit"]
    args.n_enemy=   env_info['n_enemy']

    args.reuse_network = False


    runner = Runner(env, args)

    runner.evaluate()
    env.close()
