# -*- coding: utf-8 -*-

import numpy as np
import os
import gym
import csv
import torch
from tqdm import tqdm

from common.rollout import RolloutWorker
from agent import Agents
from common.action_selecter import EpsilonGreedyActionSelector
from common.replay_buffer_prioritize import ReplayBuffer
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from env.wrapper_grf_3vs2 import make_football_env_3vs1
#from env.wrapper_grf_3_vs_2_full import make_football_env_3vs3_full

from env.wrapper_grf_4vs3 import make_football_env_4vs3
from env.wrapper_grf_3vs3 import make_football_env_3vs3
from env.wrapper_grf_3vs4_BackBall import make_football_env_3vs4_BackBall


from time import strftime, localtime
from env.gym_foo.gym_foo.envs.pac_men import CustomEnv


class Runner:

    def __init__(self, env, args):
        self.env = env

        self.csv_dir = f'./csv_file/{args.env}'
        self.csv_path = f'{self.csv_dir}/seed_{args.seed}_{args.label}.csv'

        if 'pacmen' in args.env:
            self.env_evaluate = CustomEnv(4,args.seed,args.mode)
            self.env_evaluate.seed(args.seed)

        elif args.env == '3_vs_2':
            # google research football
            self.env_evaluate = make_football_env_3vs1(args.seed,
                dense_reward=args.dense_reward)
        elif args.env == '3_vs_3':
            # google research football
            self.env_evaluate = make_football_env_3vs3(args.seed,
                                                       dense_reward=args.dense_reward)
        elif args.env == '3_vs_4_BackBall':
            # google research football
            self.env_evaluate = make_football_env_3vs4_BackBall(args.seed,dense_reward=args.dense_reward)

        elif args.env == '4_vs_2_full':
            # google research football
            self.env_evaluate = make_football_env_4vs2_full(args.seed,
                dense_reward=args.dense_reward)




        elif args.env == '4_vs_3':
            # google research football
            self.env_evaluate = make_football_env_4vs3(args.seed,
                dense_reward=args.dense_reward)




        self.win_dir = f'{self.csv_dir}/win_rate'
        self.win_path = f'{self.win_dir}/sed_{args.seed}_{args.label}.csv'
        self.action_selecter = None
        self.agents = Agents(args)
        self.rolloutWorker = RolloutWorker(env, self.agents, args)
        self.evaluateWorker = RolloutWorker(
            self.env_evaluate, self.agents, args)

        self.buffer = ReplayBuffer(args)
        self.args = args
        self.save_path = f'{self.args.result_dir}'

        for item_file in [self.save_path, self.csv_dir]:
            if not os.path.exists(item_file):
                os.makedirs(item_file)
        if not os.path.exists(self.win_dir):
            os.makedirs(self.win_dir)

    def writereward(self, path, reward, step):
        if os.path.isfile(path):
            with open(path, 'a+') as f:
                csv_write = csv.writer(f)
                csv_write.writerow([step, reward])
        else:
            with open(path, 'w') as f:
                csv_write = csv.writer(f)
                csv_write.writerow(['step', 'reward'])
                csv_write.writerow([step, reward])

    def run(self, num):
        episode_rewards = []
        train_steps, self.all_steps = 0, 0
        self.action_selecter = EpsilonGreedyActionSelector(self.args)
        for epoch in tqdm(range(self.args.n_epoch)): # 20 0000
            episodes = []
            priority = []
            for _ in range(self.args.n_episodes): #1
                episode, _, step, win_or_loss = self.rolloutWorker.generate_episode(
                    self.action_selecter, self.all_steps)
                episodes.append(episode)
                self.all_steps += step
                if win_or_loss:
                    priority.append(100)
                else:
                    priority.append(0)

            episode_batch = episodes[0]
            episodes.pop(0)

            for episode in episodes:
                for key in episode_batch.keys():
                    episode_batch[key] = np.concatenate(
                        (episode_batch[key], episode[key]), axis=0)

            #self.buffer.store_episode(episode_batch, priority)
            self.buffer.store_episode(episode_batch)

            if self.buffer.can_sample(self.args.batch_size):
                for _ in range(self.args.train_steps):
                    idx, mini_batch = self.buffer.sample(
                        min(self.buffer.current_size, self.args.batch_size))
                    update_prior = self.agents.train(
                        mini_batch, train_steps, self.all_steps)
                    self.buffer.update_priority(
                        idx, update_prior.to('cpu').tolist())
                    train_steps += 1

            if epoch % self.args.evaluate_cycle == self.args.evaluate_cycle - 1:
                win_rate, episode_reward = self.evaluate()
                episode_rewards.append(episode_reward)
                self.writereward(self.csv_path, episode_reward, self.all_steps)
                self.writereward(self.win_path, win_rate, self.all_steps)

    def evaluate(self):
        if self.action_selecter==None:
            self.action_selecter = EpsilonGreedyActionSelector(self.args)
            self.all_steps=0

        win_number = 0
        episode_rewards = 0
        for _ in range(self.args.evaluate_epoch):
            _, episode_reward, _, win_or_loss = self.rolloutWorker.generate_episode(
                self.action_selecter, self.all_steps, evaluate=True)
            if win_or_loss == 1:
                win_number += 1
            episode_rewards += episode_reward

        return win_number / self.args.evaluate_epoch, episode_rewards / self.args.evaluate_epoch
