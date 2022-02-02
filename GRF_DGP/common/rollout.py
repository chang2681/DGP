# -*- coding: utf-8 -*-

import time

import numpy as np
import torch
from network.base_net import Uniform


class RolloutWorker:

    def __init__(self, env, agents, args):
        self.env = env
        self.agents = agents
        self.episode_limit = args.episode_limit
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        self.args = args
        self.n_agents = args.n_agents
        self.n_enemy=args.n_enemy
        print('Init RolloutWorker')

    def generate_episode(self, action_selecter, t_env, evaluate=False):
        o, u, r, s, avail_u, u_onehot, terminate, padded = [], [], [], [], [], [], [], []

        n_obs, global_state = self.env.reset()
        done = False
        step = 0
        episode_reward = 0

        last_action = torch.zeros((self.args.n_agents, self.args.n_actions))
        self.agents.policy.init_hidden(1)

        while not done:

            actions, avail_actions = [], []
            avail_actions = np.ones((self.n_agents, self.n_actions))  # 3*19

            inputs = torch.tensor(np.array(n_obs), dtype=torch.float32)
            avail_actions = torch.tensor(avail_actions, dtype=torch.float32)
            agent_id = torch.eye(self.n_agents)  # 单位矩阵

            if self.args.last_action:
                inputs = torch.cat((inputs, last_action), dim=-1)  # 3*26 to 3*(26+19)
            if self.args.add_id:
                inputs = torch.cat((inputs, agent_id), dim=-1)

            inputs = inputs.unsqueeze(1)  # 在一维增加一个维度(3,45) to (3,1,45)
            if self.args.cuda:
                inputs = inputs.to(torch.device(self.args.GPU))
                self.agents.policy.eval_hidden = self.agents.policy.eval_hidden.to(
                    torch.device(self.args.GPU))
            q_value_global, self.agents.policy.eval_hidden = self.agents.policy.eval_rnn.forward(
                inputs, self.agents.policy.eval_hidden)
            self.agents.policy.eval_hidden = self.agents.policy.eval_hidden.permute(
                1, 0, 2)  # (3,1,64) to (1,3,64)
            q_value_global = q_value_global.squeeze(1)

            if  self.args.lf_q == 1:
                q_value_local = torch.cat(
                    [self.agents.policy.eval_mlp[id].forward(
                        self.agents.policy.eval_hidden[:, id, :]) for id in range(self.args.n_agents)],
                    dim=0)
                q_value = q_value_local
            else:
                q_value = q_value_global

            actions = action_selecter.select_action(
                q_value, avail_actions, t_env + step, test_mode=evaluate).to('cpu')
            actions_input = actions.numpy()

            o.append(np.array(n_obs))
            s.append(np.array(global_state))

            n_obs, global_state, reward, done, _ = self.env.step(actions_input)
            if step == self.episode_limit - 1:
                done = True

            if done == True:
                win_or_loss = reward > 50

            last_action = torch.zeros(
                (self.args.n_agents, self.args.n_actions))
            last_action = last_action.scatter_(
                1, actions.unsqueeze(1).long(), 1)

            u.append(actions.unsqueeze(1).numpy())
            u_onehot.append(last_action.to('cpu').numpy())
            avail_u.append(avail_actions.to('cpu').numpy())
            r.append([reward])
            episode_reward += reward
            step += 1

        o.append(np.array(n_obs))
        s.append(np.array(global_state))
        o_next = o[1:]
        s_next = s[1:]
        o = o[:-1]
        s = s[:-1]

        avail_actions = np.ones((self.n_agents, self.n_actions))
        avail_u.append(avail_actions)
        avail_u_next = avail_u[1:]
        avail_u = avail_u[:-1]

        remain_length = self.episode_limit - step
        o = o + [np.zeros((self.n_agents, self.obs_shape))]*remain_length
        o_next = o_next + [np.zeros((self.n_agents, self.obs_shape))]*remain_length
        u = u + [np.zeros((self.n_agents, 1))]*remain_length
        u_onehot = u_onehot + \
                   [np.zeros((self.n_agents, self.n_actions))]*remain_length
        s = s + [np.zeros((self.state_shape))]*remain_length
        s_next = s_next + [np.zeros((self.state_shape))]*remain_length
        r = r + [[0] ]*remain_length
        avail_u = avail_u + [np.ones((self.n_agents, self.n_actions))]*remain_length
        avail_u_next = avail_u_next + \
                       [np.ones((self.n_agents, self.n_actions))]*remain_length
        padded = [[0.]]*step+ [[1]]*remain_length
        terminate = [[0.]]*step + [[1]]*remain_length

        o = np.stack(o, axis=0)
        o_next = np.stack(o_next, axis=0)
        s = np.stack(s, axis=0)
        s_next = np.stack(s_next, axis=0)
        u = np.stack(u, axis=0)
        u_onehot = np.stack(u_onehot, axis=0)
        avail_u = np.stack(avail_u, axis=0)
        avail_u_next = np.stack(avail_u_next, axis=0)
        r = np.array(r)
        padded = np.array(padded)
        terminate = np.array(terminate)
        self.noise = np.array([0])  # useless
        episode = dict(
            o=o[np.newaxis, :],
            s=s[np.newaxis, :],
            u=u[np.newaxis, :],
            r=r[np.newaxis, :],
            noise=self.noise[np.newaxis, :],
            avail_u=avail_u[np.newaxis, :],
            o_next=o_next[np.newaxis, :],
            s_next=s_next[np.newaxis, :],
            avail_u_next=avail_u_next[np.newaxis, :],
            u_onehot=u_onehot[np.newaxis, :],
            padded=padded[np.newaxis, :],
            terminated=terminate[np.newaxis, :])

        return episode, episode_reward, step, win_or_loss

