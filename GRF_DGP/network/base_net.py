import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np


class Uniform:

    def __init__(self, args):
        self.args = args
        self.device = torch.device('cpu')
        self.noise_distrib = torch.distributions.one_hot_categorical.OneHotCategorical(
            torch.tensor([1 / self.args.noise_dim for _ in range(self.args.noise_dim)]))

    def sample(self, state, test_mode):
        return self.noise_distrib.sample().to(self.device)

    def update_returns(self, state, noise, returns, test_mode, t):
        pass

    def to(self, device):
        self.device = device


# class RNN(nn.Module):
#
#     def __init__(self, input_shape, args):
#         super(RNN, self).__init__()
#         self.args = args
#
#         self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
#         # self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
#         self.rnn = nn.GRU(
#             input_size=args.rnn_hidden_dim,
#             num_layers=1,
#             hidden_size=args.rnn_hidden_dim,
#             batch_first=True,
#         )
#         self.fc15 = nn.Linear(2*args.rnn_hidden_dim, args.rnn_hidden_dim)
#
#         self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)
#
#     def init_hidden(self):
#         # make hidden states on same device as model
#         return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()
#
#     def forward(self, obs, hidden_state):
#         if len(hidden_state.shape) == 2:
#             hidden_state = hidden_state.unsqueeze(0) #保证尺寸(3,1,64)
#         #########################################
#         # obs_c = obs.view(-1, obs.shape[-1])
#         # x = f.relu(self.fc1(obs_c))
#         # x = x.reshape(obs.shape[0], obs.shape[1], -1)
#         # h_in = hidden_state
#         # gru_out, _ = self.rnn(x, h_in)
#         # gru_out_c = gru_out.reshape(-1, gru_out.shape[-1])
#
#         #########################################
#
#         obs_c = obs.view(-1, obs.shape[-1])
#         x = f.relu(self.fc1(obs_c))
#         x = x.reshape(obs.shape[0], obs.shape[1], -1)
#
#         h_in = hidden_state
#         gru_out, _ = self.rnn(x, h_in)
#         gru_out_c = gru_out.reshape(-1, gru_out.shape[-1])
#         Leader_id = obs.reshape(-1, self.args.n_agents, obs.shape[-2], obs.shape[-1])[..., 0].long().unsqueeze(-1)
#         gru_l = gru_out.reshape(-1, self.args.n_agents, gru_out.shape[-2], gru_out.shape[-1])
#         gru_leader = gru_l.gather(1, Leader_id.expand_as(gru_l)).reshape(-1, gru_l.shape[-1])
#         gru_out_c = torch.cat([gru_out_c, gru_leader], axis=-1)
#         gru_out_c = f.relu(self.fc15(gru_out_c))
#         ##########################################################
#
#         q = self.fc2(gru_out_c)
#         q = q.reshape(obs.shape[0], obs.shape[1], -1)
#
#         return q, gru_out

class RNN(nn.Module):
    # 不加通信
    def __init__(self, input_shape, args):
        super(RNN, self).__init__()
        self.args = args
        self.input_shape=input_shape
        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        # self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.rnn = nn.GRU(
            input_size=args.rnn_hidden_dim,
            num_layers=1,
            hidden_size=args.rnn_hidden_dim,
            batch_first=True,
        )
        self.fc15 = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)

        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    # def forward(self, obs, hidden_state):
    #     if len(hidden_state.shape) == 2:
    #         hidden_state = hidden_state.unsqueeze(0)  # 保证尺寸(3,1,64)
    #     #########################################
    #
    #     #########################################
    #
    #     obs_c = obs.view(-1, obs.shape[-1])
    #     x = f.relu(self.fc1(obs_c))
    #     x = x.reshape(obs.shape[0], obs.shape[1], -1)
    #
    #     h_in = hidden_state
    #     gru_out, _ = self.rnn(x, h_in)
    #     gru_out_c = gru_out.reshape(-1, gru_out.shape[-1])
    #     Leader_id = obs.reshape(-1, self.args.n_agents, obs.shape[-2], obs.shape[-1])[..., 0].long().unsqueeze(-1)
    #     gru_l = gru_out.reshape(-1, self.args.n_agents, gru_out.shape[-2], gru_out.shape[-1])
    #     gru_leader = gru_l.gather(1, Leader_id.expand_as(gru_l)).reshape(-1, gru_l.shape[-1])
    #     gru_out_c = gru_out_c +0.2* gru_leader
    #     #gru_out_c = f.relu(self.fc15(gru_out_c))
    #     ##########################################################
    #
    #     q = self.fc2(gru_out_c)
    #     q = q.reshape(obs.shape[0], obs.shape[1], -1)
    #
    #     return q, gru_out_c.reshape_as(gru_out)

    def forward(self, obs, hidden_state):
        if len(hidden_state.shape) == 2:
            hidden_state = hidden_state.unsqueeze(0) #保证尺寸(1,3,64)
        #########################################

        obs_c = obs.view(-1, obs.shape[-1])

        x = f.relu(self.fc1(obs_c))
        x = x.reshape(obs.shape[0], obs.shape[1], -1)
        h_in = hidden_state
        gru_out, _ = self.rnn(x, h_in)
        gru_out_c = gru_out.reshape(-1, gru_out.shape[-1])
        q = self.fc2(gru_out_c)
        q = q.reshape(obs.shape[0], obs.shape[1], -1)  #415

        return q, gru_out
class RNN1(nn.Module):
    def __init__(self, input_shape, args):
        super(RNN, self).__init__()
        self.n_agent=args.n_agents
        self.eval_mlp = nn.ModuleList([copyRNN(input_shape, args)
                                       for _ in range(args.n_agents)])
    def init_hidden(self):
        # make hidden states on same device as model
        return 1

    def forward(self, obs, hidden_state):
        if len(hidden_state.shape) == 2:
            hidden_state = hidden_state.unsqueeze(0) #保证尺寸(1,3,64)
        if obs.shape[0] >= 32:
            label=True
            obs=obs.reshape(-1,self.n_agent,obs.shape[-2],obs.shape[-1]).permute(1,0,2,3)
            hidden_state=hidden_state.reshape(-1,self.n_agent,hidden_state.shape[-1])
        else:label= False

        q=[]
        gru_out=[]
        for id in range(self.n_agent):
            q_i,gru_out_i=self.eval_mlp[id](obs[id],hidden_state[...,id,:])
            q.append(q_i)
            gru_out.append(gru_out_i)
        if label:
            return torch.cat(q).permute(1,0,2,3).reshape(-1,q_i.shape[-2],q_i.shape[-1]), torch.cat(gru_out).permute(1,0,2,3).reshape(-1,gru_out_i.shape[-2],gru_out_i.shape[-1])
        else:
            return torch.cat(q), torch.cat(gru_out)
        #########################################

class MLP(nn.Module):

    def __init__(self, args):
        super(MLP, self).__init__()
        self.args = args
        self.fc = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def forward(self, hidden_state):
        q = self.fc(hidden_state)
        return q


class MLP_2(nn.Module):

    def __init__(self, args):
        super(MLP_2, self).__init__()
        self.args = args
        self.fc_1 = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc_2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def forward(self, hidden_state):
        h1 = f.relu(self.fc_1(hidden_state))
        q = self.fc_2(h1)
        return q


class Critic(nn.Module):

    def __init__(self, input_shape, args):
        super(Critic, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(input_shape, args.critic_dim)
        self.fc2 = nn.Linear(args.critic_dim, args.critic_dim)
        self.fc3 = nn.Linear(args.critic_dim, 1)

    def forward(self, inputs):
        x = f.relu(self.fc1(inputs))
        x = f.relu(self.fc2(x))
        q = self.fc3(x)
        return q
class copyRNN(nn.Module):
    # 不加通信
    def __init__(self, input_shape, args):
        super(copyRNN, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        # self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.rnn = nn.GRU(
            input_size=args.rnn_hidden_dim,
            num_layers=1,
            hidden_size=args.rnn_hidden_dim,
            batch_first=True,
        )
        self.fc15 = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)

        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    # def forward(self, obs, hidden_state):
    #     if len(hidden_state.shape) == 2:
    #         hidden_state = hidden_state.unsqueeze(0)  # 保证尺寸(3,1,64)
    #     #########################################
    #
    #     #########################################
    #
    #     obs_c = obs.view(-1, obs.shape[-1])
    #     x = f.relu(self.fc1(obs_c))
    #     x = x.reshape(obs.shape[0], obs.shape[1], -1)
    #
    #     h_in = hidden_state
    #     gru_out, _ = self.rnn(x, h_in)
    #     gru_out_c = gru_out.reshape(-1, gru_out.shape[-1])
    #     Leader_id = obs.reshape(-1, self.args.n_agents, obs.shape[-2], obs.shape[-1])[..., 0].long().unsqueeze(-1)
    #     gru_l = gru_out.reshape(-1, self.args.n_agents, gru_out.shape[-2], gru_out.shape[-1])
    #     gru_leader = gru_l.gather(1, Leader_id.expand_as(gru_l)).reshape(-1, gru_l.shape[-1])
    #     gru_out_c = gru_out_c +0.2* gru_leader
    #     #gru_out_c = f.relu(self.fc15(gru_out_c))
    #     ##########################################################
    #
    #     q = self.fc2(gru_out_c)
    #     q = q.reshape(obs.shape[0], obs.shape[1], -1)
    #
    #     return q, gru_out_c.reshape_as(gru_out)

    def forward(self, obs, hidden_state):
        if len(hidden_state.shape) == 2:
            hidden_state = hidden_state.unsqueeze(0) #保证尺寸(3,1,64)
        #########################################
        if len(obs.shape) == 2:
            obs = obs.unsqueeze(0) #保证尺寸(3,1,64)
        obs_c = obs.contiguous().view(-1, obs.shape[-1])
        x = f.relu(self.fc1(obs_c))
        x = x.reshape(obs.shape[0], obs.shape[1], -1)
        h_in = hidden_state
        gru_out, _ = self.rnn(x, h_in)
        gru_out_c = gru_out.reshape(-1, gru_out.shape[-1])

        #########################################

        # obs_c = obs.view(-1, obs.shape[-1])
        # x = f.relu(self.fc1(obs_c))
        # x = x.reshape(obs.shape[0], obs.shape[1], -1)
        #
        # h_in = hidden_state
        # gru_out, _ = self.rnn(x, h_in)
        # gru_out_c = gru_out.reshape(-1, gru_out.shape[-1])
        # Leader_id = obs.reshape(-1, self.args.n_agents, obs.shape[-2], obs.shape[-1])[..., 0].long().unsqueeze(-1)
        # gru_l = gru_out.reshape(-1, self.args.n_agents, gru_out.shape[-2], gru_out.shape[-1])
        # gru_leader = gru_l.gather(1, Leader_id.expand_as(gru_l)).reshape(-1, gru_l.shape[-1])
        # gru_out_c = torch.cat([gru_out_c, gru_leader], axis=-1)
        # gru_out_c = f.relu(self.fc15(gru_out_c))
        ##########################################################

        q = self.fc2(gru_out_c)
        q = q.reshape(obs.shape[0], obs.shape[1], -1)

        return q, gru_out
