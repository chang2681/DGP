import torch.nn as nn
import torch as th
import torch.nn.functional as F
import numpy as np


# class QMixNet(nn.Module):
#
#     def __init__(self, args):
#         super(QMixNet, self).__init__()
#         self.args = args
#         if args.two_hyper_layers:
#             self.hyper_w1 = nn.Sequential(
#                 nn.Linear(args.state_shape, args.hyper_hidden_dim), nn.ReLU(),
#                 nn.Linear(args.hyper_hidden_dim, args.n_agents * args.qmix_hidden_dim))
#             self.hyper_w2 = nn.Sequential(
#                 nn.Linear(args.state_shape, args.hyper_hidden_dim), nn.ReLU(), nn.Linear(args.hyper_hidden_dim, args.qmix_hidden_dim))
#         else:
#             self.hyper_w1 = nn.Linear(args.state_shape, args.n_agents * args.qmix_hidden_dim)
#             self.hyper_w2 = nn.Linear(args.state_shape, args.qmix_hidden_dim * 1)
#
#         self.hyper_b1 = nn.Linear(args.state_shape, args.qmix_hidden_dim)
#         self.hyper_b2 = nn.Sequential(nn.Linear(args.state_shape, args.qmix_hidden_dim), nn.ReLU(), nn.Linear(args.qmix_hidden_dim, 1))
#
#     def forward(self, q_values, states):
#         episode_num = q_values.size(0)
#         q_values = q_values.view(-1, 1, self.args.n_agents)
#         states = states.reshape(-1, self.args.state_shape)  # (1920, 120)
#
#         w1 = torch.abs(self.hyper_w1(states))  # (1920, 160)
#         b1 = self.hyper_b1(states)  # (1920, 32)
#
#         w1 = w1.view(-1, self.args.n_agents, self.args.qmix_hidden_dim)  # (1920, 5, 32)
#         b1 = b1.view(-1, 1, self.args.qmix_hidden_dim)  # (1920, 1, 32)
#
#         hidden = F.elu(torch.bmm(q_values, w1) + b1)  # (1920, 1, 32)
#
#         w2 = torch.abs(self.hyper_w2(states))  # (1920, 32)
#         b2 = self.hyper_b2(states)  # (1920, 1)
#
#         w2 = w2.view(-1, self.args.qmix_hidden_dim, 1)  # (1920, 32, 1)
#         b2 = b2.view(-1, 1, 1)  # (1920, 1， 1)
#
#         q_total = torch.bmm(hidden, w2) + b2  # (1920, 1, 1)
#         q_total = q_total.view(episode_num, -1, 1)  # (32, 60, 1)
#
#         return q_total
class Mixer(nn.Module):
    def __init__(self, args, abs=True):
        super(Mixer, self).__init__()

        self.args = args
        self.abs = abs
        self.n_agents = args.n_agents
        self.embed_dim = args.qmix_hidden_dim
        self.input_dim = self.state_dim = int(np.prod(args.state_shape))

        # hyper w1 b1
        self.hyper_w1 = nn.Sequential(nn.Linear(self.input_dim, args.hypernet_embed),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(args.hypernet_embed, self.n_agents * self.embed_dim))
        self.hyper_b1 = nn.Sequential(nn.Linear(self.input_dim, self.embed_dim))

        # hyper w2 b2
        self.hyper_w2 = nn.Sequential(nn.Linear(self.input_dim, args.hypernet_embed),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(args.hypernet_embed, self.embed_dim))
        self.hyper_b2 = nn.Sequential(nn.Linear(self.input_dim, self.embed_dim),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(self.embed_dim, 1))

    def forward(self, qvals, states):
        # reshape
        b, t, _ = qvals.size()

        qvals = qvals.reshape(b * t, 1, self.n_agents)
        states = states.reshape(-1, self.state_dim)

        # First layer
        w1 = self.hyper_w1(states).view(-1, self.n_agents, self.embed_dim)  # b * t, n_agents, emb
        b1 = self.hyper_b1(states).view(-1, 1, self.embed_dim)

        # Second layer
        w2 = self.hyper_w2(states).view(-1, self.embed_dim, 1)  # b * t, emb, 1
        b2 = self.hyper_b2(states).view(-1, 1, 1)

        if self.abs:
            w1 = w1.abs()
            w2 = w2.abs()

        # Forward
        hidden = F.elu(th.matmul(qvals, w1) + b1)  # b * t, 1, emb
        y = th.matmul(hidden, w2) + b2  # b * t, 1, 1

        return y.view(b, t, -1)

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class QMixerCentralFF(nn.Module):
    def __init__(self, args):
        super(QMixerCentralFF, self).__init__()

        self.args = args

        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))
        self.central_action_embed=1
        self.input_dim = self.n_agents * self.central_action_embed + self.state_dim
        self.embed_dim = 256

        non_lin = nn.ReLU

        self.net = nn.Sequential(nn.Linear(self.input_dim, self.embed_dim),
                                 non_lin(),
                                 nn.Linear(self.embed_dim, self.embed_dim),
                                 non_lin(),
                                 nn.Linear(self.embed_dim, self.embed_dim),
                                 non_lin(),
                                 nn.Linear(self.embed_dim, 1))

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                               non_lin(),
                               nn.Linear(self.embed_dim, 1))

    def forward(self, agent_qs, states):
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.reshape(-1, self.n_agents * self.central_action_embed)

        inputs = th.cat([states, agent_qs], dim=1)

        advs = self.net(inputs)
        vs = self.V(states)

        y = advs + vs

        q_tot = y.view(bs, -1, 1)
        return q_tot

class QMixNet(nn.Module):
    def __init__(self, args):
        super(QMixNet, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))

        self.embed_dim = args.qmix_hidden_dim
        self.abs = getattr(self.args, 'abs', True)

        if getattr(args, "hypernet_layers", 1) == 1:
            self.hyper_w_1 = nn.Linear(self.state_dim, self.embed_dim * self.n_agents)
            self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim)
        elif getattr(args, "hypernet_layers", 1) == 2:
            hypernet_embed = self.args.hypernet_embed
            self.hyper_w_1 = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                           nn.ReLU(inplace=True),
                                           nn.Linear(hypernet_embed, self.embed_dim * self.n_agents))
            self.hyper_w_final = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                               nn.ReLU(inplace=True),
                                               nn.Linear(hypernet_embed, self.embed_dim))
        elif getattr(args, "hypernet_layers", 1) > 2:
            raise Exception("Sorry >2 hypernet layers is not implemented!")
        else:
            raise Exception("Error setting number of hypernet layers.")

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                               nn.ReLU(inplace=True),
                               nn.Linear(self.embed_dim, 1))

    def forward(self, agent_qs, states):
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.reshape(-1, 1, self.n_agents)
        # First layer
        w1 = self.hyper_w_1(states).abs() if self.abs else self.hyper_w_1(states)
        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        hidden = F.elu(th.bmm(agent_qs, w1) + b1)

        # Second layer
        w_final = self.hyper_w_final(states).abs() if self.abs else self.hyper_w_final(states)
        w_final = w_final.view(-1, self.embed_dim, 1)
        # State-dependent bias
        v = self.V(states).view(-1, 1, 1)
        # Compute final output
        y = th.bmm(hidden, w_final) + v
        # Reshape and return
        q_tot = y.view(bs, -1, 1)

        return q_tot

    def k(self, states):
        bs = states.size(0)
        w1 = th.abs(self.hyper_w_1(states))
        w_final = th.abs(self.hyper_w_final(states))
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        w_final = w_final.view(-1, self.embed_dim, 1)
        k = th.bmm(w1, w_final).view(bs, -1, self.n_agents)
        k = k / th.sum(k, dim=2, keepdim=True)
        return k

    def b(self, states):
        bs = states.size(0)
        w_final = th.abs(self.hyper_w_final(states))
        w_final = w_final.view(-1, self.embed_dim, 1)
        b1 = self.hyper_b_1(states)
        b1 = b1.view(-1, 1, self.embed_dim)
        v = self.V(states).view(-1, 1, 1)
        b = th.bmm(b1, w_final) + v
        return b
