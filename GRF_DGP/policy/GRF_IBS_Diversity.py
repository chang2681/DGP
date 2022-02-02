import copy
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from network.base_net import MLP, RNN
from network.predict_net import Predict_Network1, Predict_Network1_combine, Predict_mse, Predict_combine_mse
from network.QPLEX.dmaq_general import DMAQer
from network.QPLEX.dmaq_qatten import DMAQ_QattenMixer
from network.qmix_net import QMixNet, Mixer

from torch.optim import RMSprop
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import csv, random
import torch as th

MARKER_ENEMY_VALUE = -1
MARKER_ENEMY_ALLY = 1


def build_td_lambda_targets(rewards, terminated, mask, target_qs, n_agents, gamma, td_lambda):
    # Assumes  <target_qs > in B*T*A and <reward >, <terminated >, <mask > in (at least) B*T-1*1
    # Initialise  last  lambda -return  for  not  terminated  episodes
    ret = target_qs.new_zeros(*target_qs.shape)
    ret[:, -1] = target_qs[:, -1] * (1 - torch.sum(terminated, dim=1))
    # Backwards  recursive  update  of the "forward  view"
    for t in range(ret.shape[1] - 2, -1, -1):
        ret[:, t] = td_lambda * gamma * ret[:, t + 1] + mask[:, t] \
                    * (rewards[:, t] + (1 - td_lambda) * gamma * target_qs[:, t + 1] * (1 - terminated[:, t]))
    # Returns lambda-return from t=0 to t=T-1, i.e. in B*T-1*A
    return ret[:, 0:-1]


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class GRF_IBS_Diversity:

    def __init__(self, args):
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.n_enemy = args.n_enemy

        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        input_shape = self.obs_shape
        print('GPU:', args.GPU)
        if args.last_action:
            input_shape += self.n_actions
        if args.add_id:
            input_shape += self.n_agents
        setup_seed(args.seed)
        self.args = args
        self.ac_seq = args.ac_seq
        self.env = args.env
        self.intrinsic_clip=args.intrinsic_clip
        if args.env == '3_vs_2' or args.env == '3_vs_3':
            MINIMAP_NORM_X_MIN = 0.1
            MINIMAP_NORM_X_MAX = 1.0
            MINIMAP_NORM_Y_MIN = -0.4
            MINIMAP_NORM_Y_MAX = 0.4
            self.SMM_LENGTH = 10  # 10 13,8 11
            self.SMM_HEIGHT = 13
            self.MINIMAP_Min = np.array([MINIMAP_NORM_X_MIN, MINIMAP_NORM_Y_MIN])
            self.MINIMAP_Range = np.array(
                [MINIMAP_NORM_X_MAX - MINIMAP_NORM_X_MIN, MINIMAP_NORM_Y_MAX - MINIMAP_NORM_Y_MIN])
            self.MINIMAP_Shape = np.array([self.SMM_LENGTH, self.SMM_HEIGHT])
            col = np.linspace(.04, .08, num=self.SMM_LENGTH + 2).reshape(1, -1)
            row = np.linspace(.04, .08, num=self.SMM_HEIGHT + 2).reshape(-1, 1)
            self.env_matrix = th.FloatTensor(-row - col)
        elif args.env == '3_vs_4_BackBall' or args.env == '4_vs_3':
            MINIMAP_NORM_X_MIN = 0.1
            MINIMAP_NORM_X_MAX = 1.0
            MINIMAP_NORM_Y_MIN = -0.4
            MINIMAP_NORM_Y_MAX = 0.4
            self.SMM_LENGTH = 8  # 10 13,8 11
            self.SMM_HEIGHT = 11
            self.MINIMAP_Min = np.array([MINIMAP_NORM_X_MIN, MINIMAP_NORM_Y_MIN])
            self.MINIMAP_Range = np.array(
                [MINIMAP_NORM_X_MAX - MINIMAP_NORM_X_MIN, MINIMAP_NORM_Y_MAX - MINIMAP_NORM_Y_MIN])
            self.MINIMAP_Shape = np.array([self.SMM_LENGTH, self.SMM_HEIGHT])
            col = np.linspace(.04, .08, num=self.SMM_LENGTH + 2).reshape(1, -1)
            row = np.linspace(.04, .08, num=self.SMM_HEIGHT + 2).reshape(-1, 1)
            self.env_matrix = th.FloatTensor(-row - col)
        else:
            MINIMAP_NORM_X_MIN = -1.0
            MINIMAP_NORM_X_MAX = 1.0
            MINIMAP_NORM_Y_MIN = -0.41
            MINIMAP_NORM_Y_MAX = 0.41
            self.SMM_LENGTH = 18  # 10 13,8 11
            self.SMM_HEIGHT = 11
            self.MINIMAP_Min = np.array([MINIMAP_NORM_X_MIN, MINIMAP_NORM_Y_MIN])
            self.MINIMAP_Range = np.array(
                [MINIMAP_NORM_X_MAX - MINIMAP_NORM_X_MIN, MINIMAP_NORM_Y_MAX - MINIMAP_NORM_Y_MIN])
            self.MINIMAP_Shape = np.array([self.SMM_LENGTH, self.SMM_HEIGHT])
            col = np.linspace(.02, .1, num=self.SMM_LENGTH + 2).reshape(1, -1)
            row = np.linspace(.04, .08, num=self.SMM_HEIGHT + 2).reshape(-1, 1)
            self.env_matrix = th.FloatTensor(-row - col)
        if args.QPLEX_mixer == "dmaq":
            self.eval_mix_net = DMAQer(args)
            self.target_mix_net = DMAQer(args)
        elif args.QPLEX_mixer == 'dmaq_qatten':
            self.eval_mix_net = DMAQ_QattenMixer(args)
            self.target_mix_net = DMAQ_QattenMixer(args)
        else:
            raise ValueError(
                "Mixer {} not recognised.".format(args.QPLEX_mixer))

        self.eval_rnn = RNN(input_shape, args)
        self.target_rnn = RNN(input_shape, args)

        self.eval_mlp = nn.ModuleList([MLP(args)
                                       for _ in range(args.n_agents)])
        self.target_mlp = nn.ModuleList(
            [MLP(args) for _ in range(args.n_agents)])

        self.player_states = 9

        self.team_state = (self.SMM_HEIGHT + 2) * (self.SMM_LENGTH + 2)  # args.state_shape
        base_input_dims = self.team_state
        self.eval_leader_state_base = Predict_mse(
            base_input_dims, 128, self.player_states, False)
        self.target_leader_state_base = Predict_mse(
            base_input_dims, 128, self.player_states, False)

        input_plus_dims = args.n_agents
        self.eval_leader_state_plus = Predict_combine_mse(
            base_input_dims + input_plus_dims, 128,
            self.player_states, input_plus_dims, False)
        self.target_leader_state_plus = Predict_combine_mse(
            base_input_dims + input_plus_dims, 128,
            self.player_states, input_plus_dims, False)

        self.model_dir = f'{args.model_dir}/{args.env}/seed_{args.seed}_{args.label}'
        ############################
        if self.args.render:
            self.load_model(args.model_step)
        self.csv_dir = f'./csv_file/{args.env}/reward'
        self.csv_path = f'{self.csv_dir}/seed_{args.seed}_{args.label}.csv'
        if not os.path.exists(self.csv_dir):
            os.makedirs(self.csv_dir)
        ##############################

        if self.args.cuda:
            self.eval_rnn.to(torch.device(self.args.GPU))
            self.target_rnn.to(torch.device(self.args.GPU))
            self.eval_mix_net.to(torch.device(self.args.GPU))
            self.target_mix_net.to(torch.device(self.args.GPU))

            self.eval_mlp.to(torch.device(self.args.GPU))
            self.target_mlp.to(torch.device(self.args.GPU))

            self.eval_leader_state_plus.to(torch.device(self.args.GPU))
            self.target_leader_state_plus.to(torch.device(self.args.GPU))

            self.eval_leader_state_base.to(torch.device(self.args.GPU))
            self.target_leader_state_base.to(torch.device(self.args.GPU))

        self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
        self.target_mix_net.load_state_dict(self.eval_mix_net.state_dict())
        self.target_mlp.load_state_dict(self.eval_mlp.state_dict())
        self.target_leader_state_plus.load_state_dict(
            self.eval_leader_state_plus.state_dict())
        self.target_leader_state_base.load_state_dict(
            self.eval_leader_state_base.state_dict())
        if self.args.render:
            self.load_model(args.model_step)
        self.eval_parameters = list(self.eval_mix_net.parameters(
        )) + list(self.eval_rnn.parameters()) + list(self.eval_mlp.parameters())

        if args.optimizer == "RMS":
            self.optimizer = torch.optim.RMSprop(
                self.eval_parameters, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        self.eval_hidden = None
        self.target_hidden = None

    def mark_point(self, pos):
        point = ((pos - self.MINIMAP_Min) / self.MINIMAP_Range * self.MINIMAP_Shape).astype('int64')
        point[..., 0] = point[..., 0].clip(min=0, max=self.SMM_LENGTH - 1)
        point[..., 1] = point[..., 1].clip(min=0, max=self.SMM_HEIGHT - 1)
        return point

    def get_diversity_input(self, s):
        """Draw dots corresponding to 'points'.

        Args:
          frame: 2-d matrix representing one SMM channel ([y, x])
          points: a list of (x, y) coordinates to be marked
        """

        s = np.array(s)
        dim_batches, dim_steps, _ = s.shape
        env_matrix = self.env_matrix.reshape(1, 1, self.env_matrix.shape[0], self.env_matrix.shape[1]).repeat(
            dim_batches, dim_steps, 1, 1)  # .to(s.device)
        field_matrix = np.array(env_matrix[..., 1:-1, 1:-1].clone())

        pos_ally = s[..., :self.n_agents * 2].reshape(dim_batches, dim_steps, self.n_agents, 2)
        pos_enemy = s[..., self.n_agents * 2:self.n_agents * 2 + self.n_enemy * 2].reshape(dim_batches, dim_steps,
                                                                                           self.n_enemy, 2)
        index_batch = np.array(list(range(dim_batches))).astype('int64').reshape(dim_batches, 1, 1)
        index_steps = np.array(list(range(dim_steps))).astype('int64').reshape(1, dim_steps, 1)
        ally_point = self.mark_point(pos_ally)
        enemy_point = self.mark_point(pos_enemy)
        # ally_index=SMM_LENGTH*ally_point[...,1]+ally_point[...,0]
        field_matrix[index_batch, index_steps, enemy_point[..., 1], enemy_point[..., 0]] = -1
        field_matrix[index_batch, index_steps, ally_point[..., 1], ally_point[..., 0]] = 1
        s_diversity = np.array(env_matrix.clone())
        s_diversity[..., 1:-1, 1:-1] = field_matrix
        dx = [-1, 0, 1, -1, 0, 1, -1, 0, 1]
        dy = [-1, -1, -1, 0, 0, 0, 1, 1, 1]
        ally_obs_point = (np.array([dx, dy]).T)[np.newaxis].repeat(self.n_agents, 0).reshape(-1, 2) + ally_point.repeat(
            9, -2)
        obs_diversity = s_diversity[index_batch, index_steps, 1 + ally_obs_point[..., 1], 1 + ally_obs_point[..., 0]]

        return s_diversity.reshape(dim_batches, dim_steps, -1), obs_diversity.reshape(dim_batches, dim_steps,
                                                                                      self.n_agents, -1)

    def learn(self, batch, max_episode_len, train_step, t_env, epsilon=None):

        episode_num = batch['o'].shape[0]
        self.init_hidden(episode_num)
        for key in batch.keys():
            if key == 'u':
                batch[key] = torch.tensor(batch[key], dtype=torch.long)
            else:
                batch[key] = torch.tensor(batch[key], dtype=torch.float32)
        s, s_next, u, r, avail_u, avail_u_next, terminated = batch['s'], batch['s_next'], batch['u'], \
                                                             batch['r'], batch['avail_u'], batch['avail_u_next'], \
                                                             batch['terminated']
        mask = 1 - batch["padded"].float()
        obs, obs_next = batch['o'], batch['o_next']  # (8,63,3,26)

        q_evals, q_targets, q_evals_local, q_evals_last, intrinsic_rewards, intrinsic_1, intrinsic_2 = self.get_q_values(
            batch, max_episode_len)
        if t_env > self.args.start_anneal_time:
            if self.args.anneal_type == 'linear':
                intrinsic_rewards = max(1 - self.args.anneal_rate * (
                        t_env - self.args.start_anneal_time) / 1000000, 0) * intrinsic_rewards
            elif self.args.anneal_type == 'exp':
                exp_scaling = (-1) * (1 / self.args.anneal_rate) / np.log(0.01)
                TTT = (t_env - self.args.start_anneal_time) / 1000000
                intrinsic_rewards = intrinsic_rewards * \
                                    min(1, max(0.01, np.exp(-TTT / exp_scaling)))

        mac_out = q_evals.clone().detach()

        if self.args.cuda:
            obs = obs.to(torch.device(self.args.GPU))
            obs_next = obs.to(torch.device(self.args.GPU))
            s = s.to(torch.device(self.args.GPU))
            u = u.to(torch.device(self.args.GPU))
            r = r.to(torch.device(self.args.GPU))
            s_next = s_next.to(torch.device(self.args.GPU))
            terminated = terminated.to(torch.device(self.args.GPU))
            mask = mask.to(torch.device(self.args.GPU))

        max_action_qvals, _ = q_evals.max(dim=3)
        q_evals = torch.gather(q_evals, dim=3, index=u).squeeze(3)
        curr_actions_onehot = torch.zeros(
            u.squeeze(3).shape + (self.n_actions,))
        if self.args.cuda:
            curr_actions_onehot = curr_actions_onehot.to(
                torch.device(self.args.GPU))

        curr_actions_onehot = curr_actions_onehot.scatter_(3, u, 1)

        with torch.no_grad():

            q_targets[avail_u_next == 0.0] = -9999999

            if self.args.double_q:
                # Get actions that maximise live Q (for double q-learning)
                mac_out[avail_u == 0] = -9999999
                cur_max_actions = mac_out[:, 1:].max(dim=3, keepdim=True)[1]
                target_last_max_actions = q_evals_last.unsqueeze(
                    1).max(dim=3, keepdim=True)[1]
                double_max_actions = torch.cat(
                    [cur_max_actions, target_last_max_actions], dim=1)
                target_max_qvals = q_targets.max(dim=3)[0]
                q_targets = torch.gather(
                    q_targets, 3, double_max_actions).squeeze(3)

                cur_max_actions_onehot = torch.zeros(
                    double_max_actions.squeeze(3).shape + (self.n_actions,))
                if self.args.cuda:
                    cur_max_actions_onehot = cur_max_actions_onehot.to(
                        torch.device(self.args.GPU))
                cur_max_actions_onehot = cur_max_actions_onehot.scatter_(
                    3, double_max_actions, 1)

            else:
                q_targets = q_targets.max(dim=3)[0]
                target_max_qvals = q_targets.max(dim=3)[0]

        if self.args.QPLEX_mixer == "dmaq_qatten":
            ans_chosen, q_attend_regs, _ = self.eval_mix_net(
                q_evals, s, obs, is_v=True)
            ans_adv, _, _ = self.eval_mix_net(
                q_evals, s, obs, actions=curr_actions_onehot, max_q_i=max_action_qvals, is_v=False)
            chosen_action_qvals = ans_chosen + ans_adv
        else:
            ans_chosen = self.eval_mix_net(q_evals, s, is_v=True)
            ans_adv = self.eval_mix_net(
                q_evals, s, actions=curr_actions_onehot, max_q_i=max_action_qvals, is_v=False)
            chosen_action_qvals = ans_chosen + ans_adv

        with torch.no_grad():
            if self.args.double_q:
                if self.args.QPLEX_mixer == "dmaq_qatten":
                    target_chosen, _, _ = self.target_mix_net(
                        q_targets, s_next, obs_next, is_v=True)
                    target_adv, _, _ = self.target_mix_net(
                        q_targets, s_next, obs_next, actions=cur_max_actions_onehot, max_q_i=target_max_qvals,
                        is_v=False)
                    target_max_qvals = target_chosen + target_adv
                else:
                    target_chosen = self.target_mix_net(
                        q_targets, s_next, is_v=True)
                    target_adv = self.target_mix_net(
                        q_targets, s_next, actions=cur_max_actions_onehot, max_q_i=target_max_qvals, is_v=False)
                    target_max_qvals = target_chosen + target_adv
            else:
                target_max_qvals = self.target_mix_net(
                    target_max_qvals, s_next, is_v=True)
            #
            # if 'pacmen' in self.env:
            if False:
                targets = r + self.args.beta * \
                          intrinsic_rewards.mean(
                              dim=-2) + self.args.gamma * (1 - terminated) * target_max_qvals
            else:
                target_max_qvals = torch.cat((target_max_qvals[:, 0, ...].unsqueeze(1), target_max_qvals), dim=1)
                rewards = r + self.args.beta * intrinsic_rewards.mean(dim=-2)
                targets = build_td_lambda_targets(rewards, terminated, mask, target_max_qvals,
                                                  self.args.n_agents, self.args.gamma, 0.6)

            # #Calculate 1-step Q-Learning targets

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask
        update_prior = (masked_td_error ** 2).squeeze().sum(dim=-1,
                                                            keepdim=True) / mask.squeeze().sum(dim=-1, keepdim=True)

        # Normal L2 loss, take mean over actual data
        if self.args.QPLEX_mixer == "dmaq_qatten":
            loss = (masked_td_error ** 2).sum() / mask.sum() + q_attend_regs
        else:
            loss = (masked_td_error ** 2).sum() / mask.sum()
        norm_loss = torch.tensor([0])
        if self.args.local_q_L1loss:
            norm_loss = F.l1_loss(q_evals_local, target=torch.zeros_like(
                q_evals_local), reduction='mean')
            loss += self.args.norm_weight * norm_loss
        # Optimise
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.eval_parameters, self.args.grad_norm_clip)
        self.optimizer.step()
        if random.random() < 0.005:
            mask = mask.unsqueeze(-2)
            self.writereward(self.csv_path, mask, self.args.beta1 * self.args.beta * intrinsic_1 * mask,
                             self.args.beta2 * self.args.beta * intrinsic_2 * mask,
                             r * mask[..., 0], masked_td_error, loss, t_env)

        if train_step > 0 and train_step % self.args.target_update_cycle == 0:
            self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
            self.target_mix_net.load_state_dict(self.eval_mix_net.state_dict())
            self.target_mlp.load_state_dict(self.eval_mlp.state_dict())

            self.target_leader_state_plus.load_state_dict(
                self.eval_leader_state_plus.state_dict())
            self.target_leader_state_base.load_state_dict(
                self.eval_leader_state_base.state_dict())

        return update_prior.squeeze().detach()

    def _get_inputs_matrix(self, batch):
        obs, obs_next = batch['o'], batch['o_next']

        obs_clone = obs.clone()
        obs_next_clone = obs_next.clone()

        if self.args.last_action:
            u_onehot = batch['u_onehot']
            u_onehot_f = torch.zeros_like(u_onehot)
            u_onehot_f[:, 1:, :, :] = u_onehot[:, :-1, :, :]

            obs = torch.cat([obs, u_onehot_f],
                            dim=-1)  # 观测加上上一步动作 torch.Size([8, 63, 3, 19]) to torch.Size([8, 63, 3, 45])
            obs_next = torch.cat([obs_next, u_onehot], dim=-1)

        add_id = torch.eye(self.args.n_agents).type_as(obs).expand(
            [obs.shape[0], obs.shape[1], self.args.n_agents, self.args.n_agents])

        if self.args.add_id:  # False
            obs = torch.cat([obs, add_id], dim=-1)
            obs_next = torch.cat([obs_next, add_id], dim=-1)

        return obs, obs_next, obs_clone, obs_next_clone, add_id

    def get_q_values(self, batch, max_episode_len):

        inputs, inputs_next, obs, obs_next, add_id = self._get_inputs_matrix(
            batch)  # inputs and _next torch.Size([8, 63, 3, 45]),相比于obs增加了动作onehot
        inputs = torch.cat([inputs, inputs_next[:, -1].unsqueeze(1)], dim=1)  #
        inputs_shape = inputs.shape  # torch.Size([8, 64, 3, 45])
        mask = 1 - batch["padded"].float()  # torch.Size([8, 63, 1])
        s = batch['s']
        s_diversity, obs_diversity = self.get_diversity_input(s)
        # s_diversity.reshape(s_diversity.shape[0],s_diversity.shape[1],17,14)
        if self.args.cuda:
            s_diversity = th.FloatTensor(s_diversity).to(torch.device(self.args.GPU))
            obs_diversity = th.FloatTensor(obs_diversity).to(torch.device(self.args.GPU))
            inputs = inputs.to(torch.device(self.args.GPU))
            inputs_next = inputs_next.to(torch.device(self.args.GPU))
            obs = obs.to(torch.device(self.args.GPU))
            mask = mask.to(torch.device(self.args.GPU))
            s = s.to(torch.device(self.args.GPU))

            self.eval_hidden = self.eval_hidden.to(torch.device(self.args.GPU))  # torch.Size([8, 3, 64])
            self.target_hidden = self.target_hidden.to(
                torch.device(self.args.GPU))

        add_id = add_id.to(inputs.device)  # torch.Size([8, 3, 63, 3])智能体和步数换维度

        eval_h = self.eval_hidden.view(-1, self.args.rnn_hidden_dim)  # torch.Size([24, 64]) （batch,hidden size）
        target_h = self.target_hidden.view(-1, self.args.rnn_hidden_dim)  # torch.Size([24, 64])



        inputs = inputs.permute(0, 2, 1, 3)  # torch.Size([8, 3, 64, 45])
        inputs_next = inputs_next.permute(0, 2, 1, 3)

        inputs = inputs.reshape(-1, inputs.shape[2], inputs.shape[3])  # torch.Size([24, 64, 45])
        inputs_next = inputs_next.reshape(-1,
                                          inputs_next.shape[2], inputs_next.shape[3])

        q_eval_global, out_eval_h = self.eval_rnn(inputs, eval_h)  # torch.Size([24, 64, 19])torch.Size([24,64, 64])
        q_target_global, out_target_h = self.target_rnn(inputs_next, target_h)

        q_eval_global = q_eval_global.reshape(inputs_shape[0], inputs_shape[2], q_eval_global.shape[-2],
                                              q_eval_global.shape[-1]).permute(0, 2, 1, 3)

        out_eval_h = out_eval_h.reshape(
            inputs_shape[0], inputs_shape[2], out_eval_h.shape[-2], out_eval_h.shape[-1]).permute(0, 2, 1,
                                                                                                  3)  # torch.Size([8, 64, 3, 64])

        q_target_global = q_target_global.reshape(inputs_shape[0], inputs_shape[2], q_target_global.shape[-2],
                                                  q_target_global.shape[-1]).permute(0, 2, 1,
                                                                                     3)  # torch.Size([8, 63, 3, 19])
        out_target_h = out_target_h.reshape(inputs_shape[0], inputs_shape[2], out_target_h.shape[-2],
                                            out_target_h.shape[-1]).permute(0, 2, 1, 3)

        if self.args.lf_q == 1:
            q_eval_local = torch.stack(
                [self.eval_mlp[id](out_eval_h[:, :, id].reshape(-1, out_eval_h.shape[-1]))
                 for id in range(self.args.n_agents)],
                dim=1).reshape_as(q_eval_global)

            q_target_local = torch.stack(
                [self.target_mlp[id](out_target_h[:, :, id].reshape(-1, out_target_h.shape[-1]))
                 for id in range(self.args.n_agents)],
                dim=1).reshape_as(q_target_global)  # torch.Size([8, 63, 3, 19])

            q_eval = q_eval_local
            q_target = q_target_local
        else:
            q_eval = q_eval_global
            q_target = q_target_global
            q_eval_local = torch.zeros_like(q_eval_global)

        with torch.no_grad():
            mask = mask.unsqueeze(-2).expand(obs.shape[:-1] + mask.shape[-1:])
            mask = mask.reshape(-1, mask.shape[-1])  # torch.Size([1512, 1])
            dim0_batches, dim1_steps, dim2_agents, dim3_obs = obs.shape
            pre_output = obs_diversity
            # ra = th.rand((dim0_batches, dim1_steps, dim2_agents))
            ac_input_base = s_diversity.unsqueeze(-2).repeat(1, 1, dim2_agents, 1)  #
            ac_input_plus = add_id
            ac_input_upgrade = th.cat((ac_input_base, ac_input_plus), dim=-1)

            ac_input_base = ac_input_base.reshape(-1, ac_input_base.shape[-1])
            ac_input_plus = ac_input_plus.reshape(-1, ac_input_plus.shape[-1])
            ac_input_upgrade = ac_input_upgrade.reshape(-1, ac_input_upgrade.shape[-1])
            pre_output = pre_output.reshape(-1, pre_output.shape[-1])

            log_p_o = self.target_leader_state_base.get_log_pi(
                ac_input_base, pre_output)  # 三层神经网络（109，128）（128，128）（128，26）
            log_q_o, pre_obs_raw = self.target_leader_state_plus.get_log_pi_plus(
                ac_input_upgrade, pre_output, ac_input_plus)

            intrinsic_1 = log_q_o - log_p_o  # torch.Size([3600, 1])
            pre_obs = pre_obs_raw.reshape(dim0_batches, dim1_steps, dim2_agents, -1).repeat(1, 1, 1,
                                                                                            dim2_agents).reshape(
                dim0_batches, dim1_steps, dim2_agents * dim2_agents, -1)
            # agent_obs = pre_obs_raw.clone().detach().reshape(a, b, c, -1).repeat(1, 1, c, 1)
            agent_obs = pre_output.clone().detach().reshape(dim0_batches, dim1_steps, dim2_agents, -1).repeat(1, 1,
                                                                                                              dim2_agents,
                                                                                                              1)
            dst = (((pre_obs - agent_obs) ** 2).sum(dim=-1)).reshape(dim0_batches, dim1_steps, dim2_agents, dim2_agents)
            agent_matrix = (torch.ones((dim2_agents, dim2_agents)) - dim2_agents * torch.eye(dim2_agents)).to(
                torch.device(self.args.GPU)).expand_as(dst)
            intrinsic_2 = (agent_matrix * dst).mean(dim=-1, keepdim=True)
            intrinsic_1 = intrinsic_1.reshape(dim0_batches, dim1_steps, dim2_agents, -1)
            if self.intrinsic_clip:
                intrinsic_1=intrinsic_1.clip(max=1.25)
                intrinsic_2=intrinsic_2.clip(max=2.5)

            intrinsic_rewards = self.args.beta1 * intrinsic_1 + self.args.beta2 * intrinsic_2
            # intrinsic_rewards= torch.clip(intrinsic_rewards,-1,1)

            # intrinsic_rewards = intrinsic_rewards + self.args.beta2 * pi_diverge

        # update predict network

        for index in BatchSampler(SubsetRandomSampler(range(ac_input_base.shape[0])), 256, False):
            self.eval_leader_state_base.update(
                ac_input_base[index], pre_output[index], mask[index])
            self.eval_leader_state_plus.update(
                ac_input_upgrade[index], pre_output[index], ac_input_plus[index], mask[index])

        return q_eval[:, :-1], q_target, q_eval_local[:, :-1], q_eval[:,
                                                               -1], intrinsic_rewards.detach(), intrinsic_1.detach(), intrinsic_2.detach()

    def save_model(self, train_step):
        num = str(train_step // self.args.save_cycle)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        torch.save(self.eval_mlp.state_dict(), self.model_dir +
                   '/' + num + '_mlp_net_params.pkl')
        torch.save(self.eval_mix_net.state_dict(), self.model_dir +
                   '/' + num + '_qplex_mix_net_params.pkl')
        torch.save(self.eval_rnn.state_dict(), self.model_dir +
                   '/' + num + '_rnn_net_params.pkl')

    def load_model(self, step):
        self.eval_mlp.load_state_dict(torch.load(
            f"{self.model_dir}/{step}_mlp_net_params.pkl", map_location='cpu'))
        self.eval_mix_net.load_state_dict(torch.load(
            f"{self.model_dir}/{step}_qplex_mix_net_params.pkl", map_location='cpu'))
        self.eval_rnn.load_state_dict(torch.load(
            f"{self.model_dir}/{step}_rnn_net_params.pkl", map_location='cpu'))

    def init_hidden(self, episode_num):
        self.eval_hidden = torch.zeros(
            episode_num, self.n_agents, self.args.rnn_hidden_dim)
        self.target_hidden = torch.zeros(
            episode_num, self.n_agents, self.args.rnn_hidden_dim)
        if self.args.cuda:
            self.eval_hidden = self.eval_hidden.to(torch.device(self.args.GPU))
            self.target_hidden = self.target_hidden.to(
                torch.device(self.args.GPU))

        for i in range(episode_num):
            for j in range(self.n_agents):
                self.eval_hidden[i, j] = self.eval_rnn.init_hidden()
                self.target_hidden[i, j] = self.target_rnn.init_hidden()

    def writereward(self, path, mask, intrin_reward_1, intrin_reward_2, reward, td_error, loss, step):
        mask_elems = mask.sum()
        intrin_reward_mean_1 = intrin_reward_1.sum() / mask_elems
        intrin_reward_sum_1 = intrin_reward_1.squeeze().sum(axis=-2).reshape(-1).mean()
        intrin_reward_mean_2 = intrin_reward_2.sum() / mask_elems
        intrin_reward_sum_2 = intrin_reward_2.squeeze().sum(axis=-2).reshape(-1).mean()
        reward = reward.squeeze().sum(axis=-1).reshape(-1).mean()
        if os.path.isfile(path):
            with open(path, 'a+') as f:
                csv_write = csv.writer(f)
                csv_write.writerow(
                    [step, intrin_reward_mean_1.item(), intrin_reward_sum_1.item(), intrin_reward_mean_2.item(),
                     intrin_reward_sum_2.item(), reward.item(),
                     reward.item() + intrin_reward_sum_1.item() + intrin_reward_sum_2.item(),
                     td_error.mean().item(), loss.item()])

        else:
            with open(path, 'w') as f:
                csv_write = csv.writer(f)
                csv_write.writerow(
                    ['step', 'intrinsic1_mean', 'intrinsic1_sum', 'intrinsic2_mean', 'intrinsic2_sum', 'reward', 'sum',
                     'td_error', 'loss'])
                csv_write.writerow(
                    [step, intrin_reward_mean_1.item(), intrin_reward_sum_1.item(), intrin_reward_mean_2.item(),
                     intrin_reward_sum_2.item(), reward.item(),
                     reward.item() + intrin_reward_sum_1.item() + intrin_reward_sum_2.item(),
                     td_error.mean().item(), loss.item()])
