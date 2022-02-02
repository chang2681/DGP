import gfootball.env as football_env
from gfootball.env import observation_preprocessing
import gym
import numpy as np
import matplotlib.pyplot as plt


class GoogleFootballMultiAgentEnv(object):
    """An wrapper for GFootball to make it compatible with our codebase."""

    def __init__(self, seed,dense_reward, dump_freq, render=False):
        self.n_agents = 3
        self.time_limit = 200
        self.time_step = 0
        self.obs_dim =48 # for counterattack_easy 4 vs 2
        self.state_shape =65
        self.n_enemy = 2
        self.dense_reward = dense_reward  # select whether to use dense reward
        self.get_ball_rew=False

        self.env = football_env.create_environment(
            env_name='academy_3v2',
            stacked=False,
            representation="simple115",
            rewards='scoring',
            logdir='/home/liuboyin/MARL/Diversity/video',
            render=render,
            write_goal_dumps=render,
            write_full_episode_dumps=render,

            write_video=True,
            dump_frequency=dump_freq,
            number_of_left_players_agent_controls=self.n_agents,
            number_of_right_players_agent_controls=0,
            channel_dimensions=(observation_preprocessing.SMM_WIDTH, observation_preprocessing.SMM_HEIGHT))
        self.env.seed(seed)
        obs_space_low = self.env.observation_space.low[0][:self.obs_dim]
        obs_space_high = self.env.observation_space.high[0][:self.obs_dim]

        self.action_space = [gym.spaces.Discrete(
            self.env.action_space.nvec[1]) for _ in range(self.n_agents)]
        self.observation_space = [
            gym.spaces.Box(low=obs_space_low, high=obs_space_high, dtype=self.env.observation_space.dtype) for _ in range(self.n_agents)
        ]

    def get_obs(self, index=-1):
        full_obs = self.env.unwrapped.observation()[0]
        simple_obs = []
        ball_x, ball_y, ball_z = full_obs['ball']
        if full_obs['ball_owned_team'] == 0:
            ball_owned = [1, 0, 0]
        elif full_obs['ball_owned_team'] == 1:
            ball_owned = [0, 1, 0]
        else:
            ball_owned = [0, 0, 1]
        if index == -1:
            # global state, absolute position
            obs_left_team_pos = full_obs['left_team'][-self.n_agents:]
            gate_relative = np.array([1,0]) - full_obs['left_team'][-self.n_agents:]
            gate_relative_dst = np.linalg.norm(gate_relative, axis=-1).reshape(-1,1)
            gate_theta = np.arctan2(gate_relative[:,1], gate_relative[:,0]).reshape(-1,1)
            obs_left_team= np.concatenate(
                ( gate_relative,gate_relative_dst,gate_theta), axis=1).reshape(-1)  # 14


            obs_left_team_direction = full_obs['left_team_direction'][-self.n_agents:]
            left_team_speed = np.linalg.norm(obs_left_team_direction, axis=1, keepdims=True)

            left_team_vel = np.concatenate(
                (obs_left_team_direction, left_team_speed), axis=1).reshape(-1)  # 21
            obs_right_team_pos = np.array(full_obs['right_team'])
            obs_right_team_direction = np.array(full_obs['right_team_direction'])
            right_team_speed = np.linalg.norm(obs_right_team_direction, axis=1, keepdims=True)
            right_team_state = np.concatenate(
                (obs_right_team_direction,right_team_speed), axis=1).reshape(-1)  # 14
            # right_team_state = np.concatenate(
            #     (obs_right_team, obs_right_team_direction,right_team_speed), axis=1).reshape(-1)  # 14
            # simple_obs=ball_state



            ball_which_zone = self._encode_ball_which_zone(ball_x, ball_y)

            #
            ball_relative = full_obs['ball'][:2] - full_obs['left_team'][-self.n_agents:]
            ball_relative_dst = np.linalg.norm(ball_relative, axis=-1)
            ball_theta = np.arctan2(ball_relative[:,1], ball_relative[:,0])
            ball_rel=np.concatenate((ball_relative,ball_relative_dst[:,np.newaxis],ball_theta[:,np.newaxis]),axis=-1)
            ball_state = np.concatenate((ball_rel.flatten(),
                                         np.array(ball_which_zone),
                                         np.array(full_obs['ball_direction']),
                                         np.array(ball_owned),
                                         np.array(np.linalg.norm(full_obs['ball_direction'][:2])).reshape(-1)))  # 24

            simple_obs = np.concatenate((obs_left_team_pos.flatten(),obs_right_team_pos.flatten(),np.array(full_obs['ball']),obs_left_team.reshape(-1),left_team_vel, ball_state, right_team_state))

        else:
            # local state, relative position
            ego_position = full_obs['left_team'][-self.n_agents +
                                                 index].reshape(-1)
            simple_obs.append(ego_position)
            simple_obs.append((np.delete(
                full_obs['left_team'][-self.n_agents:], index, axis=0) - ego_position).reshape(-1))

            simple_obs.append(
                full_obs['left_team_direction'][-self.n_agents + index].reshape(-1))
            simple_obs.append(np.delete(
                full_obs['left_team_direction'][-self.n_agents:], index, axis=0).reshape(-1))

            simple_obs.append(
                (full_obs['right_team'] - ego_position).reshape(-1))
            simple_obs.append(full_obs['right_team_direction'].reshape(-1))

            ball_which_zone = self._encode_ball_which_zone(ball_x, ball_y)


            ball_relative = full_obs['ball'][:2] - full_obs['left_team'][-self.n_agents:]
            ball_relative_dst = np.linalg.norm(ball_relative, axis=-1)
            ball_theta = np.arctan2(ball_relative[:,1], ball_relative[:,0])
            ball_state = np.concatenate((np.array(full_obs['ball']),
                                         np.array(ball_which_zone),
                                         np.array(ball_relative).reshape(-1),
                                         np.array(full_obs['ball_direction']),
                                         np.array(ball_owned),
                                         ball_relative_dst,
                                         ball_theta,
                                         np.array(np.linalg.norm(full_obs['ball_direction'][:2])).reshape(-1)))  # 24
            simple_obs.append(ball_state)
            simple_obs = np.concatenate(simple_obs)



        return simple_obs

    def get_global_state(self):
        return self.get_obs(-1)
    def _encode_ball_which_zone(self, ball_x, ball_y):
        MIDDLE_X, PENALTY_X, END_X = 0.2, 0.64, 1.0
        PENALTY_Y, END_Y = 0.27, 0.42
        if (-END_X <= ball_x and ball_x < -PENALTY_X) and (-PENALTY_Y < ball_y and ball_y < PENALTY_Y):
            return [1.0, 0, 0, 0, 0, 0]
        elif (-END_X <= ball_x and ball_x < -MIDDLE_X) and (-END_Y < ball_y and ball_y < END_Y):
            return [0, 1.0, 0, 0, 0, 0]
        elif (-MIDDLE_X <= ball_x and ball_x <= MIDDLE_X) and (-END_Y < ball_y and ball_y < END_Y):
            return [0, 0, 1.0, 0, 0, 0]
        elif (PENALTY_X < ball_x and ball_x <= END_X) and (-PENALTY_Y < ball_y and ball_y < PENALTY_Y):
            return [0, 0, 0, 1.0, 0, 0]
        elif (MIDDLE_X < ball_x and ball_x <= END_X) and (-END_Y < ball_y and ball_y < END_Y):
            return [0, 0, 0, 0, 1.0, 0]
        else:
            return [0, 0, 0, 0, 0, 1.0]
    def reset(self):
        self.time_step = 0
        self.get_ball_rew=False

        self.env.reset()
        obs = np.array([self.get_obs(i) for i in range(self.n_agents)])

        return obs, self.get_global_state()

    def check_if_done(self):
        cur_obs = self.env.unwrapped.observation()[0]
        ball_loc = cur_obs['ball']
        ours_loc = cur_obs['left_team'][-self.n_agents:]

        if ball_loc[0] < 0 or any(ours_loc[:, 0] < 0):
            return True

        return False

    def step(self, actions):

        self.time_step += 1
        _, original_rewards, done, infos = self.env.step(actions)
        rewards = list(original_rewards)
        obs = np.array([self.get_obs(i) for i in range(self.n_agents)])
        full_obs = self.env.unwrapped.observation()[0]

        if self.time_step >= self.time_limit:
            done = True

        if self.check_if_done():
            done = True

        if sum(rewards) <= 0:
            if not self.get_ball_rew and full_obs['ball_owned_team'] == 0:
                self.get_ball_rew = True
               # print('get ball')
                return obs, self.get_global_state(), 5, done, infos
            else:
                return obs, self.get_global_state(), 0, done, infos

        return obs, self.get_global_state(), 80, done, infos

    def seed(self, seed):
        self.env.seed(seed)

    def close(self):
        self.env.close()

    def get_env_info(self):
        output_dict = {}
        output_dict['n_actions'] = self.action_space[0].n
        output_dict['obs_shape'] = self.obs_dim
        output_dict['n_agents'] = self.n_agents
        output_dict['state_shape'] = self.state_shape
        output_dict['episode_limit'] = self.time_limit
        output_dict['n_enemy'] = self.n_enemy

        return output_dict


# def make_football_env(seed_dir, dump_freq=1000, representation='extracted', render=False):
def make_football_env_3vs1(seed,dense_reward=False, dump_freq=1000,render=False):
    '''
    Creates a env object. This can be used similar to a gym
    environment by calling env.reset() and env.step().

    Some useful env properties:
        .observation_space  :   Returns the observation space for each agent
        .action_space       :   Returns the action space for each agent
        .nagents            :   Returns the number of Agents
    '''
    return GoogleFootballMultiAgentEnv(seed,dense_reward, dump_freq,render)
