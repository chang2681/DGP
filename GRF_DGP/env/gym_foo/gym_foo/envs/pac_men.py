import gym
import numpy as np
from gym import spaces

N_DISCRETE_ACTIONS = 5
N_OBSERVATION_SCALE = 2
WALL = -1
BLANK_SPACE = 0
REWARD_POINT = 1
AGENT_POINT = 0.1
Green_Point = 1
GreenReward = -.3

class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    # current 1: wall 0: blank space 2: reward point 3: agent point
    # in this environment agents can overlap together
    # in this environment agents and reward points can overlap together
    # don't know whether give each agent a unique label
    # if all the reward points are eaten, then reset the reward points

    def __init__(
            self,
            n_agents: int,
            seed:int,
            mode: str,
    ):
        super(CustomEnv, self).__init__()
        sa_action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        self.action_space = spaces.Tuple(tuple(n_agents * [sa_action_space]))
        np.random.seed(seed)

        sa_obs_space = spaces.Box(low=-5, high=5,
                                  shape=((2 * N_OBSERVATION_SCALE + 1) * (2 * N_OBSERVATION_SCALE + 1),),
                                  dtype=np.uint8)
        self.observation_space = spaces.Tuple(tuple(n_agents * [sa_obs_space]))

        self.n_agents = n_agents
        # Here initial maze with tini size
        # TODO: add more suitable environments

        if mode == 'tini':
            self.env_matrix = WALL * np.ones([21, 21])
        elif mode == 'small':
            self.env_matrix = WALL * np.ones([25, 25])
        elif mode == 'complex':
            self.env_matrix = WALL * np.ones([21, 21])

        self.mode = mode
        self.init_env_matrix()
        self.time_limit = 100
        self.time_step = 0

    def init_reward_point_withdiff_mode(self):
        if self.mode == 'tini':
            self.reward_point_set_current += self.init_reward_point(4, 3, 1, 2, 9)
            self.reward_point_set_current += self.init_reward_point(4, 3, 6, 15, 9)
            self.reward_point_set_current += self.init_reward_point(3, 4, 1, 10, 15)
            self.reward_point_set_current += self.init_reward_point(3, 4, 2, 10, 2)######Green Bean
            self.state_shape = 93


        elif self.mode == 'small':
            self.reward_point_set_current += self.init_reward_point(4, 3, 1, 2, 9)
            self.reward_point_set_current += self.init_reward_point(4, 3, 6, 15, 9)
            self.reward_point_set_current += self.init_reward_point(3, 4, 1, 10, 15)
            self.reward_point_set_current += self.init_reward_point(3, 4, 1, 10, 2)
            self.state_shape = 3 * 5 * 4 + self.prop * (1 + 2 + 2 + 3) + 3 * 3


        elif self.mode == 'complex':
            self.reward_point_set_current += self.init_reward_point(4, 3, 1, 2, 9)
            self.reward_point_set_current += self.init_reward_point(4, 3, 6, 15, 9)
            self.reward_point_set_current += self.init_reward_point(3, 4, 1, 10, 15)
            self.reward_point_set_current += self.init_reward_point(3, 4, 1, 10, 2)
            self.state_shape = 93

    def init_env_matrix(self):
        if self.mode == 'tini':
            assert (self.env_matrix.shape[0] == 21)

            # first initial the environment
            self.env_matrix[2:7, 9:12] = BLANK_SPACE
            self.env_matrix[7:10, 10] = BLANK_SPACE
            self.env_matrix[10:13, 9:12] = BLANK_SPACE
            self.env_matrix[11, 7:9] = BLANK_SPACE
            self.env_matrix[11, 12:14] = BLANK_SPACE
            self.env_matrix[10:13, 14:19] = BLANK_SPACE
            self.env_matrix[10:13, 2:7] = BLANK_SPACE
            self.env_matrix[13, 9:12] = BLANK_SPACE
            self.env_matrix[14:19, 9:12] = BLANK_SPACE
            self.agent_position = []

            # second initial the environment
            self.reward_point_set_current = []
            self.init_reward_point_withdiff_mode()

            # third initial agents
            self.init_agent_point(3, 3, 10, 9)
            self.prop = 1
            self.center = [11, 10]

        elif self.mode == 'small':
            assert (self.env_matrix.shape[0] == 25)

            # first initial the environment
            self.env_matrix[2:7, 11:14] = BLANK_SPACE
            self.env_matrix[7:13, 12] = BLANK_SPACE
            self.env_matrix[13:16, 11:14] = BLANK_SPACE
            self.env_matrix[14, 7:11] = BLANK_SPACE
            self.env_matrix[14, 14:18] = BLANK_SPACE
            self.env_matrix[13:16, 18:23] = BLANK_SPACE
            self.env_matrix[13:16, 2:7] = BLANK_SPACE
            self.env_matrix[16:18, 12] = BLANK_SPACE
            self.env_matrix[18:23, 11:14] = BLANK_SPACE
            self.agent_position = []

            # second initial the environment
            self.reward_point_set_current = []
            self.init_reward_point_withdiff_mode()

            # third initial agents
            self.init_agent_point(3, 3, 13, 11)
            self.prop = 2
            self.center = [14, 12]

        elif self.mode == 'complex':
            assert (self.env_matrix.shape[0] == 21)

            # first initial the environment
            self.env_matrix[2:19, 9:12] = BLANK_SPACE
            # self.env_matrix[7:10, 10] = BLANK_SPACE
            # self.env_matrix[10:13, 9:12] = BLANK_SPACE
            # self.env_matrix[11, 7:9] = BLANK_SPACE
            # self.env_matrix[11, 12:14] = BLANK_SPACE
            self.env_matrix[10:13, 2:19] = BLANK_SPACE
            # self.env_matrix[10:13, 2:7] = BLANK_SPACE
            # self.env_matrix[13, 10] = BLANK_SPACE
            # self.env_matrix[14:19, 9:12] = BLANK_SPACE
            self.agent_position = []

            # second initial the environment
            self.reward_point_set_current = []
            self.init_reward_point_withdiff_mode()

            # third initial agents
            self.init_agent_point(3, 3, 10, 9)
            self.prop = 1
            self.center = [11, 10]

    def button(self, item, init=False):

        if init:
            return item[1] < 8 #or item[1] > 12
        else:
            return item[1][1] < 8 #or item[1][1] > 12

    def init_reward_point(self, length, wide, num, init_length, init_wide):
        assert (length != 0 and wide != 0 and num != 0)
        point_set = []
        while len(point_set) != num:
            point = [np.random.randint(length) + init_length, np.random.randint(wide) + init_wide]
            if point not in point_set:
                point_set.append(point)

        for item in point_set:
            if not self.button(item, init=True):
                self.env_matrix[item[0], item[1]] = REWARD_POINT
            else:
                self.env_matrix[item[0], item[1]] = Green_Point

        return point_set

    def init_agent_point(self, length, wide, init_length, init_wide):
        assert (length != 0 and wide != 0)
        point_set = []
        while len(point_set) != self.n_agents:
            point = [14, 10]
            point_set.append(point)
            # point = [np.random.randint(length) + init_length, np.random.randint(wide) + init_wide]
            # if point not in point_set:
            #     point_set.append(point)

        for item in point_set:
            self.agent_position.append(np.array(item))
            # self.env_matrix[item[0], item[1]] = 3
            # here agents' positions are not counted in the environment matrix
            # when calculate each agent's observation
            # 1. give matrix information in raleted scale
            # 2. overlap agents' positions if they are in current agent's observation scale
            # the weight of agents is larger then it of environment information such as reward points

    def test_init_agent_point(self):
        self.agent_position[0] = np.array(self.reward_point_set_current[0])
        self.agent_position[1] = np.array(self.reward_point_set_current[0])
        self.step([0, 0, 0, 0])

    def step(self, action):
        # Execute one time step within the environment
        # Here are five actions
        # 0: up 1: down 2: right 3: left 4: eat

        assert (len(action) == self.n_agents)
        eaten_reward_points = []
        eaten_reward_points_array = []
        # return_reward = -0.025 * np.ones(4)
        return_reward = np.zeros(4)

        self.time_step += 1

        for id, item in enumerate(action):
            # first analyse

            if item == 0:
                delta_xy = np.array([-1, 0])
                ifeat = False

            elif item == 1:
                delta_xy = np.array([1, 0])
                ifeat = False

            elif item == 2:
                delta_xy = np.array([0, 1])
                ifeat = False

            elif item == 3:
                delta_xy = np.array([0, -1])
                ifeat = False

            elif item == 4:
                delta_xy = np.array([0, 0])
                ifeat = True

            else:
                raise (print(f'wrong action label: agent{id} action is {item}'))

            next_state = self.agent_position[id] + delta_xy
            if self.env_matrix[next_state[0], next_state[1]] != WALL:
                # check if next state is in the wall
                self.agent_position[id] = next_state

            if ifeat:
                if self.env_matrix[self.agent_position[id][0], self.agent_position[id][1]] >= REWARD_POINT:
                    # this agent want to eat the reward point
                    eaten_reward_points.append([id, self.agent_position[id]])
                    eaten_reward_points_array.append(self.agent_position[id])

        eaten_reward_points_array = np.array(eaten_reward_points_array)

        # exclude more then one agents eat the same reward point
        for item in eaten_reward_points:
            if not self.button(item):  # and item[1][1]<12:
                index_0 = np.where(eaten_reward_points_array[:, 0] == item[1][0])[0]
                index_1 = np.where(eaten_reward_points_array[index_0, 1] == item[1][1])[0]
                return_reward[item[0]] = 1 / len(index_1)
            else:
                return_reward[item[0]] = GreenReward
            # more agents choose to eat the same reward point
            # then the reward get by each agent shrinks proportionally

        # remove the reward point
        for item in eaten_reward_points:
            self.env_matrix[item[1][0], item[1][1]] = BLANK_SPACE
            if item[1].tolist() in self.reward_point_set_current:
                self.reward_point_set_current.remove(item[1].tolist())

        done = False
        if self.time_step >= self.time_limit:
            # done only if time steps exceed limitation
            done = True
        if len(self.reward_point_set_current) == 0:
            # all current reward points are eaten
            # regenerate reward points
            self.init_reward_point_withdiff_mode()

        next_obs = tuple([self.get_local_observation(i) for i in range(self.n_agents)])
        next_state = self.get_global_observation()
        info = {}

        return next_obs, next_state, return_reward.sum(), done, info

    def get_local_observation(self, id):
        current_position = self.agent_position[id]
        length = N_OBSERVATION_SCALE * 2 + 1
        min_index_0 = current_position[0] - N_OBSERVATION_SCALE
        min_index_1 = current_position[1] - N_OBSERVATION_SCALE

        obs = self.env_matrix[min_index_0:min_index_0 + length, min_index_1:min_index_1 + length].copy()
        for i in range(self.n_agents):
            if i != id:
                other_agent_position = self.agent_position[i]
                if other_agent_position[0] >= min_index_0 and other_agent_position[0] < min_index_0 + length:
                    if other_agent_position[1] >= min_index_1 and other_agent_position[1] < min_index_1 + length:
                        obs[other_agent_position[0] - min_index_0][other_agent_position[1] - min_index_1] = AGENT_POINT
        # onehot_id = np.zeros(self.n_agents)
        # onehot_id[id] = 1
        # obs = np.concatenate([onehot_id, obs.reshape(-1)])
        return obs.reshape(-1)

    # return  obs.reshape(-1)

    def get_global_observation(self):
        env_matrix_withagent = self.env_matrix.copy()
        for pos in self.agent_position:
            env_matrix_withagent[pos[0], pos[1]] = AGENT_POINT

        center_matrix = env_matrix_withagent[self.center[0] - 1:self.center[0] + 2,
                        self.center[1] - 1:self.center[1] + 2]
        down_room = env_matrix_withagent[self.center[0] + 2 + self.prop * 1:self.center[0] + 2 + self.prop * 1 + 5,
                    self.center[1] -
                    1:self.center[1] + 2]

        right_room = env_matrix_withagent[self.center[0] - 1:self.center[0] + 2,
                     self.center[1] + 2 + self.prop * 2:self.center[1] + 2 +
                                                        self.prop * 2 + 5]

        up_room = env_matrix_withagent[self.center[0] - 1 - self.prop * 3 - 5:self.center[0] - 1 - self.prop * 3,
                  self.center[1] -
                  1:self.center[1] + 2]

        left_room = env_matrix_withagent[self.center[0] - 1:self.center[0] + 2,
                    self.center[1] - 1 - self.prop * 2 - 5:self.center[1] - 1 -
                                                           self.prop * 2]
        # down_path = env_matrix_withagent[self.center[0] + 2:self.center[0] + 2 + self.prop * 1, self.center[1]]
        # right_path = env_matrix_withagent[self.center[0], self.center[1] + 2:self.center[1] + 2 + self.prop * 2]
        # up_path = env_matrix_withagent[self.center[0] - 1 - self.prop * 3:self.center[0] - 1, self.center[1]]
        # left_path = env_matrix_withagent[self.center[0], self.center[1] - 2 - self.prop * 2:self.center[1] - 2]
        up_path = env_matrix_withagent[self.center[0] - 1 - self.prop * 3:self.center[0] - 1,
                  self.center[1] - 1:self.center[1] + 2]
        down_path = env_matrix_withagent[self.center[0] + 2:self.center[0] + 2 + self.prop * 1,
                    self.center[1] - 1:self.center[1] + 2]
        left_path = env_matrix_withagent[self.center[0] - 1:self.center[0] + 2,
                    self.center[1] - 2 - self.prop * 2:self.center[1] - 2]
        right_path = env_matrix_withagent[self.center[0] - 1:self.center[0] + 2,
                     self.center[1] + 2:self.center[1] + 2 + self.prop * 2]
        global_state = np.r_[center_matrix.reshape(-1),
                             down_path.reshape(-1),
                             down_room.reshape(-1),
                             right_path.reshape(-1),
                             right_room.reshape(-1),
                             up_path.reshape(-1),
                             up_room.reshape(-1),
                             left_path.reshape(-1),
                             left_room.reshape(-1)]

        return global_state

    def reset(self):
        # Reset the state of the environment to an initial state
        self.init_env_matrix()
        self.time_step = 0
        obs = tuple([self.get_local_observation(i) for i in range(self.n_agents)])
        state = self.get_global_observation()

        return obs, state

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        pass

    def seed(self, seed=0):
        np.random.seed(seed)

    def close(self):
        pass

    def get_env_info(self):
        output_dict = {}
        output_dict['n_actions'] = 5
        output_dict['n_agents'] = 4
        output_dict['state_shape'] = self.state_shape
        output_dict['obs_shape'] = (N_OBSERVATION_SCALE * 2 + 1) * (N_OBSERVATION_SCALE * 2 + 1)
        output_dict['episode_limit'] = self.time_limit
        output_dict['predict_input'] = self.state_shape
        output_dict['predict_out'] = (N_OBSERVATION_SCALE * 2 + 1) * (N_OBSERVATION_SCALE * 2 + 1)
        output_dict['n_enemy'] = 0


        return output_dict
