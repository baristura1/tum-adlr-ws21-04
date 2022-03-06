import numpy as np
import gym
from gym import spaces
from collections import OrderedDict
import cv2
from bps import *
import math


class MobileFinder(gym.GoalEnv):

    def __init__(self, num_obstacles, timesteps, num_bps):
        self.num_obstacles = num_obstacles
        self.num_bps = num_bps
        self.action_space = spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0
                                                                                  ]), dtype=np.float32)

        self.observation_space = spaces.Dict({"observation": spaces.Box(low=-1.0, high=1.0,
                                                                        shape=(self.num_bps,), dtype=np.float32),
                                              "achieved_goal": spaces.Box(low=-1.0, high=1.0,
                                                                        shape=(2,), dtype=np.float32),
                                              "desired_goal": spaces.Box(low=np.array([-1.0, -1.0]),
                                                                         high=np.array([1, 1]))
                                              })
        self.basis_pts = generate_random_basis_square(n_points=self.num_bps, n_dims=2)
        self.agent_state = None
        self.obs_state = None
        self.goal = None
        self.obstacles = None
        self.start_state = None
        self.timesteps = timesteps
        self.object_size = 0.04
        self.current_step = 0
        self.fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        self.out = cv2.VideoWriter('output1m.avi', self.fourcc, 10.0, (400, 400))
        self.punishment = - 0.2
        self.payout = 100
        self.reset()


    def reset(self):
        basis = self.basis_pts
        self.current_step += 1
        self.agent_state = self.observation_space["achieved_goal"].sample()
        self.obstacles = (np.random.rand(self.num_obstacles*2,) - 0.5) * 2
        self.goal = self.observation_space["desired_goal"].sample()
        collisions = [self.check_collision(self.obstacles[i:i+2], self.goal) for i in (np.arange(self.num_obstacles))*2]

        while not np.all(collisions):
            self.goal = self.observation_space["desired_goal"].sample()
            collisions = [self.check_collision(self.obstacles[i:i+2], self.goal) for i in (np.arange(self.num_obstacles))*2]

        self.obs_state, idx = encode(np.expand_dims(self.obstacles.reshape((self.num_obstacles, 2)), axis=0),
                                     n_bps_points=self.num_bps, custom_basis=basis)
        self.obs_state = np.squeeze(np.transpose(self.obs_state), axis=-1)
        ob = self.get_obs()

        return ob

    def step(self, action):
        x_axis = [1, 0]
        ac = action
        norm_action = action / np.linalg.norm(action)
        angle = np.sign(norm_action[1]) * np.arccos(np.clip(np.dot(x_axis, norm_action), -1.0, 1.0))
        step = [0, 0]
        step[0] = np.cos(angle) * 2 * self.object_size
        step[1] = np.sin(angle) * 2 * self.object_size

        self.agent_state[0] += step[0]
        self.agent_state[1] += step[1]
        self.agent_state = np.clip(self.agent_state, -1, 1)

        info = {"collision": 0,
                "similarity": 0}
        similarity = encode(np.expand_dims(self.agent_state.reshape((1, 2)), axis=0),
                            n_bps_points=self.num_bps, custom_basis=self.basis_pts)
        info["similarity"] = similarity

        for i in (np.arange(self.num_obstacles)) * 2:
            if self.check_collision(self.agent_state, self.obstacles[i:i + 2], sampling=False):
                info["collision"] = 1

        obs = self.get_obs()
        reward = self.compute_reward(obs["achieved_goal"], obs["desired_goal"], info)
        done = True if (np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"]) <= 4 * self.object_size) else False
        return obs, reward, done, info

    def compute_reward(self, observation, desired_goal, info):

        if info["collision"] == 1:
            return float(-200)
        elif np.linalg.norm(observation - desired_goal) <= 4 * self.object_size:
            return float(1000)
        else:
            return float(-np.linalg.norm(observation - desired_goal)) * 50

    def render(self, mode="human"):
        image = np.ones((400, 400, 3), dtype=np.uint8) * 255
        image = cv2.circle(img=image,
                           center=(int((self.agent_state[0] + 1) * 200), int((self.agent_state[1] +1) * 200)),
                           radius=8,
                           color=(255, 0, 0),
                           thickness=-1)
        image = cv2.circle(img=image,
                           center=(int((self.goal[0] +1) * 200), int((self.goal[1] +1) * 200)),
                           radius=8,
                           color=(0, 0, 255),
                           thickness=-1)
        for i in (np.arange(self.num_obstacles))*2:
            image = cv2.circle(img=image,
                               center=(int((self.obstacles[i] + 1) * 200), int((self.obstacles[i+1] + 1) * 200)),
                               radius=8,
                               color=(0, 255, 0),
                               thickness=-1)
        for i in range(self.num_bps):
            image = cv2.circle(img=image,
                               center=(int((self.basis_pts[i][0] + 1) * 200), int((self.basis_pts[i][1] + 1) * 200)),
                               radius=1,
                               color=(0, 0, 0),
                               thickness=-1)
        cv2.imshow("Render", image)
        self.out.write(image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return

    def close(self):
        pass

    def get_obs(self):
        return OrderedDict([("observation", self.obs_state.copy()),
                            ("achieved_goal", self.agent_state.copy()),
                            ("desired_goal", self.goal.copy())
                            ])

    def check_collision(self, arr1, arr2, sampling=True):
        if sampling:
            return np.linalg.norm(arr1 - arr2) >= 4 * self.object_size
        else:
            return np.linalg.norm(arr1 - arr2) <= 2 * self.object_size
            
            
class PlanarFinder(gym.GoalEnv):

    def __init__(self, num_obstacles, timesteps, num_bps, use_bps, curriculum, dof):
        self.num_obstacles = num_obstacles
        self.num_bps = num_bps
        self.timesteps = timesteps
        self.dof = dof
        action_space = np.ones(self.dof) * 2
        self.curriculum = curriculum
        self.action_space = spaces.MultiDiscrete(action_space)
        self.use_bps = use_bps

        if self.use_bps:
            self.observation_space = spaces.Dict({"observation": spaces.Box(low=-1.0, high=1.0,
                                                                            shape=(self.num_bps,), dtype=np.float32),
                                                  "achieved_goal": spaces.Box(low=0, high=2 * math.pi,
                                                                            shape=(self.dof,), dtype=np.float32),
                                                  "desired_goal": spaces.Box(low=np.array(np.ones(self.dof) * -math.pi),
                                                                             high=np.array(np.ones(self.dof) * math.pi))
                                                  })
            self.basis_pts = generate_random_basis(n_points=self.num_bps, n_dims=2)

        else:
            self.observation_space = spaces.Dict({"observation": spaces.Box(low=-1.0, high=1.0,
                                                                            shape=((self.num_obstacles) * 2,),
                                                                            dtype=np.float32),
                                                  "achieved_goal": spaces.Box(low=0, high=2 * math.pi,
                                                                            shape=(self.dof,), dtype=np.float32),
                                                  "desired_goal": spaces.Box(low=np.array(np.ones(self.dof) * -math.pi),
                                                                             high=np.array(np.ones(self.dof) * math.pi))
                                                  })

        self.agent_size = 0.02
        self.link_len = 1 / self.dof
        self.robot_nodes = np.zeros((3 * self.dof + 1,2))
        self.agent_state = None
        self.angles = None
        self.obs_state = None
        self.goal = None
        self.obstacles = None
        self.obs_sizes = None
        self.start_state = None
        self.object_size = 0.04
        self.current_step = 0
        self.fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        self.out = cv2.VideoWriter('1-3-curr-dof3.avi', self.fourcc, 10.0, (400, 400))
        self.reset()

    def reset(self):
        if self.curriculum:
            if self.current_step > self.timesteps / self.num_obstacles:
                self.num_obstacles = 2
            if self.current_step > 2 * self.timesteps / self.num_obstacles:
                self.num_obstacles = 3
            if self.current_step > 3 * self.timesteps / self.num_obstacles:
                self.num_obstacles = 4
            if self.current_step > 4 * self.timesteps / self.num_obstacles:
                self.num_obstacles = 5
            if self.current_step > 5 * self.timesteps / self.num_obstacles:
                self.num_obstacles = 6
            if self.current_step > 6 * self.timesteps / self.num_obstacles:
                self.num_obstacles = 7

        self.robot_nodes = np.zeros((3 * self.dof + 1, 2))
        self.angles = np.zeros(self.dof)
        self.agent_state = self.create_robot()
        self.obstacles = generate_random_basis(n_points=self.num_obstacles, n_dims=2, random_seed=None)

        for i in range(self.num_obstacles):
            if np.linalg.norm(self.obstacles[i]) < (1.0 / self.dof + 2 * self.object_size):
                while np.linalg.norm(self.obstacles[i]) < (1.0 / self.dof + 2 * self.object_size):
                    self.obstacles[i] = generate_random_basis(n_points=1, n_dims=2, random_seed=None)[0]

        self.obstacles = self.obstacles.flatten()

        self.goal, self.goal_cartesian = self.sample_goal()
        collisions = [self.check_collision(self.obstacles[i:i + 2], self.goal_cartesian) \
                      for i in (np.arange(self.num_obstacles)) * 2]

        while not np.all(collisions):
            self.goal, self.goal_cartesian = self.sample_goal()

            collisions = [self.check_collision(self.obstacles[i:i + 2], self.goal_cartesian) \
                          for i in (np.arange(self.num_obstacles)) * 2]

        if self.use_bps:
            self.obs_state, idx = encode(np.expand_dims(self.obstacles.reshape((self.num_obstacles, 2)), axis=0),
                                         n_bps_points=self.num_bps, custom_basis=self.basis_pts)
            self.obs_state = np.squeeze(np.transpose(self.obs_state), axis=-1)

        else:
            self.obs_state = self.obstacles.flatten()

        ob = self.get_obs()
        return ob

    def step(self, action):
        for i in range(len(action)):
            if action[i] == 0:
                self.angles[i] -= math.pi / 180
            else:
                self.angles[i] += math.pi / 180
            if self.angles[i] > math.pi:
                self.angles[i] = -(2 * math.pi - self.angles[i])
            if self.angles[i] < -math.pi:
                self.angles[i] = 2 * math.pi + self.angles[i]

        ee_pos = self.robot_update(self.angles)

        self.agent_state[0] = ee_pos[0]
        self.agent_state[1] = ee_pos[1]
        info = {"collision": 0,
                "similarity": 0}

        for i in (np.arange(self.num_obstacles)) * 2:
            if np.any([self.check_collision(self.robot_nodes[j], self.obstacles[i:i+2], sampling=False) \
                         for j in range(3 * self.dof + 1)]):
                info["collision"] = 1

        self.current_step += 1
        obs = self.get_obs()
        reward = self.compute_reward(obs["achieved_goal"], obs["desired_goal"], info)

        done = True if (np.linalg.norm(self.agent_state - self.goal_cartesian) <= 4 * self.object_size) else False
        return obs, reward, done, info

    def create_robot(self):
        self.robot_nodes[0] = np.array([0,0])
        self.robot_nodes[1] = self.robot_nodes[0] + [self.link_len / 3, 0]
        self.robot_nodes[2] = self.robot_nodes[1] + [self.link_len / 3, 0]

        for i in range(3, 3 * self.dof, 3):
            self.robot_nodes[i] = self.robot_nodes[i - 3] + [self.link_len,0]
            self.robot_nodes[i + 1] = self.robot_nodes[i] + [self.link_len/3, 0]
            self.robot_nodes[i + 2] = self.robot_nodes[i + 1] + [self.link_len/3, 0]

        self.robot_nodes[-1] = self.robot_nodes[-2] + [self.link_len/3, 0]

        return np.array(self.robot_nodes[-1])

    def robot_update(self, angles):
        sum_angles = angles[0]
        for i in range(1, 3 * self.dof + 1):
            self.robot_nodes[i] = self.robot_nodes[i - 1] + [self.link_len / 3 * np.cos(sum_angles),
                                                             self.link_len / 3 * np.sin(sum_angles)]
            if i % 3 == 0 and i < 3 * self.dof:
                idx = int(i / 3)
                sum_angles += angles[idx]

        return self.robot_nodes[-1]

    def compute_reward(self, achieved_goal, desired_goal, info):

        if info["collision"] == 1:
            return float(-200)
        elif np.linalg.norm(self.goal_cartesian - self.robot_nodes[-1]) <= 4 * self.object_size:
            return float(1000)
        else:
            return float(-np.linalg.norm(desired_goal - achieved_goal) * 50)

    def render(self, mode="human"):
        image = np.ones((400, 400, 3), dtype=np.uint8) * 255
        image = cv2.circle(img=image,
                           center=(200, 200),
                           radius=200,  # scaled twice
                           color=(0, 0, 0),
                           thickness=1)

        for idx, node in enumerate(self.robot_nodes):
            image = cv2.circle(img=image,
                           center=(int((node[0] + 1) * 200), int((node[1] + 1) * 200)),
                           radius=int(self.agent_size * 400),
                           color=(255, 0, 0) if idx % 3 else (255, 0, 255),
                           thickness=-1)

        image = cv2.circle(img=image,
                           center=(int((self.goal_cartesian[0] +1) * 200), int((self.goal_cartesian[1] +1) * 200)),
                           radius=int(self.agent_size * 400),
                           color=(0, 0, 255),
                           thickness=-1)
        for i in (np.arange(self.num_obstacles))*2:
            image = cv2.circle(img=image,
                               center=(int((self.obstacles[i] + 1) * 200), int((self.obstacles[i+1] + 1) * 200)),
                               radius=int(self.object_size * 200), # int(self.obs_sizes[int(i/2)] * 200),
                               color=(0, 255, 0),
                               thickness=-1)
        if self.use_bps == True:
            for i in range(self.num_bps):
                image = cv2.circle(img=image,
                                   center=(int((self.basis_pts[i][0] + 1) * 200), int((self.basis_pts[i][1] + 1) * 200)),
                                   radius=1,
                                   color=(0, 0, 0),
                                   thickness=-1)
        cv2.imshow("Render", image)
        self.out.write(image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return

    def close(self):
        pass

    def get_obs(self):
        return OrderedDict([("observation", self.obs_state.copy()),
                            ("achieved_goal", self.angles.copy()),
                            ("desired_goal", self.goal.copy())
                            ])

    def check_collision(self, arr1, arr2, sampling=True):
        if sampling:
            return np.linalg.norm(arr1 - arr2) >= 4 * self.object_size
        else:
            return np.linalg.norm(arr1 - arr2) <= 2 * self.object_size

    def sample_goal(self):
        goal = self.observation_space["desired_goal"].sample()
        sum_angles = 0
        goal_cartesian = np.zeros(2)
        for i in range(self.dof):
            sum_angles += goal[i]
            goal_cartesian[0] += np.cos(sum_angles) * self.link_len
            goal_cartesian[1] += np.sin(sum_angles) * self.link_len

        return goal, goal_cartesian


class PlanarFinderImage(gym.Env):

    def __init__(self, num_obstacles, timesteps, num_bps, curriculum):
        self.num_obstacles = num_obstacles
        self.num_bps = num_bps
        self.timesteps = timesteps
        self.action_space = spaces.Box(low=np.array([-1, -1, -1]), high=np.array([1, 1, 1]), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=255, shape=(200, 200, 3), dtype=np.uint8)
        self.agent_size = 0.065
        self.link_len = 0.33
        self.robot_nodes = np.zeros((10,2))
        self.obs = None
        self.agent_state = None
        self.angles = None
        self.obs_state = None
        self.goal = None
        self.obstacles = None
        self.object_size = 0.07
        self.current_step = 0
        self.curriculum = curriculum
        self.fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        self.out = cv2.VideoWriter('img.avi', self.fourcc, 10.0, (400, 400))
        self.reset()

    def reset(self):
        if self.curriculum:
            if self.current_step > self.timesteps / self.num_obstacles:
                self.num_obstacles = 2
            if self.current_step > 2 * self.timesteps / self.num_obstacles:
                self.num_obstacles = 3
            if self.current_step > 3 * self.timesteps / self.num_obstacles:
                self.num_obstacles = 4
            if self.current_step > 4 * self.timesteps / self.num_obstacles:
                self.num_obstacles = 5
            if self.current_step > 5 * self.timesteps / self.num_obstacles:
                self.num_obstacles = 6
            if self.current_step > 6 * self.timesteps / self.num_obstacles:
                self.num_obstacles = 7

        self.robot_nodes = np.zeros((10,2))
        self.angles = np.zeros(3)
        self.agent_state = self.create_robot()
        self.obstacles = np.random.rand(self.num_obstacles*2) * 2 - 1
        self.obstacles = np.clip(self.obstacles, -self.link_len * 2.5, self.link_len * 2.5)

        self.goal = np.random.rand(2) * 2 - 1
        self.goal[0] = np.random.uniform(-self.link_len*2, self.link_len*2)
        self.goal[1] = np.random.uniform(-self.link_len * 2, self.link_len * 2)

        collisions = [self.check_collision(self.obstacles[i:i + 2], self.goal) \
                      for i in (np.arange(self.num_obstacles)) * 2]

        while not np.all(collisions):
            self.obstacles = np.random.rand(self.num_obstacles*2) * 2 - 1
            collisions = [self.check_collision(self.obstacles[i:i + 2], self.goal) \
                          for i in (np.arange(self.num_obstacles)) * 2]

        ob = self.render(viz=False)

        return ob

    def step(self, action):
        self.current_step += 1
        action = np.round_(action, decimals=0)

        self.angles[0] += math.pi / 180 * action[0] if action[0] != 0 else 0
        self.angles[1] += math.pi / 180 * action[1] if action[1] != 0 else 0
        self.angles[2] += math.pi / 180 * action[2] if action[2] != 0 else 0

        ee_pos = self.robot_update(self.angles)

        self.agent_state[0] = ee_pos[0]
        self.agent_state[1] = ee_pos[1]
        self.current_step += 1
        info = {"collision": 0}

        for i in (np.arange(self.num_obstacles)) * 2:
            if np.any([self.check_collision(self.robot_nodes[j], self.obstacles[i:i+2], sampling=False) \
                         for j in range(10)]):
                info["collision"] = 1

        obs = np.array(self.render()).astype(np.float32) / 255 * 2 - 1
        obs = self.render(viz=False)
        reward = self.compute_reward(ee_pos, self.goal, info)

        done = True if (np.linalg.norm(ee_pos - self.goal) <= 4 * self.agent_size) else False

        return obs, reward, done, info

    def create_robot(self):
        joint1 = np.array([0, 0])
        self.robot_nodes[0] = joint1
        joint2 = joint1 + [self.link_len, 0]
        self.robot_nodes[1] = joint2
        joint3 = joint2 + [self.link_len, 0]
        self.robot_nodes[2] = joint3

        self.robot_nodes[3] = self.robot_nodes[0] + [self.link_len/3, 0]
        self.robot_nodes[4] = self.robot_nodes[3] + [self.link_len/3, 0]

        self.robot_nodes[5] = self.robot_nodes[1] + [self.link_len/3, 0]
        self.robot_nodes[6] = self.robot_nodes[5] + [self.link_len/3, 0]

        self.robot_nodes[7] = self.robot_nodes[2] + [self.link_len/3, 0]
        self.robot_nodes[8] = self.robot_nodes[7] + [self.link_len/3, 0]

        self.robot_nodes[9] = self.robot_nodes[8] + [self.link_len/3, 0]

        return np.array(self.robot_nodes[-1])

    def robot_update(self, angles):
        self.robot_nodes[1] = [self.link_len * np.cos(angles[0]), self.link_len * np.sin(angles[0])]
        self.robot_nodes[2] = self.robot_nodes[1] + [self.link_len * np.cos(angles[1]), self.link_len * np.sin(angles[1])]

        self.robot_nodes[3] = self.robot_nodes[0] + [self.link_len/3 * np.cos(angles[0]), self.link_len/3 * np.sin(angles[0])]
        self.robot_nodes[4] = self.robot_nodes[3] + [self.link_len/3 * np.cos(angles[0]), self.link_len/3 * np.sin(angles[0])]

        self.robot_nodes[5] = self.robot_nodes[1] + [self.link_len/3 * np.cos(angles[1]), self.link_len/3 * np.sin(angles[1])]
        self.robot_nodes[6] = self.robot_nodes[5] + [self.link_len/3 * np.cos(angles[1]), self.link_len/3 * np.sin(angles[1])]

        self.robot_nodes[7] = self.robot_nodes[2] + [self.link_len/3 * np.cos(angles[2]), self.link_len/3 * np.sin(angles[2])]
        self.robot_nodes[8] = self.robot_nodes[7] + [self.link_len/3 * np.cos(angles[2]), self.link_len/3 * np.sin(angles[2])]
        self.robot_nodes[9] = self.robot_nodes[8] + [self.link_len / 3 * np.cos(angles[2]), self.link_len / 3 * np.sin(angles[2])]

        return self.robot_nodes[-1]

    def compute_reward(self, observation, desired_goal, info):

        if info["collision"] == 1:
            return float(-100)
        elif np.linalg.norm(observation - desired_goal) <= 4 * self.agent_size:
            return float(2000)
        else:
            return float(-np.linalg.norm(observation - desired_goal)) * 10

    def render(self, mode="human", viz=True):
        image = np.ones((200, 200, 3), dtype=np.uint8) * 255
        image = cv2.circle(img=image,
                           center=(100, 100),
                           radius=100,
                           color=(0, 0, 0),
                           thickness=1)
        for idx, node in enumerate(self.robot_nodes):
            image = cv2.circle(img=image,
                           center=(int((node[0] + 1) * 100), int((node[1] + 1) * 100)),
                           radius=int(self.agent_size * 100),
                           color=(255, 0, 255) if idx < 3 else (255, 0, 0),
                           thickness=-1)

        image = cv2.circle(img=image,
                           center=(int((self.goal[0] +1) * 100), int((self.goal[1] +1) * 100)),
                           radius=int(self.agent_size * 100),
                           color=(0, 0, 255),
                           thickness=-1)
        for i in (np.arange(self.num_obstacles))*2:
            image = cv2.circle(img=image,
                               center=(int((self.obstacles[i] + 1) * 100), int((self.obstacles[i+1] + 1) * 100)),
                               radius=int(self.object_size * 100), # int(self.obs_sizes[int(i/2)] * 200),
                               color=(0, 255, 0),
                               thickness=-1)
        cv2.imshow(":D", image)
        self.out.write(image)
        if not viz:
            return image
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return

    def close(self):
        pass

    def check_collision(self, arr1, arr2, sampling=True):
        if sampling:
            return np.linalg.norm(arr1 - arr2) >= (self.object_size + self.agent_size)
        else:
            return np.linalg.norm(arr1 - arr2) <= (self.object_size + self.agent_size)

    def sample_goal(self):
        goal = self.observation_space["desired_goal"].sample()

        radius = 0.5 * (goal[0] + 1)
        angle = (goal[1] + 1) * np.pi
        goal[0] = np.cos(angle) * radius
        goal[1] = np.sin(angle) * radius

        return goal
        

class PlanarFinderImage2(gym.Env):

    def __init__(self, num_obstacles, timesteps, num_bps, curriculum):
        self.num_obstacles = num_obstacles
        self.num_bps = num_bps
        self.timesteps = timesteps
        self.curriculum = curriculum
        self.action_space = spaces.Box(low=np.array([-1, -1, -1]), high=np.array([1, 1, 1]), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=255, shape=(200, 200, 3), dtype=np.uint8)
        self.agent_size = 0.065
        self.link_len = 0.33 * 3 / 2
        self.robot_nodes = np.zeros((10, 2))
        self.obs = None
        self.agent_state = None
        self.angles = None
        self.obs_state = None
        self.goal = None
        self.obstacles = None
        self.object_size = 0.07
        self.current_step = 0
        self.fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        self.out = cv2.VideoWriter('vary.avi', self.fourcc, 10.0, (400, 400))
        self.reset()

    def reset(self):
        if self.curriculum:
            if self.current_step > self.timesteps / self.num_obstacles:
                self.num_obstacles = 2
            if self.current_step > 2 * self.timesteps / self.num_obstacles:
                self.num_obstacles = 3
            if self.current_step > 3 * self.timesteps / self.num_obstacles:
                self.num_obstacles = 4
            if self.current_step > 4 * self.timesteps / self.num_obstacles:
                self.num_obstacles = 5
            if self.current_step > 5 * self.timesteps / self.num_obstacles:
                self.num_obstacles = 6
            if self.current_step > 6 * self.timesteps / self.num_obstacles:
                self.num_obstacles = 7


        self.robot_nodes = np.zeros((10, 2))
        self.angles = np.zeros(3)
        self.agent_state = self.create_robot()
        self.obstacles = np.random.rand(self.num_obstacles * 2) * 2 - 1
        self.obstacles = np.clip(self.obstacles, -self.link_len * 2.5, self.link_len * 2.5)

        self.goal = np.random.rand(2) * 2 - 1
        self.goal[0] = np.random.uniform(-self.link_len * 2, self.link_len * 2)
        self.goal[1] = np.random.uniform(-self.link_len * 2, self.link_len * 2)

        collisions = [self.check_collision(self.obstacles[i:i + 2], self.goal) \
                      for i in (np.arange(self.num_obstacles)) * 2]

        while not np.all(collisions):
            self.obstacles = np.random.rand(self.num_obstacles * 2) * 2 - 1
            collisions = [self.check_collision(self.obstacles[i:i + 2], self.goal) \
                          for i in (np.arange(self.num_obstacles)) * 2]

        ob = self.render(viz=False)

        return ob

    def step(self, action):
        self.current_step += 1
        action = np.round_(action, decimals=0)

        self.angles[0] += math.pi / 180 * action[0] if action[0] != 0 else 0
        self.angles[1] += math.pi / 180 * action[1] if action[1] != 0 else 0
        self.angles[2] += math.pi / 180 * action[2] if action[2] != 0 else 0

        ee_pos = self.robot_update(self.angles)

        self.agent_state[0] = ee_pos[0]
        self.agent_state[1] = ee_pos[1]
        self.current_step += 1
        info = {"collision": 0}

        for i in (np.arange(self.num_obstacles)) * 2:
            if np.any([self.check_collision(self.robot_nodes[j], self.obstacles[i:i + 2], sampling=False) \
                       for j in range(7)]):
                info["collision"] = 1

        obs = np.array(self.render()).astype(np.float32) / 255 * 2 - 1
        obs = self.render(viz=False)
        reward = self.compute_reward(ee_pos, self.goal, info)

        done = True if (np.linalg.norm(ee_pos - self.goal) <= 4 * self.agent_size) else False

        return obs, reward, done, info

    def create_robot(self):
        joint1 = np.array([0, 0])
        self.robot_nodes[0] = joint1
        joint2 = joint1 + [self.link_len, 0]
        self.robot_nodes[1] = joint2
        joint3 = joint2 + [self.link_len, 0]
        self.robot_nodes[2] = joint3

        self.robot_nodes[3] = self.robot_nodes[0] + [self.link_len / 3, 0]
        self.robot_nodes[4] = self.robot_nodes[3] + [self.link_len / 3, 0]

        self.robot_nodes[5] = self.robot_nodes[1] + [self.link_len / 3, 0]
        self.robot_nodes[6] = self.robot_nodes[5] + [self.link_len / 3, 0]

        return np.array(self.robot_nodes[-1])

    def robot_update(self, angles):

        self.robot_nodes[1] = [self.link_len * np.cos(angles[0]), self.link_len * np.sin(angles[0])]
        self.robot_nodes[2] = self.robot_nodes[1] + [self.link_len * np.cos(angles[1]),
                                                     self.link_len * np.sin(angles[1])]

        self.robot_nodes[3] = self.robot_nodes[0] + [self.link_len / 3 * np.cos(angles[0]),
                                                     self.link_len / 3 * np.sin(angles[0])]
        self.robot_nodes[4] = self.robot_nodes[3] + [self.link_len / 3 * np.cos(angles[0]),
                                                     self.link_len / 3 * np.sin(angles[0])]

        self.robot_nodes[5] = self.robot_nodes[1] + [self.link_len / 3 * np.cos(angles[1]),
                                                     self.link_len / 3 * np.sin(angles[1])]
        self.robot_nodes[6] = self.robot_nodes[5] + [self.link_len / 3 * np.cos(angles[1]),
                                                     self.link_len / 3 * np.sin(angles[1])]

        return self.robot_nodes[-1]

    def compute_reward(self, observation, desired_goal, info):

        if info["collision"] == 1:
            return float(-100)
        elif np.linalg.norm(observation - desired_goal) <= 4 * self.agent_size:
            return float(2000)
        else:
            return float(-np.linalg.norm(observation - desired_goal)) * 10

    def render(self, mode="human", viz=True):
        image = np.ones((200, 200, 3), dtype=np.uint8) * 255
        image = cv2.circle(img=image,
                           center=(100, 100),
                           radius=100,
                           color=(0, 0, 0),
                           thickness=1)
        for idx, node in enumerate(self.robot_nodes):
            image = cv2.circle(img=image,
                               center=(int((node[0] + 1) * 100), int((node[1] + 1) * 100)),
                               radius=int(self.agent_size * 100),
                               color=(255, 0, 255) if idx < 3 else (255, 0, 0),
                               thickness=-1)

        image = cv2.circle(img=image,
                           center=(int((self.goal[0] + 1) * 100), int((self.goal[1] + 1) * 100)),
                           radius=int(self.agent_size * 100),
                           color=(0, 0, 255),
                           thickness=-1)
        for i in (np.arange(self.num_obstacles)) * 2:
            image = cv2.circle(img=image,
                               center=(int((self.obstacles[i] + 1) * 100), int((self.obstacles[i + 1] + 1) * 100)),
                               radius=int(self.object_size * 100),  # int(self.obs_sizes[int(i/2)] * 200),
                               color=(0, 255, 0),
                               thickness=-1)
        cv2.imshow(":D", image)
        self.out.write(image)
        if not viz:
            return image
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return

    def close(self):
        pass

    def check_collision(self, arr1, arr2, sampling=True):
        if sampling:
            return np.linalg.norm(arr1 - arr2) >= (self.object_size + self.agent_size)
        else:
            return np.linalg.norm(arr1 - arr2) <= (self.object_size + self.agent_size)

    def sample_goal(self):
        goal = self.observation_space["desired_goal"].sample()

        radius = 0.5 * (goal[0] + 1)
        angle = (goal[1] + 1) * np.pi
        goal[0] = np.cos(angle) * radius
        goal[1] = np.sin(angle) * radius

        return goal
