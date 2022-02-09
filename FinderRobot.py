import numpy as np
from typing import Callable
import gym
from gym import spaces
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import A2C, PPO, DDPG, SAC, TD3
from stable_baselines3.common.noise import NormalActionNoise
from collections import OrderedDict
import cv2
import time
from datetime import datetime
from bps import *
import math
import torch

torch.cuda.empty_cache()

class GoalFinder(gym.Env):

    def __init__(self, num_obstacles, timesteps, num_bps):
        super(GoalFinder, self).__init__()
        self.num_obstacles = num_obstacles
        self.num_bps = num_bps
        self.action_space = spaces.Box(low=np.array([-1, -1, -1]), high=np.array([0, 0, 0]), dtype=np.float32)
        #self.action_space = spaces.MultiDiscrete([3,3,3])
        self.observation_space = spaces.Box(low=0, high=255, shape=(200, 200, 3), dtype=np.uint8)
        """
        self.observation_space = spaces.Dict({"observation": spaces.Box(low=-1.0, high=1.0,
                                                                        shape=(self.num_obstacles,2), dtype=np.float32),
                                              "achieved_goal": spaces.Box(low=-1.0, high=1.0,
                                                                        shape=(2,), dtype=np.float32),
                                              "desired_goal": spaces.Box(low=np.array([-1.0, -1.0]),
                                                                         high=np.array([1, 1]))
                                              })
        """
        self.basis_pts = generate_random_basis(n_points=self.num_bps, n_dims=2)
        self.agent_size = 0.065
        self.link_len = 0.33
        self.robot_nodes = np.zeros((10,2))
        self.obs = None
        self.agent_state = None
        self.angles = None
        self.obs_state = None
        self.goal = None
        self.obstacles = None
        self.timesteps = timesteps
        self.object_size = 0.15
        self.current_step = 0
        self.fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        self.out = cv2.VideoWriter('vary.avi', self.fourcc, 10.0, (400, 400))
        self.reset()


    def reset(self):
        #super().reset()
        basis = self.basis_pts
        self.current_step += 1
        self.robot_nodes = np.zeros((10,2))
        self.angles = np.zeros(3)
        self.agent_state = self.create_robot()
        self.obstacles = np.random.rand(self.num_obstacles*2) * 2 - 1
        self.obstacles = np.array([0.0, -0.7])
        self.goal = np.random.rand(2) * 2 - 1
        self.goal[0] = np.random.uniform(-self.link_len*2, self.link_len*2)
        self.goal[1] = np.random.uniform(-self.link_len * 2, self.link_len * 2)
        self.goal = np.array([-0.4, -0.4])
        collisions = [self.check_collision(self.obstacles[i:i + 2], self.goal) \
                      for i in (np.arange(self.num_obstacles)) * 2]
        """
        =======For varying obj sizes=====
        collisions = [self.check_collision(self.obstacles[i:i+2], self.goal, margin=self.obs_sizes[int(i/2)]) \
                      for i in (np.arange(self.num_obstacles))*2]
        """
        while not np.all(collisions):
            self.obstacles = np.random.rand(self.num_obstacles*2) * 2 - 1
            collisions = [self.check_collision(self.obstacles[i:i + 2], self.goal) \
                          for i in (np.arange(self.num_obstacles)) * 2]

        #self.obs_state = self.obstacles.reshape((self.num_obstacles,2))
        #self.obs_state, idx = encode(np.expand_dims(self.obstacles.reshape((self.num_obstacles, 2)), axis=0),
                                     #n_bps_points=self.num_bps, custom_basis=basis)
        #self.obs_state = np.squeeze(np.transpose(self.obs_state), axis=-1)
        """
        ======= For varying obj sizes========
        idx = np.squeeze(idx, axis=-1)
        for j in range(self.num_bps):
            self.obs_state[j] += self.obs_sizes[idx[j]]
        #print(f"Sampling done: {self.current_step}/{self.timesteps}")
        """
        #ob = np.array(self.render()).astype(np.float32) / 255 * 2 - 1
        ob = self.render(viz=False)
        #ob = np.array(ob).astype(np.float32) / 255 * 2 - 1
        #ob = self.get_obs()

        return ob

    def step(self, action):

        self.angles[0] += math.pi * action[0] if action[0] != 0 else 0
        self.angles[1] += math.pi * action[1] if action[1] != 0 else 0
        self.angles[2] += math.pi * action[2] if action[2] != 0 else 0

        ee_pos = self.robot_update(self.angles)

        self.agent_state[0] = ee_pos[0]
        self.agent_state[1] = ee_pos[1]
        #self.agent_state = np.clip(self.agent_state, -1, 1)
        self.current_step += 1
        info = {"collision": 0}

        for i in (np.arange(self.num_obstacles)) * 2:
            if np.any([self.check_collision(self.robot_nodes[j], self.obstacles[i:i+2], sampling=False) \
                         for j in range(10)]):
                info["collision"] = 1

        #obs = self.observation
        obs = np.array(self.render()).astype(np.float32) / 255 * 2 - 1
        obs = self.render(viz=False)
        reward = self.compute_reward(ee_pos, self.goal, info)

        done = True if (np.linalg.norm(ee_pos - self.goal) <= 4 * self.agent_size) else False

        return obs, reward, done, info

    def create_robot(self):
        joint1 = np.array([0,0])
        self.robot_nodes[0] = joint1
        joint2 = joint1 + [self.link_len,0]
        self.robot_nodes[1] = joint2
        joint3 = joint2 + [self.link_len,0]
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
        """
        T01 = np.array([np.cos(theta1), -np.sin(theta1), 0, 0],
                       [np.sin(theta1), np.cos(theta1), 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1])
        T12 = np.array([np.cos(theta2), -np.sin(theta2), 0, self.link_len],
                       [np.sin(theta2), np.cos(theta2), 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1])
        T23 = np.array([np.cos(theta3), -np.sin(theta3), 0, self.link_len],
                       [np.sin(theta3), np.cos(theta3), 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1])
        T34 = np.array([1, 0, 0, 0.24 / 3 / 8],
                       [0, 1, 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1])
        T3G = np.array([1, 0, 0, 0.24 / 3],
                       [0, 1, 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1])
        """

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
            return float(1000)
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
        for i in range(self.num_bps):
            image = cv2.circle(img=image,
                               center=(int((self.basis_pts[i][0] + 1) * 100), int((self.basis_pts[i][1] + 1) * 100)),
                               radius=1,
                               color=(0, 0, 0),
                               thickness=-1)
        cv2.imshow(":D", image)
        self.out.write(image)
        if not viz:
            return image
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return



    def close(self):
        pass
    """
    def get_obs(self):
        self.observation[0:3] = self.angles
        return self.observation
    
    """

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
    """
    =========For varying obj sizes===========
    def check_collision(self, arr1, arr2, margin, sampling=True):
        if sampling:
            return np.linalg.norm(arr1 - arr2) >= 2 * (margin + self.agent_size)
        else:
            return np.linalg.norm(arr1 - arr2) <= (margin + self.agent_size)
    """


timesteps = 5000
num_obstacles = 1
num_bps = 10

env = GoalFinder(num_obstacles=num_obstacles, timesteps=timesteps, num_bps=num_bps)

#check_env(env, warn=True)

obs = env.reset()
n_actions = 3
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.01 * np.ones(n_actions))
#model = PPO("CnnPolicy", env, tensorboard_log='./logs/', verbose=1).learn(timesteps)
model = SAC("CnnPolicy", buffer_size=10000, env=env, tensorboard_log='./logs/', verbose=2, action_noise=action_noise).learn(timesteps, log_interval=1)
#model = PPO("CnnPolicy", env, tensorboard_log='./logs/', verbose=1).learn(timesteps, log_interval=1)


frames = []
n_steps = 30
n_runs = 1000
n_success = 0
n_collision = 0
for run in range(n_runs):
    obs = env.reset()
    img = env.render()
    for step in range(n_steps):
        if step < 0:
            action, _ = model.predict(obs, deterministic=False)
        else:
            action, _ = model.predict(obs, deterministic=True)
        #print("###########################STEP {}################################".format(step + 1))
        #print("Action: ", action)
        #print("******************************************************************")
        obs, reward, done, info = env.step(action)
        #print('obs=', obs["observation"][0:2], 'reward=', reward, 'done=', done)
        #print("******************************************************************")
        #print("Obstacle positions: ", obs["observation"][2:4], obs["observation"][4:6], obs["observation"][6:8])
        #print("******************************************************************")
        if info["collision"] == 0:
            #print("No collision")
            pass
        else:
            #print("COLLIDED")
            n_collision += 1
            env.reset()
            break
        #print("##################################################################")
        #print("\n\n")
        env.render(mode='human')
        #time.sleep(0.03)
        if done:
            if run == n_runs - 1:
                env.reset()
            #print("Goal reached!", "reward=", reward)
            n_success += 1
            break

env.out.release()

rundate = datetime.now()
rundatestr = rundate.strftime("%d/%m/%Y %H:%M:%S")

print("\n\n########################## END RESULT ###############################")
print(f"While learning, {env.current_step} out of {timesteps} episodes ended in success.")
print(f"During evaluation, in {n_success} of {n_runs} runs the agent managed to reach the goal while colliding {n_collision} times.")
print("--------------------------Used variables:-------------------------")
print(f"Grid size: 1, Object size: {env.object_size}, Num of obstacles: {env.num_obstacles}, BPS Size: {env.num_bps}")
#print(f"Negative reward coefficient: {env.punishment} (Multiplied with L-inf norm of BPS(agent_pos) to BPS(obstacles))")
#print(f"Positive reward: {env.payout} (Multiplied with I(goal_reached))")
print("#####################################################################\n\n")


with open('logs.txt', 'a') as logs:
    logs.write("\n\n########################## OBJ SIZE ###############################\n")
    logs.write(rundatestr)
    logs.write(f"\nWhile learning, {env.current_step} out of {timesteps} episodes ended in success.\n")
    logs.write(f"During evaluation, in {n_success} of {n_runs} runs the agent managed to reach the goal while colliding {n_collision} times.\n")
    logs.write("--------------------------Used variables:-------------------------\n")
    logs.write(f"Grid size: 1, Object size: {env.object_size}, Num of obstacles: {env.num_obstacles}, BPS Size: {env.num_bps}\n")
    #logs.write(f"Negative reward coefficient: {env.punishment} (Multiplied with L-inf norm of BPS(agent_pos) to BPS(obstacles))\n")
    #logs.write(f"Positive reward: {env.payout} (Multiplied with I(goal_reached))\n")
    logs.write("#####################################################################\n\n")