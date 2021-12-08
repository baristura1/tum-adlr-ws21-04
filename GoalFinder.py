import numpy as np
import gym
from gym import spaces
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import A2C, PPO
from collections import OrderedDict
import cv2
import time
from datetime import datetime
from bps import *


class GoalFinder(gym.GoalEnv):

    def __init__(self, gridSize, num_obstacles, timesteps, num_bps):
        super(GoalFinder, self).__init__()
        self.gridSize = gridSize
        self.num_obstacles = num_obstacles
        self.num_bps = num_bps

        self.action_space = spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32)

        self.observation_space = spaces.Dict({"observation": spaces.Box(low=0.0, high=self.gridSize,
                                                                        shape=(self.num_bps,), dtype=np.float32),
                                              "achieved_goal": spaces.Box(low=0.0, high=self.gridSize,
                                                                        shape=(2,), dtype=np.float32),
                                              "desired_goal": spaces.Box(low=np.array([0.0, 0.0]),
                                                                         high=np.array([self.gridSize, self.gridSize]))#,
                                              #"obstacles": spaces.Box(low=0.0, high=self.gridSize-1,
                                              #                        shape=(self.num_obstacles*2,), dtype=np.float32)
                                              })
        self.basis_pts = generate_random_basis(n_points=self.num_bps, n_dims=2, gridSize=self.gridSize)
        self.agent_state = None
        self.obs_state = None
        self.goal = None
        self.obstacles = None
        self.start_state = None
        self.timesteps = timesteps
        self.object_size = 0.2
        self.current_step = 0
        self.fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        self.out = cv2.VideoWriter('output1m.avi', self.fourcc, 10.0, (400, 400))
        self.punishment = - 0.2
        self.payout = 100
        self.reset()


    def reset(self):
        #super().reset()
        basis = self.basis_pts
        self.current_step += 1
        self.agent_state = self.observation_space["achieved_goal"].sample()
        self.obstacles = np.random.rand(self.num_obstacles*2,)*9
        self.goal = self.observation_space["desired_goal"].sample()
        collisions = [self.check_collision(self.obstacles[i:i+2], self.goal) for i in (np.arange(self.num_obstacles))*2]

        while not np.all(collisions):
            self.goal = self.observation_space["desired_goal"].sample()
            collisions = [self.check_collision(self.obstacles[i:i+2], self.goal) for i in (np.arange(self.num_obstacles))*2]

        self.obs_state = encode(np.expand_dims(self.obstacles.reshape((self.num_obstacles, 2)), axis=0),
                                n_bps_points=self.num_bps, custom_basis=basis)
        self.obs_state = np.squeeze(np.transpose(self.obs_state), axis=-1)
        print(f"Sampling done: {self.current_step}/{self.timesteps}")
        ob = self.get_obs()

        return ob

    def step(self, action):
        self.agent_state[0] += action[0]
        self.agent_state[1] += action[1]
        self.agent_state = np.clip(self.agent_state, 0, self.gridSize)

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
        done = True if (np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"]) <= 0.6) else False
        #print(reward)
        return obs, reward, done, info

    def compute_reward(self, observation, desired_goal, info):

        return self.punishment * np.linalg.norm(np.transpose(info["similarity"]) - np.expand_dims(self.obs_state, axis=-1), ord=np.inf) + \
               int(np.linalg.norm(observation - desired_goal) <= 0.6) * self.payout

    def render(self, mode="human"):
        image = np.ones((400, 400, 3), dtype=np.uint8) * 255
        image = cv2.circle(img=image,
                           center=(int(self.agent_state[0] * 40), int(self.agent_state[1] * 40)),
                           radius=8,
                           color=(255, 0, 0),
                           thickness=-1)
        image = cv2.circle(img=image,
                           center=(int(self.goal[0] * 40), int(self.goal[1] * 40)),
                           radius=8,
                           color=(0, 0, 255),
                           thickness=-1)
        for i in (np.arange(self.num_obstacles))*2:
            image = cv2.circle(img=image,
                               center=(int(self.obstacles[i] * 40), int(self.obstacles[i+1] * 40)),
                               radius=8,
                               color=(0, 255, 0),
                               thickness=-1)
        for i in range(self.num_bps):
            image = cv2.circle(img=image,
                               center=(int(self.basis_pts[i][0] * 40), int(self.basis_pts[i][1] * 40)),
                               radius=1,
                               color=(0, 0, 0),
                               thickness=-1)
        cv2.imshow(":D", image)
        self.out.write(image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return

    def close(self):
        pass

    def get_obs(self):
        return OrderedDict([("observation", self.obs_state.copy()),
                            ("achieved_goal", self.agent_state.copy()),
                            ("desired_goal", self.goal.copy())#,
                            #("obstacles", self.obstacles.copy())
                            ])

    def check_collision(self, arr1, arr2, sampling=True):
        if sampling:
            return np.linalg.norm(arr1 - arr2) >= 4 * self.object_size
        else:
            return np.linalg.norm(arr1 - arr2) <= 2 * self.object_size

timesteps = 500000
gridSize = 10
num_obstacles = 10
num_bps = 25

env = GoalFinder(gridSize=gridSize, num_obstacles=num_obstacles, timesteps=timesteps, num_bps=num_bps)

#check_env(env, warn=True)

obs = env.reset()
model = PPO('MultiInputPolicy', env, verbose=0).learn(timesteps)
frames = []
n_steps = 30
n_runs = 100
n_success = 0
n_collision = 0
for run in range(n_runs):
    obs = env.reset()
    for step in range(n_steps):
        if step < 0:
            action, _ = model.predict(obs, deterministic=False)
        else:
            action, _ = model.predict(obs, deterministic=True)
        print("###########################STEP {}################################".format(step + 1))
        print("Action: ", action)
        print("******************************************************************")
        obs, reward, done, info = env.step(action)
        print('obs=', obs["observation"][0:2], 'reward=', reward, 'done=', done)
        print("******************************************************************")
        print("Obstacle positions: ", obs["observation"][2:4], obs["observation"][4:6], obs["observation"][6:8])
        print("******************************************************************")
        if info["collision"] == 0:
            print("No collision")
        else:
            print("COLLIDED")
            n_collision += 1
            env.reset()
            break
            #break
        print("##################################################################")
        print("\n\n")
        env.render(mode='human')
        time.sleep(0.1)
        if done:
            if run == n_runs - 1:
                env.reset()
            print("Goal reached!", "reward=", reward)
            n_success += 1
            break

env.out.release()

rundate = datetime.now()
rundatestr = rundate.strftime("%d/%m/%Y %H:%M:%S")

print("\n\n########################## END RESULT ###############################")
print(f"While learning, {env.current_step} out of {timesteps} episodes ended in success.")
print(f"During evaluation, in {n_success} of {n_runs} runs the agent managed to reach the goal while colliding {n_collision} times.")
print("--------------------------Used variables:-------------------------")
print(f"Grid size: {env.gridSize}, Object size: {env.object_size}, Num of obstacles: {env.num_obstacles}, BPS Size: {env.num_bps}")
print(f"Negative reward coefficient: {env.punishment} (Multiplied with L-inf norm of BPS(agent_pos) to BPS(obstacles))")
print(f"Positive reward: {env.payout} (Multiplied with I(goal_reached))")
print("#####################################################################\n\n")

with open('logs_marc.txt', 'a') as logs:
    logs.write("\n\n########################## END RESULT ###############################\n")
    logs.write(rundatestr)
    logs.write(f"\nWhile learning, {env.current_step} out of {timesteps} episodes ended in success.\n")
    logs.write(f"During evaluation, in {n_success} of {n_runs} runs the agent managed to reach the goal while colliding {n_collision} times.\n")
    logs.write("--------------------------Used variables:-------------------------\n")
    logs.write(f"Grid size: {env.gridSize}, Object size: {env.object_size}, Num of obstacles: {env.num_obstacles}, BPS Size: {env.num_bps}\n")
    logs.write(f"Negative reward coefficient: {env.punishment} (Multiplied with L-inf norm of BPS(agent_pos) to BPS(obstacles))\n")
    logs.write(f"Positive reward: {env.payout} (Multiplied with I(goal_reached))\n")
    logs.write("#####################################################################\n\n")

"""
OLD REWARD:

if info["collision"] == 1:
    return float(-100)
elif np.linalg.norm(observation - desired_goal) <= 0.5:
    return float(1000)
else:
    return float(-np.linalg.norm(observation - desired_goal)) * 10

"""