import numpy as np
import gym
from gym import spaces
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.env_util import make_vec_env
from collections import OrderedDict
import copy

class goalFinder(gym.GoalEnv):


    def __init__(self, gridSize):
        super(goalFinder, self).__init__()
        self.gridSize = gridSize
        self.action_space = spaces.Box(low=np.array([-1,-1]), high=np.array([1,1]))
        self.observation_space = spaces.Dict({"observation": spaces.Box(low=np.array([0.0,0.0]), high=np.array([self.gridSize-1, self.gridSize-1])),
                                              "achieved_goal": spaces.Box(low=np.array([0.0,0.0]), high=np.array([self.gridSize-1, self.gridSize-1])),
                                              "desired_goal": spaces.Box(low=np.array([0.0,0.0]), high=np.array([self.gridSize-1, self.gridSize-1]))})
        self.obstacles = np.array([[4, 4], [4, 5], [5, 4], [5, 5], [5, 6], [6, 5], [6, 6]], dtype=np.float32) #7 blocks
        self.state = np.array([0.0, 0.0], dtype=np.float32)
        self.goal = np.array([self.gridSize-1, self.gridSize-1], dtype=np.float32)
        self.start_state = np.array([0.0, 0.0], dtype=np.float32)
        self.reset()

    def reset(self):
        #super().reset()
        self.state = np.array([0.0, 0.0], dtype=np.float32)
        self.goal = self.goal
        ob = self.get_obs()
        return ob

    def step(self, action):
        self.state[0] += action[0]
        self.state[1] += action[1]

        """
        if action == 0: #right
            self.state = np.add(self.state, np.array([1.0, 0.0]))
        elif action == 1: #left
            self.state = np.add(self.state, np.array([-1.0, 0.0]))
        elif action == 2: #up
            self.state = np.add(self.state, np.array([0.0, 1.0]))
        elif action == 3: #down
            self.state = np.add(self.state, np.array([0.0, -1.0]))
        else:
            raise ValueError("INVALID ACTION")
        """
        self.state = np.clip(self.state, 0, self.gridSize-1)
        ###implement clipping
        info = {"collision": 1} if self.check_collision() else {"collision": 0}
        reward = self.compute_reward(self.state, self.goal, info)
        obs = self.get_obs()
        done = reward == 10

        return obs, reward, done, info

    def compute_reward(self, observation, desired_goal, info):
        if np.array_equal(observation, desired_goal):
            reward = 10
        elif info["collision"] == 1:
            reward = -1
        else:
            reward = 0
        return float(reward)

    def render(self, mode="human"):
        pass

    def close(self):
        pass

    def check_collision(self):
        if np.array_equal(self.state, self.obstacles[0]):
            return True
        elif np.array_equal(self.state, self.obstacles[1]):
            return True
        elif np.array_equal(self.state, self.obstacles[2]):
            return True
        elif np.array_equal(self.state, self.obstacles[3]):
            return True
        elif np.array_equal(self.state, self.obstacles[4]):
            return True
        elif np.array_equal(self.state, self.obstacles[5]):
            return True
        elif np.array_equal(self.state, self.obstacles[6]):
            return True
        else:
            return False

    def get_obs(self):
        return OrderedDict([("desired_goal", self.goal.copy()),
                            ("achieved_goal", self.state.copy()),
                            ("observation", self.state.copy())])


env = goalFinder(gridSize=10)

check_env(env, warn=True)

#env = make_vec_env(lambda: env, n_envs=1)


obs = env.reset()
#print(obs)
model = PPO('MultiInputPolicy', env, verbose=0).learn(50000)
obs = env.reset()
n_steps = 50

for step in range(n_steps):
    if step == 0:
        obs = env.reset()
    action, _ = model.predict(obs, deterministic=False)
    print("Step {}".format(step + 1))
    print("Action: ", action)
    obs, reward, done, info = env.step(action)
    print('obs=', obs["observation"], 'reward=', reward, 'done=', done)

    #env.render(mode='human')
    if done:
        env.reset()
        print("Goal reached!", "reward=", reward)
        break

"""
print(env.observation_space["achieved_goal"])
print(env.observation_space.spaces["achieved_goal"])
env.observation_space.spaces["achieved_goal"] = [0.0,0.0]
print(env.observation_space.spaces["achieved_goal"])
print(env.observation_space["achieved_goal"])
print(env.observation_space)
"""
