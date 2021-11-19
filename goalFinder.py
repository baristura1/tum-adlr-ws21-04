import numpy as np
import gym
from gym import spaces
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import A2C, PPO
from collections import OrderedDict

class Obstacle:

    def __init__(self, gridSize, tag: int, radius=None):
        self.tag = tag
        self.gridSize = gridSize
        self.radius = radius if radius is not None else 1
        self.spawn_space = spaces.Box(low=np.array([self.radius, self.radius]),
                                      high=np.array([self.gridSize-1-self.radius, self.gridSize-1-self.radius]),
                                      dtype=np.float32)
        self.vel_space = spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), dtype=np.float32)
        self.position = None
        self.velocity = None
        self.spawn()

    def spawn(self) -> None:
        self.position = self.spawn_space.sample()
        self.velocity = self.vel_space.sample()

    def update_pos(self) -> None:
        self.position[0] = self.position[0] + self.velocity[0]
        self.position[1] = self.position[1] + self.velocity[1]
        self.position = np.clip(self.position, self.radius, self.gridSize-1-self.radius)


class goalFinder(gym.GoalEnv):

    def __init__(self, gridSize):
        super(goalFinder, self).__init__()
        self.gridSize = gridSize
        self.action_space = spaces.Box(low=np.array([-1,-1]), high=np.array([1,1]))
        self.observation_space = spaces.Dict({"observation": spaces.Box(low=np.array([0.0,0.0]), high=np.array([self.gridSize-1, self.gridSize-1])),
                                              "achieved_goal": spaces.Box(low=np.array([0.0,0.0]), high=np.array([self.gridSize-1, self.gridSize-1])),
                                              "desired_goal": spaces.Box(low=np.array([0.0,0.0]), high=np.array([self.gridSize-1, self.gridSize-1]))})
        #self.obstacles = np.array([[4, 4], [4, 5], [5, 4], [5, 5], [5, 6], [6, 5], [6, 6]], dtype=np.float32) #7 blocks
        self.obstacles = np.array([Obstacle(gridSize=self.gridSize, tag=i) for i in range(3)])
        self.state = np.array([0.0, 0.0], dtype=np.float32)
        self.goal = np.array([self.gridSize-1, self.gridSize-1], dtype=np.float32)
        self.start_state = np.array([0.0, 0.0], dtype=np.float32)
        self.agent_size = 1
        self.reset()
        #for obstacle in self.obstacles:
        #    obstacle.spawn()

    def reset(self):
        #super().reset()
        self.state = np.array([0.0, 0.0], dtype=np.float32)
        self.goal = self.goal
        ob = self.get_obs()
        return ob

    def step(self, action):
        self.state[0] += action[0]
        self.state[1] += action[1]
        self.state = np.clip(self.state, 0, self.gridSize - 1)
        info = {"collision": 0}
        for obstacle in self.obstacles:
            obstacle.update_pos()
            if np.linalg.norm(env.state - obstacle.position) <= env.agent_size + obstacle.radius:
                info["collision"] = 1

        #
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


        #info = {"collision": 1} if self.is_colliding() else {"collision": 0}
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

    def is_colliding(self, obstacle):
        return np.linalg.norm(env.state - obstacle.position) <= env.agent_size + obstacle.radius

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


obs = env.reset()

model = PPO('MultiInputPolicy', env, verbose=0).learn(500000)

obs = env.reset()

n_steps = 50

for obstacle in env.obstacles:
    obstacle.spawn() #have to reset after training

for step in range(n_steps):
    if step == 0:
        obs = env.reset()
    action, _ = model.predict(obs, deterministic=False)
    print("###########################STEP {}################################".format(step + 1))
    print("Action: ", action)
    print("******************************************************************")
    obs, reward, done, info = env.step(action)
    print('obs=', obs["observation"], 'reward=', reward, 'done=', done)
    print("******************************************************************")
    print("Obstacle positions: ", env.obstacles[0].position, env.obstacles[1].position, env.obstacles[2].position)
    print("Obstacle vels: ", env.obstacles[0].velocity, env.obstacles[1].velocity, env.obstacles[2].velocity)
    print("******************************************************************")
    if info["collision"] == 0:
        print("No collision")
    else:
        print("COLLIDED")
    print("##################################################################")
    print("\n\n")
    #env.render(mode='human')
    if done:
        env.reset()
        print("Goal reached!", "reward=", reward)
        break
