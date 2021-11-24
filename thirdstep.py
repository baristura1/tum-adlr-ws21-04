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

        if np.any(self.position == 0.0) or \
                np.any(self.position == self.spawn_space.high):
            self.velocity = self.vel_space.sample()


class goalFinder(gym.GoalEnv):

    def __init__(self, gridSize, num_obstacles):
        super(goalFinder, self).__init__()
        self.gridSize = gridSize
        self.num_obstacles = num_obstacles
        self.action_space = spaces.Box(low=np.array([-1,-1]), high=np.array([1,1]))
        self.observation_space = spaces.Dict({"observation": spaces.Box(low=0.0, high=self.gridSize-1,
                                                                        shape=(self.num_obstacles+1, 4), dtype=np.float32),
                                              "achieved_goal": spaces.Box(low=0.0, high=self.gridSize-1,
                                                                        shape=(self.num_obstacles+1, 4), dtype=np.float32),
                                              "desired_goal": spaces.Box(low=np.array([0.0, 0.0, 0.0, 0.0]),
                                                                         high=np.array([self.gridSize-1, self.gridSize-1, 0.0, 0.0]))
                                              })

        # self.obstacles = np.array([Obstacle(gridSize=self.gridSize, tag=i) for i in range(3)])
        self.state = None
        self.goal = np.array([self.gridSize-1, self.gridSize-1, 0.0, 0.0], dtype=np.float32)
        self.start_state = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.object_size = 1
        self.reset()
        # for obstacle in self.obstacles:
        #    obstacle.spawn()

    def reset(self):
        #super().reset()
        self.state = self.observation_space["observation"].sample()
        self.state[0] = self.start_state
        self.state[1:, 2:] = self.state[1:, 2:] / 10
        self.goal = self.goal
        ob = self.get_obs()
        return ob


    def step(self, action):
        self.state[0, 0] += action[0]
        self.state[0, 1] += action[1]
        self.state[0] = np.clip(self.state[0], 0, self.gridSize - 1)

        info = {"collision": 0}

        for i in (np.arange(self.num_obstacles) + 1):
            self.state[i, 0] = self.state[i, 0] + self.state[i, 2]
            self.state[i, 1] = self.state[i, 1] + self.state[i, 3]
            self.state[i] = np.clip(self.state[i], 0, self.gridSize - 1)
            if np.linalg.norm(self.state[0] - self.state[i]) <= 2 * env.object_size:
                info["collision"] = 1

        # info = {"collision": 1} if self.is_colliding() else {"collision": 0}
        obs = self.get_obs()
        reward = self.compute_reward(obs["achieved_goal"][0], obs["desired_goal"], info)
        done = reward == 0

        return obs, reward, done, info

    def compute_reward(self, observation, desired_goal, info):
        if info["collision"] == 1:
            return float(-100)
        else:
            return float(-np.linalg.norm(observation - desired_goal))

        """
        if np.array_equal(observation, desired_goal):
            reward = 10
        elif info["collision"] == 1:
            reward = -1
        else:
            reward = 0
        return float(reward)
        """

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
        return OrderedDict([("observation", self.state.copy()),
                            ("achieved_goal", self.state.copy()),
                            ("desired_goal", self.goal.copy())])


env = goalFinder(gridSize=10, num_obstacles=3)

#check_env(env, warn=True)


obs = env.reset()

model = PPO('MultiInputPolicy', env, verbose=0).learn(5000000)

obs = env.reset()

n_steps = 50

for step in range(n_steps):
    if step == 0:
        obs = env.reset()
    action, _ = model.predict(obs, deterministic=False)
    print("###########################STEP {}################################".format(step + 1))
    print("Action: ", action)
    print("******************************************************************")
    obs, reward, done, info = env.step(action)
    print('obs=', obs["observation"][0], 'reward=', reward, 'done=', done)
    print("******************************************************************")
    print("Obstacle positions: ", obs["observation"][1, :2], obs["observation"][2, :2], obs["observation"][3, :2])
    print("Obstacle vels: ", obs["observation"][1, 2:], obs["observation"][1, 2:], obs["observation"][1, 2:])
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
