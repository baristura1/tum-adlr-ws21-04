import numpy as np
import gym
from gym import spaces
from collections import OrderedDict
import cv2
import copy

class contEnv(gym.GoalEnv):

    def __init__(self):
        super(contEnv, self).__init__()
        self.action_space = spaces.Box(
            np.array([-2, -2]).astype(np.float32),
            np.array([2, 2]).astype(np.float32))
        pos_x_space = spaces.Box(low=-1.0, high=1.0, shape=(1,))
        pos_y_space = spaces.Box(low=-1.0, high=1.0, shape=(1,))
        tar_x_space = spaces.Box(low=-0.5, high=0.5, shape=(1,))
        tar_y_space = spaces.Box(low=-0.5, high=0.5, shape=(1,))
        self.observation_space = spaces.Dict({"Pos_X": pos_x_space, "Pos_Y": pos_y_space, "Tar_X": tar_x_space, "Tar_Y": tar_y_space})
        self.obstacles = np.array([[4, 4], [4, 5], [5, 4], [5, 5], [5, 6], [5, 7], [5, 8], [5, 9], [6, 5], [6, 6]], dtype=np.float32) #7 blocks
        self.pos = np.array([0.0, 0.0], dtype=np.float32)
        self.target = np.array([0.0, 0.0], dtype=np.float32)
        self.state = None
        self.timestep = 0.01
        self.reset()

    def reset(self):
        #super().reset()
        self.pos[0] = np.random.uniform(0.75, 1)
        self.pos[1] = np.random.uniform(0.75, 1)
        self.target[0] = np.random.uniform(-1, -0.75)
        self.target[1] = np.random.uniform(-1, -0.75)
        obs = self.get_obs()
        return obs

    def step(self, action):
        if abs(action[0]) > 2 or abs(action[1] > 2):
            raise ValueError("INVALID ACTION")
        else:
            vel_x = action[0]
            vel_y = action[1]
            #compute new positions
            self.pos[0] = self.pos[0] + vel_x * self.timestep
            self.pos[1] = self.pos[1] + vel_y * self.timestep

        self.pos = np.clip(self.pos, -1, 1) #clipping position
        info = {"collision": 1} if self.check_collision() else {"collision": 0}
        reward = self.compute_reward(self.pos, self.target, info)
        obs = self.get_obs()
        done = reward > -0.05

        return obs, reward, done, info

    def compute_reward(self, observation, desired_goal, info):
        reward = -np.linalg.norm(desired_goal - observation)
        """
        if np.allclose(observation, desired_goal, atol=0.1):
            reward = 10
        elif info["collision"] == 1:
            reward = -1
        else:
            reward = 0
        """
        return float(reward)

    def render(self, mode="human"):
        image = np.ones((400, 400, 3), dtype=np.uint8) * 255
        image = cv2.circle(img=image,
                           center=(int(self.pos[0] * 200) + 200, int(self.pos[1] * 200) + 200),
                           radius=10,
                           color=(255, 0, 0),
                           thickness=1)
        image = cv2.circle(img=image,
                           center=(int(self.target[0] * 200) + 200, int(self.target[1] * 200) + 200),
                           radius=5,
                           color=(0, 0, 255),
                           thickness=-1)
        cv2.imshow("PipeLine", image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return

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
        return OrderedDict([("Pos_X", self.pos[0].copy()),
                            ("Pos_Y", self.pos[1].copy()),
                            ("Tar_X", self.target[0].copy()),
                            ("Tar_Y", self.target[1].copy())])
