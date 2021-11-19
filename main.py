from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import DQN, A2C, TD3, DDPG
from stable_baselines3.common.env_util import make_vec_env
from myEnv import contEnv

# Instantiate the env
env = contEnv()
# wrap it
env = make_vec_env(lambda: env, n_envs=1)
obs = env.reset()

#Train the agent
model = A2C('MultiInputPolicy', env, verbose=1).learn(50000)

# Test the trained agent
obs = env.reset()

n_steps = 500
for step in range(n_steps):
  env.render()
  action, _ = model.predict(obs, deterministic=True)
  print("Step {}".format(step + 1))
  print("Action: ", action)
  obs, reward, done, info = env.step(action)
  print('obs=', obs, 'reward=', reward, 'done=', done)
  if done:
    # Note that the VecEnv resets automatically
    # when a done signal is encountered
    print("Goal reached!", "reward=", reward)
    break

