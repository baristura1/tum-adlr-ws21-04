from stable_baselines3 import PPO, DDPG, SAC
import cv2
import time
from models import *
from parser import args


def run():
    pol = "CnnPolicy" if args.env_rep == 'img' else "MultiInputPolicy"
    curr = True if args.runtime == 'curriculum' else False
    if args.robot == 'joint':
        if args.env_rep == 'pos':
            env = PlanarFinder(num_obstacles=args.num_obstacles, timesteps=args.timesteps, num_bps=args.num_bps,
                               use_bps=False, curriculum=curr, dof=args.dof)
        elif args.env_rep == 'img':
            if args.dof == 2:
                env = PlanarFinderImage2(num_obstacles=args.num_obstacles, timesteps=args.timesteps, num_bps=args.num_bps,
                                         curriculum=curr)
            else:
                env = PlanarFinderImage(num_obstacles=args.num_obstacles, timesteps=args.timesteps, num_bps=args.num_bps,
                                        curriculum=curr)
        else:
            env = PlanarFinder(num_obstacles=args.num_obstacles, timesteps=args.timesteps, num_bps=args.num_bps,
                               use_bps=True, curriculum=curr, dof=args.dof)
    else:
        env = MobileFinder(num_obstacles=args.num_obstacles, timesteps=args.timesteps, num_bps=args.num_bps)

    obs = env.reset()

    if args.algorithm == 'PPO':
        model = PPO(pol, env, tensorboard_log='./logs/', verbose=1).learn(args.timesteps)
    elif args.algorithm == 'SAC':
        model = SAC(pol, env, tensorboard_log='./logs/',verbose=1).learn(args.timesteps)
    else:
        model = DDPG(pol, env, tensorboard_log='./logs/', verbose=1).learn(args.timesteps)

    frames = []
    n_steps = 360
    n_runs = args.eval_steps
    n_success = 0
    n_collision = 0
    for run in range(n_runs):
        if run % 500 == 0:
            print(f"Evaluation {run} / {args.eval_steps}.\n")
        obs = env.reset()
        env.render()
        for step in range(n_steps):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            if info["collision"] == 0:
                pass
            else:
                print("Collided.")
                n_collision += 1
                env.reset()
                break
            env.render(mode='human')
            time.sleep(0.02)
            if done:
                if run == n_runs - 1:
                    env.reset()
                print("Goal reached!")
                n_success += 1
                break

    env.out.release()

    print("\n\n########################## END RESULT ###############################")
    print(
        f"During evaluation, in {n_success} of {n_runs} runs the agent managed to reach the goal while colliding {n_collision} times.")
    print("--------------------------Used variables:-------------------------")
    print(f"Object size: {env.object_size}, Num of obstacles: {env.num_obstacles}, Robot type: {args.robot}")
    print(
        f"Environment representation: {args.env_rep}, Learning type: {args.runtime}, DoF: {args.dof}")
    print("#####################################################################\n\n")


if __name__ == '__main__':
    run()

