import argparse


parser = argparse.ArgumentParser(description='Arguments for running the scripts')

parser.add_argument('--robot', type=str, default='joint', help='robot type, mobile or joint')
parser.add_argument('--env_rep', type=str, default='bps', choices=['pos', 'bps', 'img'],
                    help="environment representation, positional or image or bps")
parser.add_argument('--runtime', type=str, default='vanilla', help="runtime options, vanilla or curriculum")
parser.add_argument('--dof', type=int, default=2, choices=[2,3], help='degrees of freedom, 2 or 3')
parser.add_argument('--num_obstacles', type=int, default=2, help='number of obstacles to use')


parser.add_argument('--timesteps', type=int, default=300000, help='total timesteps for training')
parser.add_argument('--num_bps', type=int, default=10, help='number of basis pts')
parser.add_argument('--algorithm', type=str, default='PPO', help='algorithm to use, PPO or DDPG or SAC')
parser.add_argument('--eval_steps', type=int, default=1000, help='number of runs for evaluation')


args = parser.parse_args()




