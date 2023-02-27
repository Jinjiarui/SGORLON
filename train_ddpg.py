import argparse
import gym
import torch
import random
import numpy as np
import os
from models.ddpg import DDPG
from env import *
from env.mujoco import *

def get_args():
    parser = argparse.ArgumentParser()
    # the environment setting
    parser.add_argument('--env-name', type=str, default='GoalPlane-v0', help='the environment name')
    parser.add_argument('--test', type=str, default='GoalPlaneTest-v0')
    parser.add_argument('--n-epochs', type=int, default=30000, help='the number of epochs to train the agent')
    parser.add_argument('--n-batches', type=int, default=50, help='the times to update the network')
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--dis_s', type=int, default=10, help='discrete state dim')
    parser.add_argument('--dis_a', type=int, default=10, help='discrete action dim')

    parser.add_argument('--replay-strategy', type=str, default='graph', help='the HER strategy')
    parser.add_argument('--save-dir', type=str, default='saved_models/', help='the path to save the models')

    parser.add_argument('--noise-eps', type=float, default=0.2, help='noise factor for Gaussian')
    parser.add_argument('--random-eps', type=float, default=0.2, help="prob for acting randomly")

    parser.add_argument('--buffer-size', type=int, default=int(1e6), help='the size of the buffer')
    parser.add_argument('--warmup', type=int, default=10, help='the initial size of buffer')
    parser.add_argument('--replay-k', type=int, default=5, help='ratio to be replaced')
    parser.add_argument('--future-step', type=int, default=200, help='future step to be sampled')
    parser.add_argument('--batch-size', type=int, default=256, help='the sample batch size')
    parser.add_argument('--gamma', type=float, default=0.98, help='the discount factor')
    parser.add_argument('--action-l2', type=float, default=0.5, help='l2 reg')
    parser.add_argument('--lr-actor', type=float, default=0.0002, help='the learning rate of the actor')
    parser.add_argument('--lr-critic', type=float, default=0.0002, help='the learning rate of the critic')
    parser.add_argument('--polyak', type=float, default=0.98, help='the average coefficient')
    parser.add_argument('--n-test-rollouts', type=int, default=10, help='the number of tests')

    parser.add_argument('--metric', type=str, default='MLP', help='the metric for the distance embedding')
    parser.add_argument('--device', type=str, default="cpu", help='cuda device')

    parser.add_argument('--lr-decay-actor', type=int, default=3000, help='actor learning rate decay')
    parser.add_argument('--lr-decay-critic', type=int, default=3000, help='critic learning rate decay')

    parser.add_argument('--period', type=int, default=3, help='target update period')
    parser.add_argument('--distance', type=float, default=0.1,  help='distance threshold for HER')

    parser.add_argument('--resume', action='store_true', help='resume or not')
    # Will be considered only if resume is True
    parser.add_argument('--resume-epoch', type=int, default=10000, help='resume epoch')
    parser.add_argument('--resume-path', type=str, default='saved_models/', help='resume path')

    #args for the goal generator
    parser.add_argument('--lr-goal', type=float, default=0.0002, help='the learning rate of the goal generation')
    parser.add_argument('--type', type=str, default='neighbor', help='the metric for graph partition:neighbor/certainty')
    parser.add_argument('--trajectory-part', type=int, default=3, help='the number of parts to divide the graph')
    parser.add_argument('--is-random', action='store_true', help='random or qmax')

    args = parser.parse_args()
    return args


def get_env_params(env):
    obs = env.reset()
    params = {'obs': obs['observation'].shape[0],
              'goal': obs['desired_goal'].shape[0],
              'action': env.action_space.shape[0],
              'action_max': env.action_space.high[0],
              }
    params['max_timesteps'] = env._max_episode_steps
    return params


def launch(args):
    env = gym.make(args.env_name)
    test_env = gym.make(args.test)
    env.seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device is not 'cpu':
        torch.cuda.manual_seed(args.seed)
    # get the environment parameters
    env_params = get_env_params(env)
    env_params['max_test_timesteps'] = test_env._max_episode_steps
    # create the ddpg agent to interact with the environment
    ddpg_trainer = DDPG(args, env, env_params, test_env)
    ddpg_trainer.learn()


if __name__ == '__main__':
    args = get_args()
    launch(args)