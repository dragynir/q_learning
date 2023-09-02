import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from deep_qlearning.config import Config
from deep_qlearning.model import DQN
from deep_qlearning.sampling import ReplayMemory


def train():
    config = Config()

    env = gym.make("CartPole-v1")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get the number of state observations and number of actions
    state, info = env.reset()
    n_observations = len(state)
    n_actions = env.action_space.n

    print('Number of observations:', n_observations)
    print('Number of actions:', n_actions)

    # create the policy network to choose actions
    policy_net = DQN(n_observations, n_actions).to(device)

    # create the target network to provide target values
    target_net = DQN(n_observations, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.AdamW(policy_net.parameters(), lr=config.learning_rate, amsgrad=True)
    memory = ReplayMemory(config.replay_memory_capacity)




if __name__ == '__main__':
    train()
