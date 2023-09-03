import os

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
from tqdm import tqdm

from deep_qlearning.config import Config
from deep_qlearning.evaluate import evaluate_agent, show_episode
from deep_qlearning.model import DQN
from deep_qlearning.sampling import ReplayMemory, ActionSelector, Transition
from deep_qlearning.visualization import plot_values


def train_one_step(
    memory: ReplayMemory,
    policy_net: DQN,
    target_net: DQN,
    device: torch.device,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    batch_size: int,
    gamma: float,
):
    """"""
    transitions = memory.sample(batch_size)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which episode simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)

    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the 'older' target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(batch_size, device=device)
    with torch.no_grad():
        # perform greedy-policy with max to select action (off-policy)
        # our expected reward on next state
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0]

    # Compute the expected Q values
    expected_state_action_values = (gamma * next_state_values) + reward_batch

    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

    return loss


def train():
    config = Config()

    env = gym.make(config.env_name, render_mode=config.render_mode)
    env.reset()
    env.render()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    action_selector = ActionSelector(
        env,
        policy_net,
        config.eps_start,
        config.eps_end,
        config.eps_decay,
        device,
    )
    plot_values(
        *action_selector.get_info(config.eps_decay),
        title='Epsilon on step (Exploration vs Exploitation)',
        xlabel='step',
        ylabel='epsilon',
    )
    # Compute Huber loss https://pytorch.org/docs/stable/generated/torch.nn.SmoothL1Loss.html
    criterion = nn.SmoothL1Loss()

    print('Training..')
    for i_episode in tqdm(range(config.num_episodes), total=config.num_episodes):
        # Initialize the environment and get it's state
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        for t in count():
            action = action_selector.select_action(state)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            if len(memory) >= config.batch_size:
                loss = train_one_step(
                    memory,
                    policy_net,
                    target_net,
                    device,
                    optimizer,
                    criterion,
                    config.batch_size,
                    config.gamma,
                )

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = target_net.state_dict()
                policy_net_state_dict = policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key] * config.tau + target_net_state_dict[key] * (1 - config.tau)

                target_net.load_state_dict(target_net_state_dict)

            if done:
                break

    episodes = evaluate_agent(env, policy_net, config.max_eval_steps, config.n_eval_episodes, device)

    video_save_path = './replays/trained.mp4'
    if os.path.exists(video_save_path):
        os.remove(video_save_path)
    show_episode(env, episodes[0], config.max_eval_steps, video_save_path, fps=1)


if __name__ == '__main__':
    # TODO install pytorch-gpu to env: done
    # TODO debug environment: done
    # TODO visualize and evaluate environment (track episodes with bad and good results)

    # TODO log metrics to wandb
    # TODO train cart environment
    # TODO test on new environments
    train()
