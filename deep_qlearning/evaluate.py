from dataclasses import dataclass
from typing import List

import imageio as imageio
import numpy as np
import torch
from gym import Env
from torch import nn
from tqdm import tqdm

from deep_qlearning.model import DQN

@dataclass
class Episode:
    episode: int
    actions: List[int]
    total_reward: float


def next_action(state: np.ndarray, policy_net: nn.Module, device: torch.device) -> int:
    """Model adapter to get next action for gym environment"""
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    return policy_net(state).max(1)[1].item()


def show_episode(env: Env, episode: Episode, max_steps: int, video_save_path: str, fps: int = 1) -> None:
    images = []

    env.reset(seed=episode.episode)
    img = env.render()
    images.append(img)

    for step in range(max_steps):
        action = episode.actions[step]
        new_state, reward, terminated, truncated, info = env.step(action)
        img = env.render()
        images.append(img)

        if terminated or truncated:
            break

    imageio.mimsave(video_save_path, [np.array(img) for i, img in enumerate(images)], fps=fps)


def evaluate_agent(env: Env, policy_net: nn.Module, max_steps: int, n_episodes: int, device: torch.device) -> List[Episode]:
    """Evaluate the agent for ``n_episodes`` episodes and returns average reward and std of reward.

    :param env: The evaluation environment
    :param policy_net: trained policy network
    :param n_episodes: Number of episode to evaluate the agent
    :param max_steps: max_steps in episode
    :param policy_net: policy network to choose actions
    :param device: device to run
    """
    print('Evaluation...')

    episode_rewards = []
    episodes_tracks = []
    for episode in tqdm(range(n_episodes), total=n_episodes):
        state, info = env.reset(seed=episode)

        truncated = False
        terminated = False
        total_rewards_ep = 0

        actions = []
        for step in range(max_steps):
            action = next_action(state, policy_net, device)
            actions.append(action)
            new_state, reward, terminated, truncated, info = env.step(action)
            total_rewards_ep += reward

            if terminated or truncated:
                break

            state = new_state

        episodes_tracks.append(Episode(actions=actions, total_reward=total_rewards_ep, episode=episode))
        episode_rewards.append(total_rewards_ep)

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")
    print(f'Max possible reward: ', max_steps)  # for cart-pole

    return episodes_tracks
