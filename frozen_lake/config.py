from dataclasses import dataclass
from typing import List


@dataclass
class Config:
    env_name: str = 'FrozenLake-v1'
    map_name: str = '4x4'
    is_slippery: bool = True
    render_mode: str = 'rgb_array'

    # Training parameters
    n_training_episodes: int = 10000  # Total training episodes
    learning_rate: float = 0.7  # Learning rate

    # Evaluation parameters
    n_eval_episodes: int = 100  # Total number of test episodes

    # Environment parameters
    max_steps: int = 99  # Max steps per episode
    gamma: float = 0.95  # Discounting rate

    # Exploration parameters
    max_epsilon: float = 1.0  # Exploration probability at start
    min_epsilon: float = 0.05  # Minimum exploration probability
    decay_rate: float = 0.0005  # Exponential decay rate for exploration prob
