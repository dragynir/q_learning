from dataclasses import dataclass
from typing import List


@dataclass
class Config:
    env_name: str = 'Taxi-v3'
    render_mode: str = 'rgb_array'

    # Training parameters
    n_training_episodes: int = 25000  # Total training episodes
    learning_rate: float = 0.7  # Learning rate

    # Evaluation parameters
    n_eval_episodes: int = 100  # Total number of test episodes

    # DO NOT MODIFY EVAL_SEED
    eval_seed = [16, 54, 165, 177, 191, 191, 120, 80, 149, 178, 48, 38, 6, 125, 174, 73, 50, 172, 100, 148, 146, 6, 25,
                 40, 68, 148, 49, 167, 9, 97, 164, 176, 61, 7, 54, 55,
                 161, 131, 184, 51, 170, 12, 120, 113, 95, 126, 51, 98, 36, 135, 54, 82, 45, 95, 89, 59, 95, 124, 9,
                 113, 58, 85, 51, 134, 121, 169, 105, 21, 30, 11, 50, 65, 12, 43, 82, 145, 152, 97, 106, 55, 31, 85, 38,
                 112, 102, 168, 123, 97, 21, 83, 158, 26, 80, 63, 5, 81, 32, 11, 28,
                 148]  # Evaluation seed, this ensures that all classmates agents are trained on the same taxi starting position
    # Each seed has a specific starting state

    # Environment parameters
    max_steps: int = 99  # Max steps per episode
    gamma: float = 0.95  # Discounting rate

    # Exploration parameters
    max_epsilon: float = 1.0  # Exploration probability at start
    min_epsilon: float = 0.05  # Minimum exploration probability
    decay_rate: float = 0.005  # Exponential decay rate for exploration prob
