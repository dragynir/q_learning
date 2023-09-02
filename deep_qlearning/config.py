from dataclasses import dataclass


@dataclass
class Config:
    """Config for CartPole env.

    https://gymnasium.farama.org/environments/classic_control/cart_pole/
    """
    env_name: str = "CartPole-v1"
    batch_size: int = 128
    replay_memory_capacity: int = 10000
    num_episodes: int = 600

    gamma: float = 0.99  # how future is important
    eps_start = 0.9  # for epsilon gready strategy
    eps_end = 0.05  # for epsilon gready strategy
    eps_decay: int = 1000
    tau = 0.005
    learning_rate: float = 1e-4
