import math
from collections import namedtuple, deque
import random
from typing import List, Tuple

import torch
from gym import Env
from torch import nn

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory:
    """Replay memory class.

    We use deque to simulate the memory of length=capacity.
    When another transition added to the memory, we remove the oldest one
    from the front of the deque if the memory is full (equal to maxlen).

    This structure helps to avoid the problem of forgetting oldest transitions
    by the neural network.
    """
    def __init__(self, capacity: int):
        """Init memory deque.

        :param capacity: memory capacity
        """
        self.memory = deque([], maxlen=capacity)

    def push(self, *args) -> None:
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size: int) -> List[Transition]:
        """Sample a mini-batch of transitions"""
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)


class ActionSelector:
    """Epsilon-greedy strategy selector."""

    def __init__(
        self,
        env: Env,
        policy_net: nn.Module,
        eps_start: float = 0.95,
        eps_end: float = 0.05,
        eps_decay: int = 1000,
        device: torch.device = torch.device('cpu'),
    ) -> None:
        self.device = device
        self.env = env
        self.policy_net = policy_net
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.steps_done = 0

    def epsilon_on_step(self, steps_done: int):
        """Returns exploration vs exploitation epsilon threshold."""
        return self.eps_end + (self.eps_start - self.eps_end) * \
                                math.exp(-1. * steps_done / self.eps_decay)

    def get_info(self, max_steps: int) -> Tuple[List[float], List[float]]:
        """Returns all possible steps and epsilon values."""
        steps = [i for i in range(max_steps * 10)]
        eps = [self.epsilon_on_step(i) for i in steps]
        return steps, eps

    def select_action(self, state: torch.Tensor) -> torch.Tensor:
        """Select an action using epsilon-greedy strategy.

        :param state: current environment state
        :param steps_done: number of steps done
        """
        sample = random.random()
        eps_threshold_on_step = self.epsilon_on_step(self.steps_done)
        self.steps_done += 1

        if sample > eps_threshold_on_step:
            # exploitation strategy (use neural net to sample action)
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)

        # exploration strategy (sample random action)
        return torch.tensor([[self.env.action_space.sample()]], device=self.device, dtype=torch.long)
