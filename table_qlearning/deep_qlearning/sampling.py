from collections import namedtuple, deque
import random
from typing import List

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
