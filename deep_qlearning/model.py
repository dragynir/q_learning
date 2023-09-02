import torch
from torch import nn
import torch.nn.functional as F


class DQN(nn.Module):
    """Deep Q-Network."""
    def __init__(self, n_observations: int, n_actions: int) -> None:
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict next action using input observation.

        Called with either one element to determine next action, or a batch
        during optimization. Returns tensor([[left0exp,right0exp]...]).
        :param x: observation
        :return: action
        """
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
