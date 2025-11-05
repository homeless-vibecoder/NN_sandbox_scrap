import os
from typing import Dict, Iterable, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_mlp(input_dim: int, hidden_sizes: Iterable[int], output_dim: int) -> nn.Sequential:
    layers = []
    prev = input_dim
    for hidden in hidden_sizes:
        layers.append(nn.Linear(prev, hidden))
        layers.append(nn.Tanh())
        prev = hidden
    layers.append(nn.Linear(prev, output_dim))
    return nn.Sequential(*layers)


class BallPlatePolicy(nn.Module):
    """
    Simple MLP policy mapping observation (y, v, p, p_dot, t) -> action p'' in [-1, 1].

    - action_scale can be used to scale output range to match environment limits.
    """

    def __init__(self,
                 hidden_sizes: Tuple[int, int] = (64, 64),
                 action_scale: float = 1.0):
        super().__init__()
        self.net = _build_mlp(input_dim=5, hidden_sizes=hidden_sizes, output_dim=1)
        self.action_scale = float(action_scale)

    def forward(self, obs_tensor: torch.Tensor) -> torch.Tensor:
        """
        obs_tensor: shape (batch, 5)
        returns: shape (batch, 1) in [-action_scale, action_scale]
        """
        logits = self.net(obs_tensor)
        action = torch.tanh(logits) * self.action_scale
        return action

    @torch.no_grad()
    def act_from_obs_dict(self, obs: Dict[str, float]) -> float:
        """
        Convert observation dict to tensor and produce a scalar action.
        Keys expected: 'y', 'v', 'p', 'p_dot', 't'
        """
        x = torch.tensor([[obs["y"], obs["v"], obs["p"], obs["p_dot"], obs["t"]]], dtype=torch.float32)
        a = self.forward(x)
        return float(a.item())

    def save_weights(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load_weights(self, path: str, map_location: str = "cpu") -> None:
        state = torch.load(path, map_location=map_location)
        self.load_state_dict(state)


def get_or_create_policy(weights_path: Optional[str] = None,
                         hidden_sizes: Tuple[int, int] = (64, 64),
                         action_scale: float = 1.0) -> Tuple[BallPlatePolicy, str]:
    """
    Load policy weights if available; otherwise initialize and save to default.

    Returns (policy, resolved_weights_path).
    """
    default_dir = os.path.join(os.path.dirname(__file__), "NN_weights")
    default_path = os.path.join(default_dir, "policy.pt")
    resolved_path = weights_path or default_path

    policy = BallPlatePolicy(hidden_sizes=hidden_sizes, action_scale=action_scale)

    if os.path.exists(resolved_path):
        policy.load_weights(resolved_path, map_location="cpu")
    else:
        os.makedirs(os.path.dirname(resolved_path), exist_ok=True)
        policy.save_weights(resolved_path)

    return policy, resolved_path


def _demo():
    policy, path = get_or_create_policy(weights_path=None, hidden_sizes=(64, 64), action_scale=1.0)
    obs = {"y": 1.0, "v": 0.0, "p": 0.5, "p_dot": 0.0, "t": 0.0}
    a = policy.act_from_obs_dict(obs)
    print("weights:", path)
    print("sample action:", a)


if __name__ == "__main__":
    _demo()


