import os
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn


def _build_mlp(input_dim: int, hidden_sizes: Tuple[int, int], output_dim: int) -> nn.Sequential:
    layers: List[nn.Module] = []
    prev = input_dim
    for h in hidden_sizes:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.Tanh())
        prev = h
    layers.append(nn.Linear(prev, output_dim))
    return nn.Sequential(*layers)


class CategoricalPolicy(nn.Module):
    """
    MLP mapping an observation vector to 4 independent 3-way categoricals
    for actions in {-1, 0, +1} per joint: [left_hip, left_knee, right_hip, right_knee].
    """

    def __init__(self, input_dim: int = 15, hidden_sizes: Tuple[int, int] = (128, 128)):
        super().__init__()
        # 4 joints * 3 classes = 12 logits
        self.net = _build_mlp(input_dim=input_dim, hidden_sizes=hidden_sizes, output_dim=12)

    def forward(self, obs_tensor: torch.Tensor) -> torch.Tensor:
        # returns logits of shape (batch, 12)
        return self.net(obs_tensor)

    def act_vec(self, obs_vec: List[float]) -> Tuple[Dict[str, int], torch.Tensor]:
        # Keep graph for policy gradient; callers can wrap with torch.no_grad() for inference
        x = torch.tensor([obs_vec], dtype=torch.float32)  # (1, D)
        logits = self.forward(x).view(1, 4, 3)  # (1, 4, 3)
        dists = [torch.distributions.Categorical(logits=logits[0, i]) for i in range(4)]
        samples = [d.sample() for d in dists]  # each in {0,1,2}
        log_probs = [d.log_prob(s) for d, s in zip(dists, samples)]
        # map class index -> {-1,0,+1}; 0->-1, 1->0, 2->+1
        idx_to_act = {0: -1, 1: 0, 2: +1}
        action = {
            "left_hip": idx_to_act[int(samples[0].item())],
            "left_knee": idx_to_act[int(samples[1].item())],
            "right_hip": idx_to_act[int(samples[2].item())],
            "right_knee": idx_to_act[int(samples[3].item())],
        }
        return action, torch.stack(log_probs).sum()

    @torch.no_grad()
    def act_vec_eval(self, obs_vec: List[float]) -> Dict[str, int]:
        x = torch.tensor([obs_vec], dtype=torch.float32)
        logits = self.forward(x).view(1, 4, 3)
        action = {}
        idx_to_act = {0: -1, 1: 0, 2: +1}
        for i, name in enumerate(["left_hip", "left_knee", "right_hip", "right_knee"]):
            cls = torch.argmax(logits[0, i]).item()
            action[name] = idx_to_act[int(cls)]
        return action

    def save_weights(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load_weights(self, path: str, map_location: str = "cpu") -> None:
        state = torch.load(path, map_location=map_location)
        self.load_state_dict(state)


def get_or_create_policy(weights_path: Optional[str] = None,
                         hidden_sizes: Tuple[int, int] = (128, 128),
                         input_dim: int = 15) -> Tuple[CategoricalPolicy, str]:
    # Place weights under RL_medium/NN_weights/RL_policies
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    default_dir = os.path.join(root, "NN_weights", "RL_policies")
    os.makedirs(default_dir, exist_ok=True)
    default_path = os.path.join(default_dir, "qwop_policy_conventional.pt")
    resolved = weights_path or default_path

    policy = CategoricalPolicy(input_dim=input_dim, hidden_sizes=hidden_sizes)
    if os.path.exists(resolved):
        try:
            policy.load_weights(resolved, map_location="cpu")
        except Exception:
            # shape mismatch or corrupted file; overwrite with fresh weights
            policy.save_weights(resolved)
    else:
        policy.save_weights(resolved)
    return policy, resolved


