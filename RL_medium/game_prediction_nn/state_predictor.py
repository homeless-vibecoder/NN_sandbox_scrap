import os
from typing import Tuple

import torch
import torch.nn as nn


class FixedWidthMLP(nn.Module):
    """
    Simple predictor MLP with fixed-width hidden layers.

    Architecture: input_dim -> width -> [width]* (depth-2) -> width -> output_dim
    - The first and last hidden layers have the same dimension ("width").
    - Depth >= 2 gives at least one hidden layer before output; set higher for more capacity.
    """

    def __init__(self, input_dim: int, output_dim: int, width: int = 128, depth: int = 4):
        super().__init__()
        assert depth >= 2, "depth must be >= 2"

        layers = []
        # Input -> first hidden
        layers.append(nn.Linear(input_dim, width))
        layers.append(nn.Tanh())
        # Middle hidden layers (fixed width)
        for _ in range(depth - 2):
            layers.append(nn.Linear(width, width))
            layers.append(nn.Tanh())
        # Last hidden -> output
        layers.append(nn.Linear(width, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def get_or_create_predictor(weights_path: str | None,
                             input_dim: int,
                             output_dim: int,
                             width: int = 128,
                             depth: int = 4) -> Tuple[FixedWidthMLP, str]:
    """
    Create predictor and load weights if present; otherwise initialize and save.
    """
    # Save under RL_medium/NN_weights/game_prediction
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    default_dir = os.path.join(root, "NN_weights", "game_prediction")
    os.makedirs(default_dir, exist_ok=True)
    default_path = os.path.join(default_dir, "state_predictor.pt")
    resolved = weights_path or default_path

    model = FixedWidthMLP(input_dim=input_dim, output_dim=output_dim, width=width, depth=depth)
    if os.path.exists(resolved):
        try:
            state = torch.load(resolved, map_location="cpu")
            model.load_state_dict(state)
        except Exception:
            # On mismatch/corruption, overwrite with fresh weights
            torch.save(model.state_dict(), resolved)
    else:
        torch.save(model.state_dict(), resolved)
    return model, resolved


