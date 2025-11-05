import os
import math
import json
from typing import List, Tuple, Optional

import torch
import torch.nn as nn

from nerual_network_class import (
    ResidualMLP,
    compose_fn,
    load_model_weights,
    _reduce_fraction_str,
    _ensure_dir,
    build_residual_mlp,
)


CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")
with open(CONFIG_PATH, "r") as f:
    CFG = json.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weights_dir = os.path.join(os.path.dirname(__file__), CFG["paths"]["weights_dir"]) 

# Base NN shape (must match trained models)
dim = CFG["model"]["dim"]
hidden_dim = CFG["model"]["hidden_dim"]
num_layers = CFG["model"]["num_layers"]

# Components to apply (ordered). Example: [[4,5],[4,5]] => F^(4/5) then F^(4/5)
components_cfg = CFG["run"].get("components", [])
components: List[Tuple[int, int]] = [(int(p), int(q)) for p, q in components_cfg]


def _load_fraction_model(dim: int, hidden_dim: int, num_layers: int, p: int, q: int, device: torch.device) -> Optional[nn.Module]:
    key = _reduce_fraction_str(p, q)
    path = os.path.join(weights_dir, f"F^({key}).pt")
    model = build_residual_mlp(dim, hidden_dim, num_layers).to(device)
    if load_model_weights(model, path, device):
        return model
    return None


def _compose_fraction_list(fracs: List[Tuple[int, int]]) -> Tuple[int, int]:
    # Multiply fractions p/q sequentially: overall exponent numerator/denominator
    num = 1
    den = 1
    for p, q in fracs:
        num *= p
        den *= q
    g = math.gcd(num, den)
    return num // g, den // g


def _compose_models(models: List[nn.Module]) -> nn.Module:
    class Chain(nn.Module):
        def __init__(self, ms: List[nn.Module]):
            super().__init__()
            self.ms = nn.ModuleList(ms)
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = x
            for m in self.ms:
                out = m(out)
            return out
    return Chain(models)


if __name__ == "__main__":
    # Validate components exist and load models in order
    models: List[nn.Module] = []
    for p, q in components:
        m = _load_fraction_model(dim, hidden_dim, num_layers, p, q, device)
        if m is None:
            raise FileNotFoundError(f"Missing trained component F^({p}_{q}).pt in {weights_dir}")
        models.append(m)

    # Compose models in sequence
    if len(models) == 0:
        print({"warning": "No components specified in config"})
    composed = _compose_models(models).to(device)

    # Demo: run on random inputs
    x = torch.randn(5, dim, device=device)
    with torch.no_grad():
        y = composed(x)
    overall = _compose_fraction_list(components) if len(components) else (0, 1)
    print({
        "components": components,
        "overall_fraction": overall,
        "input_shape": tuple(x.shape),
        "output_shape": tuple(y.shape),
    })


