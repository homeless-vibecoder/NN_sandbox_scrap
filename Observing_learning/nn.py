import os
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


def _get_activation(name: str) -> nn.Module:
    name_l = name.lower()
    if name_l in ("relu",):
        return nn.ReLU()
    if name_l in ("tanh",):
        return nn.Tanh()
    if name_l in ("gelu",):
        return nn.GELU()
    if name_l in ("silu", "swish"):
        return nn.SiLU()
    if name_l in ("leaky_relu", "lrelu"):
        return nn.LeakyReLU(negative_slope=0.01)
    raise ValueError(f"Unsupported activation: {name}")


def compose(model: nn.Module, x: torch.Tensor, k: int) -> torch.Tensor:
    if k < 1:
        raise ValueError("k must be >= 1")
    out = x
    for _ in range(k):
        out = model(out)
    return out


class ParamTracer:
    """Track specific parameter elements across snapshots.

    Stores traces for elements identified by (parameter_name, flat_index) pairs.
    Use with `NN.select_param_slices` and `NN.snapshot_params`.
    """

    def __init__(self, param_slices: List[Tuple[str, int]]):
        self.param_slices: List[Tuple[str, int]] = list(param_slices)
        self.traces: List[List[float]] = [[] for _ in self.param_slices]

    def add_snapshot(self, name_to_vec: Dict[str, torch.Tensor]) -> None:
        for i, (nm, li) in enumerate(self.param_slices):
            if nm not in name_to_vec:
                raise KeyError(f"Parameter '{nm}' not found during snapshot")
            vec = name_to_vec[nm]
            if li < 0 or li >= vec.numel():
                raise IndexError(f"Index {li} out of bounds for parameter '{nm}' of size {vec.numel()}")
            self.traces[i].append(float(vec[li].item()))


class NN(nn.Module):
    """Flexible MLP for observation and experimentation.

    Features:
    - Choose architecture via `hidden_layers` (explicit) or `width`+`depth` (fixed width)
    - Activation selection (relu/tanh/gelu/silu/leaky_relu)
    - Optional LayerNorm and Dropout between layers
    - Save/load helpers
    - Freeze/unfreeze utilities
    - Parameter tracing helpers for observing training dynamics
    - Composition helper via `forward_k`
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        hidden_layers: Optional[Sequence[int]] = None,
        width: int = 32,
        depth: int = 3,
        activation: str = "tanh",
        use_layernorm: bool = False,
        dropout: float = 0.0,
        final_activation: Optional[str] = None,
    ) -> None:
        super().__init__()

        if hidden_layers is None:
            if depth < 1:
                raise ValueError("depth must be >= 1")
            hidden_layers = [width] * max(0, depth - 1)

        if any(h <= 0 for h in hidden_layers):
            raise ValueError("hidden layer sizes must be positive")

        act = _get_activation(activation)
        final_act = _get_activation(final_activation) if final_activation else None

        layers: List[nn.Module] = []
        prev = input_dim
        for i, h in enumerate(hidden_layers):
            layers.append(nn.Linear(prev, h))
            if use_layernorm:
                layers.append(nn.LayerNorm(h))
            layers.append(_get_activation(activation)) if i < len(hidden_layers) else None
            if dropout and dropout > 0.0:
                layers.append(nn.Dropout(p=float(dropout)))
            prev = h

        layers.append(nn.Linear(prev, output_dim))
        if final_act is not None:
            layers.append(final_act)

        self.net = nn.Sequential(*[m for m in layers if m is not None])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @torch.no_grad()
    def forward_k(self, x: torch.Tensor, k: int) -> torch.Tensor:
        return compose(self, x, k)

    # ---------- Convenience utilities ----------
    def num_parameters(self, trainable_only: bool = False) -> int:
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def freeze(self) -> None:
        for p in self.parameters():
            p.requires_grad_(False)

    def unfreeze(self) -> None:
        for p in self.parameters():
            p.requires_grad_(True)

    def save_weights(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)

    def load_weights(self, path: str, map_location: str | torch.device = "cpu", strict: bool = True) -> None:
        state = torch.load(path, map_location=map_location)
        self.load_state_dict(state, strict=strict)

    # ---------- Tracing utilities ----------
    def _flat_trainable_registry(self) -> List[Tuple[str, torch.Tensor]]:
        return [(name, p.view(-1)) for name, p in self.named_parameters() if p.requires_grad]

    def select_param_slices(self, k: int, seed: int = 42) -> ParamTracer:
        """Select `k` random parameter elements to trace.

        Returns a ParamTracer with (name, flat_index) for stable identification.
        """
        rng = torch.Generator()
        rng.manual_seed(int(seed))

        flat_registry = self._flat_trainable_registry()
        if not flat_registry:
            raise RuntimeError("No trainable parameters to trace.")

        total = sum(vec.numel() for _, vec in flat_registry)
        k_eff = min(k, total)
        # Sample unique global indices
        perm = torch.randperm(total, generator=rng)
        chosen = perm[:k_eff].tolist()
        chosen.sort()

        # Map global -> (name, local_index)
        slices: List[Tuple[str, int]] = []
        offset = 0
        j = 0
        for name, vec in flat_registry:
            n = vec.numel()
            while j < len(chosen) and chosen[j] < offset + n:
                local_idx = int(chosen[j] - offset)
                slices.append((name, local_idx))
                j += 1
            offset += n

        return ParamTracer(slices)

    @torch.no_grad()
    def snapshot_params(self, tracer: ParamTracer) -> None:
        name_to_vec = {name: p.view(-1).detach().clone() for name, p in self.named_parameters() if p.requires_grad}
        tracer.add_snapshot(name_to_vec)


# Optional small sanity check when run directly (no argparse, uses defaults)
if __name__ == "__main__":
    model = NN(input_dim=8, output_dim=8, hidden_layers=[64, 64], activation="tanh", use_layernorm=False, dropout=0.0)
    x = torch.randn(4, 8)
    y1 = model(x)
    y3 = model.forward_k(x, 3)
    tracer = model.select_param_slices(k=16, seed=0)
    model.snapshot_params(tracer)
    print({
        "out_shape": tuple(y1.shape),
        "out3_shape": tuple(y3.shape),
        "num_params": model.num_parameters(),
        "num_trainable": model.num_parameters(trainable_only=True),
        "traced": len(tracer.param_slices),
    })


