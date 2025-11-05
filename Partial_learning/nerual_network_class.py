import os
import math
import copy
from typing import Callable, Tuple, List, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualMLP(nn.Module):
    """A simple residual MLP mapping R^d -> R^d.

    Uses a residual connection to encourage near-identity behavior for small step sizes,
    which is useful for learning fractional powers of functions/networks.
    """

    def __init__(self, dim: int, hidden_dim: int = 128, num_layers: int = 2):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        layers = []
        in_dim = dim
        for layer_idx in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual: y = x + f(x)
        return x + self.net(x)


def compose_fn(fn: Callable[[torch.Tensor], torch.Tensor], times: int) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return a function applying `fn` repeatedly `times` times.

    times >= 1.
    """
    if times < 1:
        raise ValueError("times must be >= 1")

    def _composed(x: torch.Tensor) -> torch.Tensor:
        out = x
        for _ in range(times):
            out = fn(out)
        return out

    return _composed


def make_synthetic_target(dim: int, hidden_dim: int = 128, num_layers: int = 2, seed: int = 0) -> ResidualMLP:
    """Create a frozen target network F: R^d -> R^d to stand in for the mystery NN.

    In practice, you'd replace this with your real F. Here we initialize random
    weights to simulate a fixed black-box function.
    """
    torch.manual_seed(seed)
    target = ResidualMLP(dim=dim, hidden_dim=hidden_dim, num_layers=num_layers)
    for p in target.parameters():
        p.requires_grad_(False)
    return target


@torch.no_grad()
def sample_inputs(batch_size: int, dim: int, scale: float = 1.0, device: torch.device | None = None) -> torch.Tensor:
    dev = device if device is not None else torch.device("cpu")
    return torch.randn(batch_size, dim, device=dev) * scale


def _ensure_dir(path: str) -> None:
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def _reduce_fraction_str(numerator: int, denominator: int) -> str:
    g = math.gcd(numerator, denominator)
    return f"{numerator // g}_{denominator // g}"


def save_model_weights(model: nn.Module, path: str) -> None:
    _ensure_dir(os.path.dirname(path))
    torch.save(model.state_dict(), path)


def load_model_weights(model: nn.Module, path: str, device: torch.device) -> bool:
    if not os.path.isfile(path):
        return False
    state = torch.load(path, map_location=device)
    model.load_state_dict(state, strict=True)
    return True


def build_residual_mlp(dim: int, hidden_dim: int, num_layers: int) -> ResidualMLP:
    return ResidualMLP(dim=dim, hidden_dim=hidden_dim, num_layers=num_layers)


def load_or_generate_F(
    dim: int,
    hidden_dim: int,
    num_layers: int,
    device: torch.device,
    weights_dir: str,
    filename: str = "F^(1).pt",
    seed: int = 0,
) -> ResidualMLP:
    """Load F from weights if present, else generate synthetic F, save, and return it."""
    F_model = build_residual_mlp(dim, hidden_dim, num_layers).to(device)
    weights_path = os.path.join(weights_dir, filename)
    loaded = load_model_weights(F_model, weights_path, device)
    if not loaded:
        F_model = make_synthetic_target(dim=dim, hidden_dim=hidden_dim, num_layers=num_layers, seed=seed).to(device)
        save_model_weights(F_model, weights_path)
    for p_ in F_model.parameters():
        p_.requires_grad_(False)
    return F_model


def train_fractional_match(
    F_model: nn.Module,
    dim: int,
    p: int,
    q: int,
    steps: int = 2000,
    batch_size: int = 128,
    lr: float = 1e-3,
    input_scale: float = 1.0,
    device: torch.device | None = None,
) -> Tuple[nn.Module, list[float]]:
    """Train G (same architecture as F) so that F^p ≈ G^q on random inputs.

    Returns (G, losses).
    """
    if p < 1 or q < 1:
        raise ValueError("p and q must be >= 1")

    dev = device if device is not None else torch.device("cpu")

    # Initialize G as a deep copy of F (same architecture and weights)
    G_model = copy.deepcopy(F_model).to(dev)
    # Ensure G is trainable (F may be frozen)
    for param in G_model.parameters():
        param.requires_grad_(True)
    F_model = F_model.to(dev)
    for p_ in F_model.parameters():
        p_.requires_grad_(False)

    optimizer = torch.optim.Adam(G_model.parameters(), lr=lr)

    F_p = compose_fn(F_model, p)
    G_q = compose_fn(G_model, q)

    losses: list[float] = []
    for step in range(steps):
        x = sample_inputs(batch_size, dim, input_scale, dev)
        with torch.no_grad():
            target = F_p(x)
        pred = G_q(x)
        loss = F.mse_loss(pred, target)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        losses.append(float(loss.detach().cpu()))
    return G_model, losses


def _infer_dim_from_model(model: nn.Module) -> Optional[int]:
    # Best-effort inference for ResidualMLP
    if isinstance(model, ResidualMLP):
        # Last Linear out_features equals dimension
        last_linear = None
        for m in model.net.modules():
            if isinstance(m, nn.Linear):
                last_linear = m
        if last_linear is not None:
            return last_linear.out_features
    return None


def learn_fractional_power(
    F_model: Optional[nn.Module],
    p: int,
    q: int,
    *,
    dim: Optional[int] = None,
    steps: int = 1500,
    batch_size: int = 128,
    lr: float = 1e-3,
    input_scale: float = 1.0,
    device: Optional[torch.device] = None,
    weights_dir: Optional[str] = None,
    save: bool = True,
) -> Tuple[nn.Module, List[float], Optional[str]]:
    """Train and optionally save G such that F^p ≈ G^q.

    - If F_model is None, a synthetic F is loaded/generated and frozen.
    - epsilon used in naming is p/q (reduced). Saved filename: F^(p_q).pt
    """
    dev = device if device is not None else torch.device("cpu")
    wdir = weights_dir if weights_dir is not None else os.path.join(os.path.dirname(__file__), "weights")

    if F_model is None:
        # Default dimensions if not provided
        if dim is None:
            dim = 8
        F_model = load_or_generate_F(dim=dim, hidden_dim=128, num_layers=3, device=dev, weights_dir=wdir, filename="F^(1).pt", seed=0)
    else:
        # Infer dim if not provided
        if dim is None:
            dim = _infer_dim_from_model(F_model)
            if dim is None:
                raise ValueError("dim must be provided when F_model type is unknown")

    # Train
    G_model, losses = train_fractional_match(
        F_model=F_model,
        dim=int(dim),
        p=p,
        q=q,
        steps=steps,
        batch_size=batch_size,
        lr=lr,
        input_scale=input_scale,
        device=dev,
    )

    # Print a few sample losses
    idxs = [0, 10, 100, len(losses) - 1]
    seen = set()
    for idx in idxs:
        if 0 <= idx < len(losses) and idx not in seen:
            seen.add(idx)
            print({"loss_step": idx, "loss": losses[idx]})

    out_path: Optional[str] = None
    if save:
        eps_str = _reduce_fraction_str(p, q)
        out_path = os.path.join(wdir, f"F^({eps_str}).pt")
        save_model_weights(G_model, out_path)

    return G_model, losses, out_path


def learn_fractional_chain(
    F_model: Optional[nn.Module],
    p: int,
    q: int,
    *,
    dim: Optional[int] = None,
    steps: int = 1500,
    batch_size: int = 128,
    lr: float = 1e-3,
    input_scale: float = 1.0,
    device: Optional[torch.device] = None,
    weights_dir: Optional[str] = None,
    save: bool = True,
    num_iters: Optional[int] = None,
    min_alpha: Optional[float] = None,
) -> Dict[str, Dict[str, float | str]]:
    """Iteratively learn powers by repeatedly treating latest G as new F.

    On iteration k (1-indexed): trains G_k so that F_{k-1}^p ≈ G_k^q, where
    F_0 = F (input), and F_k := G_k. This yields G_k ≈ F^( (p/q)^k ).

    Stopping: provide num_iters, min_alpha (stop when (p/q)^k < min_alpha), or both.
    Saves each iteration as F^(p^k_q^k).pt if save=True.
    """
    if num_iters is None and min_alpha is None:
        raise ValueError("Specify num_iters or min_alpha (or both)")
    if p < 1 or q < 1:
        raise ValueError("p and q must be >= 1")

    dev = device if device is not None else torch.device("cpu")
    wdir = weights_dir if weights_dir is not None else os.path.join(os.path.dirname(__file__), "weights")

    # Initialize base F
    if F_model is None:
        if dim is None:
            dim = 8
        F_curr = load_or_generate_F(dim=dim, hidden_dim=128, num_layers=3, device=dev, weights_dir=wdir, filename="F^(1).pt", seed=0)
    else:
        F_curr = F_model.to(dev)
        if dim is None:
            dim = _infer_dim_from_model(F_curr)
            if dim is None:
                raise ValueError("dim must be provided when F_model type is unknown")

    results: Dict[str, Dict[str, float | str]] = {}
    k = 1
    p_pow = p
    q_pow = q

    def current_alpha() -> float:
        return (p_pow / q_pow)

    while True:
        if num_iters is not None and k > num_iters:
            break
        if min_alpha is not None and current_alpha() < min_alpha:
            break

        # Train next G from current base F
        G_next, losses = train_fractional_match(
            F_model=F_curr,
            dim=int(dim),
            p=p,
            q=q,
            steps=steps,
            batch_size=batch_size,
            lr=lr,
            input_scale=input_scale,
            device=dev,
        )

        # Print sample losses
        idxs = [0, 10, 100, len(losses) - 1]
        printed = set()
        for idx in idxs:
            if 0 <= idx < len(losses) and idx not in printed:
                printed.add(idx)
                print({"iter": k, "loss_step": idx, "loss": losses[idx]})

        # Save with exponent p^k / q^k
        eps_key = _reduce_fraction_str(p_pow, q_pow)
        out_path = os.path.join(wdir, f"F^({eps_key}).pt")
        if save:
            save_model_weights(G_next, out_path)

        results[eps_key] = {
            "iter": float(k),
            "alpha": float(current_alpha()),
            "last_loss": losses[-1] if len(losses) else float("nan"),
            "weights_path": out_path,
        }

        # Prepare for next iteration: F becomes G_next; update powers
        F_curr = G_next
        k += 1
        p_pow *= p
        q_pow *= q

    return results

