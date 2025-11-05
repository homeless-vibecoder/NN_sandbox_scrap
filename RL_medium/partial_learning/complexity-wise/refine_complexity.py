import os
import sys
from typing import Dict, List, Tuple

import torch
import torch.nn as nn


# Project roots for imports
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from game_prediction_nn.state_predictor import FixedWidthMLP

GAME_DIR = os.path.join(BASE_DIR, "game")
if GAME_DIR not in sys.path:
    sys.path.insert(0, GAME_DIR)
from qwop_env import QWOPRunnerEnv

RL_CONV_DIR = os.path.join(BASE_DIR, "RL_training", "conventional_method")
if RL_CONV_DIR not in sys.path:
    sys.path.insert(0, RL_CONV_DIR)
from train_qwop import build_obs_vec


def gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return abs(a)


def compose_core(model: FixedWidthMLP, h: torch.Tensor, k: int) -> torch.Tensor:
    y = h
    for _ in range(k):
        y = forward_core(model, y)
    return y


def collect_states(env: QWOPRunnerEnv, num_steps: int, action_repeat: int = 3) -> torch.Tensor:
    import random
    rng = random.Random(0)
    env.reset()
    seq: List[List[float]] = []
    for _ in range(num_steps):
        seq.append(build_obs_vec(env))
        action = {
            "left_hip": rng.choice([-1, 0, +1]),
            "left_knee": rng.choice([-1, 0, +1]),
            "right_hip": rng.choice([-1, 0, +1]),
            "right_knee": rng.choice([-1, 0, +1]),
        }
        done = False
        for _ in range(action_repeat):
            _, _, done, _ = env.step(action)
            if done:
                env.reset()
                break
    return torch.tensor(seq, dtype=torch.float32)


def list_complexity_models() -> Dict[Tuple[int, int], str]:
    """
    Return mapping (p,q) -> weight_path found under the complexity-wise partial folder only.
    The folder should contain files named: state_predictor_partial_complexity-wise_{p}_{q}.pt
    """
    weights_root = os.path.join(BASE_DIR, "NN_weights", "game_prediction")
    cx_dir = os.path.join(weights_root, "partial_state_prediction_complexity-wise")
    out: Dict[Tuple[int, int], str] = {}
    if os.path.isdir(cx_dir):
        for fname in os.listdir(cx_dir):
            if not fname.startswith("state_predictor_partial_complexity-wise_") or not fname.endswith(".pt"):
                continue
            stem = fname[len("state_predictor_partial_complexity-wise_"):-3]
            parts = stem.split("_")
            if len(parts) != 2:
                continue
            try:
                p = int(parts[0])
                q = int(parts[1])
                out[(p, q)] = os.path.join(cx_dir, fname)
            except ValueError:
                continue
    return out


def _infer_arch_from_state(state: dict) -> Tuple[int, int, int]:
    """
    Infer (input_dim, width, depth) from a FixedWidthMLP state_dict.
    depth equals number of Linear layers in the Sequential.
    """
    linear_indices: List[int] = []
    for k in state.keys():
        if not k.startswith("net.") or not k.endswith(".weight"):
            continue
        try:
            idx = int(k.split(".")[1])
        except Exception:
            continue
        linear_indices.append(idx)
    if not linear_indices:
        raise ValueError("Could not infer architecture from state_dict")
    first_idx = min(linear_indices)
    last_idx = max(linear_indices)
    input_dim = int(state[f"net.{first_idx}.weight"].shape[1])
    width = int(state[f"net.{first_idx}.weight"].shape[0])
    depth = len(set(linear_indices))
    return input_dim, width, depth


def load_model(path: str) -> FixedWidthMLP:
    state = torch.load(path, map_location="cpu")
    input_dim, width, depth = _infer_arch_from_state(state)
    model = FixedWidthMLP(input_dim=input_dim, output_dim=input_dim, width=width, depth=depth)
    model.load_state_dict(state)
    model.eval()
    return model


def forward_in_adapter(model: FixedWidthMLP, x: torch.Tensor) -> torch.Tensor:
    # net[0]: Linear(in->width); net[1]: Tanh
    h = model.net[0](x)
    h = model.net[1](h)
    return h


def forward_core(model: FixedWidthMLP, h: torch.Tensor) -> torch.Tensor:
    # Apply middle hidden stack: model.net[2:-1]
    y = h
    for layer in list(model.net)[2:-1]:
        y = layer(y)
    return y


def forward_out_adapter(model: FixedWidthMLP, h: torch.Tensor) -> torch.Tensor:
    # net[-1]: Linear(width->out)
    return model.net[-1](h)


def get_core_parameters(model: FixedWidthMLP) -> List[nn.Parameter]:
    params: List[nn.Parameter] = []
    for idx, layer in enumerate(model.net):
        # Skip input adapter (0,1) and output adapter (last)
        if idx < 2 or idx == len(model.net) - 1:
            continue
        # Only Linear layers have parameters
        if isinstance(layer, nn.Linear):
            params += list(layer.parameters())
    return params


def refine_pair_core(high_path: str, low_path: str, a: int, b: int, c: int, d: int,
                     X: torch.Tensor, epochs: int, lr: float, batch_size: int) -> None:
    """
    Refine the LOWER model core so that Core_low^(b*c/g)(h0) ≈ Core_high^(a*d/g)(h0), g=gcd(a*d,b*c),
    where h0 = InAdapter_low(x). Only core parameters are updated; adapters are frozen.
    Saves back to low_path.
    """
    # Exponents reduced by gcd
    e1 = a * d
    e2 = b * c
    g = gcd(e1, e2)
    e1 //= g
    e2 //= g

    high = load_model(high_path)
    low = load_model(low_path)

    optimizer = torch.optim.Adam(get_core_parameters(low), lr=lr)
    loss_fn = nn.MSELoss()

    num_samples = X.shape[0]
    num_batches = max(1, num_samples // batch_size)
    for epoch in range(1, epochs + 1):
        perm = torch.randperm(num_samples)
        X_epoch = X[perm]
        epoch_loss = 0.0
        for bidx in range(num_batches):
            xb = X_epoch[bidx * batch_size:(bidx + 1) * batch_size]
            with torch.no_grad():
                h0 = forward_in_adapter(low, xb)
                target_h = compose_core(high, h0, e1)
            pred_h = compose_core(low, h0, e2)
            loss = loss_fn(pred_h, target_h)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(get_core_parameters(low), max_norm=1.0)
            optimizer.step()
            epoch_loss += float(loss.item())
        if epoch % max(1, epochs // 5) == 0:
            print(f"refine-core ({a}/{b}) vs ({c}/{d}) -> loss={epoch_loss/num_batches:.6f}")

    torch.save(low.state_dict(), low_path)
    print("Refined core and saved:", low_path)


def run_refinement():
    # Training hyperparameters (specific to this refinement process)
    epochs_between = 100
    lr = 3e-4
    batch_size = 256

    # Load available fractions and sort descending
    frac_to_path = list_complexity_models()
    fracs = sorted(frac_to_path.keys(), key=lambda t: (t[0] / t[1]), reverse=True)

    # Data
    env = QWOPRunnerEnv()
    X = collect_states(env, num_steps=10000, action_repeat=3)

    # Iterate over adjacent pairs (largest -> smaller); refine only the smaller
    for i in range(len(fracs) - 1):
        (a, b) = fracs[i]
        (c, d) = fracs[i + 1]
        if (a / b) <= (c / d):
            continue
        high_path = frac_to_path[(a, b)]
        low_path = frac_to_path[(c, d)]
        print(f"Refine core between ({a}/{b}) and ({c}/{d})")
        number_of_iterations = max(int(epochs_between * (a / b) ** 4), 1)
        print(f"Number of iterations: {number_of_iterations}")
        refine_pair_core(high_path, low_path, a, b, c, d, X, number_of_iterations, lr, batch_size)


if __name__ == "__main__":
    while True:
        run_refinement()


