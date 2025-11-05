import os
from typing import Set

import torch

# Local import without argparse/config machinery (hardcoded per project preference)
from nn import NN


# ---------------------- Configuration (edit as needed) ----------------------
# Small but deep architecture: same input/output dimension, many narrow layers
INPUT_DIM = 4
OUTPUT_DIM = 4
WIDTH = 32
DEPTH = 9  # number of layers including the output linear; hidden layers = DEPTH - 1
ACTIVATION = "tanh"
USE_LAYERNORM = False
DROPOUT = 0.0

# Base directory for saving weights under Observing_learning
BASE_WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "NN_weights")


def _shape_string(input_dim: int, width: int, depth: int, output_dim: int) -> str:
    if depth < 1:
        raise ValueError("DEPTH must be >= 1")
    hidden_count = max(0, depth - 1)
    parts = [input_dim] + [width] * hidden_count + [output_dim]
    return "_".join(str(n) for n in parts)


def _next_available_numeric_name(dir_path: str) -> str:
    os.makedirs(dir_path, exist_ok=True)
    used: Set[int] = set()
    for fname in os.listdir(dir_path):
        base, ext = os.path.splitext(fname)
        if ext == ".pt" and base.isdigit():
            try:
                used.add(int(base))
            except Exception:
                pass
    n = 1
    while n in used:
        n += 1
    return f"{n}.pt"


def main() -> None:
    # Build the model
    model = NN(
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
        width=WIDTH,
        depth=DEPTH,
        activation=ACTIVATION,
        use_layernorm=USE_LAYERNORM,
        dropout=DROPOUT,
    )

    # Compose folder name by architecture (e.g., 8_32_32_..._8)
    shape = _shape_string(INPUT_DIM, WIDTH, DEPTH, OUTPUT_DIM)
    shape_dir = os.path.join(BASE_WEIGHTS_DIR, shape)
    os.makedirs(shape_dir, exist_ok=True)

    # Pick the smallest available numeric filename
    filename = _next_available_numeric_name(shape_dir)
    out_path = os.path.join(shape_dir, filename)

    # Save weights
    model.save_weights(out_path)

    # Brief confirmation
    print({
        "shape": shape,
        "saved_as": out_path,
        "num_params": model.num_parameters(),
        "num_trainable": model.num_parameters(trainable_only=True),
    })


if __name__ == "__main__":
    # Optional: deterministic init if desired
    torch.manual_seed(0)
    main()


