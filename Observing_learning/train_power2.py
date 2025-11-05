import os
import shutil
from typing import List, Tuple

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from nn import NN


# ---------------------- Configuration (hardcoded) ----------------------
BASE_WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "NN_weights")

# Training hyperparameters
STEPS_PER_EPOCH = 2
EPOCHS = 120
BATCH_SIZE = 256
LR = 8e-3
INPUT_SCALE = 1.0

# Tracing
NUM_TRACED_PARAMS = 32


def _parse_shape_from_dirname(dirname: str) -> Tuple[int, List[int], int] | None:
    """Parse a folder name like '4_8_8_8_4' into (input_dim, hidden_layers, output_dim)."""
    try:
        parts = [int(tok) for tok in dirname.split("_") if tok.strip()]
        if len(parts) < 2:
            return None
        input_dim = parts[0]
        output_dim = parts[-1]
        hidden_layers = parts[1:-1]
        return input_dim, hidden_layers, output_dim
    except Exception:
        return None


def _compose_twice(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return model(model(x))


def _snapshot(model: NN, tracer) -> None:
    model.snapshot_params(tracer)


def _train_power2_in_shape_dir(shape_dir: str) -> None:
    # Require 1.pt as base
    f_path = os.path.join(shape_dir, "1.pt")
    if not os.path.exists(f_path):
        print(f"skip (no 1.pt): {shape_dir}")
        return

    # Reconstruct architecture from folder name
    dirname = os.path.basename(shape_dir)
    parsed = _parse_shape_from_dirname(dirname)
    if parsed is None:
        print(f"skip (unable to parse shape): {shape_dir}")
        return
    input_dim, hidden_layers, output_dim = parsed

    # Build F and load 1.pt
    F = NN(input_dim=input_dim, output_dim=output_dim, hidden_layers=hidden_layers, activation="tanh")
    F.load_weights(f_path, map_location="cpu", strict=True)
    F.eval()
    for p in F.parameters():
        p.requires_grad_(False)

    # Build G with same architecture
    G = NN(input_dim=input_dim, output_dim=output_dim, hidden_layers=hidden_layers, activation="tanh")
    g_existing_path = os.path.join(shape_dir, "2.pt")
    if os.path.exists(g_existing_path):
        try:
            # Continue training from existing 2.pt
            G.load_weights(g_existing_path, map_location="cpu", strict=True)
            print("loaded existing 2.pt:", g_existing_path)
        except Exception:
            # If loading fails (e.g., mismatch), fall back to initializing from F
            G.load_state_dict(F.state_dict(), strict=True)
            print("failed to load existing 2.pt; initialized from 1.pt")
    else:
        # No 2.pt yet: initialize from F (1.pt)
        G.load_state_dict(F.state_dict(), strict=True)
    G.train()

    optimizer = torch.optim.Adam(G.parameters(), lr=LR)
    # Alternative optimizer (SGD): uncomment to use instead of Adam
    optimizer = torch.optim.SGD(G.parameters(), lr=LR, momentum=0.0)
    loss_fn = nn.MSELoss()

    # Setup tracing
    tracer = G.select_param_slices(k=NUM_TRACED_PARAMS, seed=0)
    _snapshot(G, tracer)  # initial

    # Training loop
    for epoch in range(1, EPOCHS + 1):
        epoch_loss = 0.0
        for _ in range(STEPS_PER_EPOCH):
            x = torch.randn(BATCH_SIZE, input_dim) * INPUT_SCALE
            with torch.no_grad():
                target = _compose_twice(F, x)
            pred = G(x)
            loss = loss_fn(pred, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += float(loss.item())
        _snapshot(G, tracer)
        print({"shape": dirname, "epoch": epoch, "avg_loss": epoch_loss / max(1, STEPS_PER_EPOCH)})

    # Save 2.pt (backup existing if present)
    g_path = os.path.join(shape_dir, "2.pt")
    if os.path.exists(g_path):
        bak = g_path + ".bak"
        try:
            shutil.copyfile(g_path, bak)
            print("backed up existing 2.pt to:", bak)
        except Exception:
            pass
    G.eval()
    torch.save(G.state_dict(), g_path)
    print("saved:", g_path)

    # Plot traces
    try:
        plt.figure(figsize=(9, 5))
        for series in tracer.traces:
            plt.plot(series, alpha=0.85)
        plt.xlabel("Snapshot (epoch index)")
        plt.ylabel("Weight value")
        plt.title(f"Training traces for 2.pt in {dirname}")
        plt.tight_layout()
        out_png = os.path.join(shape_dir, "weights_trace_2.png")
        plt.savefig(out_png, dpi=140)
        plt.close()
        print("saved plot:", out_png)
    except Exception as e:
        print("plot failed:", e)


def main() -> None:
    if not os.path.isdir(BASE_WEIGHTS_DIR):
        print("No NN_weights directory found:", BASE_WEIGHTS_DIR)
        return

    # Iterate all shape directories
    entries = sorted(os.listdir(BASE_WEIGHTS_DIR))
    shape_dirs = [os.path.join(BASE_WEIGHTS_DIR, d) for d in entries if os.path.isdir(os.path.join(BASE_WEIGHTS_DIR, d))]

    if not shape_dirs:
        print("No shape folders inside:", BASE_WEIGHTS_DIR)
        return

    for sd in shape_dirs:
        _train_power2_in_shape_dir(sd)


if __name__ == "__main__":
    torch.manual_seed(0)
    main()


