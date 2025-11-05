import os
import sys
import shutil
from typing import Dict, Tuple


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


def _list_timewise_models() -> Dict[Tuple[int, int], str]:
    weights_dir = os.path.join(BASE_DIR, "NN_weights", "game_prediction", "partial_state_prediction_time-wise")
    os.makedirs(weights_dir, exist_ok=True)
    out: Dict[Tuple[int, int], str] = {}
    for fname in os.listdir(weights_dir):
        if not fname.startswith("state_predictor_partial_time-wise_") or not fname.endswith(".pt"):
            continue
        stem = fname[len("state_predictor_partial_time-wise_"):-3]
        parts = stem.split("_")
        if len(parts) != 2:
            continue
        try:
            p = int(parts[0]); q = int(parts[1])
        except ValueError:
            continue
        out[(p, q)] = os.path.join(weights_dir, fname)
    return out


def insert_one(p: int, q: int) -> None:
    """
    Create state_predictor_partial_time-wise_{p}_{q}.pt by copying from the closest
    higher (p',q') available. If none exists, copy from base state_predictor.pt.
    If the target already exists, do nothing.
    """
    weights_dir = os.path.join(BASE_DIR, "NN_weights", "game_prediction", "partial_state_prediction_time-wise")
    os.makedirs(weights_dir, exist_ok=True)

    dst_path = os.path.join(weights_dir, f"state_predictor_partial_time-wise_{p}_{q}.pt")
    if os.path.exists(dst_path):
        print("Exists, skipping:", dst_path)
        return

    # Find source: closest available fraction to target p/q; break ties by picking the larger value
    target_val = p / q
    models = _list_timewise_models()
    best_path = None
    best_val = None
    best_diff = None
    for (pp, qq), path in models.items():
        val = pp / qq
        diff = abs(val - target_val)
        if best_diff is None or diff < best_diff or (abs(diff - best_diff) < 1e-12 and (best_val is None or val > best_val)):
            best_diff = diff
            best_val = val
            best_path = path

    if best_path is None:
        # Fallback to base predictor
        base_path = os.path.join(BASE_DIR, "NN_weights", "game_prediction", "state_predictor.pt")
        if not os.path.exists(base_path):
            raise FileNotFoundError("Base predictor not found: " + base_path)
        src_path = base_path
    else:
        src_path = best_path

    shutil.copyfile(src_path, dst_path)
    print("Created:", dst_path, "from", src_path)


if __name__ == "__main__":
    # Set desired p, q here (no argparse per project preference)
    TARGET_P = 18
    TARGET_Q = 20
    insert_one(TARGET_P, TARGET_Q)


