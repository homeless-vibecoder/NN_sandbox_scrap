import os
import sys
from typing import Tuple

import torch


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from game_prediction_nn.state_predictor import get_or_create_predictor


def seed_complexity(p: int = 1, q: int = 2, threshold: float = 1.0 / 20.0) -> None:
    """
    Copy the base predictor weights into complexity-wise partial files for powers of (p/q):
    1_1, p_q, p^2_q^2, ... until (p/q)^k <= threshold.

    Stored under NN_weights/game_prediction/partial_state_prediction_complexity-wise
    with filenames: state_predictor_partial_complexity-wise_{num}_{den}.pt
    """
    # Base predictor
    _, base_path = get_or_create_predictor(weights_path=None, input_dim=15, output_dim=15, width=20, depth=3)

    out_dir = os.path.join(BASE_DIR, "NN_weights", "game_prediction", "partial_state_prediction_complexity-wise")
    os.makedirs(out_dir, exist_ok=True)

    def copy_from(src_path: str, num: int, den: int) -> str:
        dst_path = os.path.join(out_dir, f"state_predictor_partial_complexity-wise_{num}_{den}.pt")
        torch.save(torch.load(src_path, map_location="cpu"), dst_path)
        return dst_path

    # Start with 1_1 — if exists, don't overwrite; set it as the current source
    num, den = 1, 1
    path_1_1 = os.path.join(out_dir, f"state_predictor_partial_complexity-wise_{num}_{den}.pt")
    if os.path.exists(path_1_1):
        current_src = path_1_1
        print("Exists, skipping:", path_1_1)
    else:
        current_src = copy_from(base_path, num, den)
        print("Seeded:", current_src)

    # Powers of p/q — at each step, copy content from the previous level
    value = 1.0
    while value > threshold:
        num *= p
        den *= q
        value *= (p / q)
        dst_path = os.path.join(out_dir, f"state_predictor_partial_complexity-wise_{num}_{den}.pt")
        if os.path.exists(dst_path):
            print("Exists, skipping:", dst_path)
            current_src = dst_path
        else:
            current_src = copy_from(current_src, num, den)
            print("Seeded:", current_src)


if __name__ == "__main__":
    seed_complexity()


