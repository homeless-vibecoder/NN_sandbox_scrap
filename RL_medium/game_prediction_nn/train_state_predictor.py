import os
import sys
import math
import random
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

# Import game env from new folder
CURRENT_DIR = os.path.dirname(__file__)
GAME_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../game"))
if GAME_DIR not in sys.path:
    sys.path.insert(0, GAME_DIR)

from qwop_env import QWOPRunnerEnv
from state_predictor import get_or_create_predictor

# Reuse feature builder from RL conventional method
RL_CONV_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../RL_training/conventional_method"))
if RL_CONV_DIR not in sys.path:
    sys.path.insert(0, RL_CONV_DIR)
from train_qwop import build_obs_vec # we only import for copying the dimensions, otherwise, we don't use the NN


def collect_rollout(env: QWOPRunnerEnv,
                    num_steps: int,
                    horizon_steps: int,
                    action_repeat: int = 3,
                    rng: random.Random | None = None) -> Tuple[List[List[float]], List[List[float]]]:
    """
    Collect (state_t, state_t+H) pairs using random actions.
    - state_t is the 15D feature vector used in RL.
    - H = horizon_steps * (env.step dt).
    - We use simple random {-1,0,+1} actions per joint, held for action_repeat frames.
    """
    # If rng not provided, use system randomness (not a fixed seed) for diversity
    if rng is None:
        rng = random.Random()

    env.reset()
    xs: List[List[float]] = []
    ys: List[List[float]] = []

    buffer: List[List[float]] = []
    # Generate a trajectory longer than num_steps + horizon to allow pairing
    total_frames = num_steps + horizon_steps

    # Warm up from a random transient so we don't always start from the same pose
    warmup_steps = rng.randint(0, 300)
    prev_action = {"left_hip": 0, "left_knee": 0, "right_hip": 0, "right_knee": 0}

    def correlated_action(prev: dict) -> dict:
        # Temporal correlation: each joint keeps previous value with high prob, otherwise flips or changes
        action = {}
        for name in ("left_hip", "left_knee", "right_hip", "right_knee"):
            if rng.random() < 0.7:
                action[name] = prev[name]
            else:
                action[name] = rng.choice([-1, 0, +1])
        return action

    # Warmup to diversify the state distribution
    for _ in range(warmup_steps):
        prev_action = correlated_action(prev_action)
        repeat = rng.randint(1, 5)
        for _ in range(repeat):
            _, _, done, _ = env.step(prev_action)
            if done:
                env.reset()
                prev_action = {k: 0 for k in prev_action}
                break

    for t in range(total_frames):
        # Build feature at current time
        obs_vec = build_obs_vec(env)
        buffer.append(obs_vec)

        # Correlated random action with random repeat per frame for richer dynamics
        action = correlated_action(prev_action)
        prev_action = action
        repeat = rng.randint(1, 5)
        done = False
        for _ in range(repeat):
            _, _, done, _ = env.step(action)
            if done:
                env.reset()
                prev_action = {"left_hip": 0, "left_knee": 0, "right_hip": 0, "right_knee": 0}
                break

    # Build supervised pairs (state_t -> state_{t+H})
    for t in range(num_steps):
        xs.append(buffer[t])
        ys.append(buffer[t + horizon_steps])

    return xs, ys


def train():
    # Hardcoded config (no argparse)
    input_dim = 15
    output_dim = 15
    width = 64
    depth = 4
    learning_rate = 3e-4
    batch_size = 128
    epochs = 15000
    horizon_seconds = 0.1

    env = QWOPRunnerEnv()
    dt = 1.0 / env.FPS
    horizon_steps = int(round(horizon_seconds / dt))

    model, weights_path = get_or_create_predictor(weights_path=None, input_dim=input_dim, output_dim=output_dim, width=width, depth=depth)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    # Dataset generation
    # Collect multiple diverse rollouts to form a training set
    xs_all: List[List[float]] = []
    ys_all: List[List[float]] = []
    master_rng = random.Random(12345)
    num_rollouts = 60  # increase for more diversity
    for i in range(num_rollouts):
        rollout_seed = master_rng.randint(0, 2**31 - 1)
        rng = random.Random(rollout_seed)
        xs, ys = collect_rollout(env, num_steps=1000, horizon_steps=horizon_steps, action_repeat=rng.randint(1, 5), rng=rng)
        xs_all.extend(xs)
        ys_all.extend(ys)

    X = torch.tensor(xs_all, dtype=torch.float32)
    Y = torch.tensor(ys_all, dtype=torch.float32)

    # Simple train loop
    num_samples = X.shape[0]
    num_batches = max(1, num_samples // batch_size)
    for epoch in range(1, epochs + 1):
        perm = torch.randperm(num_samples)
        X = X[perm]
        Y = Y[perm]
        epoch_loss = 0.0

        for b in range(num_batches):
            start = b * batch_size
            end = start + batch_size
            xb = X[start:end]
            yb = Y[start:end]

            pred = model(xb)
            loss = loss_fn(pred, yb)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += float(loss.item())

        epoch_loss /= num_batches
        print(f"epoch={epoch} loss={epoch_loss:.6f} samples={num_samples}")

        if epoch % 2 == 0:
            torch.save(model.state_dict(), weights_path)

    torch.save(model.state_dict(), weights_path)
    print("Training complete. Weights saved to:", weights_path)


if __name__ == "__main__":
    train()


