import os
import random
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from ball_plate_env import BallPlateEnv
from nn_policy import BallPlatePolicy, get_or_create_policy


def select_action(policy: BallPlatePolicy, obs: Dict[str, float], stddev: float) -> Tuple[float, torch.Tensor]:
    x = torch.tensor([[obs["y"], obs["v"], obs["p"], obs["p_dot"], obs["t"]]], dtype=torch.float32)
    mean = policy(x)  # shape (1,1)
    dist = torch.distributions.Normal(loc=mean.squeeze(0), scale=torch.tensor([stddev]))
    action = dist.sample()  # shape (1)
    log_prob = dist.log_prob(action).sum()
    return float(action.item()), log_prob


def run_episode(env: BallPlateEnv, policy: BallPlatePolicy, stddev: float, max_steps: int) -> Tuple[List[torch.Tensor], List[float]]:
    log_probs: List[torch.Tensor] = []
    rewards: List[float] = []
    # Randomize initial conditions each episode (world coordinates)
    plate_y = random.uniform(env.plate_min, env.plate_max)
    plate_v = random.uniform(0.0, 1.0)
    ball_y = plate_y + random.uniform(0.3, 1.5)
    ball_v = random.uniform(-1.0, 1.0)
    obs = env.reset(height=ball_y, velocity=ball_v, position=plate_y, velocity_plate=plate_v, contact_time=0.0)

    for _ in range(max_steps):
        action, log_prob = select_action(policy, obs, stddev)
        obs, reward, done, _ = env.step(action)
        log_probs.append(log_prob)
        rewards.append(reward)
        if done:
            break
    return log_probs, rewards


def compute_returns(rewards: List[float], gamma: float) -> torch.Tensor:
    G = 0.0
    returns: List[float] = []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.append(G)
    returns.reverse()
    return torch.tensor(returns, dtype=torch.float32)


def train():
    # Hardcoded config
    total_episodes = 300
    max_steps_per_episode = 800
    gamma = 0.99
    lr = 3e-5
    action_stddev = 0.3
    autosave_every = 50

    # Env and policy
    env = BallPlateEnv()
    policy, weights_path = get_or_create_policy(weights_path=None, hidden_sizes=(64, 64), action_scale=1.0)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    for episode in range(1, total_episodes + 1):
        log_probs, rewards = run_episode(env, policy, action_stddev, max_steps_per_episode)

        if len(rewards) == 0:
            continue

        returns = compute_returns(rewards, gamma)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        loss = []
        for log_prob, Gt in zip(log_probs, returns):
            loss.append(-log_prob * Gt)
        loss = torch.stack(loss).sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if episode % autosave_every == 0:
            # Save to default path; directory handled in get_or_create_policy when first called
            policy.save_weights(weights_path)

        if episode % 10 == 0:
            ep_return = sum(rewards)
            print(f"episode={episode} return={ep_return:.2f} loss={float(loss.item()):.4f}")

    # Final save
    policy.save_weights(weights_path)
    print("Training complete. Weights saved to:", weights_path)


if __name__ == "__main__":
    train()


