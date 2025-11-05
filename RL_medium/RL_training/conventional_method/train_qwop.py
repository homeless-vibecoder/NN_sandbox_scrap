import os
import sys
import math
import random
from typing import Dict, List, Tuple

import torch
import torch.optim as optim

# Ensure we can import the game environment from the new folder structure
CURRENT_DIR = os.path.dirname(__file__)
GAME_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../../game"))
if GAME_DIR not in sys.path:
    sys.path.insert(0, GAME_DIR)

from qwop_env import QWOPRunnerEnv
from nn_qwop import CategoricalPolicy, get_or_create_policy


# Policy now imported from nn_qwop.py


def compute_returns(rewards: List[float], gamma: float) -> torch.Tensor:
    G = 0.0
    out: List[float] = []
    for r in reversed(rewards):
        G = r + gamma * G
        out.append(G)
    out.reverse()
    return torch.tensor(out, dtype=torch.float32)


def _joint_angles(env: QWOPRunnerEnv) -> Tuple[float, float, float, float]:
    r = env.runner
    torso = r["torso"]
    lt = r["left_thigh"]
    ls = r["left_shin"]
    rt = r["right_thigh"]
    rs = r["right_shin"]
    l_hip = float(lt.angle - torso.angle)
    l_knee = float(ls.angle - lt.angle)
    r_hip = float(rt.angle - torso.angle)
    r_knee = float(rs.angle - rt.angle)
    return l_hip, l_knee, r_hip, r_knee


def _joint_ang_vels(env: QWOPRunnerEnv) -> Tuple[float, float, float, float]:
    r = env.runner
    torso = r["torso"]
    lt = r["left_thigh"]
    ls = r["left_shin"]
    rt = r["right_thigh"]
    rs = r["right_shin"]
    l_hip_v = float(lt.angular_velocity - torso.angular_velocity)
    l_knee_v = float(ls.angular_velocity - lt.angular_velocity)
    r_hip_v = float(rt.angular_velocity - torso.angular_velocity)
    r_knee_v = float(rs.angular_velocity - rt.angular_velocity)
    return l_hip_v, l_knee_v, r_hip_v, r_knee_v


def build_obs_vec(env: QWOPRunnerEnv) -> List[float]:
    o = env.get_obs()
    dx = float(o["x"] - env.start_x)
    y = float(o["y"]) / 200.0
    vx = float(o["vx"]) / 400.0
    vy = float(o["vy"]) / 400.0
    ang = float(o["angle"])
    sin_t = math.sin(ang)
    cos_t = math.cos(ang)
    ang_vel = float(o["ang_vel"]) / 10.0
    l_hip, l_knee, r_hip, r_knee = _joint_angles(env)
    l_hip_v, l_knee_v, r_hip_v, r_knee_v = _joint_ang_vels(env)
    # Normalize joint velocities
    l_hip_v /= 10.0
    l_knee_v /= 10.0
    r_hip_v /= 10.0
    r_knee_v /= 10.0
    # No contact bits for now (could be added later)
    return [dx / 200.0, y, vx, vy, sin_t, cos_t, ang_vel,
            l_hip, l_knee, r_hip, r_knee,
            l_hip_v, l_knee_v, r_hip_v, r_knee_v]


def run_episode(env: QWOPRunnerEnv, policy: CategoricalPolicy, max_steps: int,
                action_repeat: int = 3) -> Tuple[List[torch.Tensor], List[float]]:
    logps: List[torch.Tensor] = []
    rewards: List[float] = []

    env.reset()
    for _ in range(max_steps):
        obs_vec = build_obs_vec(env)
        action, logp = policy.act_vec(obs_vec)
        # Apply action repeat to reduce control frequency
        total_reward = 0.0
        done = False
        for _rep in range(action_repeat):
            _, reward, done, _ = env.step(action)
            total_reward += float(reward)
            if done:
                break
        logps.append(logp)
        rewards.append(total_reward)
        if done:
            break
    return logps, rewards


def train():
    # Hardcoded config (no argparse per user preference)
    total_episodes = 15000
    max_steps_per_episode = 1500
    gamma = 0.995
    lr = 3e-4
    autosave_every = 50

    # Environment and policy
    env = QWOPRunnerEnv()
    policy, weights_path = get_or_create_policy(weights_path=None, hidden_sizes=(128, 128), input_dim=15)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    # weights_path already resolved by get_or_create_policy

    for episode in range(1, total_episodes + 1):
        logps, rewards = run_episode(env, policy, max_steps_per_episode, action_repeat=3)
        if len(rewards) == 0:
            continue

        returns = compute_returns(rewards, gamma)
        # Normalize for stability
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        loss_terms: List[torch.Tensor] = []
        for logp, Gt in zip(logps, returns):
            loss_terms.append(-logp * Gt)
        loss = torch.stack(loss_terms).sum()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer.step()

        if episode % 10 == 0:
            ep_return = sum(rewards)
            print(f"episode={episode} return={ep_return:.2f} loss={float(loss.item()):.4f}")

        if episode % autosave_every == 0:
            torch.save(policy.state_dict(), weights_path)

    # Final save
    torch.save(policy.state_dict(), weights_path)
    print("Training complete. Weights saved to:", weights_path)


if __name__ == "__main__":
    train()


