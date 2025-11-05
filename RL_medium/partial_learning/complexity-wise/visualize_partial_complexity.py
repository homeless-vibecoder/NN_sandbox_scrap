import math
import os
import sys
from typing import List, Tuple, Dict
import torch

import pygame
from pymunk.vec2d import Vec2d

# Project roots for imports
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

GAME_DIR = os.path.join(BASE_DIR, "game")
RL_CONV_DIR = os.path.join(BASE_DIR, "RL_training", "conventional_method")
for d in (GAME_DIR, RL_CONV_DIR):
    if d not in sys.path:
        sys.path.insert(0, d)

from qwop_env import QWOPRunnerEnv
from train_qwop import build_obs_vec
from game_prediction_nn.state_predictor import FixedWidthMLP


def _infer_arch_from_state(state: dict) -> Tuple[int, int, int]:
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
    input_dim = int(state[f"net.{first_idx}.weight"].shape[1])
    width = int(state[f"net.{first_idx}.weight"].shape[0])
    depth = len(set(linear_indices))
    return input_dim, width, depth


def _list_complexity_models() -> Dict[Tuple[int, int], str]:
    weights_dir = os.path.join(BASE_DIR, "NN_weights", "game_prediction", "partial_state_prediction_complexity-wise")
    out: Dict[Tuple[int, int], str] = {}
    if os.path.isdir(weights_dir):
        for fname in os.listdir(weights_dir):
            if not fname.startswith("state_predictor_partial_complexity-wise_") or not fname.endswith(".pt"):
                continue
            stem = fname[len("state_predictor_partial_complexity-wise_"):-3]
            parts = stem.split("_")
            if len(parts) != 2:
                continue
            try:
                p = int(parts[0]); q = int(parts[1])
            except ValueError:
                continue
            out[(p, q)] = os.path.join(weights_dir, fname)
    return out


def _pick_closest_fraction(target: float) -> Tuple[int, int, str]:
    models = _list_complexity_models()
    if not models:
        raise FileNotFoundError("No complexity-wise partial predictor files found.")
    best = None
    best_diff = float("inf")
    for (p, q), path in models.items():
        val = p / q
        diff = abs(val - target)
        if diff < best_diff:
            best_diff = diff
            best = (p, q, path)
    assert best is not None
    return best


def load_partial_core_by_target(target_fraction: float) -> Tuple[FixedWidthMLP, int, int]:
    p, q, path = _pick_closest_fraction(target_fraction)
    state = torch.load(path, map_location="cpu")
    input_dim, width, depth = _infer_arch_from_state(state)
    model = FixedWidthMLP(input_dim=input_dim, output_dim=input_dim, width=width, depth=depth)
    model.load_state_dict(state)
    model.eval()
    return model, p, q


def visualize_partial_complexity():
    """
    Visualize using the stitched network: OutAdapter( Core^1( InAdapter(x) ) )
    where Core is taken from a complexity-wise partial model at closest p/q to TARGET_FRACTION.
    """
    TARGET_FRACTION = 1.0
    core_model, p, q = load_partial_core_by_target(TARGET_FRACTION)

    # Env and UI
    env = QWOPRunnerEnv()
    pygame.init()
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 18)

    # Initialize from real env state; then step with predictor only
    env.reset()
    start_x = env.start_x
    lengths = env.runner["lengths"]
    thick = env.runner["thickness"]
    head_r = 12
    obs_vec = build_obs_vec(env)

    def to_pygame(pnt: Vec2d) -> Tuple[int, int]:
        return int(pnt.x - env.camera_x), int(env.HEIGHT - pnt.y)

    def forward_in_adapter(model: FixedWidthMLP, x):
        h = model.net[0](x)
        h = model.net[1](h)
        return h

    def forward_core(model: FixedWidthMLP, h):
        y = h
        for layer in list(model.net)[2:-1]:
            y = layer(y)
        return y

    def forward_out_adapter(model: FixedWidthMLP, h):
        return model.net[-1](h)

    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    running = False
                elif ev.key == pygame.K_r:
                    env.reset()
                    start_x = env.start_x
                    lengths = env.runner["lengths"]
                    thick = env.runner["thickness"]
                    obs_vec = build_obs_vec(env)

        with torch.no_grad():
            x = torch.tensor([obs_vec], dtype=torch.float32)
            h0 = forward_in_adapter(core_model, x)
            h1 = forward_core(core_model, h0)
            y = forward_out_adapter(core_model, h1).detach().numpy()[0].tolist()
        obs_vec = y

        # Decode predicted features
        dx = obs_vec[0] * 200.0
        y_world = obs_vec[1] * 200.0
        sin_t = obs_vec[4]
        cos_t = obs_vec[5]
        torso_angle = math.atan2(sin_t, cos_t)
        l_hip, l_knee, r_hip, r_knee = obs_vec[7], obs_vec[8], obs_vec[9], obs_vec[10]

        torso_center = Vec2d(start_x + dx, y_world)
        env.camera_x = torso_center.x - env.WIDTH * 0.3

        pts: dict = {}
        reconstruct_skeleton(pts, lengths, torso_center, torso_angle, l_hip, l_knee, r_hip, r_knee)

        # Draw
        screen.fill((240, 240, 240))
        pygame.draw.rect(screen, (180, 180, 180), (0, env.HEIGHT - env.GROUND_Y, env.WIDTH, env.GROUND_Y))
        tick_spacing = 120
        offset = int(env.camera_x) % tick_spacing
        for x in range(-offset, env.WIDTH, tick_spacing):
            pygame.draw.line(screen, (150, 150, 150), (x, env.HEIGHT - env.GROUND_Y), (x, env.HEIGHT - env.GROUND_Y - 15), 2)

        pygame.draw.line(screen, (20, 20, 120), to_pygame(pts["torso_a"]), to_pygame(pts["torso_b"]), 6)
        up = Vec2d(0, 1).rotated(torso_angle)
        head_center = torso_center + up * (lengths["torso"] * 0.6 + head_r)
        pygame.draw.circle(screen, (230, 200, 160), to_pygame(head_center), head_r)
        pygame.draw.circle(screen, (0, 0, 0), to_pygame(head_center), head_r, 2)

        pygame.draw.line(screen, (34, 139, 34), to_pygame(pts["hip_L"]), to_pygame(pts["knee_L"]), 6)
        pygame.draw.line(screen, (255, 140, 0), to_pygame(pts["knee_L"]), to_pygame(pts["foot_L"]), 6)
        pygame.draw.line(screen, (34, 139, 34), to_pygame(pts["hip_R"]), to_pygame(pts["knee_R"]), 6)
        pygame.draw.line(screen, (255, 140, 0), to_pygame(pts["knee_R"]), to_pygame(pts["foot_R"]), 6)

        radius = max(4, int(thick * 0.6))
        for p2 in (to_pygame(pts["hip_L"]), to_pygame(pts["knee_L"]), to_pygame(pts["foot_L"]),
                   to_pygame(pts["hip_R"]), to_pygame(pts["knee_R"]), to_pygame(pts["foot_R"])):
            pygame.draw.circle(screen, (0, 0, 0), p2, radius)

        dist = dx
        hud = font.render(f"Complexity partial (p/q={p}/{q}) | ESC quit, R reset", True, (10,10,10))
        screen.blit(hud, (10, 10))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


def reconstruct_skeleton(points: dict,
                         lengths: dict,
                         torso_center: Vec2d,
                         torso_angle: float,
                         l_hip: float,
                         l_knee: float,
                         r_hip: float,
                         r_knee: float) -> None:
    L_torso = lengths["torso"]
    L_thigh = lengths["left_thigh"]
    L_shin = lengths["left_shin"]

    torso_dir = Vec2d(1, 0).rotated(torso_angle)
    torso_left = torso_center + (-torso_dir) * (L_torso / 2)
    torso_right = torso_center + (torso_dir) * (L_torso / 2)
    points["torso_a"] = torso_left
    points["torso_b"] = torso_right

    hip_off = 12
    hip_y = -10
    left_hip_anchor = torso_center + Vec2d(-hip_off, hip_y).rotated(torso_angle)
    right_hip_anchor = torso_center + Vec2d(hip_off, hip_y).rotated(torso_angle)

    left_thigh_angle = torso_angle + l_hip
    left_thigh_dir = Vec2d(1, 0).rotated(left_thigh_angle)
    left_knee_pos = left_hip_anchor + (-left_thigh_dir) * L_thigh
    left_shin_angle = left_thigh_angle + l_knee
    left_shin_dir = Vec2d(1, 0).rotated(left_shin_angle)
    left_foot_pos = left_knee_pos + (-left_shin_dir) * L_shin

    right_thigh_angle = torso_angle + r_hip
    right_thigh_dir = Vec2d(1, 0).rotated(right_thigh_angle)
    right_knee_pos = right_hip_anchor + (-right_thigh_dir) * L_thigh
    right_shin_angle = right_thigh_angle + r_knee
    right_shin_dir = Vec2d(1, 0).rotated(right_shin_angle)
    right_foot_pos = right_knee_pos + (-right_shin_dir) * L_shin

    points["hip_L"] = left_hip_anchor
    points["knee_L"] = left_knee_pos
    points["foot_L"] = left_foot_pos
    points["hip_R"] = right_hip_anchor
    points["knee_R"] = right_knee_pos
    points["foot_R"] = right_foot_pos


def main():
    visualize_partial_complexity()


if __name__ == "__main__":
    main()


