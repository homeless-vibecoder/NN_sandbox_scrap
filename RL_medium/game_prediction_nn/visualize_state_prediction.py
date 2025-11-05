import math
import os
import sys
from typing import List, Tuple
import torch

import pygame
from pymunk.vec2d import Vec2d

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
from train_qwop import build_obs_vec
from nn_qwop import get_or_create_policy


def reconstruct_skeleton(points: dict,
                         lengths: dict,
                         torso_center: Vec2d,
                         torso_angle: float,
                         l_hip: float,
                         l_knee: float,
                         r_hip: float,
                         r_knee: float) -> None:
    """
    Fill 'points' with kinematic positions for drawing based on joint angles.
    Uses the same anchor conventions as the env: hip anchors are offset on the
    torso, thigh length aligns with its local +x (hip at +x end), knee at -x end;
    shin hip/knee anchors mirror those used in env's joints.
    """
    L_torso = lengths["torso"]
    L_thigh = lengths["left_thigh"]  # same for right
    L_shin = lengths["left_shin"]     # same for right

    # Torso endpoints
    torso_dir = Vec2d(1, 0).rotated(torso_angle)
    torso_left = torso_center + (-torso_dir) * (L_torso / 2)
    torso_right = torso_center + (torso_dir) * (L_torso / 2)
    points["torso_a"] = torso_left
    points["torso_b"] = torso_right

    # Hip anchors on torso (match env local anchors)
    hip_off = 12
    hip_y = -10
    left_hip_anchor = torso_center + Vec2d(-hip_off, hip_y).rotated(torso_angle)
    right_hip_anchor = torso_center + Vec2d(hip_off, hip_y).rotated(torso_angle)

    # Left leg FK: thigh angle relative to torso
    left_thigh_angle = torso_angle + l_hip
    left_thigh_dir = Vec2d(1, 0).rotated(left_thigh_angle)
    left_knee_pos = left_hip_anchor + (-left_thigh_dir) * L_thigh
    left_shin_angle = left_thigh_angle + l_knee
    left_shin_dir = Vec2d(1, 0).rotated(left_shin_angle)
    left_foot_pos = left_knee_pos + (-left_shin_dir) * L_shin

    # Right leg FK
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


def visualize_prediction() -> None:
    env = QWOPRunnerEnv()
    pygame.init()
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 18)

    # Predictor
    input_dim = 15
    output_dim = 15
    model, _ = get_or_create_predictor(weights_path=None, input_dim=input_dim, output_dim=output_dim, width=48, depth=4)
    model.eval()

    # RL policy (for showing suggested actions given the current features)
    policy, _ = get_or_create_policy(weights_path=None, hidden_sizes=(128, 128), input_dim=input_dim)
    policy.eval()

    # Initialize from real env state, then switch to prediction-only updates
    obs = env.reset()
    start_x = env.start_x
    lengths = env.runner["lengths"]
    thick = env.runner["thickness"]
    head_r = 12

    # Initial feature vector
    obs_vec = build_obs_vec(env)

    # Each model application predicts ~1.0s into the future (per training).
    # Chain multiple applications per render frame to run longer/faster.
    steps_per_frame = 1
    paused = False
    total_predicted_seconds = 0
    show_policy_actions = False
    last_step_time_ms = 0
    last_delta_norm = 0.0

    def to_pygame(p: Vec2d) -> Tuple[int, int]:
        return int(p.x - env.camera_x), int(env.HEIGHT - p.y)

    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    running = False
                elif ev.key == pygame.K_r:
                    obs = env.reset()
                    start_x = env.start_x
                    lengths = env.runner["lengths"]
                    thick = env.runner["thickness"]
                    obs_vec = build_obs_vec(env)
                    total_predicted_seconds = 0
                elif ev.key == pygame.K_a:
                    show_policy_actions = not show_policy_actions
                elif ev.key == pygame.K_SPACE:
                    paused = not paused
                elif ev.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    steps_per_frame = min(60, steps_per_frame + 1)
                elif ev.key in (pygame.K_MINUS, pygame.K_UNDERSCORE, pygame.K_KP_MINUS):
                    steps_per_frame = max(0, steps_per_frame - 1)
                elif ev.key == pygame.K_1:
                    steps_per_frame = 1
                elif ev.key == pygame.K_0:
                    steps_per_frame = 0

        # Advance prediction chain: each application is ~+1s into future
        if not paused and steps_per_frame > 0:
            with torch.no_grad():
                if steps_per_frame == 1:
                    # Real-time: 1 step per ~1000 ms
                    now_ms = pygame.time.get_ticks()
                    if now_ms - last_step_time_ms >= 1000:
                        x0 = obs_vec
                        x = torch.tensor([x0], dtype=torch.float32)
                        y = model(x).detach().numpy()[0].tolist()
                        # Keep orientation features well-formed: normalize (sin, cos)
                        s, c = float(y[4]), float(y[5])
                        n = math.hypot(s, c)
                        if n > 1e-6:
                            y[4] = s / n
                            y[5] = c / n
                        last_delta_norm = math.sqrt(sum((yi - xi) * (yi - xi) for xi, yi in zip(x0, y)))
                        obs_vec = y
                        total_predicted_seconds += 1
                        last_step_time_ms = now_ms
                else:
                    # Fast-forward: N steps per rendered frame
                    for _ in range(steps_per_frame):
                        x0 = obs_vec
                        x = torch.tensor([x0], dtype=torch.float32)
                        y = model(x).detach().numpy()[0].tolist()
                        s, c = float(y[4]), float(y[5])
                        n = math.hypot(s, c)
                        if n > 1e-6:
                            y[4] = s / n
                            y[5] = c / n
                        last_delta_norm = math.sqrt(sum((yi - xi) * (yi - xi) for xi, yi in zip(x0, y)))
                        obs_vec = y
                    total_predicted_seconds += steps_per_frame

        # Decode predicted features
        dx = obs_vec[0] * 200.0
        y_world = obs_vec[1] * 200.0
        vx = obs_vec[2] * 400.0
        vy = obs_vec[3] * 400.0
        sin_t = obs_vec[4]
        cos_t = obs_vec[5]
        torso_angle = math.atan2(sin_t, cos_t)
        ang_vel = obs_vec[6] * 10.0
        l_hip, l_knee, r_hip, r_knee = obs_vec[7], obs_vec[8], obs_vec[9], obs_vec[10]

        # Torso center from start_x and dx
        torso_center = Vec2d(start_x + dx, y_world)
        env.camera_x = torso_center.x - env.WIDTH * 0.3

        # Reconstruct kinematic points
        pts: dict = {}
        reconstruct_skeleton(pts, lengths, torso_center, torso_angle, l_hip, l_knee, r_hip, r_knee)

        # Draw
        screen.fill((240, 240, 240))
        pygame.draw.rect(screen, (180, 180, 180), (0, env.HEIGHT - env.GROUND_Y, env.WIDTH, env.GROUND_Y))
        tick_spacing = 120
        offset = int(env.camera_x) % tick_spacing
        for x in range(-offset, env.WIDTH, tick_spacing):
            pygame.draw.line(screen, (150, 150, 150), (x, env.HEIGHT - env.GROUND_Y), (x, env.HEIGHT - env.GROUND_Y - 15), 2)

        # Torso line
        pygame.draw.line(screen, (20, 20, 120), to_pygame(pts["torso_a"]), to_pygame(pts["torso_b"]), 6)
        # Head
        up = Vec2d(0, 1).rotated(torso_angle)
        head_center = torso_center + up * (lengths["torso"] * 0.6 + head_r)
        pygame.draw.circle(screen, (230, 200, 160), to_pygame(head_center), head_r)
        pygame.draw.circle(screen, (0, 0, 0), to_pygame(head_center), head_r, 2)

        # Legs
        pygame.draw.line(screen, (34, 139, 34), to_pygame(pts["hip_L"]), to_pygame(pts["knee_L"]), 6)
        pygame.draw.line(screen, (255, 140, 0), to_pygame(pts["knee_L"]), to_pygame(pts["foot_L"]), 6)
        pygame.draw.line(screen, (34, 139, 34), to_pygame(pts["hip_R"]), to_pygame(pts["knee_R"]), 6)
        pygame.draw.line(screen, (255, 140, 0), to_pygame(pts["knee_R"]), to_pygame(pts["foot_R"]), 6)

        radius = max(4, int(thick * 0.6))
        for p in (to_pygame(pts["hip_L"]), to_pygame(pts["knee_L"]), to_pygame(pts["foot_L"]),
                  to_pygame(pts["hip_R"]), to_pygame(pts["knee_R"]), to_pygame(pts["foot_R"])):
            pygame.draw.circle(screen, (0, 0, 0), p, radius)

        # Optional: overlay RL policy's suggested actions on joints
        action_overlay_text = ""
        if show_policy_actions:
            a = policy.act_vec_eval(obs_vec)
            # Color map for action {-1,0,+1}
            def color_for(u: int):
                return (214, 40, 40) if u < 0 else ((100, 100, 100) if u == 0 else (40, 120, 220))

            # Overlay thinner, color-coded segments on top of limbs
            pygame.draw.line(screen, color_for(a["left_hip"]), to_pygame(pts["hip_L"]), to_pygame(pts["knee_L"]), 3)
            pygame.draw.line(screen, color_for(a["left_knee"]), to_pygame(pts["knee_L"]), to_pygame(pts["foot_L"]), 3)
            pygame.draw.line(screen, color_for(a["right_hip"]), to_pygame(pts["hip_R"]), to_pygame(pts["knee_R"]), 3)
            pygame.draw.line(screen, color_for(a["right_knee"]), to_pygame(pts["knee_R"]), to_pygame(pts["foot_R"]), 3)

            action_overlay_text = f" | Actions (LHip/LKnee/RHip/RKnee): {a['left_hip']}/{a['left_knee']}/{a['right_hip']}/{a['right_knee']}"

        dist = dx
        spf = steps_per_frame
        if paused or spf == 0:
            mode_text = "Paused"
        elif spf == 1:
            mode_text = "Real-time (1s/step)"
        else:
            mode_text = f"Fast x{spf}/frame"
        hud = font.render(f"Predict-only | Dist: {dist:.0f}  Step: 1s  {mode_text}  t={total_predicted_seconds}s  Δ={last_delta_norm:.4f}  [+/-] speed, SPACE pause, A actions, R reset, ESC quit{action_overlay_text}", True, (10,10,10))
        screen.blit(hud, (10, 10))

        pygame.display.flip()
        # Draw at monitor refresh; logic steps happen at 1 Hz via timer
        clock.tick(60)

    pygame.quit()


def main():
    visualize_prediction()


if __name__ == "__main__":
    main()


