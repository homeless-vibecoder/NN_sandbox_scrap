import math
import os
import sys
from typing import List, Tuple, Dict

import pygame
from pymunk.vec2d import Vec2d

# Ensure we can import the game environment from the new folder structure
CURRENT_DIR = os.path.dirname(__file__)
GAME_DIR = os.path.abspath(os.path.join(CURRENT_DIR, "../../game"))
if GAME_DIR not in sys.path:
    sys.path.insert(0, GAME_DIR)

from qwop_env import QWOPRunnerEnv
from nn_qwop import get_or_create_policy, CategoricalPolicy


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
    l_hip_v /= 10.0
    l_knee_v /= 10.0
    r_hip_v /= 10.0
    r_knee_v /= 10.0
    return [dx / 200.0, y, vx, vy, sin_t, cos_t, ang_vel,
            l_hip, l_knee, r_hip, r_knee,
            l_hip_v, l_knee_v, r_hip_v, r_knee_v]


def visualize_agent(policy: CategoricalPolicy) -> None:
    env = QWOPRunnerEnv()
    pygame.init()
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 18)

    env.reset()
    running = True

    def to_pygame(p: Vec2d) -> Tuple[int, int]:
        return int(p.x - env.camera_x), int(env.HEIGHT - p.y)

    def world(body, local):
        return body.local_to_world(local)

    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    running = False
                elif ev.key == pygame.K_r:
                    env.reset()

        # Agent action
        obs_vec = build_obs_vec(env)
        action = policy.act_vec_eval(obs_vec)
        _, reward, done, info = env.step(action)

        # Draw
        screen.fill((240, 240, 240))
        # ground and marks
        pygame.draw.rect(screen, (180, 180, 180), (0, env.HEIGHT - env.GROUND_Y, env.WIDTH, env.GROUND_Y))
        tick_spacing = 120
        offset = int(env.camera_x) % tick_spacing
        for x in range(-offset, env.WIDTH, tick_spacing):
            pygame.draw.line(screen, (150, 150, 150), (x, env.HEIGHT - env.GROUND_Y), (x, env.HEIGHT - env.GROUND_Y - 15), 2)

        # Skeleton rendering (copied from env.visualize)
        J = env.runner["joints"]
        L = env.runner["lengths"]
        thick = env.runner["thickness"]

        torso = env.runner["torso"]
        t_a = world(torso, Vec2d(-L["torso"]/2, 0))
        t_b = world(torso, Vec2d(L["torso"]/2, 0))
        pygame.draw.line(screen, (20, 20, 120), to_pygame(t_a), to_pygame(t_b), 6)
        torso_center = torso.position
        up = Vec2d(0, 1).rotated(torso.angle)
        head_r = 12
        head_center = torso_center + up * (L["torso"] * 0.6 + head_r)
        pygame.draw.circle(screen, (230, 200, 160), to_pygame(head_center), head_r)
        pygame.draw.circle(screen, (0, 0, 0), to_pygame(head_center), head_r, 2)

        lt = env.runner["left_thigh"]
        ls = env.runner["left_shin"]
        hip_L = world(torso, J["left_hip"].anchor_a)
        knee_L = world(lt, J["left_knee"].anchor_a)
        foot_L = world(ls, (-L["left_shin"] / 2, 0))
        pygame.draw.line(screen, (34, 139, 34), to_pygame(hip_L), to_pygame(knee_L), 6)
        pygame.draw.line(screen, (255, 140, 0), to_pygame(knee_L), to_pygame(foot_L), 6)

        rt = env.runner["right_thigh"]
        rs = env.runner["right_shin"]
        hip_R = world(torso, J["right_hip"].anchor_a)
        knee_R = world(rt, J["right_knee"].anchor_a)
        foot_R = world(rs, (-L["right_shin"] / 2, 0))
        pygame.draw.line(screen, (34, 139, 34), to_pygame(hip_R), to_pygame(knee_R), 6)
        pygame.draw.line(screen, (255, 140, 0), to_pygame(knee_R), to_pygame(foot_R), 6)

        radius = max(4, int(thick * 0.6))
        for p in (to_pygame(hip_L), to_pygame(knee_L), to_pygame(foot_L), to_pygame(hip_R), to_pygame(knee_R), to_pygame(foot_R)):
            pygame.draw.circle(screen, (0, 0, 0), p, radius)

        hud = font.render(f"Dist: {info['distance']:.0f}   Score: {info['distance']:.0f}   Steps: {env.steps}/{env.max_steps}   ESC quit, R restart", True, (10,10,10))
        screen.blit(hud, (10, 10))

        pygame.display.flip()
        clock.tick(env.FPS)

        if done:
            msg = font.render(f"Episode over. Score (distance): {info['distance']:.1f}  Press R to restart, ESC to quit.", True, (0,0,0))
            screen.blit(msg, (40, 50))
            pygame.display.flip()
            waiting = True
            while waiting:
                for ev in pygame.event.get():
                    if ev.type == pygame.QUIT:
                        waiting = False
                        running = False
                    elif ev.type == pygame.KEYDOWN:
                        if ev.key == pygame.K_r:
                            env.reset()
                            waiting = False
                        elif ev.key == pygame.K_ESCAPE:
                            waiting = False
                            running = False
                clock.tick(30)

    pygame.quit()


def main():
    policy, _ = get_or_create_policy(weights_path=None, hidden_sizes=(128, 128), input_dim=15)
    policy.eval()
    visualize_agent(policy)


if __name__ == "__main__":
    main()


