import math
from typing import Dict, Tuple, Optional

import pygame
import pymunk
from pymunk.vec2d import Vec2d


class QWOPRunnerEnv:
    """
    QWOP-like 2D runner built with Pymunk physics and Pygame rendering.

    - Controls: Q/W control left hip/knee, O/P control right hip/knee.
    - RL API: reset() -> obs, step(action_dict) -> (obs, reward, done, info)
      where action_dict is optional (if None, uses keyboard input when visualize=True).
    - Observation: dict with torso pose/vel, joint angles, and linear velocity.
    - Reward: forward torso x-velocity; penalties on tilt; episode ends on fall.
    """

    def __init__(self):
        # Hardcoded config (no argparse per user preference)
        self.WIDTH, self.HEIGHT = 1000, 600
        self.FPS = 60
        self.SCALE = 30.0  # px per meter for drawing (kept for future use)
        self.GROUND_Y = 100
        self.GRAVITY = 10.0#900.0

        self.MOTOR_RATE = 6.0
        self.MOTOR_FORCE = 2e5

        self.MAX_TILT = math.radians(80)

        # Physics world
        self.space = None
        self.runner = None
        self.camera_x = 0.0
        self.distance = 0.0
        self.start_x = 0.0
        self.max_steps = 3000  # hardcoded episode limit
        self.steps = 0

    # --- Build world ---
    def _add_ground(self):
        body = pymunk.Body(body_type=pymunk.Body.STATIC)
        seg = pymunk.Segment(body, (-10000, self.GROUND_Y), (10000, self.GROUND_Y), 1.0)
        seg.friction = 1.2
        self.space.add(body, seg)

    def _make_box(self, pos: Tuple[float, float], length: float, thickness: float, mass: float):
        inertia = self._moment_for_box_compat(mass, (length, thickness))
        body = pymunk.Body(mass, inertia)
        body.position = Vec2d(*pos)
        shape = pymunk.Poly.create_box(body, (length, thickness))
        shape.friction = 1.0
        # Prevent self-collision among runner parts by grouping them
        shape.filter = pymunk.ShapeFilter(group=1)
        self.space.add(body, shape)
        return body

    def _moment_for_box_compat(self, mass: float, size: Tuple[float, float]) -> float:
        """Compatibility wrapper for pymunk.moment_for_box across versions.
        Older Pymunk expects (mass, w, h); newer accepts (mass, (w, h)).
        """
        try:
            return pymunk.moment_for_box(mass, size)
        except TypeError:
            w, h = size
            return pymunk.moment_for_box(mass, w, h)

    def _set_motor_max_force(self, motor: pymunk.SimpleMotor, force: float) -> None:
        """Set motor max force handling older Pymunk attribute naming."""
        try:
            motor.max_force = force
        except AttributeError:
            # Older versions used camelCase
            motor.maxForce = force

    def _build_runner(self, x=300, y=None) -> Dict:
        if y is None:
            y = self.GROUND_Y + 150

        torso_len = 40
        thigh_len = 40
        shin_len = 40
        thick = 10

        torso = self._make_box((x, y + 40), torso_len, int(thick * 1.2), mass=3.0)

        lt = self._make_box((x - 12, y - thigh_len/2), thigh_len, thick, mass=1.0)
        ls = self._make_box((x - 12, y - thigh_len - shin_len/2 - 5), shin_len, thick, mass=1.0)

        rt = self._make_box((x + 12, y - thigh_len/2), thigh_len, thick, mass=1.0)
        rs = self._make_box((x + 12, y - thigh_len - shin_len/2 - 5), shin_len, thick, mass=1.0)

        # Joints
        hip_off = 12
        hip_y = -10
        left_hip_torso_local = Vec2d(-hip_off, hip_y)
        # Align initial limb poses so anchors coincide to avoid startup tilt
        torso.angle = 0.0
        # Left side placement (vertical thigh and shin)
        left_hip_world = torso.local_to_world(left_hip_torso_local)
        lt.angle = math.pi / 2
        lt.position = left_hip_world - Vec2d(thigh_len / 2, 0).rotated(lt.angle)
        knee_L_world = lt.local_to_world((-thigh_len / 2, 0))
        ls.angle = math.pi / 2
        ls.position = knee_L_world - Vec2d(shin_len / 2, 0).rotated(ls.angle)
        left_hip = pymunk.PivotJoint(torso, lt, left_hip_torso_local, (thigh_len/2, 0))
        left_hip_lim = pymunk.RotaryLimitJoint(torso, lt, math.radians(30), math.radians(150))
        left_hip_m = pymunk.SimpleMotor(torso, lt, 0.0)
        self._set_motor_max_force(left_hip_m, self.MOTOR_FORCE)
        self.space.add(left_hip, left_hip_lim, left_hip_m)

        left_knee = pymunk.PivotJoint(lt, ls, (-thigh_len/2, 0), (shin_len/2, 0))
        left_knee_lim = pymunk.RotaryLimitJoint(lt, ls, math.radians(-120), math.radians(30))
        left_knee_m = pymunk.SimpleMotor(lt, ls, 0.0)
        self._set_motor_max_force(left_knee_m, self.MOTOR_FORCE)
        self.space.add(left_knee, left_knee_lim, left_knee_m)

        right_hip_torso_local = Vec2d(hip_off, hip_y)
        # Right side placement (vertical thigh and shin)
        right_hip_world = torso.local_to_world(right_hip_torso_local)
        rt.angle = math.pi / 2
        rt.position = right_hip_world - Vec2d(thigh_len / 2, 0).rotated(rt.angle)
        knee_R_world = rt.local_to_world((-thigh_len / 2, 0))
        rs.angle = math.pi / 2
        rs.position = knee_R_world - Vec2d(shin_len / 2, 0).rotated(rs.angle)
        right_hip = pymunk.PivotJoint(torso, rt, right_hip_torso_local, (thigh_len/2, 0))
        right_hip_lim = pymunk.RotaryLimitJoint(torso, rt, math.radians(30), math.radians(150))
        right_hip_m = pymunk.SimpleMotor(torso, rt, 0.0)
        self._set_motor_max_force(right_hip_m, self.MOTOR_FORCE)
        self.space.add(right_hip, right_hip_lim, right_hip_m)

        right_knee = pymunk.PivotJoint(rt, rs, (-thigh_len/2, 0), (shin_len/2, 0))
        right_knee_lim = pymunk.RotaryLimitJoint(rt, rs, math.radians(-120), math.radians(30))
        right_knee_m = pymunk.SimpleMotor(rt, rs, 0.0)
        self._set_motor_max_force(right_knee_m, self.MOTOR_FORCE)
        self.space.add(right_knee, right_knee_lim, right_knee_m)

        return {
            "torso": torso,
            "left_thigh": lt,
            "left_shin": ls,
            "right_thigh": rt,
            "right_shin": rs,
            "motors": {
                "left_hip": left_hip_m,
                "left_knee": left_knee_m,
                "right_hip": right_hip_m,
                "right_knee": right_knee_m,
            },
            "joints": {
                "left_hip": left_hip,
                "left_knee": left_knee,
                "right_hip": right_hip,
                "right_knee": right_knee,
            },
            "lengths": {
                "torso": torso_len,
                "left_thigh": thigh_len,
                "left_shin": shin_len,
                "right_thigh": thigh_len,
                "right_shin": shin_len,
            },
            "thickness": thick,
        }

    # --- RL API ---
    def reset(self) -> Dict[str, float]:
        self.space = pymunk.Space()
        self.space.gravity = (0.0, -self.GRAVITY)
        # Increase solver accuracy and add global damping for stability
        self.space.iterations = 40
        self.space.damping = 0.99
        self._add_ground()
        self.runner = self._build_runner(x=300, y=self.GROUND_Y + 120)
        self.camera_x = 0.0
        torso = self.runner["torso"]
        self.start_x = float(torso.position.x)
        self.distance = 0.0
        self.steps = 0
        return self.get_obs()

    def get_obs(self) -> Dict[str, float]:
        torso = self.runner["torso"]
        obs = {
            "x": float(torso.position.x),
            "y": float(torso.position.y),
            "vx": float(torso.velocity.x),
            "vy": float(torso.velocity.y),
            "angle": float(torso.angle),
            "ang_vel": float(torso.angular_velocity),
        }
        return obs

    def step(self, action: Optional[Dict[str, int]]) -> Tuple[Dict[str, float], float, bool, Dict]:
        # Update motors
        motors = self.runner["motors"]
        names = ("left_hip", "left_knee", "right_hip", "right_knee")
        if action is None:
            # No action provided; do nothing (keyboard handled in visualize())
            for n in names:
                motors[n].rate = 0.0
        else:
            for n in names:
                # Clamp to {-1, 0, +1}
                u = int(action.get(n, 0))
                u = -1 if u < 0 else (1 if u > 0 else 0)
                # Apply directly; direction is encoded in the key mapping
                motors[n].rate = self.MOTOR_RATE * u

        # Step physics with substeps for constraint stability
        dt = 1.0 / self.FPS
        substeps = 5
        h = dt / substeps
        for _ in range(substeps):
            self.space.step(h)

        # Mild damping
        for k, body in self.runner.items():
            if isinstance(body, pymunk.Body):
                body.angular_velocity *= 0.995
                body.velocity = body.velocity * 0.998

        torso = self.runner["torso"]
        self.camera_x = torso.position.x - self.WIDTH * 0.3
        self.distance = float(torso.position.x - self.start_x)
        self.steps += 1

        # Reward and termination
        reward = float(torso.velocity.x) * dt
        fallen = (torso.position.y < self.GROUND_Y + 5) or (abs(torso.angle) > self.MAX_TILT)
        time_up = self.steps >= self.max_steps
        done = bool(fallen or time_up)
        if fallen:
            reward -= 10.0

        return self.get_obs(), reward, done, {"distance": self.distance, "score": self.distance, "steps": self.steps}

    # --- Visualization ---
    def visualize(self) -> None:
        pygame.init()
        screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        clock = pygame.time.Clock()
        font = pygame.font.SysFont("Arial", 18)

        self.reset()
        running = True
        pressed = set()

        # Classic QWOP-style: Q/W drive hips in opposite directions; O/P drive knees
        # Each key applies +1 to one side and -1 to the other side
        KEYS = {
            pygame.K_q: [("left_hip", 1), ("right_hip", -1)],
            pygame.K_w: [("left_hip", -1), ("right_hip", 1)],
            pygame.K_o: [("left_knee", 1), ("right_knee", -1)],
            pygame.K_p: [("left_knee", -1), ("right_knee", 1)],
        }

        def to_pygame(p: Vec2d) -> Tuple[int, int]:
            return int(p.x - self.camera_x), int(self.HEIGHT - p.y)

        while running:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    running = False
                elif ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_ESCAPE:
                        running = False
                    elif ev.key == pygame.K_r:
                        self.reset()
                        pressed.clear()
                    elif ev.key in KEYS:
                        pressed.add(ev.key)
                elif ev.type == pygame.KEYUP:
                    if ev.key in KEYS and ev.key in pressed:
                        pressed.remove(ev.key)

            # Build action dict from pressed
            action = {"left_hip": 0, "right_hip": 0, "left_knee": 0, "right_knee": 0}
            for key in list(pressed):
                if key in KEYS:
                    for name, direction in KEYS[key]:
                        action[name] += direction
            # Clamp to [-1, 1]
            for k in action:
                action[k] = -1 if action[k] < 0 else (1 if action[k] > 0 else 0)

            obs, reward, done, info = self.step(action)
            # Debug overlay (toggle-able if needed): body centers and joint anchors
            # centers = [self.runner["torso"].position, self.runner["left_thigh"].position, self.runner["left_shin"].position, self.runner["right_thigh"].position, self.runner["right_shin"].position]
            # for c in centers:
            #     pygame.draw.circle(screen, (200, 0, 0), to_pygame(c), 3)

            # Draw
            screen.fill((240, 240, 240))
            # ground (with tick marks for motion cues)
            pygame.draw.rect(screen, (180, 180, 180), (0, self.HEIGHT - self.GROUND_Y, self.WIDTH, self.GROUND_Y))
            tick_spacing = 120
            offset = int(self.camera_x) % tick_spacing
            for x in range(-offset, self.WIDTH, tick_spacing):
                pygame.draw.line(screen, (150, 150, 150), (x, self.HEIGHT - self.GROUND_Y), (x, self.HEIGHT - self.GROUND_Y - 15), 2)

            # skeleton rendering with constraint-anchored joints to keep lengths visually exact
            J = self.runner["joints"]
            L = self.runner["lengths"]
            thick = self.runner["thickness"]

            def world(body: pymunk.Body, local):
                # Accept tuple or Vec2d; convert via Pymunk utility
                return body.local_to_world(local)

            # Torso line and head from torso body's transform to ensure stable attachment
            torso = self.runner["torso"]
            t_a = world(torso, Vec2d(-L["torso"]/2, 0))
            t_b = world(torso, Vec2d(L["torso"]/2, 0))
            pygame.draw.line(screen, (20, 20, 120), to_pygame(t_a), to_pygame(t_b), 6)
            # Head anchored to torso center + rotated up vector
            torso_center = torso.position
            up = Vec2d(0, 1).rotated(torso.angle)
            head_r = 12
            head_center = torso_center + up * (L["torso"] * 0.6 + head_r)
            pygame.draw.circle(screen, (230, 200, 160), to_pygame(head_center), head_r)
            pygame.draw.circle(screen, (0, 0, 0), to_pygame(head_center), head_r, 2)

            # Left leg: hip on torso, knee from thigh, foot from shin using their constraint anchors
            lt = self.runner["left_thigh"]
            ls = self.runner["left_shin"]
            hip_L = world(torso, J["left_hip"].anchor_a)
            knee_L = world(lt, J["left_knee"].anchor_a)
            foot_L = world(ls, (-L["left_shin"] / 2, 0))
            pygame.draw.line(screen, (34, 139, 34), to_pygame(hip_L), to_pygame(knee_L), 6)
            pygame.draw.line(screen, (255, 140, 0), to_pygame(knee_L), to_pygame(foot_L), 6)

            # Right leg
            rt = self.runner["right_thigh"]
            rs = self.runner["right_shin"]
            hip_R = world(torso, J["right_hip"].anchor_a)
            knee_R = world(rt, J["right_knee"].anchor_a)
            foot_R = world(rs, (-L["right_shin"] / 2, 0))
            pygame.draw.line(screen, (34, 139, 34), to_pygame(hip_R), to_pygame(knee_R), 6)
            pygame.draw.line(screen, (255, 140, 0), to_pygame(knee_R), to_pygame(foot_R), 6)

            # Joints as circles to visually connect
            radius = max(4, int(thick * 0.6))
            for p in (to_pygame(hip_L), to_pygame(knee_L), to_pygame(foot_L), to_pygame(hip_R), to_pygame(knee_R), to_pygame(foot_R)):
                pygame.draw.circle(screen, (0, 0, 0), p, radius)

            # HUD
            hud = font.render(f"Dist: {info['distance']:.0f}   Score: {info['distance']:.0f}   Steps: {self.steps}/{self.max_steps}   Q/W/O/P to move, R restart", True, (10,10,10))
            screen.blit(hud, (10, 10))

            pygame.display.flip()
            clock.tick(self.FPS)

            if done:
                # Show message and wait for R/ESC
                msg = font.render(f"Episode over. Score (distance): {self.distance:.1f}  Press R to restart, ESC to quit.", True, (0,0,0))
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
                                self.reset()
                                pressed.clear()
                                waiting = False
                            elif ev.key == pygame.K_ESCAPE:
                                waiting = False
                                running = False
                    clock.tick(30)

        pygame.quit()


def main():
    env = QWOPRunnerEnv()
    env.visualize()


if __name__ == "__main__":
    main()


