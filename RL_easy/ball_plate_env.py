import math
import random
from dataclasses import dataclass
from typing import Dict, Tuple

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


@dataclass
class BallPlateState:
    # Ball in world coordinates
    ball_y: float
    ball_v: float
    # Plate in world coordinates
    plate_y: float
    plate_v: float
    # Contact dwell time (time effectively resting on plate)
    contact_time: float


class BallPlateEnv:
    """
    1D vertical ball with gravity and a controllable plate (world coordinates).

    State variables:
      - ball_y (y): ball height [meters]
      - ball_v (v): ball vertical velocity [m/s] (positive up)
      - plate_y (p): plate height [meters] (bounded)
      - plate_v (p'): plate velocity [m/s]
      - contact_time (t): seconds in continuous contact or quasi-rest

    Action:
      - acceleration command for plate (p'') in [-1, 1] m/s^2

    Reward (only when in contact): 20*(v - p')^2 - t^2
      (Using world velocities; reward is 0 when not in contact.)

    Notes:
      - Semi-implicit Euler integration in world coordinates.
      - Elastic collision controlled by coefficient of restitution `restitution` in [0,1].
        On collision: (v_ball_post - v_plate) = -e * (v_ball_pre - v_plate).
      - Gravity acts on the ball only.
    """

    def __init__(self,
                 dt: float = 0.05,
                 gravity: float = 3.0,
                 plate_min: float = 0.0,
                 plate_max: float = 2.0,
                 plate_accel_limit: float = 1.0,
                 restitution: float = 0.8):
        self.dt = dt
        self.g = gravity
        self.plate_min = plate_min
        self.plate_max = plate_max
        self.plate_accel_limit = plate_accel_limit
        self.restitution = max(0.0, min(1.0, restitution))
        self.state = BallPlateState(ball_y=1.0, ball_v=0.0, plate_y=0.5, plate_v=0.0, contact_time=0.0)
        self.last_action = 0.0  # store last p'' (acceleration command)

    def reset(self,
              height: float = 1.0,
              velocity: float = 0.0,
              position: float = 0.5,
              velocity_plate: float = 0.0,
              contact_time: float = 0.0) -> Dict[str, float]:
        # Keep parameter names for backwards-compat; map to world coordinates
        self.state = BallPlateState(ball_y=height,
                                    ball_v=velocity,
                                    plate_y=position,
                                    plate_v=velocity_plate,
                                    contact_time=contact_time)
        return self._obs()

    def _obs(self) -> Dict[str, float]:
        return {
            "y": self.state.ball_y,
            "v": self.state.ball_v,
            "p": self.state.plate_y,
            "p_dot": self.state.plate_v,
            "t": self.state.contact_time,
        }

    def step(self, action: float) -> Tuple[Dict[str, float], float, bool, Dict]:
        # Clamp action to [-1, 1]
        if action is None:
            action = 0.0
        action = max(-self.plate_accel_limit, min(self.plate_accel_limit, float(action)))
        self.last_action = action

        s = self.state
        dt = self.dt

        # Plate dynamics (world coords)
        s.plate_v += action * dt
        s.plate_y += s.plate_v * dt
        if s.plate_y < self.plate_min:
            s.plate_y = self.plate_min
            s.plate_v = 0.0
        if s.plate_y > self.plate_max:
            s.plate_y = self.plate_max
            s.plate_v = 0.0

        # Ball dynamics (world coords)
        s.ball_v += -self.g * dt
        pre_v_ball = s.ball_v
        s.ball_y += s.ball_v * dt

        # Contact and collision
        contact = s.ball_y <= s.plate_y
        if contact:
            s.ball_y = s.plate_y
            rel_pre = pre_v_ball - s.plate_v
            if rel_pre < 0.0:
                # Elastic collision: reflect relative velocity with restitution
                s.ball_v = s.plate_v - self.restitution * rel_pre
            else:
                # Separating (or equal). Preserve ball upward velocity so it can lift off
                s.ball_v = pre_v_ball

            # Accumulate dwell time only when nearly sticking
            if abs(s.ball_v - s.plate_v) < 1e-3:
                s.contact_time += dt
            else:
                s.contact_time = 0.0
        else:
            s.contact_time = 0.0

        # Reward only when in contact
        reward = 0.0
        if contact:
            dv = (s.ball_v - s.plate_v)
            reward = -(20.0 * (dv * dv) + (s.contact_time * s.contact_time))

        done = False
        info = {"contact": contact}
        return self._obs(), reward, done, info

    # --- Visualization ---
    def visualize(self, steps: int = 600, policy: str = "random") -> None:
        """
        Visualize the system using matplotlib animation.
        policy: "random" (default) -> action sampled uniformly in [-1, 1]
        """
        fig, ax = plt.subplots(figsize=(4, 6))
        supports_blit = bool(getattr(fig.canvas, "supports_blit", False))
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(self.plate_min - 0.5, self.plate_max + 1.0)
        ax.set_xlabel("x (ignored)")
        ax.set_ylabel("height")
        ax.set_title("1D Ball-Plate")

        # Artists
        plate_line, = ax.plot([-0.4, 0.4], [self.state.plate_y, self.state.plate_y], lw=4, color="black")
        ball_point, = ax.plot([0.0], [self.state.ball_y], marker="o", color="tab:blue")
        text = ax.text(-0.45, self.plate_max + 0.6, "", fontsize=9)
        if supports_blit:
            plate_line.set_animated(True)
            ball_point.set_animated(True)
            text.set_animated(True)

        def policy_action() -> float:
            if policy == "random":
                return random.uniform(-1.0, 1.0)
            return 0.0

        step_counter = {"k": 0}

        def init():
            # Return the artists that will be updated each frame
            return plate_line, ball_point, text

        def update(i):
            a = policy_action()
            obs, reward, _, info = self.step(a)

            # Update artists
            y_plate = self.state.plate_y
            y_ball = self.state.ball_y
            plate_line.set_ydata([y_plate, y_plate])
            ball_point.set_data([0.0], [y_ball])

            text.set_text(
                f"step={i+1}\n"
                f"p''={self.last_action:+.2f}  contact={info['contact']}\n"
                f"y={obs['y']:.2f} v={obs['v']:.2f}\n"
                f"p={obs['p']:.2f} p'={obs['p_dot']:.2f} t={obs['t']:.2f}\n"
                f"r={reward:.2f}"
            )

            return plate_line, ball_point, text

        _ = FuncAnimation(
            fig,
            update,
            init_func=init,
            frames=steps,
            interval=int(self.dt * 1000),
            blit=supports_blit,
            cache_frame_data=False,
            repeat=False,
        )
        plt.tight_layout()
        plt.show()


def demo():
    env = BallPlateEnv()
    env.reset(height=2.5, velocity=0.0, position=0.0, velocity_plate=0.0)
    env.visualize(steps=800, policy="random")


if __name__ == "__main__":
    demo()


