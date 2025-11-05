import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from ball_plate_env import BallPlateEnv
from nn_policy import get_or_create_policy


def visualize_with_policy(steps: int = 800) -> None:
    env = BallPlateEnv()
    # Load or create weights in default location
    policy, weights_path = get_or_create_policy(weights_path=None, hidden_sizes=(64, 64), action_scale=1.0)

    # Reset environment
    obs = env.reset(height=2.5, velocity=0.0, position=0.0, velocity_plate=0.0, contact_time=0.0)

    fig, ax = plt.subplots(figsize=(4, 6))
    supports_blit = bool(getattr(fig.canvas, "supports_blit", False))
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(env.plate_min - 0.5, env.plate_max + 1.0)
    ax.set_xlabel("x (ignored)")
    ax.set_ylabel("height")
    ax.set_title("NN Policy: 1D Ball-Plate")

    # Artists
    plate_line, = ax.plot([-0.4, 0.4], [env.state.plate_y, env.state.plate_y], lw=4, color="black")
    ball_point, = ax.plot([0.0], [env.state.ball_y], marker="o", color="tab:blue")
    text = ax.text(-0.45, env.plate_max + 0.6, f"weights: {weights_path}", fontsize=9)
    if supports_blit:
        plate_line.set_animated(True)
        ball_point.set_animated(True)
        text.set_animated(True)

    # Keep obs mutable inside closure
    buf = {"obs": obs}

    def init():
        return plate_line, ball_point, text

    def update(i):
        # NN policy action
        action = policy.act_from_obs_dict(buf["obs"])  # in [-1, 1]
        obs, reward, _, info = env.step(action)
        buf["obs"] = obs

        # Update artists
        y_plate = env.state.plate_y
        y_ball = env.state.ball_y
        plate_line.set_ydata([y_plate, y_plate])
        ball_point.set_data([0.0], [y_ball])

        text.set_text(
            f"step={i+1}\n"
            f"y={obs['y']:.2f} v={obs['v']:.2f}\n"
            f"p={obs['p']:.2f} p'={obs['p_dot']:.2f} t={obs['t']:.2f}\n"
            f"p''={env.last_action:+.2f}  contact={info['contact']}\n"
            f"r={reward:.2f}\n"
            f"weights: {weights_path}"
        )

        return plate_line, ball_point, text

    anim = FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=steps,
        interval=int(env.dt * 1000),
        blit=supports_blit,
        cache_frame_data=False,
        repeat=False,
    )
    plt.tight_layout()
    plt.show()


def main():
    visualize_with_policy(steps=800)


if __name__ == "__main__":
    main()


