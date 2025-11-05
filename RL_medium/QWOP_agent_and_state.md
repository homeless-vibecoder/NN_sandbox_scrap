## Overview

This QWOP-like environment is a 2D ragdoll runner built with Pymunk physics and Pygame rendering. The agent controls four motors at the hips and knees; forward progress is rewarded and episodes end on falls or a step limit.

## What the agent controls

- **Joints (4)**: `left_hip`, `left_knee`, `right_hip`, `right_knee`.
- **Action per joint**: 3-way command in {-1, 0, +1} mapped to a joint motor rate (angular velocity target) with a fixed magnitude.
  - **Hips**: +1 flexes one side and extends the other (signs mirrored left/right).
  - **Knees**: +1 extends on one side and flexes on the other (signs mirrored left/right).
- **Execution rate**: Physics at 60 Hz with 3 substeps; typical action repeat of 3 → control ~20 Hz.

Practical meaning: per joint, −1 = drive one direction (e.g., flex), 0 = hold, +1 = drive the opposite direction (e.g., extend). Alternating stance/swing leg with coordinated hips/knees produces forward motion while keeping the torso upright.

## Game state: core signals

Minimal observation from `env.get_obs()` (torso only):
- **Torso pose/vel**: `x, y, vx, vy, angle, ang_vel`.

Highly useful derived signals for control/prediction:
- **Relative torso position**: `dx = x - start_x` (progress along track).
- **Orientation (robust)**: `sin(angle), cos(angle)` to avoid angle wrap.
- **Joint angles (relative)**:
  - `left_hip = angle(left_thigh) - angle(torso)`
  - `left_knee = angle(left_shin) - angle(left_thigh)`
  - `right_hip = angle(right_thigh) - angle(torso)`
  - `right_knee = angle(right_shin) - angle(right_thigh)`
- **Joint angular velocities (relative)**:
  - e.g., `left_knee_vel = ang_vel(left_shin) - ang_vel(left_thigh)` (similarly for others)
- (Optional) **Foot contacts**: left/right foot touching ground (binary) or contact impulse.

## Recommended feature vector (15D)

Used by the training code and suitable for prediction models:
- `[dx/200, y/200, vx/400, vy/400, sin(angle), cos(angle), ang_vel/10,`
  `left_hip, left_knee, right_hip, right_knee,`
  `left_hip_vel/10, left_knee_vel/10, right_hip_vel/10, right_knee_vel/10]`

Notes:
- The joint angles are in radians and typically fall within joint limits; velocities are scaled for numeric stability.
- Add 2 bits for foot contacts if you expose them.

## Reward and termination (for context)

- **Reward**: Integrated forward velocity (progress). A fall incurs a penalty.
- **Done**: If the torso tilts too much or drops near the ground, or when the step limit is reached.

## Physics and timing

- **Gravity**: Downward; ground at a fixed `GROUND_Y` with friction.
- **Damping/iterations**: Global space damping and higher solver iterations for stability.
- **Self-collision**: Disabled between body parts to prevent jitter.
- **Joint limits**: Rotary limits on hips/knees constrain motion to reasonable ranges.

## Building a state predictor (forecasting)

Define what to predict and at what horizon:
- **Inputs**: Current state features (15D) and, if forecasting under policy, include the **action** that will be applied for the next K frames (or the policy’s action distribution). Without action, the prediction is a marginal forecast over unknown inputs.
- **Targets**: Next-state features at horizon `H` (e.g., 1, 3, 5, 10 control steps ahead). For multi-step, train on several horizons or roll your model forward autoregressively.

Recommendations:
- **Condition on actions**: Use the 4 joint commands in {-1,0,+1} (one-hot per joint, 3D each → 12D) or as integers in {-1,0,+1}.
- **Normalize**: Keep the same scaling used above for stability.
- **Dataset**: Collect (state, action) → (next_state) tuples by rolling the environment under a mix of random and learned policies. Include falls and recoveries for robustness.
- **Curriculum**: Start with short horizons (e.g., H=1–3) and extend as the model learns.

## Quick reference

- **Controls**: 4 joints, each in {-1,0,+1}; action repeat ≈ 3; control ≈ 20 Hz.
- **Core state**: torso pose/vel; add joint angles/vels, sin/cos(orientation), dx.
- **Feature dim**: 15 (or 17 with foot contacts).
- **Good targets**: next 15D state at horizon H; include actions when forecasting.


