The plan is to have a simple game:
Ball goes up and down, and we have a control over the bottom plate which can kick it, as the ball falls down - it is like a simple physics simulation.

We have h, v, a of the ball, p, p' of the plate (p bounded in [0, 5]), and we have control over p'', with range [-1, 1].
We also have t - how long the ball has been sitting on the plate (this is to prevent staying on the floor).

Score is generated only when ball is in contact with the plate 20*(v-p')^2-t^2. This score discourages large t and large difference in velcity - we want smooth landing of the ball, but also we don't want optimal strategy to be stationary ball.

First, do RL to find optimal strategy




For now, ignore the following:
Now, let us do it differently:

Let us declare that the "default" control is p'' = unif[-1, 1], and stays constant.

