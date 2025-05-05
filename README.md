# PendulumSwing-ActorCriticSoftmax

# Average Reward Softmax Actor-Critic — Pendulum Swing-Up

This repository contains an implementation of the **Average Reward Softmax Actor-Critic** algorithm applied to the **Pendulum Swing-Up** task. This assignment explores reinforcement learning in a **continuing task** setting using **differential TD learning** and a **softmax-parameterized policy**.

---

## Overview

The Pendulum Swing-Up task requires an agent to swing a pendulum from a hanging position and stabilize it in an upright position using discrete torque actions. The state is continuous, consisting of the pendulum's angle and angular velocity.

**Why is this problem hard?**  
The action space is not strong enough to immediately swing the pendulum upward. The agent must learn to gain momentum through oscillatory motion and balance the pendulum once upright — a classic underactuated control challenge.

---
## Environment

### Pendulum Swing-Up

- **State space**:  
  - Angle `β ∈ [−π, π]`  
  - Angular velocity `β̇ ∈ [−2π, 2π]`
- **Action space**:  
  - Discrete torques `{−1, 0, 1}`
- **Reward**:  
  - `Rₜ = −|βₜ|` (maximize time near upright position)

### Packages Used

- `numpy` — numerical computation
- `matplotlib` — plotting
- `tiles3` — tile coding
- `rl_glue` — RL framework
- `pendulum_env` — custom pendulum environment
- `plot_script` — experiment visualizations
- 
---

## Agent: Average Reward Actor-Critic

- **Policy (Actor)**:  
  Parameterized using softmax over linear action preferences `θᵀ x(s, a)`

- **Value function (Critic)**:  
  Estimates differential value using TD(0) and linear function approximation

- **Learning signal (TD-error)**:  
```
δₜ = Rₜ₊₁ − R̄ + v̂(Sₜ₊₁) − v̂(Sₜ)
R̄ ← R̄ + ᾱ * δₜ
```

- **Actor update (semi-gradient)**: ```θ ← θ + α * δₜ * (x(s, a) - ∑ₐ' π(a'|s, θ) x(s, a'))```

---

## Tile Coding (with Wrapping)

- **Tile coder** wraps the angle dimension** to maintain circular continuity:
- E.g. `−π` and `π` map to similar features
- `tiles3.tileswrap()` used to implement wrapping over angular dimension

---

## Experimentation

### Experiment Setup

- 50 independent runs
- 20,000 time steps per run
- 32 tilings × 8×8 tiles
- Tuned step-sizes:
- `actor_step_size = 0.25`
- `critic_step_size = 2.0`
- `avg_reward_step_size = 2⁻⁶`

### Metrics Evaluated

- **Total Return**: Sum of negative angle deviations over time
- **Exponential Average Reward**: Bias-corrected metric for long-run policy quality

> Final plots are generated using `plot_script.plot_result()`.

---

## Repository Structure

```
├── agent.py # ActorCriticSoftmaxAgent implementation
├── tiles3.py # Tile coding (with wrap support)
├── pendulum_env.py # Pendulum Swing-Up environment
├── rl_glue.py # RL-Glue framework
├── plot_script.py # Plotting utility for reward curves
├── experiments.ipynb # All experiments and results
├── results/ # Generated reward curves and .npy files
├── requirements.txt
└── README.md # This file
```
---

## Results Preview

### Exponential Average Reward Over Time:

- The best meta-parameter configuration learned to balance the pendulum upright
- Rewards converged close to `0.0` indicating effective swing-up and stabilization

---

## Insights

- **Wrapping** is essential for handling circular state features like angles.
- **Actor-Critic** methods are naturally suited for continuing tasks.
- **Meta-parameter sensitivity** was low, especially for average reward step-size — demonstrating robustness.
- **Softmax policies** allow probabilistic exploration and smooth learning in discrete action spaces.

---

## References

- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.), Chapters 9–10.
- Santamaría, R., Sutton, R. S., & Ram, A. (1998). *Experiments with reinforcement learning in problems with continuous state and action spaces.*

---

## Acknowledgements

- University of Alberta — Reinforcement Learning Specialization (Course 3)
