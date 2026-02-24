# CS272 Final Project — Autonomous Driving with Deep RL (HighwayEnv + Custom Accident Env)

This project trains and evaluates Deep Reinforcement Learning (DRL) agents for autonomous driving tasks using **HighwayEnv** and a **custom accident-avoidance environment**. We compare multiple on-policy algorithms (A2C, PPO, TRPO), perform hyperparameter tuning with **Optuna**, and report results using learning curves and deterministic performance tests.

---

## Project Goals

- Train DRL agents to drive safely in highway-style environments.
- Extend the base highway environment with a **custom “accident scene”** where two crashed vehicles block adjacent lanes.
- Compare algorithms and tuning strategies using consistent evaluation:
  - **Learning curve:** Mean episodic return vs training episodes  
  - **Performance test:** Violin plot of episodic returns over **500 deterministic episodes** (no exploration)

---

## Environments

### 1) Custom Accident Environment (`AccidentEnv`)
A custom HighwayEnv-style environment with a fixed crash zone:
- Two crashed vehicles placed at fixed longitudinal positions (e.g., around x=500 and x=505)
- Crash blocks **two adjacent lanes** (lane pair may vary per episode)
- Agent must:
  - move away from blocked lanes
  - slow down near the crash
  - continue driving after clearing
  - prefer right-most lane when safe
  - avoid tailgating and off-road behavior

### 2) HighwayEnv Baselines
We also train on standard HighwayEnv tasks (depending on experiment setup):
- Highway
- Merge
- Intersection  
with multiple observation types:
- **LiDAR** (vector observation)
- **Grayscale** (image observation + frame stacking)

---

## Algorithms

We compare:
- **A2C** (baseline, fast but higher variance)
- **PPO** (clipped objective for stable updates)
- **TRPO** (trust-region updates for stability)

We tune PPO using **Optuna** with optional pruning to stop weak trials early.

---

## Evaluation Outputs

For each environment + observation setup we generate:
1. **Learning Curve**  
   Mean episodic training return vs training episodes  
   (from Monitor CSV logs, or eval `.npz` when available)
2. **Deterministic Performance Test (500 episodes)**  
   Violin plot of episodic returns using `deterministic=True`

Total expected plots:
- **3 environments × 2 observation types × 2 plots = 12 plots**

---
