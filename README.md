# 🚀 Reinforcement Learning-Based Rocket Landing using PPO

**Author:** Ayo-Oluwa Sodipe  
**Institution:** Georgia State University  
**Contact:** [asodipe1@student.gsu.edu](mailto:asodipe1@student.gsu.edu)

---

## 🚀 Project Overview

This project investigates how **reward function design** influences the learning behavior of a Proximal Policy Optimization (PPO) agent trained to land a rocket in a custom 2D simulation environment.

Three progressively complex reward function variants were implemented and compared:

| Version | Reward Signal |
|---------|--------------|
| **V1** | Distance-only penalty |
| **V2** | Distance + vertical velocity penalty |
| **V3** | Distance + vertical velocity + horizontal position + horizontal velocity penalties |

Each environment was built from scratch using the **Gymnasium** framework, with agents trained via **Stable-Baselines3's PPO** implementation. The central research question: *Are dense reward signals alone sufficient for learning stable rocket landing behavior?*

---

## ⚡ Quick Start

```bash
git clone https://github.com/AAAY1119/rocket-landing-ppo.git
cd rocket_landing_project
pip install -r requirements.txt
cd src

# Evaluate models
python evaluate.py v1
python evaluate.py v2
python evaluate.py v3

---

## 🧠 Motivation

Autonomous rocket landing is one of the most demanding control problems in modern aerospace engineering. Systems like **SpaceX's Falcon 9** must execute precise, real-time adjustments across multiple axes — managing velocity, orientation, fuel consumption, and environmental disturbances — all within a narrow margin of error.

Reinforcement learning presents a compelling alternative to classical control methods, offering the ability to learn complex policies directly from environmental feedback. However, the design of the reward function is a make-or-break factor: a poorly specified reward signal can prevent convergence entirely, even with a powerful algorithm like PPO.

This project explores that challenge in a simplified 2D setting — serving as a testbed for understanding how reward shaping decisions translate into agent behavior, and laying the groundwork for more sophisticated landing systems.

---

## 📁 Project Structure

```
rocket_landing_project/
│
├── data/
├── figures/
│   ├── evaluation_rewards.png
│   ├── evaluation_rewards_v1.png
│   ├── evaluation_rewards_v2.png
│   ├── evaluation_rewards_v3.png
│   ├── sample_trajectories.png
│   ├── sample_trajectories_v1.png
│   ├── sample_trajectories_v2.png
│   └── sample_trajectories_v3.png
│
├── models/
│   ├── ppo_v1.zip
│   ├── ppo_v2.zip
│   └── ppo_v3.zip
│
├── notebooks/
├── paper/
│   └── IEEE_Paper.pdf
│
├── presentation/
│   └── Research Presentation.pptx
│
├── results/
├── src/
│   ├── env_v1.py
│   ├── env_v2.py
│   ├── env_v3.py
│   ├── train.py
│   ├── evaluate.py
│   └── test_env.py
│
├── requirements.txt
└── README.md
```

**Key directories at a glance:**

- `src/` — All source code: environment definitions, training, and evaluation scripts
- `models/` — Saved PPO model checkpoints for each reward variant (`.zip`)
- `figures/` — Auto-generated plots from evaluation runs
- `paper/` — Full IEEE-formatted research paper
- `presentation/` — Slide deck used for project presentation

---

## ⚙️ Installation

### Prerequisites

- Python 3.8 or higher
- pip

### Steps

**1. Clone or download the repository:**

```bash
git clone https://github.com/your-username/rocket_landing_project.git
cd rocket_landing_project
```

**2. Install all dependencies:**

```bash
pip install -r requirements.txt
```

This installs the full stack: `gymnasium`, `stable-baselines3`, `numpy`, and `matplotlib`.

---

## ▶️ How to Run

### Training

Navigate to the `src/` directory and run the training script:

```bash
cd src
python train.py
```

This will train PPO agents for all three environment variants and save the resulting model files to the `models/` directory as `ppo_v1.zip`, `ppo_v2.zip`, and `ppo_v3.zip`.

> **Note:** Training can be time-intensive depending on your hardware. Timestep counts and hyperparameters can be adjusted directly in `train.py`.

---

### Evaluation

Evaluate each trained model individually by passing the version flag as a command-line argument:

```bash
python evaluate.py v1
python evaluate.py v2
python evaluate.py v3
```

Each evaluation run loads the corresponding saved model, runs it through several episodes, and saves output plots to the `figures/` directory.

---

## 📊 Output Explanation

After running evaluation, two types of plots are generated per version:

### `evaluation_rewards_v*.png`
Displays the **total cumulative reward per episode** across the evaluation run. Use this to assess whether the agent is improving over time and how stable its performance is. Flat or declining reward curves indicate the agent has failed to converge on a meaningful policy.

### `sample_trajectories_v*.png`
Visualizes the **physical path of the rocket** across multiple episodes. This makes it immediately clear whether the agent is guiding the rocket toward the landing zone or drifting unpredictably. Diverging or chaotic trajectories confirm poor policy learning.

All figures are saved to:

```
rocket_landing_project/figures/
```

---

## 🧪 Experiment Design

Each environment version builds progressively on the last, incorporating more signal components to guide the agent's behavior.

### V1 — Distance-Only Penalty

```
reward = -distance
```

The simplest possible formulation. The agent is penalized proportionally to its distance from the landing target. No velocity or positional axis is explicitly addressed. Intended as a baseline to measure how far a minimal signal can take learning.

---

### V2 — Distance + Vertical Velocity Penalty

```
reward = -distance - 0.5 * |vy|
```

Extends V1 by penalizing excessive vertical velocity. This encourages the agent to descend more slowly and smoothly — a critical real-world constraint to prevent crash landings.

---

### V3 — Full Multi-Component Penalty

```
reward = -distance - 0.5 * |vy| - 0.3 * |x| - 0.2 * |vx|
```

The most informative reward signal of the three. V3 adds penalties for horizontal displacement and lateral velocity, pressuring the agent to maintain vertical alignment and reduce sideways drift throughout the descent.

The rationale behind this progression: each version adds a new behavioral constraint, allowing us to isolate exactly which components of the reward signal contribute to — or fail to produce — stable landing behavior.

---

## 📈 Results Summary

Despite the progressive complexity of the reward functions, all three variants failed to produce a successful landing policy:

- **Success rate:** 0% across all three versions
- **Reward trends:** Persistently negative across all evaluation episodes, with no convergence signal
- **Trajectory behavior:** Rocket paths were inconsistent and non-directed, indicating random or near-random action selection

These results held across V1, V2, and V3, suggesting the issue is systemic rather than tied to any specific reward variant.

---

## ⚠️ Core Limitation

The primary finding of this project is that **dense penalties alone are insufficient to guide PPO toward a successful landing policy.**

Three structural problems were identified:

**1. No terminal success incentive.**  
The reward function never explicitly rewards reaching the landing zone. The agent has no signal differentiating "close to landing" from "actually landed." Without a positive terminal reward, there is no gradient pulling the agent toward task completion.

**2. Reward signal is dense but not directional.**  
Penalizing distance at every timestep creates a learning signal, but not one that teaches the agent *what actions* to take. The agent receives the same type of feedback regardless of whether it's improving or worsening — making it difficult to extract a meaningful control gradient.

**3. PPO cannot recover from an underspecified objective.**  
PPO is a powerful policy optimization algorithm, but it optimizes what it is given. If the reward landscape does not guide behavior toward the goal, PPO will learn to minimize penalties in ways that do not correspond to landing — or fail to learn a coherent strategy at all.

---

## 🔧 Future Improvements

The following enhancements are likely to produce meaningful improvements in agent performance:

- **Terminal landing bonus** — Add a large positive reward upon successful touchdown within the target zone, giving the agent a clear objective to optimize toward
- **Curriculum learning** — Start the rocket at short distances from the landing pad and progressively increase difficulty as the agent improves
- **Increased training timesteps** — Extend training well beyond current limits; complex control tasks routinely require millions of timesteps to converge
- **Potential-based reward shaping** — Replace raw distance penalties with potential-based shaping functions, which are proven not to alter the optimal policy while improving sample efficiency

---

## 📄 Additional Materials

This project is accompanied by a full research paper and slide deck:

- **IEEE Research Paper:** [`paper/IEEE_Paper.pdf`](paper/IEEE_Paper.pdf)  
  A formal write-up of the methodology, experiment design, results, and analysis formatted to IEEE standards.

- **Presentation Slides:** [`presentation/Research Presentation.pptx`](presentation/Research%20Presentation.pptx)  
  The slide deck used to present findings, including visual summaries of trajectories and reward curves.

---

## 👤 Author

**Ayo-Oluwa Sodipe**  
B.S. Computer Science — Georgia State University  
📧 [asodipe1@student.gsu.edu](mailto:asodipe1@student.gsu.edu)

---

*This project was developed as part of an independent research investigation into reinforcement learning reward design for autonomous control tasks.*
