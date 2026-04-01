# Reinforcement Learning & MuJoCo — Zero to Understanding

A complete guide written specifically for this project. Every concept here
directly maps to something in `mujoco_env.py`, `train_ppo.py`, `reward.py`,
`curriculum.py`, or the MuJoCo world files.

---

## Table of Contents

1. [What is Reinforcement Learning?](#1-what-is-reinforcement-learning)
2. [Key RL Concepts](#2-key-rl-concepts)
3. [How Neural Networks Fit In](#3-how-neural-networks-fit-in)
4. [PPO — The Algorithm We Use](#4-ppo--the-algorithm-we-use)
5. [What is MuJoCo?](#5-what-is-mujoco)
6. [MuJoCo Internals](#6-mujoco-internals)
7. [Gymnasium — The RL Environment API](#7-gymnasium--the-rl-environment-api)
8. [How This Project Uses All of the Above](#8-how-this-project-uses-all-of-the-above)
9. [Reading the Training Output](#9-reading-the-training-output)
10. [Common Problems and What They Mean](#10-common-problems-and-what-they-mean)
11. [Hyperparameter Tuning Intuition](#11-hyperparameter-tuning-intuition)
12. [Learning Resources](#12-learning-resources)

---

## 1. What is Reinforcement Learning?

Reinforcement Learning (RL) is a way to teach a computer to make decisions
by **trial and error** — exactly how you learned to ride a bike.

The core idea:

```
Agent takes action → Environment changes → Agent gets reward → Repeat
```

- The **agent** is your robot (or any decision-maker).
- The **environment** is the world it lives in (MuJoCo maze in our case).
- The **action** is what the robot does (move forward, turn left, etc.).
- The **reward** is a score: positive for doing good things, negative for bad.
- The **goal** is to learn a strategy that maximises total reward over time.

### Analogy: Teaching a Dog

| Dog training      | Reinforcement Learning    |
|-------------------|---------------------------|
| Dog               | Agent (robot)             |
| Room              | Environment (maze)        |
| Sit / roll over   | Action (vx, vy, yaw)      |
| Treat / no treat  | Reward (+1 / -1)          |
| Training sessions | Episodes                  |
| Learned behaviour | Policy (neural network)   |

---

## 2. Key RL Concepts

### Episode
One complete run from start to finish. In our project:
- **Start**: robot spawned at a random position in the maze
- **End**: robot reaches goal (success), hits a wall (collision), runs out of
  time (timeout), or gets stuck

Each episode gives the agent new experience to learn from.

### Observation (State)
What the robot can "see" at each step. In our project:

```
observation = {
    'costmap':     (84, 84, 1)  — bird's-eye obstacle map around robot
    'scan':        (360,)       — LiDAR distances in every direction (0–1 normalised)
    'goal_vector': (2,)         — direction + distance to goal in robot's frame
    'velocity':    (3,)         — current speed (vx, vy, yaw rate)
}
```

Think of this as the robot's "senses" at one instant.

### Action
What the robot decides to do. In our project:

```
action = [vx, vy, vyaw]  — all normalised to [-1, 1]
```

- `vx`  = forward/backward speed
- `vy`  = left/right strafe speed (holonomic wheels)
- `vyaw` = rotation speed

The neural network outputs these three numbers at every step.

### Reward
A score given to the agent after each action. Designed by us in `reward.py`:

| Component       | When triggered                        | Value     |
|-----------------|---------------------------------------|-----------|
| Progress        | Every step closer to goal             | +up to 5  |
| Heading         | Facing toward goal                    | +0.2 max  |
| Near-goal       | Within 1.0 m of goal (linear)         | +0→3      |
| Goal reached    | Robot centre < 0.5 m from goal        | +10       |
| Collision       | LiDAR < 0.15 m or wall contact        | −5        |
| Proximity       | LiDAR < 0.4 m (soft warning)          | −0.5 max  |
| Smoothness      | Jerky velocity changes                | −0.1 max  |
| Step cost       | Every step taken (encourages speed)   | −0.01     |

The agent learns to **maximise the sum of these rewards** over an episode.

### Policy
The strategy the agent follows — implemented as a neural network. It maps
observations → actions:

```
observation  →  [neural network]  →  action
```

Before training: random actions.
After training: smooth, goal-directed navigation.

In `train_ppo.py` the policy is `MultiInputPolicy` — a CNN that processes
the costmap image plus MLP layers for the other inputs.

### Value Function
A neural network that predicts "how much total future reward will I get from
this state?" It is trained alongside the policy and used to reduce the
variance of learning updates.

```
observation  →  [value network]  →  expected future reward
```

In our training output you see `explained_variance` — how well the value
network's predictions match actual rewards. Closer to 1.0 = better.

### Return (Discounted)
The total reward over an episode, but future rewards are worth slightly less
than immediate ones:

```
Return = r₁ + γ·r₂ + γ²·r₃ + ...    (γ = 0.99 in our config)
```

`γ` (gamma) = 0.99 means a reward 100 steps in the future is worth
`0.99^100 ≈ 0.37` of an immediate reward. This makes the agent care about
long-term success without completely ignoring the present.

---

## 3. How Neural Networks Fit In

The policy is a neural network with this architecture in our project:

```
Costmap (84×84×1)  →  CNN (feature_extractor.py)  ─┐
Scan (360,)        →  MLP                           ├→  shared features (128d)
Goal vector (2,)   →  MLP                           │      │
Velocity (3,)      →  MLP                          ─┘      ├→ Policy head [256,256] → action (3,)
                                                            └→ Value head  [256,256] → value (1,)
```

During training:
1. Robot acts in the environment → collects experience (observations, actions, rewards)
2. PPO computes how much better/worse the actions were than expected
3. Gradients flow backward through the network
4. Weights update slightly to make good actions more likely

This repeats for 500,000 steps in our config.

---

## 4. PPO — The Algorithm We Use

**PPO (Proximal Policy Optimization)** is currently one of the most popular
RL algorithms. It was published by OpenAI in 2017 and is the default choice
for continuous control tasks (like robot navigation).

### Why PPO?

| Property         | PPO                                         |
|------------------|---------------------------------------------|
| Stability        | High — "proximal" = small, safe updates     |
| Sample efficiency| Medium — needs many steps but is reliable   |
| Implementation   | Relatively simple                           |
| Continuous actions | Native support (our [vx, vy, vyaw])       |

### How PPO Works (simplified)

**Step 1: Collect experience**
Run the current policy for `n_steps × n_envs` steps. In our config:
`2048 × 8 = 16,384 transitions` per update.

**Step 2: Compute advantage**
For each step, compute how much better the action was compared to what the
value network predicted. Positive advantage = better than expected.

```
Advantage = (actual return) − (predicted value)
```

**Step 3: Update the policy**
Nudge the policy to make high-advantage actions more likely. BUT — here is
the key PPO trick — **clip the update** so it doesn't change too much:

```
ratio = new_policy_prob / old_policy_prob
clipped_ratio = clip(ratio, 1−ε, 1+ε)   ε = 0.2 in our config
loss = −min(ratio × advantage, clipped_ratio × advantage)
```

The clip prevents catastrophic updates (one bad batch can't destroy a good
policy). This is what "proximal" means.

**Step 4: Repeat**
Do `n_epochs = 5` passes over the collected data, then go back to Step 1.

### Key hyperparameters in our `training_config.yaml`

| Parameter     | Value   | Meaning                                           |
|---------------|---------|---------------------------------------------------|
| `n_steps`     | 2048    | Steps collected per env before each update        |
| `batch_size`  | 64      | Mini-batch size for gradient updates              |
| `n_epochs`    | 5       | How many passes over collected data               |
| `gamma`       | 0.99    | Future reward discount factor                     |
| `gae_lambda`  | 0.95    | Bias/variance trade-off for advantage estimation  |
| `clip_range`  | 0.2     | Max policy change per update (ε above)            |
| `ent_coef`    | 0.01    | Entropy bonus — encourages exploration            |
| `learning_rate`| 3e-4   | How fast the network weights change               |

---

## 5. What is MuJoCo?

**MuJoCo** (Multi-Joint dynamics with Contact) is a physics simulator
originally developed at UMass and now maintained by Google DeepMind.
It is the standard simulator for robotics RL research.

### Why MuJoCo instead of Gazebo for training?

| Feature           | Gazebo                       | MuJoCo                     |
|-------------------|------------------------------|----------------------------|
| Physics speed     | ~5 steps/sec (real-time)     | 100,000+ steps/sec          |
| ROS2 required     | Yes                          | No                         |
| Headless training | Difficult                    | Native                     |
| Accuracy          | Very high (for deployment)   | High (sufficient for RL)   |
| Use case          | Final deployment/testing     | Fast training               |

The saved `.zip` model is **sim-agnostic** — we train in MuJoCo, deploy in
Gazebo/real robot. The robot only sees normalised observations, so it doesn't
know which simulator it's in.

### MuJoCo's place in the pipeline

```
[Train fast in MuJoCo]  →  [.zip policy file]  →  [Deploy in Gazebo / real robot]
     100k+ fps                sim-agnostic               existing ROS2 stack
```

---

## 6. MuJoCo Internals

Understanding MuJoCo means understanding a few key objects.

### MjModel
The **static description** of the world — loaded once from the XML file.
Contains geometry, masses, joint limits, etc. Never changes during simulation.

```python
model = mujoco.MjModel.from_xml_path('mujoco_maze.xml')
```

### MjData
The **dynamic state** of the simulation — changes every step.
Contains positions, velocities, forces, contacts, sensor readings.

```python
data = mujoco.MjData(model)
```

### Key MjData fields used in our project

| Field            | Shape     | Meaning                                    |
|------------------|-----------|--------------------------------------------|
| `data.qpos`      | (7,)      | Robot pose: [x, y, z, qw, qx, qy, qz]     |
| `data.qvel`      | (6,)      | Velocity: [vx, vy, vz, wx, wy, wz] world  |
| `data.ncon`      | scalar    | Number of active contacts                  |
| `data.contact`   | array     | Details of each contact (bodies, force)    |
| `data.site_xpos` | (n, 3)    | World position of each named site          |
| `data.site_xmat` | (n, 9)    | World rotation matrix of each site         |

### Free Joint
Our robot has a `<freejoint/>` — it can move freely in 3D space (like a
floating body). This is the simplest way to model a wheeled robot:

- No wheel joints needed
- We directly write velocity to `qvel` every step
- `qpos[0:3]` = world position, `qpos[3:7]` = quaternion orientation

### mj_forward vs mj_step

```python
mujoco.mj_forward(model, data)  # Recompute state without advancing time
                                 # Used after teleporting the robot (reset)

mujoco.mj_step(model, data)     # Advance physics by one timestep (0.05s)
                                 # Used every step during an episode
```

### mj_ray — How our LiDAR works
There is no physical LiDAR in MuJoCo. We simulate it in software:

```python
for i in range(360):
    angle = 2π * i / 360
    direction = [cos(angle), sin(angle), 0]   # horizontal ray in body frame
    direction_world = rotation_matrix @ direction
    distance = mujoco.mj_ray(model, data, lidar_pos, direction_world, ...)
```

360 rays × 0.05s timestep = the same data as a 20Hz real LiDAR.

### MJCF — The XML Format
MuJoCo worlds are described in **MJCF** (MuJoCo XML format):

```xml
<mujoco model="my_world">
  <option timestep="0.05" integrator="implicitfast"/>

  <worldbody>
    <geom name="floor" type="plane"/>               <!-- infinite floor -->

    <body name="wall_north" pos="0 12.5 1.25">
      <geom type="box" size="12.8 0.15 1.25"/>      <!-- half-extents! -->
    </body>

    <body name="base_link" pos="0 0 0.15">
      <freejoint/>                                   <!-- mobile robot -->
      <geom type="box" size="0.25 0.15 0.12"/>
      <site name="lidar_site" pos="0 0 0.175"/>      <!-- LiDAR position -->
    </body>
  </worldbody>
</mujoco>
```

Key difference from SDF (Gazebo): box `size` in MJCF is **half-extents**
(so `size="1 1 1"` = a 2×2×2 metre cube).

---

## 7. Gymnasium — The RL Environment API

**Gymnasium** (formerly OpenAI Gym) is the standard interface every RL
environment must follow. Stable-Baselines3 expects this API.

### The five methods every environment must implement

```python
class MyEnv(gym.Env):

    def reset(self, seed=None, options=None):
        """Start a new episode. Returns (observation, info)."""
        ...
        return obs, {}

    def step(self, action):
        """Apply action, advance simulation, return results."""
        ...
        return obs, reward, terminated, truncated, info

    def close(self):
        """Clean up resources."""
        pass
```

### terminated vs truncated

```
terminated = True   →  episode ended naturally (goal reached OR collision)
truncated  = True   →  episode ended by external limit (timeout OR stuck)
```

This distinction matters for computing correct returns: a truncated episode
has a non-zero value at the end (the robot could have continued), while a
terminated episode does not.

### Observation and Action Spaces

```python
# Our observation space
self.observation_space = gym.spaces.Dict({
    'costmap':     gym.spaces.Box(0, 255, (84, 84, 1), np.uint8),
    'scan':        gym.spaces.Box(0.0, 1.0, (360,), np.float32),
    'goal_vector': gym.spaces.Box(-1.0, 1.0, (2,), np.float32),
    'velocity':    gym.spaces.Box(-1.0, 1.0, (3,), np.float32),
})

# Our action space
self.action_space = gym.spaces.Box(-1.0, 1.0, (3,), np.float32)
```

These tell Stable-Baselines3 the shape and range of all inputs/outputs.
The policy network is automatically sized to match.

---

## 8. How This Project Uses All of the Above

### Full Training Loop (what happens during `train_ppo.py --sim mujoco`)

```
train_ppo.py
│
├── Creates 8 MuJoCoExplorerEnv instances (SubprocVecEnv — 8 parallel processes)
├── Creates 1 eval env
├── Wraps with VecTransposeImage (costmap: HWC → CHW for CNN)
├── Wraps with VecNormalize (normalises rewards)
│
└── model.learn(500_000 steps)
        │
        └── PPO loop:
              │
              ├── [COLLECT] 2048 steps × 8 envs = 16,384 transitions
              │     │
              │     └── For each step in each env:
              │           1. policy.predict(obs) → action [vx, vy, vyaw]
              │           2. env.step(action):
              │                a. Rotate body→world frame velocity
              │                b. Write to data.qvel
              │                c. mj_step()           ← physics advances 0.05s
              │                d. simulate_lidar()    ← 360 × mj_ray()
              │                e. lidar_to_costmap()  ← 84×84 grid
              │                f. build_observation() ← normalise all inputs
              │                g. compute_reward()    ← reward.py
              │                h. check termination   ← collision / goal / stuck
              │           3. Store (obs, action, reward, done, next_obs)
              │
              ├── [UPDATE] 5 epochs over 16,384 transitions in 64-sample batches
              │     │
              │     └── Compute advantages → clip → gradient update
              │
              ├── [EVAL] Every 25,000 steps: run 10 episodes in eval env
              │     └── Save best_model.zip if mean_reward improves
              │
              └── [CHECKPOINT] Every 50,000 steps: save ppo_step_N.zip
```

### Curriculum Learning (`curriculum.py`)

Instead of jumping straight to long-distance navigation, we start easy:

```
Stage 0 (Bootstrap) → Stage 1 (Easy) → Stage 2 (Medium) → Stage 3 (Hard)
  goals: 0.3–0.8m      0.8–2.0m         2.0–4.0m           3.0–6.0m
  steps: 50             100              200                 200
  advance when: 40% / 60% / 50% success rate over last 50 episodes
```

This is called **curriculum learning** — the same reason you learn arithmetic
before calculus. A random policy has a reasonable chance of accidentally
reaching a 0.4 m goal (stage 0) but zero chance of reaching a 5 m goal.

### Domain Randomization (`curriculum.py` stage 3)

In stage 3, the simulator deliberately adds noise to make the policy robust:
- **Velocity scaling** (0.8–1.2×): motors aren't perfectly calibrated
- **Action delay** (0–2 steps): control latency on real hardware
- **Scan noise**: LiDAR readings have noise in the real world

This is why a policy trained in simulation can work on a real robot — it has
seen enough variation to generalise.

---

## 9. Reading the Training Output

Here is what each number in the training console actually means:

```
-----------------------------------------
| curriculum/             |             |
|    stage                | 0           |  ← Which difficulty level (0–3)
| episode/                |             |
|    collision            | 0           |  ← Fraction of episodes ending in collision
|    goal_distance        | 0.462       |  ← Mean distance to goal at episode end (metres)
|    goal_reached         | 0           |  ← Fraction of episodes where goal was reached ✓
| safety/                 |             |
|    min_lidar_range      | 0.855       |  ← Closest the robot got to a wall (metres)
| time/                   |             |
|    fps                  | 164         |  ← Environment steps per second (target: >50)
|    iterations           | 3           |  ← Number of PPO update cycles done so far
|    total_timesteps      | 49152       |  ← Total steps taken across all envs
| train/                  |             |
|    explained_variance   | 0.491       |  ← Value network quality: 0=random, 1=perfect
|    entropy_loss         | -4.23       |  ← Exploration level (more negative = less random)
|    clip_fraction        | 0.0653      |  ← How often the PPO clip triggers (target: <0.3)
|    approx_kl            | 0.007       |  ← Policy change size (target: <0.05)
|    value_loss           | 0.373       |  ← How wrong the value network is (lower = better)
-----------------------------------------
```

### What good training looks like over time

| Metric                | Early training  | Healthy progress  | Converged     |
|-----------------------|-----------------|-------------------|---------------|
| `explained_variance`  | 0.0 – 0.3       | 0.5 – 0.8         | 0.8 – 0.95    |
| `entropy_loss`        | -4.3 (random)   | -4.0 to -3.5      | -3.0 to -2.5  |
| `goal_reached`        | 0.0             | 0.1 – 0.5         | 0.6+          |
| `goal_distance`       | 1.0+            | 0.3 – 0.6         | < 0.4         |
| `curriculum/stage`    | 0               | 0 → 1 → 2         | 3             |

---

## 10. Common Problems and What They Mean

### `goal_reached: 0` for a long time
**What it means:** The robot is learning but not quite reaching goals.
**Typical cause:** Goals are at the edge of what the policy can do.
**What to watch:** `goal_distance` trending downward + `explained_variance`
going up = healthy, just needs more time.

### `collision: high` (e.g. 0.8)
**What it means:** Robot hits walls constantly.
**Typical cause:** Too aggressive learning rate, or reward collision penalty
too low.
**Fix:** Training usually self-corrects. If it persists, check that
`min_lidar_range` is > 0.3 m before collisions.

### `explained_variance` stuck near 0
**What it means:** Value network can't predict returns at all.
**Typical cause:** Reward signal is too noisy or episodes too short.

### `clip_fraction` > 0.5 consistently
**What it means:** Policy is changing too fast — updates are too aggressive.
**Fix:** Reduce `learning_rate` or increase `batch_size` in config.

### `EOFError` on exit
**What it means:** Ctrl+C killed subprocess workers before clean shutdown.
**Is it a problem?** No. Model is saved before this error appears.

---

## 11. Hyperparameter Tuning Intuition

This section documents how to read training failures and know what to change.
Everything here was learned from actual failed runs on this project.

---

### Every metric is a symptom, not the problem

You read the dashboard like a doctor reading vitals:

| Metric | What it actually tells you |
|---|---|
| `entropy_loss` | How random the policy is. Still dropping fast = policy locking in prematurely. Still rising = exploration bonus dominating |
| `train/std` | Same as entropy but for continuous actions. >1.2 = nearly random. <0.3 = too deterministic |
| `explained_variance` | How well the value function predicts returns. <0.5 = value net is lost, policy cannot learn properly |
| `value_loss` | Is the value net converging? Flat + high = something is wrong with the reward signal itself |
| `clip_fraction` | How aggressively the policy is updating each step. >0.3 = updates too large, unstable |
| `goal_distance` trend | Is the policy actually improving? Oscillating = stuck in local optimum. Trending to 0 = working |

---

### The three core tensions in PPO

Almost every tuning problem is one of these three:

**1. Exploration vs exploitation — controlled by `ent_coef`**

PPO's total loss is:
```
L = −policy_gradient_loss + value_coef × value_loss − ent_coef × entropy
```

The `ent_coef` term is *subtracted* — the policy is literally rewarded for being random. This means:

- Too low → policy collapses early, locks into the nearest local optimum before finding goals
- Too high → policy stays random, value function cannot learn from random actions
- The right value depends on your reward density — **richer reward signal = you need less entropy bonus**

Real example from this project:
- `ent_coef: 0.01` → entropy collapsed run 1 (−4.25 → −2.14), never reached goals
- `ent_coef: 0.03` → entropy slowed but still collapsed, 3 goal reaches in 22 iterations
- `ent_coef: 0.05` → entropy exploded upward (−4.78 → −5.50), std hit 1.53, policy became pure noise
- `ent_coef: 0.02` → current target after adding near-goal reward shaping

**Why 0.05 worked in theory but failed in practice:** After adding the `near_goal` reward term (+3.0/step at close range), `VecNormalize` adapted its running stats to the higher per-episode reward magnitude. But the entropy loss is applied *after* normalization in SB3's loss computation — so the 0.05 entropy bonus became relatively larger compared to the scaled policy gradient. The policy was rewarded more for being random than for reaching goals.

**2. Credit assignment — controlled by episode length, `gamma`, `gae_lambda`**

The policy needs to know *which* action caused success. Long episodes with sparse rewards = the policy cannot tell.

- `max_steps` in each curriculum stage directly controls this. Stage 0 had `max_steps=50` — too short for a worst-case goal orientation, causing frequent timeouts even for physically reachable goals. Raised to 80.
- `gamma = 0.99` means a reward 100 steps in the future is worth `0.99^100 ≈ 0.37` of an immediate reward. Lower gamma = more myopic, learns faster but ignores long-term consequences.
- `gae_lambda = 0.95` trades off bias vs variance in advantage estimates. Lower = lower variance but more biased (underestimates long-horizon returns).

**3. Signal-to-noise in the reward**

The value function can only learn if the reward signal is meaningful. Flat reward landscapes = policy has no gradient to follow, converges to the nearest local optimum.

Real example: the binary `goal_reached` bonus (+10.0 at <0.5 m) left zero gradient in the 0.5–1.0 m approach zone. The progress reward `5.0 × (delta_dist / initial_dist)` only gives ~0.025/step for a 0.1 m move on a 0.8 m goal — too small to dominate entropy noise. The policy learned to hover at ~0.5 m because getting closer gave almost no extra signal.

Fix: add a near-goal shaping term that grows linearly in the 0–1.0 m zone:
```python
r_near_goal = 3.0 × max(0, 1 − goal_dist / 1.0)
```
At 0.5 m distance this contributes +1.5/step — large enough to distinguish "almost there" from "far away" and provide a gradient through the dead zone.

This is called **potential-based reward shaping** (Ng 1999) — it provably does not change the optimal policy, it only fills in gradient where the base reward was flat.

---

### How to build this intuition

**Read the SB3 loss computation once.** When you see the actual formula:
```
L = −policy_gradient + value_coef × value_loss − ent_coef × entropy
```
...it becomes obvious why raising `ent_coef` while also adding a larger reward signal can cause entropy to explode. It is not magic — it is algebra.

**Change one thing at a time.** When we went from run 2 to run 3, we changed `ent_coef` AND added `near_goal` shaping simultaneously. When 0.05 blew up it took extra reasoning to isolate the cause — the VecNormalize interaction was subtle.

**Keep a run log.** Not in TensorBoard — a plain text note. For each run: what you changed, what you expected, what actually happened. After 5–6 runs you pattern-match faster than any paper explains.

**Intentionally break things.** Set `ent_coef: 0.5` and watch the policy go random. Set it to `0.0` and watch it collapse in 50k steps. Watching metrics respond to known changes builds the mental map far faster than reading documentation.

---

### When to use an entropy schedule instead of a fixed value

If a fixed `ent_coef` is hard to get right, SB3 accepts a callable:

```python
from stable_baselines3.common.utils import get_linear_fn

ent_coef_schedule = get_linear_fn(start=0.05, end=0.005, end_fraction=1.0)
model = PPO(..., ent_coef=ent_coef_schedule)
```

This gives high exploration early (stage 0 bootstrap) and tighter actions by stage 2–3 (where precision matters more than exploration). Use this if a static value either collapses early or stays random too long.

---

### Papers that become readable after real failures

| Paper | Why it matters | When to read it |
|---|---|---|
| PPO (Schulman 2017) — arxiv 1707.06347 | Explains clip fraction and why small updates matter | After seeing clip_fraction > 0.3 |
| GAE (Schulman 2015) — arxiv 1506.02438 | Explains the bias/variance trade-off in `gae_lambda` | After struggling with sparse rewards |
| Reward shaping (Ng 1999) | Proves potential-based shaping doesn't change the optimal policy | After adding any shaped reward term |
| 37 PPO tricks (ICLR blog 2022) | What actually makes PPO work in practice | After your first few failed runs |

---

## 12. Learning Resources

### MuJoCo

| Resource | URL | What you'll learn |
|----------|-----|-------------------|
| Official docs | https://mujoco.readthedocs.io/en/stable/ | Everything — XML format, Python API, physics |
| Python bindings | https://mujoco.readthedocs.io/en/stable/python.html | MjModel, MjData, mj_step, mj_ray |
| Interactive tutorial (Colab) | https://colab.research.google.com/github/deepmind/dm_control/blob/main/dm_control/mujoco/tutorial.ipynb | Hands-on MuJoCo in your browser |
| GitHub | https://github.com/google-deepmind/mujoco | Source, issues, examples |

### Reinforcement Learning Fundamentals

| Resource | URL | What you'll learn |
|----------|-----|-------------------|
| Spinning Up (OpenAI) | https://spinningup.openai.com/en/latest/spinningup/rl_intro.html | RL from scratch — policy, reward, value, return |
| Hugging Face Deep RL Course | https://huggingface.co/learn/deep-rl-course/en/unit0/introduction | Free 8-unit course with code notebooks |
| Sutton & Barto textbook | https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf | The foundational RL textbook (free PDF) |
| David Silver's UCL lectures | http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html | 10-lecture university course from a top researcher |

### PPO Specifically

| Resource | URL | What you'll learn |
|----------|-----|-------------------|
| Original PPO paper | https://arxiv.org/abs/1707.06347 | The algorithm itself |
| Hugging Face PPO explained | https://huggingface.co/blog/deep-rl-ppo | Clear walkthrough of how PPO works |
| Spinning Up PPO guide | https://spinningup.openai.com/en/latest/algorithms/ppo.html | Implementation details |
| 37 PPO implementation tricks | https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/ | What actually makes PPO work in practice |

### Stable-Baselines3 (the library we use)

| Resource | URL | What you'll learn |
|----------|-----|-------------------|
| Official docs | https://stable-baselines3.readthedocs.io/ | Full API reference |
| Quick start | https://stable-baselines3.readthedocs.io/en/master/guide/quickstart.html | Train your first agent in 10 lines |
| Examples | https://stable-baselines3.readthedocs.io/en/master/guide/examples.html | Ready-to-run training scripts |
| RL Baselines3 Zoo | https://github.com/DLR-RM/rl-baselines3-zoo | Pre-trained models, hyperparameter tuning |

### Gymnasium (environment API)

| Resource | URL | What you'll learn |
|----------|-----|-------------------|
| Official docs | https://gymnasium.farama.org/ | How to build and use RL environments |
| Environment API | https://gymnasium.farama.org/api/env/ | reset(), step(), spaces |

### Videos

| Channel / Series | URL | Best for |
|-----------------|-----|---------|
| Hugging Face Deep RL Course | https://huggingface.co/learn/deep-rl-course | Structured learning, best starting point |
| Machine Learning with Phil | https://www.youtube.com/@PhilTabor | PPO, SAC, TD3 code walkthroughs |
| Two Minute Papers | https://www.youtube.com/@TwoMinutePapers | Staying up-to-date with RL research |

### Suggested Learning Order (complete beginner)

1. **Week 1** — Read Spinning Up RL Intro + watch first 2 Hugging Face course units
2. **Week 2** — Do Hugging Face units 3–5 (PPO, Actor-Critic)
3. **Week 3** — Read the MuJoCo Python docs + run the Colab tutorial
4. **Week 4** — Read `mujoco_env.py` and `train_ppo.py` in this project — you'll understand every line

---

*Generated for the ros2-autonomous-explorer project — 2026-03-29*
