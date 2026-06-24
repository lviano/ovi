# Gym Experiments
This directory contains the code for reproducing the gymnasium experiments comparing four
imitation learning algorithms in the limited expert-data regime.

---

## Commands used to generate the plots

### Acrobot-v1

```bash
python run_traj_experiment.py \
  --env Acrobot-v1 \
  --subsampling_freq 15 \
  --l_alg4 50 \
  --algs alg4 alg5 alg6 alg7 \
  --n_trajs 1 2 3 5 7 10 \
  --sizes large small verysmall veryverysmall tiny \
```

### CartPole-v1

```bash
python run_traj_experiment.py \
  --env CartPole-v1 \
  --subsampling_freq 25 \
  --l_alg4 50 \
  --algs alg4 alg5 alg6 alg7 \
  --n_trajs 1 2 3 5 7 10 \
  --sizes large small verysmall veryverysmall tiny \
```
### Pendulum-v1

```
python run_pendulum_experiment.py \                                              
  --subsampling_freq 5 \
  --algs alg4 alg5 alg6 alg7 \
  --n_trajs 1 3 5 7 10 15 --sizes large small verysmall veryverysmall tiny
```

### LunarLander-v2

```
python run_traj_experiment.py \
    --env LunarLander-v2 \ 
    --subsampling_freq 5 \
    --n_trajs 2 3 5 10 15 \
    --sizes large small verysmall veryverysmall tiny \
    --l_alg4 50
```

---

## Action space discretization (Pendulum-v1)

Pendulum-v1 has a continuous torque action in `[-2, 2]`.  The
`DiscretizedPendulumWrapper` (in `pendulum_wrapper.py`) converts this into a
`Discrete(n_bins)` space so the same discrete-action IL algorithms (Alg 4–7)
can be applied without modification.

**Mapping:** action index `i` → torque `= np.linspace(-2, 2, n_bins)[i]`

With the default `n_bins = 11` the 11 discrete actions correspond to torques

```
-2.0  -1.6  -1.2  -0.8  -0.4  0.0  0.4  0.8  1.2  1.6  2.0   (N·m)
```

giving a uniform step size of **0.4 N·m**.  The observation space
`(cos θ, sin θ, θ̇)` is left unchanged.

---

## Algorithm parameters

Hyperparameters that are identical across all environments are set in
`run_traj_experiment.py`.  Parameters that differ by environment are specified in
`env_configs.py` and loaded automatically at runtime.

### Parameters shared across all environments

| Parameter | Value | Description |
|---|---|---|
| `eta` | 0.1 | Mirror-ascent / softmax step size (η) |
| `pi_lr` | 1e-3 | Adam learning rate for policy networks |
| `pi_steps` | 10 | Gradient steps per policy update for Alg 4/5 |
| `seeds` | 0–49 | 50 random seeds |
| `n_eval_episodes` | 10 | Episodes used to evaluate each trained policy |
| **Large network** | [64, 64] | 2 hidden layers of 64 units (realizable class) |
| **Small network** | [16, 16] | 2 hidden layers of 16 units (misspecified class) |

### Per-environment parameters (`env_configs.py`)

| Parameter | Acrobot-v1 | CartPole-v1 | Pendulum-v1 | Description |
|---|---|---|---|---|
| `q_lr` | 1e-3 | 1e-3 | 1e-3 | Adam learning rate for Q networks (Alg 4 & 5) |
| `q_steps` | 10 | 10 | 10 | Gradient steps per Q update (Alg 4 & 5) |
| `bc_steps` | 100 | 100 | **1500** | Gradient steps for Alg 6 (BC) |
| `dagger_pi_steps` | 10 | 20 | **50** | Gradient steps per outer iter for Alg 7 (DAgger) |
| `k_alg5` | 100 | 100 | **200** | Outer iterations for Alg 5 (SPOIL) |


### Algorithm 4 – Interactive SPOIL

Interactive IL algorithm with Q^{π_E} realizability.
At each outer iteration the learner collects data by rolling out its current policy in the
environment (with expert labels queried on-the-fly), mixed with samples from the offline
expert dataset.  A Q network and policy are then updated for L inner steps on the aggregated
dataset.

| Parameter | Acrobot-v1 / CartPole-v1 | Pendulum-v1 |
|---|---|---|
| Outer iterations K | N (number of trajectories) | N (number of trajectories) |
| Inner loop L | **50** (set via `--l_alg4 50`) | **50** (set via `--l_alg4 50`) |
| Q learning rate (`q_lr`) | 1e-3 | 1e-3 |
| Q gradient steps (`q_steps`) | 10 | 10 |
| Policy update mode | softmax (default) | softmax (default) |
| Data split per outer iter | 50% learner rollout / 50% offline expert dataset | 50% learner rollout / 50% offline expert dataset |
| Dataset aggregation | cumulative (DAgger-style) | cumulative (DAgger-style) |

### Algorithm 5 – SPOIL (offline)

Offline variant of Algorithm 4.  Uses a fixed pre-collected expert dataset throughout;
no additional environment interaction.  Alternates between Q updates and policy updates
for K outer iterations.

| Parameter | Acrobot-v1 / CartPole-v1 | Pendulum-v1 |
|---|---|---|
| Outer iterations K | 100 | **200** |
| Q learning rate (`q_lr`) | 1e-3 | 1e-3 |
| Q gradient steps (`q_steps`) | 10 | 10 |
| Policy update mode | softmax (default) | softmax (default) |

### Algorithm 6 – BC (Behavioural Cloning)

Single MLE pass over the offline expert dataset.

| Parameter | Acrobot-v1 / CartPole-v1 | Pendulum-v1 |
|---|---|---|
| Gradient steps | 100 | **1500** |
| Loss | Cross-entropy (log-likelihood) | Cross-entropy (log-likelihood) |

### Algorithm 7 – DAgger

DAgger with a decaying expert mixture.  At iteration k the environment is rolled out under
the mixture policy `(1−(1−α)^k) π^k + (1−α)^k π_E`; the expert is queried at every visited
state for its action label.  A fresh policy is then fitted via MLE on the aggregated dataset.

| Parameter | Acrobot-v1 / CartPole-v1 | Pendulum-v1 |
|---|---|---|
| Outer iterations K | N (number of trajectories) | N (number of trajectories) |
| Mixture decay α | 0.1 | 0.1 |
| Gradient steps per iter | 10 | **50** |

---

## Expert training

Experts are trained once and cached in `expert_model/`.

| Environment | Algorithm | Timesteps | Notes |
|---|---|---|---|
| Acrobot-v1 | PPO | 1 000 000 | 8 parallel envs, entropy coef 0.01 |
| CartPole-v1 | PPO | 1 000 000 | 8 parallel envs, entropy coef 0.01 |
| Pendulum-v1 | DQN | 300 000 | Trained on discretized env (11 bins); no reward shaping needed |

All experts use an MLP with 2 hidden layers of 64 units and tanh activations.
PPO uses `n_steps=2048`, `batch_size=64`, `n_epochs=10`, `lr=3e-4`, `γ=0.99`,
`gae_λ=0.95`, `clip_range=0.2`.
DQN (Pendulum-v1) uses `lr=1e-3`, `batch_size=64`, `buffer_size=50 000`, `γ=0.99`,
`train_freq=4`, `target_update_interval=500`, `exploration_fraction=0.3`,
`exploration_final_eps=0.05`.

---

## Offline dataset

A single large offline dataset is collected once per experiment (fixed seed 0) and cached
in `results/offline_dataset_{env}_{Nmax}_sub{freq}.pt`.  Smaller trajectory budgets are
obtained by taking strict prefixes of this dataset, ensuring nested subsets.

---

## Subsampling

`--subsampling_freq k` records only every k-th `(state, expert_action)` pair within each
trajectory.  This makes the imitation learning task harder while keeping the same trajectory
budget, and is intended to simulate covariate shift.

---

## Replotting saved results

`replot.py` regenerates figures from the per-config JSON files saved in `results/` without
re-running any training.  This is useful for adjusting plot style or adding new baselines
to an existing experiment.

```bash
# Regenerate the plot for a specific environment and subsampling frequency
python replot.py --env Acrobot-v1 --subsampling_freq 15

# Regenerate plots for all experiments found in results/
python replot.py
```

## Generating LaTeX figures

### `make_latex_figure.py` – IL comparison grid

Generates a compilable LaTeX/pgfplots document (`results/combined_figure.tex`) containing
a **2 × 4 grid** of normalized-return vs. number-of-trajectories plots, one column per
environment and one row per learner network size.

```bash
python make_latex_figure.py
pdflatex results/combined_figure.tex
```

Row labels are controlled by `--rows`; the default produces two rows:
- **Row 1** (`large`, 64×64 network): policy class contains the expert (`π_E ∈ Π_E`)
- **Row 2** (`small`, 16×16 network): misspecified policy class

| Flag | Default | Description |
|---|---|---|
| `--rows SIZE ...` | `large small` | Size keys to use as rows, in order. Valid keys: `large`, `small`, `verysmall`, `veryverysmall`, `tiny`. |
| `--divide` | off | Use standard error of the mean (std/√n) for shaded bands instead of std. |
| `--max_seeds N` | all | Cap the number of seeds used per (N, config) entry. |

The script reads per-config JSON files from `results/` (as produced by `run_traj_experiment.py`)
and normalizes returns between the uniform-random-policy baseline and the expert return.

### `make_latex_scaling_plot.py` – capacity scaling figure

Generates a compilable LaTeX/pgfplots document (`results/size_scaling_figure.tex`) containing
a **1 × 4 grid** of normalized-return vs. learner network hidden-layer width plots, one
column per environment.  The x-axis is log-scaled over hidden sizes 2, 4, 8, 16, 64; all
plots are evaluated at a fixed trajectory budget of **N = 10**.

```bash
python make_latex_scaling_plot.py --divide
pdflatex results/size_scaling_figure.tex
```

| Flag | Default | Description |
|---|---|---|
| `--divide` | off | Use standard error of the mean (std/√n) for shaded bands instead of std. |

The script reads the same per-config JSON files as `make_latex_figure.py` and applies the
same normalization scheme.

In particular, for the figures in the paper do
```
python make_latex_figure.py --divide --rows large small verysmall veryverysmall tiny
```
and, then
```
pdflatex results/combined_figure.tex
```

For the size vs suboptimality experiment. First create the latex code by
```
python make_latex_scaling_plot.py --divide --rows large small verysmall veryverysmall tiny                                     
```
and then compile as
``` 
pdflatex results/size_scaling_figure.tex
```
---

### Options

| Flag | Default | Description |
|---|---|---|
| `--env` | *(auto-discover)* | Environment name, e.g. `Acrobot-v1`. If omitted, all experiments in `results/` are processed. |
| `--subsampling_freq` | 1 | Must match the value used when the experiment was run. |
| `--n_trajs` | 1 2 3 5 7 10 30 50 100 | Trajectory counts to show on the x-axis. Values not present in the saved data are silently skipped. |

Output files (`{env_tag}{freq_tag}.pdf` and `.png`) are written to `results/`.

---

## File overview

| File | Description |
|---|---|
| `run_traj_experiment.py` | Main experiment script: trains/loads expert, sweeps over trajectory counts and seeds, saves per-config JSON results, plots. |
| `run_pendulum_experiment.py` | Pendulum-v1 experiment script: wraps the continuous action space, then runs the same IL sweep. |
| `algorithms.py` | Implementations of Algorithms 4–7. |
| `expert.py` | Expert training (PPO / DQN), loading, data collection utilities. |
| `env_configs.py` | Per-environment hyperparameter overrides (bc_steps, dagger_pi_steps, k_alg5). |
| `networks.py` | `PolicyNet` and `QNet` MLP definitions. |
| `pendulum_wrapper.py` | `DiscretizedPendulumWrapper`: converts Pendulum-v1's continuous torque space into `Discrete(n_bins)`. |
| `replot.py` | Regenerates plots from saved JSON results. |
