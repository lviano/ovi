#!/usr/bin/env python3
"""
compute_uniform_returns.py
--------------------------
Evaluates the uniform random policy in each environment and writes the
mean returns into the UNIFORM_RETURN dict inside make_latex_figure.py.

Usage:
  python compute_uniform_returns.py [--episodes 500] [--seed 0]
"""

import argparse
import re
import numpy as np
import gymnasium as gym

# Map from the key used in make_latex_figure.py to the gymnasium env id
ENV_MAP = {
    "Acrobot_v1":     "Acrobot-v1",
    "CartPole_v1":    "CartPole-v1",
    "MountainCar_v0": "MountainCar-v0",
    "LunarLander_v2": "LunarLander-v2",
}

MAKE_LATEX_PATH = "make_latex_figure.py"


def make_env(env_id: str) -> gym.Env:
    try:
        return gym.make(env_id)
    except gym.error.DeprecatedEnv:
        # Try bumping the version number by 1
        import re as _re
        newer = _re.sub(r"v(\d+)$", lambda m: f"v{int(m.group(1)) + 1}", env_id)
        print(f"  {env_id} is deprecated, falling back to {newer}")
        return gym.make(newer)


def evaluate_uniform(env_id: str, n_episodes: int, seed: int) -> float:
    env = make_env(env_id)
    env.reset(seed=seed)
    returns = []
    for ep in range(n_episodes):
        obs, _ = env.reset()
        total, done = 0.0, False
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            total += reward
            done = terminated or truncated
        returns.append(total)
        if (ep + 1) % 100 == 0:
            print(f"  {env_id}  episode {ep+1}/{n_episodes}  "
                  f"running mean={np.mean(returns):.2f}")
    env.close()
    return float(np.mean(returns))


def patch_make_latex(results: dict[str, float]) -> None:
    with open(MAKE_LATEX_PATH, "r") as f:
        src = f.read()

    for key, value in results.items():
        # Replace:  "Acrobot_v1":    None,
        # With:     "Acrobot_v1":    -487.23,
        pattern = rf'("{key}":\s*)None'
        replacement = rf'\g<1>{value:.4f}'
        src, n = re.subn(pattern, replacement, src)
        if n == 0:
            print(f"  WARNING: could not find entry for {key!r} in {MAKE_LATEX_PATH}")

    with open(MAKE_LATEX_PATH, "w") as f:
        f.write(src)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=500,
                        help="Number of episodes per environment (default: 500)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    results = {}
    for key, env_id in ENV_MAP.items():
        print(f"\nEvaluating uniform policy: {env_id} ({args.episodes} episodes) ...")
        mean_return = evaluate_uniform(env_id, args.episodes, args.seed)
        print(f"  => mean return = {mean_return:.4f}")
        results[key] = mean_return

    print(f"\nPatching {MAKE_LATEX_PATH} ...")
    patch_make_latex(results)
    print("Done. Re-run:")
    print("  python make_latex_figure.py && cd results && pdflatex combined_figure.tex")


if __name__ == "__main__":
    main()
