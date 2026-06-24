"""
evaluate_epsilon_policy.py
--------------------------
Evaluate a mixed policy on MountainCar-v0:
  - with probability epsilon=0.05: uniform random action
  - with probability 1-epsilon=0.95: expert action (from expert_model)

Runs 200 episodes and reports mean ± std return.
"""

import numpy as np
import gymnasium as gym

from expert import load_expert, get_expert_action, evaluate_policy_fn

ENV_NAME = "MountainCar-v0"
EPSILON = 0.0
N_EPISODES = 200
EXPERT_PATH = "expert_model"


def make_epsilon_policy(expert_model, env: gym.Env, epsilon: float):
    def policy(obs: np.ndarray) -> int:
        if np.random.random() < epsilon:
            return int(env.action_space.sample())
        return get_expert_action(expert_model, obs, deterministic=True)
    return policy


def main():
    expert_model = load_expert(save_path=EXPERT_PATH, env_name=ENV_NAME)
    env = gym.make(ENV_NAME)

    policy = make_epsilon_policy(expert_model, env, EPSILON)

    returns = []
    for ep in range(N_EPISODES):
        obs, _ = env.reset()
        episode_return = 0.0
        done = False
        while not done:
            action = policy(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_return += reward
            done = terminated or truncated
        returns.append(episode_return)
        if (ep + 1) % 20 == 0:
            print(f"  Episode {ep + 1}/{N_EPISODES} | running mean: {np.mean(returns):.2f}")

    env.close()

    returns = np.array(returns)
    print(f"\nResults over {N_EPISODES} episodes (epsilon={EPSILON}):")
    print(f"  Mean return : {returns.mean():.2f}")
    print(f"  Std  return : {returns.std():.2f}")
    print(f"  Min  return : {returns.min():.2f}")
    print(f"  Max  return : {returns.max():.2f}")


if __name__ == "__main__":
    main()
