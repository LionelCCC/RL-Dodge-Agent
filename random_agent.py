"""
Sanity check: run DodgeEnv with a random agent for a few episodes.
This is also our baseline. Anything we train must beat this.

Pick the env with --env {2d, 3d}. With more actions in 3D (27 vs 9), a random
agent has more ways to do nothing useful — the 3D baseline is usually weaker.
"""

import argparse
import numpy as np

from env_factory import make_env, ENV_NAMES


def run_random_agent(env_name="2d", num_episodes=20, render=False):
    env = make_env(env_name, render_mode="human" if render else None)
    episode_lengths = []
    episode_rewards = []

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=ep)  # use ep as seed for reproducibility
        total_reward = 0.0
        steps = 0
        terminated = False

        while True:
            action = env.action_space.sample()  # uniformly random action
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1

            if render:
                env.render()
                if env.closed_by_user:
                    print("Window closed — stopping random baseline early.")
                    break

            if terminated or truncated:
                break

        episode_lengths.append(steps)
        episode_rewards.append(total_reward)
        print(f"Episode {ep+1:3d}: steps = {steps:4d}, reward = {total_reward:6.1f}, "
              f"died = {terminated}")
        if env.closed_by_user:
            break

    env.close()
    lengths = np.array(episode_lengths)
    rewards = np.array(episode_rewards)
    print(f"\n--- Summary over {num_episodes} episodes (env={env_name}) ---")
    print(f"Mean steps:   {lengths.mean():.1f}  (std {lengths.std():.1f})")
    print(f"Mean reward:  {rewards.mean():.1f}")
    print(f"Min / Max:    {lengths.min()} / {lengths.max()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--env", choices=list(ENV_NAMES), default="2d",
                        help="which env to baseline.")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--render", action="store_true",
                        help="open a window during the baseline run (slow).")
    args = parser.parse_args()
    run_random_agent(env_name=args.env, num_episodes=args.episodes, render=args.render)
