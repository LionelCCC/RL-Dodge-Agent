"""
Sanity check: run DodgeEnv with a random agent for a few episodes.
This is also our baseline. Anything we train must beat this.
"""

import numpy as np
from dodge_env import DodgeEnv


def run_random_agent(num_episodes=20, render=False):
    env = DodgeEnv(render_mode="human" if render else None)
    episode_lengths = []
    episode_rewards = []

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=ep)  # use ep as seed for reproducibility
        total_reward = 0.0
        steps = 0

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
    print(f"\n--- Summary over {num_episodes} episodes ---")
    print(f"Mean steps:   {lengths.mean():.1f}  (std {lengths.std():.1f})")
    print(f"Mean reward:  {rewards.mean():.1f}")
    print(f"Min / Max:    {lengths.min()} / {lengths.max()}")


if __name__ == "__main__":
    # First, headless: just get the numbers.
    run_random_agent(num_episodes=20, render=False)
