"""
Watch a trained PPO agent play DodgeEnv.

Two modes:
  --mode human  -> opens a pygame window (run this locally on your machine)
  --mode gif    -> saves a GIF of one episode to <out> (works headless)
"""

import argparse
import numpy as np
import torch

from dodge_env import DodgeEnv
from ppo_agent import load_agent, CHECKPOINT_PATH


def run_episode(agent, render_mode, seed=None, deterministic=True, max_steps=1000):
    """Run one episode. Returns (steps, frames). frames is a list of HxWx3 uint8 arrays
    if render_mode == 'rgb_array', else None."""
    env = DodgeEnv(render_mode=render_mode)
    obs, _ = env.reset(seed=seed)
    frames = [] if render_mode == "rgb_array" else None
    steps = 0
    while True:
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits = agent.actor(agent.trunk(obs_t))
            if deterministic:
                action = int(logits.argmax(dim=-1).item())
            else:
                from torch.distributions import Categorical
                action = int(Categorical(logits=logits).sample().item())

        obs, _, terminated, truncated, _ = env.step(action)
        steps += 1

        if render_mode is not None:
            frame = env.render()
            if frames is not None and frame is not None:
                frames.append(frame)

        if terminated or truncated or steps >= max_steps:
            break

    env.close()
    return steps, frames


def save_gif(frames, path, fps=30):
    """Save a list of HxWx3 uint8 frames as a GIF. Subsamples to keep file size sane."""
    from PIL import Image
    # GIFs above ~15 FPS get bloated; downsample to ~15 FPS.
    keep_every = max(1, fps // 15)
    keep = frames[::keep_every]
    imgs = [Image.fromarray(f) for f in keep]
    imgs[0].save(
        path,
        save_all=True,
        append_images=imgs[1:],
        duration=int(1000 / (fps / keep_every)),
        loop=0,
        optimize=False,
    )
    print(f"Saved {len(keep)} frames to {path}")


def evaluate(agent, num_episodes=20):
    """Headless eval: report mean episode length."""
    lengths = []
    for ep in range(num_episodes):
        steps, _ = run_episode(agent, render_mode=None, seed=ep, deterministic=True)
        lengths.append(steps)
    lengths = np.array(lengths)
    print(f"Eval over {num_episodes} episodes: "
          f"mean={lengths.mean():.1f}  std={lengths.std():.1f}  "
          f"min={lengths.min()}  max={lengths.max()}")
    return lengths


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=CHECKPOINT_PATH)
    parser.add_argument("--mode", choices=["human", "gif", "eval"], default="human",
                        help="human=live window, gif=save GIF, eval=headless stats")
    parser.add_argument("--out", default="trained_agent.gif")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=20,
                        help="how many episodes for eval mode")
    parser.add_argument("--stochastic", action="store_true",
                        help="sample actions instead of taking argmax")
    args = parser.parse_args()

    agent = load_agent(args.checkpoint)

    if args.mode == "eval":
        evaluate(agent, num_episodes=args.episodes)
    elif args.mode == "human":
        steps, _ = run_episode(
            agent, render_mode="human", seed=args.seed,
            deterministic=not args.stochastic,
        )
        print(f"Episode lasted {steps} steps")
    elif args.mode == "gif":
        steps, frames = run_episode(
            agent, render_mode="rgb_array", seed=args.seed,
            deterministic=not args.stochastic,
        )
        print(f"Episode lasted {steps} steps. Rendering {len(frames)} frames -> {args.out}")
        save_gif(frames, args.out)
