"""
Watch a trained PPO agent play DodgeEnv.

Modes:
  --mode human  -> opens a pygame window. Close the X or hit ESC to stop.
                   Use --episodes N to play multiple back-to-back.
  --mode gif    -> saves a GIF of one episode (works headless)
  --mode eval   -> headless stats over N episodes
"""

import argparse
import numpy as np
import torch

from dodge_env import DodgeEnv
from ppo_agent import load_agent, CHECKPOINT_PATH


def pick_action(agent, obs, deterministic=True):
    """Return one integer action given a (single) observation array."""
    obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        logits = agent.actor(agent.trunk(obs_t))
        if deterministic:
            return int(logits.argmax(dim=-1).item())
        from torch.distributions import Categorical
        return int(Categorical(logits=logits).sample().item())


def run_episode(env, agent, deterministic=True, max_steps=1000,
                collect_frames=False, seed=None):
    """Run one episode using the given (already-constructed) env.
    Returns (steps, frames_or_None, user_quit_bool)."""
    obs, _ = env.reset(seed=seed)
    if obs.shape[0] != agent.obs_dim:
        raise RuntimeError(
            f"Checkpoint expects obs_dim={agent.obs_dim}, but the current env "
            f"returns obs_dim={obs.shape[0]}. The environment changed; retrain "
            "from scratch with `python3 ppo_agent.py` before watching/evaluating."
        )
    frames = [] if collect_frames else None
    steps = 0
    user_quit = False
    while True:
        action = pick_action(agent, obs, deterministic=deterministic)
        obs, _, terminated, truncated, _ = env.step(action)
        steps += 1

        if env.render_mode is not None:
            frame = env.render()
            if frames is not None and frame is not None:
                frames.append(frame)
            if env.render_mode == "human" and env.closed_by_user:
                user_quit = True
                break

        if terminated or truncated or steps >= max_steps:
            break

    return steps, frames, user_quit


def save_gif(frames, path, fps=30):
    """Save a list of HxWx3 uint8 frames as a GIF. Subsamples to keep file size sane."""
    from PIL import Image
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


def watch_live(agent, episodes, deterministic, seed):
    """Open a window and play N episodes back-to-back. Stops cleanly on window close / ESC."""
    env = DodgeEnv(render_mode="human")
    try:
        for ep in range(1, episodes + 1):
            steps, _, user_quit = run_episode(
                env,
                agent,
                deterministic=deterministic,
                seed=seed + ep,  # different seed per episode for variety
            )
            print(f"Episode {ep:3d}/{episodes}: survived {steps} steps")
            if user_quit:
                print("Window closed — stopping.")
                break
    finally:
        env.close()


def evaluate(agent, num_episodes=20):
    """Headless eval: report mean episode length."""
    env = DodgeEnv(render_mode=None)
    lengths = []
    try:
        for ep in range(num_episodes):
            steps, _, _ = run_episode(env, agent, deterministic=True, seed=ep)
            lengths.append(steps)
    finally:
        env.close()
    lengths = np.array(lengths)
    print(f"Eval over {num_episodes} episodes: "
          f"mean={lengths.mean():.1f}  std={lengths.std():.1f}  "
          f"min={lengths.min()}  max={lengths.max()}")
    return lengths


def make_gif(agent, out_path, seed, deterministic):
    env = DodgeEnv(render_mode="rgb_array")
    try:
        steps, frames, _ = run_episode(
            env,
            agent,
            deterministic=deterministic,
            collect_frames=True,
            seed=seed,
        )
    finally:
        env.close()
    print(f"Episode lasted {steps} steps. Rendering {len(frames)} frames -> {out_path}")
    save_gif(frames, out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", default=CHECKPOINT_PATH)
    parser.add_argument("--mode", choices=["human", "gif", "eval"], default="human",
                        help="human=live window, gif=save GIF, eval=headless stats")
    parser.add_argument("--out", default="trained_agent.gif",
                        help="output path for --mode gif")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=5,
                        help="how many episodes (used by --mode human and --mode eval)")
    parser.add_argument("--stochastic", action="store_true",
                        help="sample actions instead of taking argmax")
    args = parser.parse_args()

    agent = load_agent(args.checkpoint)
    deterministic = not args.stochastic

    if args.mode == "eval":
        evaluate(agent, num_episodes=args.episodes)
    elif args.mode == "human":
        watch_live(agent, episodes=args.episodes, deterministic=deterministic, seed=args.seed)
    elif args.mode == "gif":
        make_gif(agent, out_path=args.out, seed=args.seed, deterministic=deterministic)
