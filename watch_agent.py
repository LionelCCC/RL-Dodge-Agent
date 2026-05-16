"""
Watch a trained PPO agent play DodgeEnv.

Modes:
  --mode human  -> opens a pygame window. Close the X or hit ESC to stop.
                   Use --episodes N to play multiple back-to-back.
  --mode gif    -> saves a GIF of one episode (works headless)
  --mode eval   -> headless stats over N episodes
"""

import argparse
import os
import numpy as np
import torch

from dodge_env import (
    DodgeEnv,
    CURRICULUM_PRESETS,
    CHECKPOINT_PATH,
    BEST_CHECKPOINT_PATH,
    DEFAULT_GIF_PATH,
)
from ppo_agent import load_agent


def default_checkpoint():
    """Prefer the best-by-mean checkpoint; fall back to latest if missing."""
    return BEST_CHECKPOINT_PATH if os.path.exists(BEST_CHECKPOINT_PATH) else CHECKPOINT_PATH


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
            f"returns obs_dim={obs.shape[0]}. The env changed since this "
            f"checkpoint was trained — retrain or use a matching checkpoint."
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


def _resolve_spawn(spawn_min, spawn_max):
    smin_d, smax_d = CURRICULUM_PRESETS["target"]
    return (smin_d if spawn_min is None else spawn_min,
            smax_d if spawn_max is None else spawn_max)


def watch_live(agent, episodes, deterministic, seed,
               spawn_distance_min=None, spawn_distance_max=None):
    smin, smax = _resolve_spawn(spawn_distance_min, spawn_distance_max)
    env = DodgeEnv(render_mode="human",
                   spawn_distance_min=smin, spawn_distance_max=smax)
    try:
        for ep in range(1, episodes + 1):
            steps, _, user_quit = run_episode(
                env, agent, deterministic=deterministic, seed=seed + ep,
            )
            print(f"Episode {ep:3d}/{episodes}: survived {steps} steps")
            if user_quit:
                print("Window closed — stopping.")
                break
    finally:
        env.close()


def evaluate(agent, num_episodes=20,
             spawn_distance_min=None, spawn_distance_max=None):
    smin, smax = _resolve_spawn(spawn_distance_min, spawn_distance_max)
    env = DodgeEnv(render_mode=None,
                   spawn_distance_min=smin, spawn_distance_max=smax)
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


def make_gif(agent, out_path, seed, deterministic,
             spawn_distance_min=None, spawn_distance_max=None):
    smin, smax = _resolve_spawn(spawn_distance_min, spawn_distance_max)
    env = DodgeEnv(render_mode="rgb_array",
                   spawn_distance_min=smin, spawn_distance_max=smax)
    try:
        steps, frames, _ = run_episode(
            env, agent, deterministic=deterministic,
            collect_frames=True, seed=seed,
        )
    finally:
        env.close()
    # Make sure the output folder exists (assets/ by default).
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    print(f"Episode lasted {steps} steps. Rendering {len(frames)} frames -> {out_path}")
    save_gif(frames, out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="checkpoint to load. Defaults to the best (if it exists) else latest.",
    )
    parser.add_argument("--mode", choices=["human", "gif", "eval"], default="human",
                        help="human=live window, gif=save GIF, eval=headless stats")
    parser.add_argument("--out", default=None,
                        help=f"output path for --mode gif. Default: {DEFAULT_GIF_PATH}.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=5,
                        help="how many episodes (used by --mode human and --mode eval)")
    parser.add_argument("--stochastic", action="store_true",
                        help="sample actions instead of taking argmax")
    parser.add_argument("--curriculum", choices=list(CURRICULUM_PRESETS.keys()),
                        default="target", help="spawn-distance preset.")
    parser.add_argument("--spawn-distance-min", type=float, default=None,
                        help="override curriculum preset minimum.")
    parser.add_argument("--spawn-distance-max", type=float, default=None,
                        help="override curriculum preset maximum.")
    args = parser.parse_args()

    preset_min, preset_max = CURRICULUM_PRESETS[args.curriculum]
    spawn_distance_min = (
        args.spawn_distance_min if args.spawn_distance_min is not None else preset_min
    )
    spawn_distance_max = (
        args.spawn_distance_max if args.spawn_distance_max is not None else preset_max
    )

    checkpoint_path = args.checkpoint or default_checkpoint()
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Spawn distance: {spawn_distance_min:.0f}..{spawn_distance_max:.0f}")
    agent = load_agent(checkpoint_path)
    deterministic = not args.stochastic

    if args.mode == "eval":
        evaluate(
            agent, num_episodes=args.episodes,
            spawn_distance_min=spawn_distance_min,
            spawn_distance_max=spawn_distance_max,
        )
    elif args.mode == "human":
        watch_live(
            agent,
            episodes=args.episodes, deterministic=deterministic, seed=args.seed,
            spawn_distance_min=spawn_distance_min,
            spawn_distance_max=spawn_distance_max,
        )
    elif args.mode == "gif":
        out_path = args.out or DEFAULT_GIF_PATH
        make_gif(
            agent, out_path=out_path, seed=args.seed, deterministic=deterministic,
            spawn_distance_min=spawn_distance_min,
            spawn_distance_max=spawn_distance_max,
        )
