"""
Classify what the trained agent is doing on the boundary.

Given the position diagnostic already showed mean ||pos||/R = 0.89 and 92% of
steps near the surface, the question now is *which* boundary behavior:

  sitting    — agent picks one point on the sphere and barely moves
               (low speed, low position variance, low angular spread)
  sliding    — agent moves along the surface, but in a roughly fixed direction
               (high speed, position changes, angular spread grows over time)
  orbiting   — agent traces a closed path on the surface
               (high speed, ||pos|| nearly constant, returns to start)
  bouncing   — agent repeatedly hits the surface and rebounds inward
               (high std of ||pos||/R)

We compute per-episode summaries and label each episode. The headline is the
modal label across episodes.
"""

import argparse
import os
import numpy as np

from dodge_env import DodgeEnv, CHECKPOINT_PATH, BEST_CHECKPOINT_PATH


def run_episodes(checkpoint_path, num_episodes, seed=0):
    """Run N deterministic episodes; return per-step positions + velocities."""
    from ppo_agent import load_agent
    from watch_agent import pick_action

    agent = load_agent(checkpoint_path)
    env = DodgeEnv(render_mode=None)
    episodes = []
    for ep in range(num_episodes):
        obs, _ = env.reset(seed=seed + ep)
        positions = []
        velocities = []
        while True:
            positions.append(env.agent_pos.copy())
            velocities.append(env.agent_vel.copy())
            action = pick_action(agent, obs, deterministic=True)
            obs, _r, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
        episodes.append({
            "positions": np.stack(positions, axis=0),
            "velocities": np.stack(velocities, axis=0),
            "length": len(positions),
            "terminated_by_hit": bool(terminated),
        })
    env.close()
    return episodes


def classify_episode(ep, R, surface_threshold=0.85, motion_threshold=2.0):
    """Label one episode based on its trajectory statistics."""
    pos = ep["positions"]
    vel = ep["velocities"]
    if len(pos) < 30:
        return "too_short"

    # Strip the initial "transit from origin to boundary" — first ~30 steps.
    pos = pos[30:]
    vel = vel[30:]
    if len(pos) < 20:
        return "too_short"

    radii = np.linalg.norm(pos, axis=1)
    norm_radii = radii / R
    speed = np.linalg.norm(vel, axis=1)

    # Mostly-near-surface check: if it spent most of the post-transit time
    # well inside, the agent isn't actually boundary-camping.
    frac_near_surface = float((norm_radii >= surface_threshold).mean())
    if frac_near_surface < 0.5:
        return "interior"

    # Position spread: how far the agent wandered.
    pos_center = pos.mean(axis=0)
    pos_spread = float(np.linalg.norm(pos - pos_center, axis=1).mean())
    # Angular spread on the surface: how much the unit direction changes.
    dirs = pos / (radii[:, None] + 1e-8)
    mean_dir = dirs.mean(axis=0)
    mean_dir_norm = float(np.linalg.norm(mean_dir))   # 1 = always same dir, 0 = uniform
    # Radial variability: bouncing vs sliding.
    radial_std = float(np.std(norm_radii))

    mean_speed = float(speed.mean())

    if radial_std > 0.08:
        return "bouncing"
    if mean_speed < motion_threshold:
        return "sitting"
    # On the surface, moving fast, low radial variability.
    if mean_dir_norm > 0.85:
        # Always in roughly the same direction → sliding without orbiting
        return "sliding"
    return "orbiting"


def main(checkpoint_path, num_episodes, seed):
    episodes = run_episodes(checkpoint_path, num_episodes, seed)
    R = 200.0  # ARENA_RADIUS; could pull from env if it ever varies

    labels = [classify_episode(ep, R) for ep in episodes]
    lengths = [ep["length"] for ep in episodes]

    print(f"=== Trajectory classifier ===")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Episodes:   {num_episodes}    seed: {seed}")
    print()
    counts = {}
    for lab in labels:
        counts[lab] = counts.get(lab, 0) + 1
    for lab, c in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {lab:>11}  {c:3d} / {num_episodes}  ({100*c/num_episodes:.0f}%)")

    print()
    print(f"Mean ep length: {np.mean(lengths):.1f}   "
          f"std {np.std(lengths):.1f}   "
          f"min {min(lengths)}   max {max(lengths)}")
    print()
    print("Per-episode detail (post-transit-window, first 30 steps stripped):")
    print(f"  {'ep':>3}  {'len':>5}  {'label':>10}  "
          f"{'mean_speed':>10}  {'mean_||pos||/R':>14}  "
          f"{'radial_std':>10}  {'dir_concentration':>17}")
    for i, ep in enumerate(episodes):
        pos = ep["positions"][30:]
        vel = ep["velocities"][30:]
        if len(pos) < 20:
            print(f"  {i:>3}  {ep['length']:>5}  {'too_short':>10}")
            continue
        radii = np.linalg.norm(pos, axis=1)
        nr = radii / R
        speed = np.linalg.norm(vel, axis=1)
        dirs = pos / (radii[:, None] + 1e-8)
        mean_dir_norm = float(np.linalg.norm(dirs.mean(axis=0)))
        print(f"  {i:>3}  {ep['length']:>5}  {labels[i]:>10}  "
              f"{speed.mean():>10.2f}  {nr.mean():>14.3f}  "
              f"{np.std(nr):>10.3f}  {mean_dir_norm:>17.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    ckpt = args.checkpoint or (
        BEST_CHECKPOINT_PATH if os.path.exists(BEST_CHECKPOINT_PATH) else CHECKPOINT_PATH
    )
    main(ckpt, args.episodes, args.seed)
