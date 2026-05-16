"""
Position diagnostic for the env.

What we want to know: is the agent's spatial distribution symmetric and
*not pinned to the boundary*?

There are no corners or walls in this arena — only the inner surface of
a sphere. The signals we track:

  - mean ||pos|| / R              (0 = always at center, 1 = always at surface)
  - histogram of ||pos|| / R      (the *shape* — flat / unimodal / pinned)
  - mean direction vector / R     (should be ~0 along all axes if symmetric)
  - % steps within margin of surface

Compared baselines you'd expect:
  - A pure-random walk in 3D drifts but does not pin: the radial-distance
    distribution should be roughly unimodal, peaking somewhere in the
    middle of the sphere.
  - A trained dodger should *also* not pin to the surface — pinning means
    the agent has nowhere left to retreat to and is being hit.
  - A reward-hacking agent that found a quiet spot will have its mean
    direction off-center (one axis nonzero) and a sharp histogram peak
    there.

Usage:
  # random baseline
  python3 position_diagnostic.py --policy random --episodes 30

  # trained checkpoint (uses checkpoints/ppo_dodge_best.pt by default)
  python3 position_diagnostic.py --policy ppo --episodes 30
  python3 position_diagnostic.py --policy ppo --checkpoint path/to.pt
"""

import argparse
import os
import numpy as np

from dodge_env import (
    DodgeEnv,
    CURRICULUM_PRESETS,
    CHECKPOINT_PATH,
    BEST_CHECKPOINT_PATH,
)


def _print_histogram(values, n_bins=10, low=0.0, high=1.0, width=40):
    """ASCII histogram of `values` over [low, high]."""
    counts, edges = np.histogram(values, bins=n_bins, range=(low, high))
    total = max(counts.sum(), 1)
    peak = max(counts.max(), 1)
    for i, c in enumerate(counts):
        bar = "#" * int(round(width * c / peak))
        pct = 100.0 * c / total
        print(f"  [{edges[i]:.2f}, {edges[i+1]:.2f})  {bar:<{width}}  "
              f"{c:6d}  ({pct:5.1f}%)")


def run_diagnostic(policy, num_episodes, checkpoint_path,
                   spawn_distance_min, spawn_distance_max,
                   surface_margin_frac=0.1, seed=0):
    """Roll out `num_episodes` and aggregate per-step positions.

    surface_margin_frac: a step counts as "near surface" if
        ||pos|| / R >= (1 - surface_margin_frac).
    """
    env = DodgeEnv(
        render_mode=None,
        spawn_distance_min=spawn_distance_min,
        spawn_distance_max=spawn_distance_max,
    )
    R = float(env.arena_radius)

    pick = _make_action_picker(env, policy, checkpoint_path)

    all_positions = []     # list of (3,) np.float32 per step, across episodes
    ep_lengths = []

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=seed + ep)
        steps = 0
        while True:
            all_positions.append(env.agent_pos.copy())
            action = pick(obs)
            obs, _r, terminated, truncated, _ = env.step(action)
            steps += 1
            if terminated or truncated:
                break
        ep_lengths.append(steps)

    env.close()

    positions = np.stack(all_positions, axis=0)        # (N, 3)
    radii = np.linalg.norm(positions, axis=1)          # (N,)
    norm_radii = radii / R
    mean_dir = positions.mean(axis=0) / R              # (3,)

    near_surface_mask = norm_radii >= (1.0 - surface_margin_frac)
    near_surface_frac = float(near_surface_mask.mean())

    ep_lengths = np.array(ep_lengths)

    print(f"\n=== Position diagnostic (policy={policy}) ===")
    print(f"Episodes: {num_episodes}   total steps: {len(positions)}")
    print(f"Spawn shell: {env.spawn_shell_min:.0f}..{env.spawn_shell_max:.0f}   "
          f"arena R: {R:.0f}   perception R: {env.perception_radius:.0f}")
    print(f"Mean ep length: {ep_lengths.mean():.1f}   "
          f"std {ep_lengths.std():.1f}   "
          f"min {ep_lengths.min()}   max {ep_lengths.max()}")
    print()
    print(f"||pos||      mean {radii.mean():7.1f}   std {radii.std():6.1f}   "
          f"max {radii.max():7.1f}   (R = {R:.0f})")
    print(f"||pos|| / R  mean {norm_radii.mean():7.3f}   "
          f"std {norm_radii.std():6.3f}   max {norm_radii.max():6.3f}")
    print(f"Mean direction / R   "
          f"x={mean_dir[0]:+.3f}  y={mean_dir[1]:+.3f}  z={mean_dir[2]:+.3f}")
    print(f"% steps within outer {surface_margin_frac*100:.0f}% of R "
          f"(near surface): {100*near_surface_frac:5.1f}%")
    print()
    print("Histogram of ||pos|| / R:")
    _print_histogram(norm_radii, n_bins=10, low=0.0, high=1.0, width=40)

    return {
        "mean_ep_len": float(ep_lengths.mean()),
        "mean_radius": float(radii.mean()),
        "mean_norm_radius": float(norm_radii.mean()),
        "mean_direction": mean_dir.tolist(),
        "near_surface_frac": near_surface_frac,
    }


def _make_action_picker(env, policy, checkpoint_path):
    """Return a callable obs -> action."""
    if policy == "random":
        def pick(_obs):
            return env.action_space.sample()
        return pick

    if policy == "ppo":
        # Deferred import so the diagnostic works with just numpy+gym when
        # we only want the random baseline.
        from ppo_agent import load_agent
        from watch_agent import pick_action
        agent = load_agent(checkpoint_path)
        if env.observation_space.shape[0] != agent.obs_dim:
            raise RuntimeError(
                f"Checkpoint obs_dim={agent.obs_dim} doesn't match env "
                f"obs_dim={env.observation_space.shape[0]}."
            )

        def pick(obs):
            return pick_action(agent, obs, deterministic=True)
        return pick

    raise ValueError(f"Unknown policy: {policy!r}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--policy", choices=["random", "ppo"], default="random")
    parser.add_argument("--checkpoint", default=None,
                        help="checkpoint path (used when --policy ppo). "
                             "Default: best, falling back to latest.")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--surface-margin", type=float, default=0.1,
                        help="fraction of R that counts as 'near surface'.")
    parser.add_argument("--curriculum", choices=list(CURRICULUM_PRESETS.keys()),
                        default="target")
    parser.add_argument("--spawn-distance-min", type=float, default=None)
    parser.add_argument("--spawn-distance-max", type=float, default=None)
    args = parser.parse_args()

    preset_min, preset_max = CURRICULUM_PRESETS[args.curriculum]
    spawn_min = (args.spawn_distance_min
                 if args.spawn_distance_min is not None else preset_min)
    spawn_max = (args.spawn_distance_max
                 if args.spawn_distance_max is not None else preset_max)

    ckpt = args.checkpoint
    if args.policy == "ppo" and ckpt is None:
        ckpt = BEST_CHECKPOINT_PATH if os.path.exists(BEST_CHECKPOINT_PATH) else CHECKPOINT_PATH

    run_diagnostic(
        policy=args.policy,
        num_episodes=args.episodes,
        checkpoint_path=ckpt,
        spawn_distance_min=spawn_min,
        spawn_distance_max=spawn_max,
        surface_margin_frac=args.surface_margin,
        seed=args.seed,
    )
