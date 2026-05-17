"""
Threat-distribution diagnostic.

The trained agent slides along the spherical boundary. This script answers:
**Is the boundary actually safer than the center?** Or is the agent in a
spec-gaming local optimum that happens to be stable but not advantageous?

Method: place the agent at a fixed position, force it to take action 13
("stay still" — the all-zeros action) every step, and let the projectile
spawner run normally. Track:
  - mean survival steps (hit-truncated at MAX_EPISODE_STEPS=1000)
  - hit rate (fraction of "episodes" that ended in a collision)
  - mean closest-approach distance of projectiles that came within range

We compare several positions:
  - origin                       (0, 0, 0)
  - half-radius along +x         (R/2, 0, 0)
  - near boundary along +x       (R-AGENT_RADIUS, 0, 0)
  - the policy's favorite octant ~(+0.587, +0.538, +0.262) normalized * (R-AGENT_RADIUS)

If "near boundary" survives longer than "origin," the boundary is
objectively safer and we have a real env-design issue (the sphere isn't
actually symmetric *from the agent's reward-perspective*). If the boundary
is *worse* or *equal*, then the trained agent is in a local optimum that
doesn't reflect a real safety asymmetry — and the fix is curriculum, frame
randomization, or different exploration, not a shaping penalty.

Usage:
  python3 threat_diagnostic.py --episodes 100
"""

import argparse
import numpy as np

from dodge_env import DodgeEnv, ARENA_RADIUS, AGENT_RADIUS, PROJECTILE_RADIUS


STAY_STILL_ACTION = 13   # index of (0, 0, 0) in the 27-action grid


def run_fixed_position(agent_pos, num_episodes, seed, max_steps=1000):
    """Place the agent at `agent_pos`, force action 13 every step,
    run `num_episodes` simulated episodes. Return per-episode stats."""
    env = DodgeEnv(render_mode=None)
    lengths = []
    hits = 0
    closest_approaches = []   # one per episode: min(||proj - agent||) seen

    for ep in range(num_episodes):
        env.reset(seed=seed + ep)
        # Override the env's "agent starts at origin." This is safe because
        # we own the env state — no rendering or test-suite depends on it.
        env.agent_pos = np.array(agent_pos, dtype=np.float32).copy()
        env.agent_vel = np.zeros(3, dtype=np.float32)

        ep_closest = float("inf")
        steps = 0
        terminated = False
        for step in range(max_steps):
            _obs, _r, terminated, truncated, _ = env.step(STAY_STILL_ACTION)
            steps = step + 1
            # Closest approach over the lifetime of this episode.
            if env.projectiles:
                d = min(
                    float(np.linalg.norm(p["pos"] - env.agent_pos))
                    for p in env.projectiles
                )
                if d < ep_closest:
                    ep_closest = d
            if terminated or truncated:
                break

        lengths.append(steps)
        if terminated:
            hits += 1
        closest_approaches.append(ep_closest if np.isfinite(ep_closest) else max_steps)

    env.close()
    return {
        "lengths": np.array(lengths),
        "hit_rate": hits / num_episodes,
        "closest_approaches": np.array(closest_approaches),
    }


def summarize(label, agent_pos, stats, R):
    lengths = stats["lengths"]
    closest = stats["closest_approaches"]
    finite_closest = closest[np.isfinite(closest)]
    pos_norm = float(np.linalg.norm(agent_pos))
    print(
        f"  {label:<30}  "
        f"pos=({agent_pos[0]:+5.0f},{agent_pos[1]:+5.0f},{agent_pos[2]:+5.0f})  "
        f"||pos||/R={pos_norm/R:.2f}   "
        f"mean_steps={lengths.mean():6.1f}   "
        f"std={lengths.std():5.1f}   "
        f"hit_rate={100*stats['hit_rate']:5.1f}%   "
        f"mean_closest={finite_closest.mean():5.1f}"
    )


def main(num_episodes, seed):
    R = ARENA_RADIUS
    boundary = R - AGENT_RADIUS

    # Policy-favorite octant direction (from the position-diagnostic on
    # the trained agent: mean direction +0.587, +0.538, +0.262).
    fav_dir = np.array([0.587, 0.538, 0.262], dtype=np.float32)
    fav_dir = fav_dir / np.linalg.norm(fav_dir)

    positions = [
        ("origin",                   np.array([0.0, 0.0, 0.0], dtype=np.float32)),
        ("half-radius +x",           np.array([R/2, 0.0, 0.0], dtype=np.float32)),
        ("near-boundary +x",         np.array([boundary, 0.0, 0.0], dtype=np.float32)),
        ("near-boundary +y",         np.array([0.0, boundary, 0.0], dtype=np.float32)),
        ("near-boundary -x",         np.array([-boundary, 0.0, 0.0], dtype=np.float32)),
        ("policy-favorite octant",   fav_dir * boundary),
        ("opposite octant",         -fav_dir * boundary),
    ]

    print(f"=== Threat-distribution diagnostic (stay-still policy) ===")
    print(f"Episodes per position: {num_episodes}   seed start: {seed}   "
          f"R={R}   AGENT_RADIUS={AGENT_RADIUS}   collision<{AGENT_RADIUS+PROJECTILE_RADIUS}\n")
    print(f"  {'label':<30}  {'pos':<22}  {'rel_R':<7}  "
          f"{'mean_steps':<10}  {'std':<5}  {'hit_rate':<8}  {'mean_closest':<10}")
    for label, pos in positions:
        stats = run_fixed_position(pos, num_episodes, seed)
        summarize(label, pos, stats, R)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=50,
                        help="how many simulated episodes per position.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args.episodes, args.seed)
