"""
Env factory + per-env defaults.

All call sites (`ppo_agent.py`, `watch_agent.py`, `random_agent.py`) go through
these helpers, so no script has to know which file defines which env class.

Why this is a separate module:
  - Keeps the 2D and 3D env files independent (neither imports the other).
  - Avoids pulling torch when random_agent.py just needs an env.
  - Centralizes the per-env defaults (spawn distance, curriculum presets,
    checkpoint paths) so the rules don't drift across scripts.
"""

ENV_NAMES = ("2d", "3d")


def make_env(env_name, **kwargs):
    """Instantiate the env. kwargs are forwarded to the env constructor
    (e.g. render_mode, spawn_distance_min, spawn_distance_max)."""
    if env_name == "2d":
        from dodge_env import DodgeEnv
        return DodgeEnv(**kwargs)
    if env_name == "3d":
        from dodge_env_3d import DodgeEnv3D
        return DodgeEnv3D(**kwargs)
    raise ValueError(f"Unknown env: {env_name!r}. Use one of {list(ENV_NAMES)}.")


def env_defaults(env_name):
    """Return everything per-env: (spawn_min, spawn_max, presets, ckpt, best_ckpt).

    Naming convention for checkpoints:
      - 2D keeps the original ppo_dodge.pt / ppo_dodge_best.pt (backward compat)
      - 3D uses ppo_dodge_3d.pt / ppo_dodge_3d_best.pt
    Different files mean a fresh 2D run won't clobber a 3D run, and vice versa.
    """
    if env_name == "2d":
        from dodge_env import (
            SPAWN_DISTANCE_MIN as smin,
            SPAWN_DISTANCE_MAX as smax,
            CURRICULUM_PRESETS as presets,
        )
        return smin, smax, presets, "ppo_dodge.pt", "ppo_dodge_best.pt"
    if env_name == "3d":
        from dodge_env_3d import (
            SPAWN_DISTANCE_MIN as smin,
            SPAWN_DISTANCE_MAX as smax,
            CURRICULUM_PRESETS as presets,
        )
        return smin, smax, presets, "ppo_dodge_3d.pt", "ppo_dodge_3d_best.pt"
    raise ValueError(f"Unknown env: {env_name!r}. Use one of {list(ENV_NAMES)}.")
