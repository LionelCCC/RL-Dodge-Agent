"""Generate the two front-page GIFs that tell the project's story.

  before_predictive_aim_slide.gif
    A hand-crafted constant-action policy in the OLD env (ballistic aim,
    no frame randomization, PROJECTILE_SPEED=4.0). The agent slides along
    the inner boundary and survives meaningfully longer than random,
    because each projectile aims where the agent WAS at spawn time and
    the slide moves it ~150 px out of the aim cone before impact.

  after_predictive_aim_random.gif
    The current trained PPO checkpoint in the CURRENT env (predictive aim
    with boundary clipping, frame randomization, PROJECTILE_SPEED=6.0).
    Sampling from the (near-uniform) policy produces a random-walk that
    defeats the predictor. The "trained agent" is approximately a
    randomness-generator under stochastic action selection.

This script is a one-off; it's safe to delete after the GIFs are committed.
Run with:  python3 make_journey_gifs.py
"""

import os
import numpy as np
import torch
from torch.distributions import Categorical

from dodge_env import DodgeEnv
from ppo_agent import load_agent
from watch_agent import save_gif


def collect_frames(env, action_fn, seed, max_steps=750):
    """Roll out one episode collecting RGB frames."""
    obs, _ = env.reset(seed=seed)
    frames = []
    for _ in range(max_steps):
        action = action_fn(obs)
        obs, _r, term, trunc, _ = env.step(action)
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        if term or trunc:
            break
    return frames


def render_before():
    """Slide policy in ballistic-aim env."""
    print("--- Rendering before_predictive_aim_slide.gif ---")
    env = DodgeEnv(
        render_mode="rgb_array",
        randomize_frame=False,
        predictive_aim=False,
        projectile_speed=4.0,
    )
    # Action 4 = (-1, 0, 0) -> constant -x slide. Any single axial action works.
    slide_action = 4
    frames = collect_frames(env, lambda _obs: slide_action, seed=3, max_steps=750)
    env.close()
    out = "assets/before_predictive_aim_slide.gif"
    print(f"  collected {len(frames)} frames, saving -> {out}")
    save_gif(frames, out)


def render_after():
    """Current trained PPO, stochastic actions, in current env."""
    print("--- Rendering after_predictive_aim_random.gif ---")
    agent = load_agent("checkpoints/ppo_dodge_best.pt")
    env = DodgeEnv(render_mode="rgb_array")   # all defaults

    torch.manual_seed(0)
    np.random.seed(0)

    def stochastic_pick(obs):
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits = agent.actor(agent.trunk(obs_t))
            return int(Categorical(logits=logits).sample().item())

    frames = collect_frames(env, stochastic_pick, seed=7, max_steps=750)
    env.close()
    out = "assets/after_predictive_aim_random.gif"
    print(f"  collected {len(frames)} frames, saving -> {out}")
    save_gif(frames, out)


if __name__ == "__main__":
    os.makedirs("assets", exist_ok=True)
    render_before()
    render_after()
    print("Done.")
