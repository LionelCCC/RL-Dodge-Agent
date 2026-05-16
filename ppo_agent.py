"""
Single-file PPO for DodgeEnv. Modeled on CleanRL's ppo.py.

Reads top-to-bottom:
  1. Hyperparameters
  2. ActorCritic network
  3. Rollout buffer collection
  4. GAE advantages + returns
  5. PPO update (clip loss + value loss + entropy bonus)
  6. Main loop that ties them together

CLI cheatsheet (see also README.md):
  python3 ppo_agent.py                       # fresh training, no window
  python3 ppo_agent.py --render              # fresh training + live window (slow)
  python3 ppo_agent.py --resume              # continue from checkpoints/ppo_dodge.pt
  python3 ppo_agent.py --steps 1_000_000     # longer run
  python3 ppo_agent.py --curriculum easy     # easier spawn shell
  Ctrl-C or close the window -> graceful save to checkpoints/ppo_dodge.pt.
"""

import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from dodge_env import (
    DodgeEnv,
    CURRICULUM_PRESETS,
    CHECKPOINT_DIR,
    CHECKPOINT_PATH,
    BEST_CHECKPOINT_PATH,
    DEFAULT_GIF_PATH,
    SPAWN_DISTANCE_MIN,
    SPAWN_DISTANCE_MAX,
)


# ============================================================
# Hyperparameters (see GLOSSARY.md for what each one means)
# ============================================================
# Vocabulary note:
#   "step"  = one env interaction (one call to env.step)
#   "episode" = one play from reset to terminal/truncated
#   "rollout" = a batch of NUM_STEPS env interactions collected before an update
#   "update" = one PPO update (UPDATE_EPOCHS passes over the rollout)
#   "total_timesteps" = total env steps for the WHOLE training run
#
# Number of updates = TOTAL_TIMESTEPS / NUM_STEPS
# Number of episodes is data-dependent (~ TOTAL_TIMESTEPS / mean_episode_length)
#
# MAX_EPISODE_STEPS lives in dodge_env.py and caps a SINGLE episode (1000 steps).
# TOTAL_TIMESTEPS lives here and controls how long we train OVERALL.

TOTAL_TIMESTEPS  = 300_000      # total env steps to train for in this session
NUM_STEPS        = 2048         # steps collected per rollout before each update
NUM_MINIBATCHES  = 32           # rollout is split into this many minibatches
UPDATE_EPOCHS    = 10           # how many passes over each rollout
LR               = 3e-4
ANNEAL_LR        = False        # linearly decay LR to 0 over training.
                                # Disabled: this is the CleanRL Atari default
                                # (designed for 10M+ step runs). On short ~1M-step
                                # runs it (a) flattens learning right when you'd
                                # want late corrections, and (b) breaks --resume
                                # because the loaded optimizer state has LR≈0.
GAMMA            = 0.99         # discount factor
GAE_LAMBDA       = 0.95         # GAE smoothing
CLIP_COEF        = 0.2          # PPO clip range
ENT_COEF         = 0.005        # entropy bonus weight (encourages exploration).
                                # Tried 0.002 as a polish-stage value and it
                                # did NOT improve over 0.005 in 3D (peak
                                # rolling-mean briefly hit 674 but 30-ep
                                # deterministic eval came in at 573.7 vs the
                                # 0.005 baseline's 587.0 on the same seeds).
                                # See README "What we learned from 3D
                                # training" for the full story.
VF_COEF          = 0.5          # value loss weight
MAX_GRAD_NORM    = 0.5          # global grad clip
HIDDEN_DIM       = 64

BATCH_SIZE       = NUM_STEPS                   # one env, so batch == rollout length
MINIBATCH_SIZE   = BATCH_SIZE // NUM_MINIBATCHES

# Checkpoint paths live in dodge_env.py (CHECKPOINT_PATH / BEST_CHECKPOINT_PATH)
# so every script sees the same canonical location (checkpoints/).
CHECKPOINT_INTERVAL  = 10        # save every N updates (so progress survives crashes/Ctrl-C)
MIN_EPISODES_FOR_BEST = 10       # need at least this many finished episodes in
                                 # the window before "best mean" is meaningful


# ============================================================
# Network
# ============================================================
def layer_init(layer, std=np.sqrt(2.0), bias=0.0):
    """Orthogonal init with a tunable gain — standard PPO practice.
    Small gain (0.01) on the policy head keeps initial actions ~uniform."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


class ActorCritic(nn.Module):
    """Shared MLP trunk, separate linear heads for policy and value."""

    def __init__(self, obs_dim, n_actions, hidden=HIDDEN_DIM):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.n_actions = int(n_actions)
        self.hidden_dim = int(hidden)

        self.trunk = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)),
            nn.Tanh(),
        )
        # Policy head: tiny init so logits start ~0 → action distribution ~uniform.
        self.actor = layer_init(nn.Linear(hidden, n_actions), std=0.01)
        # Value head: unit gain.
        self.critic = layer_init(nn.Linear(hidden, 1), std=1.0)

    def get_value(self, x):
        return self.critic(self.trunk(x)).squeeze(-1)

    def get_action_and_value(self, x, action=None):
        z = self.trunk(x)
        logits = self.actor(z)
        value = self.critic(z).squeeze(-1)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value


# ============================================================
# Checkpoint helpers
# ============================================================
def save_checkpoint(
    agent,
    optimizer,
    global_step,
    path,
    best_mean_len=None,
    spawn_distance_min=None,
    spawn_distance_max=None,
):
    """Persist enough state to resume training later."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "model_state_dict": agent.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "global_step": int(global_step),
            "best_mean_len": None if best_mean_len is None else float(best_mean_len),
            "spawn_distance_min": spawn_distance_min,
            "spawn_distance_max": spawn_distance_max,
            "obs_dim": agent.obs_dim,
            "n_actions": agent.n_actions,
            "hidden_dim": agent.hidden_dim,
        },
        path,
    )


def load_agent(path, device="cpu"):
    """Load a trained ActorCritic from a checkpoint (weights only)."""
    # weights_only=False is safe here — we wrote this file ourselves.
    ckpt = torch.load(path, map_location=device, weights_only=False)
    agent = ActorCritic(int(ckpt["obs_dim"]), int(ckpt["n_actions"]),
                        hidden=int(ckpt["hidden_dim"])).to(device)
    agent.load_state_dict(ckpt["model_state_dict"])
    agent.eval()
    return agent


def auto_eval_and_gif(agent, n_episodes=20, gif_seed=7,
                      gif_path=None,
                      spawn_distance_min=None,
                      spawn_distance_max=None):
    """Post-training: print deterministic-eval stats and overwrite the GIF.

    Prefers the BEST checkpoint over the in-memory (latest) agent. PPO can
    regress after finding a good policy; if we eval the *latest* weights
    we measure that regression. Eval'ing the saved best gives us a stable
    picture of the best policy this run produced.

    Deferred import to dodge the circular dependency on watch_agent.
    """
    from watch_agent import evaluate, make_gif
    smin = SPAWN_DISTANCE_MIN if spawn_distance_min is None else spawn_distance_min
    smax = SPAWN_DISTANCE_MAX if spawn_distance_max is None else spawn_distance_max
    if gif_path is None:
        gif_path = DEFAULT_GIF_PATH

    eval_agent = agent
    eval_source = "latest (in-memory weights)"
    if os.path.exists(BEST_CHECKPOINT_PATH):
        try:
            eval_agent = load_agent(BEST_CHECKPOINT_PATH)
            eval_source = BEST_CHECKPOINT_PATH
        except Exception as e:
            print(f"Could not load {BEST_CHECKPOINT_PATH} ({e}); falling back to in-memory latest.")
    print(f"\n=== Auto-eval (deterministic, source: {eval_source}) ===")
    evaluate(
        eval_agent,
        num_episodes=n_episodes,
        spawn_distance_min=smin,
        spawn_distance_max=smax,
    )
    print(f"\n=== Auto-GIF: seed={gif_seed} -> {gif_path} ===")
    try:
        make_gif(
            eval_agent,
            out_path=gif_path,
            seed=gif_seed,
            deterministic=True,
            spawn_distance_min=smin,
            spawn_distance_max=smax,
        )
    except Exception as e:
        # GIF failure shouldn't crash the whole training session.
        print(f"GIF generation failed: {e}")


# ============================================================
# Training
# ============================================================
def train(total_timesteps=TOTAL_TIMESTEPS, seed=0, render=False,
          resume_path=None, verbose=True,
          spawn_distance_min=None,
          spawn_distance_max=None):
    # Seeding for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    # CPU is plenty for a tiny MLP — and avoids MPS overhead on small batches.
    device = torch.device("cpu")

    if spawn_distance_min is None:
        spawn_distance_min = SPAWN_DISTANCE_MIN
    if spawn_distance_max is None:
        spawn_distance_max = SPAWN_DISTANCE_MAX
    checkpoint_path = CHECKPOINT_PATH
    best_checkpoint_path = BEST_CHECKPOINT_PATH

    render_mode = "human" if render else None
    env = DodgeEnv(
        render_mode=render_mode,
        spawn_distance_min=spawn_distance_min,
        spawn_distance_max=spawn_distance_max,
    )
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    agent = ActorCritic(obs_dim, n_actions).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=LR, eps=1e-5)

    # ---- Optional resume ----
    if resume_path and os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        if int(ckpt.get("obs_dim", obs_dim)) != obs_dim:
            raise RuntimeError(
                f"Checkpoint expects obs_dim={ckpt.get('obs_dim')}, but the "
                f"current env returns obs_dim={obs_dim}. The environment changed; "
                "start a fresh run without --resume."
            )
        agent.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        # Loading optimizer state restores LR from whatever the prior session
        # ended at — which could be ~0 if it had been annealing. Force LR back
        # to the configured value so resumed runs actually learn.
        for g in optimizer.param_groups:
            g["lr"] = LR
        print(f"Resumed from {resume_path} (trained for {ckpt.get('global_step', '?')} prior steps)")
    elif resume_path:
        print(f"--resume given but {resume_path} not found; starting fresh.")

    # ---- Rollout storage (preallocated tensors) ----
    obs_buf      = torch.zeros((NUM_STEPS, obs_dim), device=device)
    actions_buf  = torch.zeros(NUM_STEPS, dtype=torch.long, device=device)
    logprobs_buf = torch.zeros(NUM_STEPS, device=device)
    rewards_buf  = torch.zeros(NUM_STEPS, device=device)
    dones_buf    = torch.zeros(NUM_STEPS, device=device)
    values_buf   = torch.zeros(NUM_STEPS, device=device)

    # ---- Episode bookkeeping for logging ----
    ep_lengths_window = []   # recent finished-episode lengths
    ep_rewards_window = []
    cur_ep_len = 0
    cur_ep_rew = 0.0
    episodes_done = 0

    # ---- Best-checkpoint tracking ----
    # Best is scoped to THIS training command and THIS difficulty setting.
    # Curriculum transfer should usually be:
    #   1. train easy -> ppo_dodge_best.pt
    #   2. optionally copy that source checkpoint elsewhere
    #   3. train target --resume ppo_dodge_best.pt
    # During step 3, the first target-stage improvement replaces the easy-stage
    # best with a target-stage best, which is what watch/eval should headline.
    if os.path.exists(best_checkpoint_path):
        print(f"Removing stale {best_checkpoint_path} before this run.")
        os.remove(best_checkpoint_path)
    best_mean_len = -1.0

    # Initial state
    next_obs_np, _ = env.reset(seed=seed)
    next_obs = torch.tensor(next_obs_np, dtype=torch.float32, device=device)
    next_done = torch.tensor(0.0, device=device)

    num_updates = max(1, total_timesteps // NUM_STEPS)
    global_step = 0
    start_time = time.time()
    should_stop = False  # set if window is closed or Ctrl-C

    if render:
        print("\nLIVE RENDER ON. Training will be slow (rendering caps at 30 FPS).")
        print("Close the window or press ESC (or Ctrl-C) to stop and save.\n")

    print(f"Will run {num_updates} updates of {NUM_STEPS} steps = {num_updates * NUM_STEPS} total steps")
    print(f"Env: DodgeEnv | obs_dim {obs_dim} | n_actions {n_actions}")
    print(f"Spawn shell: {spawn_distance_min:.0f}..{spawn_distance_max:.0f} outside R={env.arena_radius:.0f}")
    print(f"Checkpoint every {CHECKPOINT_INTERVAL} updates -> {checkpoint_path}\n")

    try:
        for update in range(1, num_updates + 1):
            if should_stop:
                break

            # Linear LR anneal
            if ANNEAL_LR:
                frac = 1.0 - (update - 1.0) / num_updates
                optimizer.param_groups[0]["lr"] = frac * LR

            # ============================================================
            # PHASE 1: Collect a rollout of NUM_STEPS transitions
            # ============================================================
            for step in range(NUM_STEPS):
                global_step += 1
                obs_buf[step] = next_obs
                dones_buf[step] = next_done

                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(next_obs)
                values_buf[step] = value
                actions_buf[step] = action
                logprobs_buf[step] = logprob

                next_obs_np, reward, terminated, truncated, _ = env.step(action.item())
                done = terminated or truncated
                rewards_buf[step] = float(reward)

                cur_ep_len += 1
                cur_ep_rew += float(reward)

                if render:
                    env.render()
                    if env.closed_by_user:
                        print("\nWindow closed by user — stopping training.")
                        should_stop = True
                        break

                if done:
                    episodes_done += 1
                    ep_lengths_window.append(cur_ep_len)
                    ep_rewards_window.append(cur_ep_rew)
                    if len(ep_lengths_window) > 50:
                        ep_lengths_window.pop(0)
                        ep_rewards_window.pop(0)
                    # Per-episode line so you can see progress as it happens.
                    print(
                        f"ep {episodes_done:5d} | step {global_step:7d} | "
                        f"survived {cur_ep_len:4d} | "
                        f"rollout {update}/{num_updates}"
                    )
                    cur_ep_len = 0
                    cur_ep_rew = 0.0
                    next_obs_np, _ = env.reset()

                next_obs = torch.tensor(next_obs_np, dtype=torch.float32, device=device)
                next_done = torch.tensor(1.0 if done else 0.0, device=device)

            if should_stop:
                break

            # ============================================================
            # PHASE 2: Compute GAE advantages and returns
            # ============================================================
            # GAE: A_t = sum_{l>=0} (gamma*lambda)^l * delta_{t+l}
            # where delta_t = r_t + gamma * V(s_{t+1}) * (1 - done) - V(s_t)
            with torch.no_grad():
                next_value = agent.get_value(next_obs)
                advantages = torch.zeros(NUM_STEPS, device=device)
                last_gae = 0.0
                for t in reversed(range(NUM_STEPS)):
                    if t == NUM_STEPS - 1:
                        next_nonterminal = 1.0 - next_done
                        next_v = next_value
                    else:
                        next_nonterminal = 1.0 - dones_buf[t + 1]
                        next_v = values_buf[t + 1]
                    delta = rewards_buf[t] + GAMMA * next_v * next_nonterminal - values_buf[t]
                    last_gae = delta + GAMMA * GAE_LAMBDA * next_nonterminal * last_gae
                    advantages[t] = last_gae
                returns = advantages + values_buf

            # ============================================================
            # PHASE 3: PPO update — multiple epochs over minibatches
            # ============================================================
            b_obs       = obs_buf
            b_actions   = actions_buf
            b_logprobs  = logprobs_buf
            b_advs      = advantages
            b_returns   = returns
            b_values    = values_buf

            idx = np.arange(BATCH_SIZE)
            clipfracs = []
            for _ in range(UPDATE_EPOCHS):
                if render and env.poll_close_event():
                    print("\nWindow closed by user during PPO update — stopping after this checkpoint.")
                    should_stop = True
                    break
                np.random.shuffle(idx)
                for start in range(0, BATCH_SIZE, MINIBATCH_SIZE):
                    if render and env.poll_close_event():
                        print("\nWindow closed by user during PPO update — stopping after this checkpoint.")
                        should_stop = True
                        break
                    mb_idx = idx[start:start + MINIBATCH_SIZE]

                    _, new_logprob, entropy, new_value = agent.get_action_and_value(
                        b_obs[mb_idx], b_actions[mb_idx]
                    )
                    logratio = new_logprob - b_logprobs[mb_idx]
                    ratio = logratio.exp()

                    # Normalize advantages within the minibatch (variance reduction).
                    mb_advs = b_advs[mb_idx]
                    mb_advs = (mb_advs - mb_advs.mean()) / (mb_advs.std() + 1e-8)

                    # PPO clipped surrogate objective
                    pg_loss1 = -mb_advs * ratio
                    pg_loss2 = -mb_advs * torch.clamp(ratio, 1 - CLIP_COEF, 1 + CLIP_COEF)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss (clipped, as in CleanRL)
                    v_clipped = b_values[mb_idx] + torch.clamp(
                        new_value - b_values[mb_idx], -CLIP_COEF, CLIP_COEF
                    )
                    v_loss_unclipped = (new_value - b_returns[mb_idx]).pow(2)
                    v_loss_clipped = (v_clipped - b_returns[mb_idx]).pow(2)
                    v_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                    ent_loss = entropy.mean()
                    loss = pg_loss - ENT_COEF * ent_loss + VF_COEF * v_loss

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), MAX_GRAD_NORM)
                    optimizer.step()

                    with torch.no_grad():
                        clipfracs.append(((ratio - 1.0).abs() > CLIP_COEF).float().mean().item())

                if should_stop:
                    break

            if should_stop:
                break

            # ---- Per-update summary line ----
            mean_len_now = (
                float(np.mean(ep_lengths_window)) if ep_lengths_window else float("nan")
            )
            if verbose:
                mean_rew = np.mean(ep_rewards_window) if ep_rewards_window else float("nan")
                sps = int(global_step / (time.time() - start_time))
                print(
                    f"=== upd {update:4d}/{num_updates} | step {global_step:7d} | "
                    f"mean_ep_len {mean_len_now:6.1f} | mean_ep_rew {mean_rew:6.1f} | "
                    f"pg_loss {pg_loss.item():+.3f} | v_loss {v_loss.item():.3f} | "
                    f"ent {ent_loss.item():.3f} | clipfrac {np.mean(clipfracs):.3f} | "
                    f"sps {sps} ==="
                )

            # ---- Track best mean_ep_len so we have a stable peak to evaluate ----
            # Without this, the final checkpoint is whatever the policy regressed to.
            if (len(ep_lengths_window) >= MIN_EPISODES_FOR_BEST
                    and mean_len_now > best_mean_len):
                best_mean_len = mean_len_now
                save_checkpoint(agent, optimizer, global_step,
                                path=best_checkpoint_path,
                                best_mean_len=best_mean_len,
                                spawn_distance_min=spawn_distance_min,
                                spawn_distance_max=spawn_distance_max)
                print(f"   ^ new best mean_ep_len {best_mean_len:.1f} "
                      f"(saved -> {best_checkpoint_path})")

            # ---- Periodic checkpoint (latest, used by --resume) ----
            if update % CHECKPOINT_INTERVAL == 0:
                save_checkpoint(
                    agent,
                    optimizer,
                    global_step,
                    path=checkpoint_path,
                    spawn_distance_min=spawn_distance_min,
                    spawn_distance_max=spawn_distance_max,
                )

    except KeyboardInterrupt:
        print("\nCtrl-C — stopping training.")
        should_stop = True

    finally:
        # Always save before exiting, even on Ctrl-C / window close.
        save_checkpoint(
            agent,
            optimizer,
            global_step,
            path=checkpoint_path,
            spawn_distance_min=spawn_distance_min,
            spawn_distance_max=spawn_distance_max,
        )
        print(f"\nSaved checkpoint to {checkpoint_path} (global_step = {global_step})")
        env.close()

    return agent


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Train PPO on DodgeEnv (3D spherical arena).",
    )
    parser.add_argument("--steps", type=int, default=TOTAL_TIMESTEPS,
                        help="total env steps to train for in this session")
    parser.add_argument("--seed", type=int, default=0,
                        help="random seed for reproducibility")
    parser.add_argument("--render", action="store_true",
                        help="open a live pygame window during training "
                             "(slow! caps at 30 FPS). For a fast snapshot view, "
                             "train headless and run watch_agent.py separately.")
    parser.add_argument("--resume", nargs="?", const="auto", default=None,
                        help="continue training from an existing checkpoint. "
                             f"`--resume` loads {CHECKPOINT_PATH}. "
                             "`--resume path/to/file.pt` loads a specific file.")
    parser.add_argument("--no-auto-eval", action="store_true",
                        help="skip the post-training eval + GIF generation.")
    parser.add_argument("--eval-episodes", type=int, default=20,
                        help="how many episodes the post-training eval uses.")
    parser.add_argument("--gif-seed", type=int, default=7,
                        help="seed used for the post-training GIF episode.")
    parser.add_argument(
        "--curriculum",
        choices=list(CURRICULUM_PRESETS.keys()),
        default="target",
        help="spawn-shell preset (distance OUTSIDE the arena radius).",
    )
    parser.add_argument(
        "--spawn-distance-min",
        type=float,
        default=None,
        help="override curriculum preset minimum spawn-shell distance.",
    )
    parser.add_argument(
        "--spawn-distance-max",
        type=float,
        default=None,
        help="override curriculum preset maximum spawn-shell distance.",
    )
    args = parser.parse_args()

    preset_min, preset_max = CURRICULUM_PRESETS[args.curriculum]
    spawn_distance_min = (
        args.spawn_distance_min if args.spawn_distance_min is not None else preset_min
    )
    spawn_distance_max = (
        args.spawn_distance_max if args.spawn_distance_max is not None else preset_max
    )

    # `--resume` with no path means "use the canonical latest checkpoint".
    resume_path = args.resume
    if resume_path == "auto":
        resume_path = CHECKPOINT_PATH

    agent = train(total_timesteps=args.steps, seed=args.seed,
                  render=args.render, resume_path=resume_path,
                  spawn_distance_min=spawn_distance_min,
                  spawn_distance_max=spawn_distance_max)
    if not args.no_auto_eval:
        auto_eval_and_gif(agent,
                          n_episodes=args.eval_episodes,
                          gif_seed=args.gif_seed,
                          spawn_distance_min=spawn_distance_min,
                          spawn_distance_max=spawn_distance_max)
