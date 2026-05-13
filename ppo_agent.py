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
  python3 ppo_agent.py --resume              # continue from ppo_dodge.pt
  python3 ppo_agent.py --steps 1_000_000     # longer run
  Ctrl-C or close the window -> graceful save to ppo_dodge.pt.
"""

import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from dodge_env import DodgeEnv


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
ANNEAL_LR        = True         # linearly decay LR to 0 over training
GAMMA            = 0.99         # discount factor
GAE_LAMBDA       = 0.95         # GAE smoothing
CLIP_COEF        = 0.2          # PPO clip range
ENT_COEF         = 0.01         # entropy bonus weight (encourages exploration)
VF_COEF          = 0.5          # value loss weight
MAX_GRAD_NORM    = 0.5          # global grad clip
HIDDEN_DIM       = 64

BATCH_SIZE       = NUM_STEPS                   # one env, so batch == rollout length
MINIBATCH_SIZE   = BATCH_SIZE // NUM_MINIBATCHES

CHECKPOINT_PATH      = "ppo_dodge.pt"
CHECKPOINT_INTERVAL  = 10        # save every N updates (so progress survives crashes/Ctrl-C)


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
def save_checkpoint(agent, optimizer, global_step, path=CHECKPOINT_PATH):
    """Persist enough state to resume training later."""
    torch.save(
        {
            "model_state_dict": agent.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "global_step": int(global_step),
            "obs_dim": agent.obs_dim,
            "n_actions": agent.n_actions,
            "hidden_dim": agent.hidden_dim,
        },
        path,
    )


def load_agent(path=CHECKPOINT_PATH, device="cpu"):
    """Load a trained ActorCritic from a checkpoint (weights only)."""
    # weights_only=False is safe here — we wrote this file ourselves.
    ckpt = torch.load(path, map_location=device, weights_only=False)
    agent = ActorCritic(int(ckpt["obs_dim"]), int(ckpt["n_actions"]),
                        hidden=int(ckpt["hidden_dim"])).to(device)
    agent.load_state_dict(ckpt["model_state_dict"])
    agent.eval()
    return agent


# ============================================================
# Training
# ============================================================
def train(total_timesteps=TOTAL_TIMESTEPS, seed=0, render=False,
          resume_path=None, verbose=True):
    # Seeding for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    # CPU is plenty for a tiny MLP — and avoids MPS overhead on small batches.
    device = torch.device("cpu")

    render_mode = "human" if render else None
    env = DodgeEnv(render_mode=render_mode)
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    agent = ActorCritic(obs_dim, n_actions).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=LR, eps=1e-5)

    # ---- Optional resume ----
    if resume_path and os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        agent.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
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
    print(f"Checkpoint every {CHECKPOINT_INTERVAL} updates -> {CHECKPOINT_PATH}\n")

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
                np.random.shuffle(idx)
                for start in range(0, BATCH_SIZE, MINIBATCH_SIZE):
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

            # ---- Per-update summary line ----
            if verbose:
                mean_len = np.mean(ep_lengths_window) if ep_lengths_window else float("nan")
                mean_rew = np.mean(ep_rewards_window) if ep_rewards_window else float("nan")
                sps = int(global_step / (time.time() - start_time))
                print(
                    f"=== upd {update:4d}/{num_updates} | step {global_step:7d} | "
                    f"mean_ep_len {mean_len:6.1f} | mean_ep_rew {mean_rew:6.1f} | "
                    f"pg_loss {pg_loss.item():+.3f} | v_loss {v_loss.item():.3f} | "
                    f"ent {ent_loss.item():.3f} | clipfrac {np.mean(clipfracs):.3f} | "
                    f"sps {sps} ==="
                )

            # ---- Periodic checkpoint ----
            if update % CHECKPOINT_INTERVAL == 0:
                save_checkpoint(agent, optimizer, global_step)

    except KeyboardInterrupt:
        print("\nCtrl-C — stopping training.")
        should_stop = True

    finally:
        # Always save before exiting, even on Ctrl-C / window close.
        save_checkpoint(agent, optimizer, global_step)
        print(f"\nSaved checkpoint to {CHECKPOINT_PATH} (global_step = {global_step})")
        env.close()

    return agent


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Train PPO on DodgeEnv.",
    )
    parser.add_argument("--steps", type=int, default=TOTAL_TIMESTEPS,
                        help="total env steps to train for in this session")
    parser.add_argument("--seed", type=int, default=0,
                        help="random seed for reproducibility")
    parser.add_argument("--render", action="store_true",
                        help="open a live pygame window during training "
                             "(slow! caps at 30 FPS). For a fast snapshot view, "
                             "train headless and run watch_agent.py separately.")
    parser.add_argument("--resume", nargs="?", const=CHECKPOINT_PATH, default=None,
                        help="continue training from an existing checkpoint. "
                             "Use `--resume` to load ppo_dodge.pt, or "
                             "`--resume path/to/file.pt` for a specific file.")
    args = parser.parse_args()
    train(total_timesteps=args.steps, seed=args.seed,
          render=args.render, resume_path=args.resume)
