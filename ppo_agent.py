"""
Single-file PPO for DodgeEnv. Modeled on CleanRL's ppo.py.

Reads top-to-bottom:
  1. Hyperparameters
  2. ActorCritic network
  3. Rollout buffer collection
  4. GAE advantages + returns
  5. PPO update (clip loss + value loss + entropy bonus)
  6. Main loop that ties them together
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


# ---- Hyperparameters ----
# Most are CleanRL PPO defaults. NUM_STEPS / batch size tuned for a single env.
TOTAL_TIMESTEPS  = 300_000      # total env steps to train for
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

CHECKPOINT_PATH  = "ppo_dodge.pt"


# ---- Network ----
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


# ---- Training ----
def train(total_timesteps=TOTAL_TIMESTEPS, seed=0, verbose=True):
    # Seeding for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    # CPU is plenty for a tiny MLP — and avoids MPS overhead on small batches.
    device = torch.device("cpu")

    env = DodgeEnv()
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    agent = ActorCritic(obs_dim, n_actions).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=LR, eps=1e-5)

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

    # Initial state
    next_obs_np, _ = env.reset(seed=seed)
    next_obs = torch.tensor(next_obs_np, dtype=torch.float32, device=device)
    next_done = torch.tensor(0.0, device=device)

    num_updates = total_timesteps // NUM_STEPS
    global_step = 0
    start_time = time.time()

    for update in range(1, num_updates + 1):
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
            if done:
                ep_lengths_window.append(cur_ep_len)
                ep_rewards_window.append(cur_ep_rew)
                if len(ep_lengths_window) > 50:
                    ep_lengths_window.pop(0)
                    ep_rewards_window.pop(0)
                cur_ep_len = 0
                cur_ep_rew = 0.0
                next_obs_np, _ = env.reset()

            next_obs = torch.tensor(next_obs_np, dtype=torch.float32, device=device)
            next_done = torch.tensor(1.0 if done else 0.0, device=device)

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

        # ---- Logging ----
        if verbose:
            mean_len = np.mean(ep_lengths_window) if ep_lengths_window else float("nan")
            mean_rew = np.mean(ep_rewards_window) if ep_rewards_window else float("nan")
            sps = int(global_step / (time.time() - start_time))
            print(
                f"upd {update:4d} | step {global_step:7d} | "
                f"mean_ep_len {mean_len:6.1f} | mean_ep_rew {mean_rew:6.1f} | "
                f"pg_loss {pg_loss.item():+.3f} | v_loss {v_loss.item():.3f} | "
                f"ent {ent_loss.item():.3f} | clipfrac {np.mean(clipfracs):.3f} | "
                f"sps {sps}"
            )

    env.close()

    torch.save(
        {
            "model_state_dict": agent.state_dict(),
            "obs_dim": obs_dim,
            "n_actions": n_actions,
            "hidden_dim": HIDDEN_DIM,
        },
        CHECKPOINT_PATH,
    )
    print(f"\nSaved checkpoint to {CHECKPOINT_PATH}")
    return agent


def load_agent(path=CHECKPOINT_PATH, device="cpu"):
    """Load a trained ActorCritic from a checkpoint."""
    # weights_only=False is safe here — we wrote this file ourselves.
    ckpt = torch.load(path, map_location=device, weights_only=False)
    agent = ActorCritic(int(ckpt["obs_dim"]), int(ckpt["n_actions"]),
                        hidden=int(ckpt["hidden_dim"])).to(device)
    agent.load_state_dict(ckpt["model_state_dict"])
    agent.eval()
    return agent


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=TOTAL_TIMESTEPS,
                        help="total env steps to train for")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    train(total_timesteps=args.steps, seed=args.seed)
