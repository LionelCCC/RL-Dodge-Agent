# RL Dodge Agent

A small PPO agent that learns to dodge projectiles in a 2D arena.

- **`dodge_env.py`** — the environment (Gymnasium-compatible)
- **`ppo_agent.py`** — PPO trainer + the ActorCritic network
- **`watch_agent.py`** — load a trained checkpoint, watch it / save GIF / eval
- **`random_agent.py`** — the random-action baseline (sanity check)
- **`GLOSSARY.md`** — what every term in the logs means
- **`ppo_dodge.pt`** — saved checkpoint (created/updated when you train)

---

## Setup

```bash
python3 -m pip install --user gymnasium pygame torch pillow numpy
```

---

## How a session works (the mental model)

You'll usually run **two terminals** side by side:

| Terminal 1 — training | Terminal 2 — viewing |
|---|---|
| `python3 ppo_agent.py` | `python3 watch_agent.py --mode human` |
| Prints `ep` and `=== upd` lines as it learns. Saves `ppo_dodge.pt` every 10 updates. | Opens a window and plays the latest saved checkpoint. Reopen this any time to see how the agent has improved. |

This gives you fast training (Terminal 1) *and* a live view (Terminal 2) without
forcing the training loop itself to render every frame.

If you'd rather watch training happen *inside* the training loop (slow but
self-contained), use `python3 ppo_agent.py --render` — see below.

---

## Common commands

### Train

```bash
# Default: 300k steps, headless, ~70s on CPU.
python3 ppo_agent.py

# Longer training:
python3 ppo_agent.py --steps 1_000_000

# Resume from the existing ppo_dodge.pt (continues from saved weights):
python3 ppo_agent.py --resume

# Resume from a specific file:
python3 ppo_agent.py --resume my_old_run.pt

# Train with a live window (slow — 30 fps cap). Close the window or hit ESC
# (or Ctrl-C) to save & stop.
python3 ppo_agent.py --render
```

### Stop training cleanly

- **Ctrl-C** in the training terminal → saves checkpoint, exits.
- If `--render` is on: **close the window** or **press ESC** → same thing.
- The checkpoint also auto-saves every 10 updates, so a crash loses at most ~20s of work.

### Watch the trained agent

```bash
# Live window, 5 episodes back-to-back, deterministic policy:
python3 watch_agent.py --mode human

# Single episode, sampling actions instead of argmax (looks more natural):
python3 watch_agent.py --mode human --episodes 1 --stochastic

# Different scenarios via seeds:
python3 watch_agent.py --mode human --seed 42

# Close the window or press ESC to stop early.
```

### Save a GIF

```bash
python3 watch_agent.py --mode gif --out trained_agent.gif --seed 7
```

### Headless evaluation (just numbers)

```bash
# Mean / std / min / max episode length over 30 episodes:
python3 watch_agent.py --mode eval --episodes 30
```

### Sanity-check the env with the random baseline

```bash
python3 random_agent.py
```
Random survives ~225 steps on average. Whatever you train must clearly beat that.

---

## Knobs you'll actually want to tune

All in `dodge_env.py` (top of file):

| Knob | What changes | Try |
|---|---|---|
| `SPAWN_PROB` | How often new projectiles appear | 0.03 (easier) → 0.08 (harder) |
| `PROJECTILE_SPEED` | How fast projectiles move | 3.0 (easier) → 6.0 (harder) |
| `AIM_RADIUS` | How precisely projectiles target the agent | 200 (sprays, easier) → 50 (surgical, harder) |
| `AGENT_SPEED` | How fast the agent moves | 4.0 (harder) → 7.0 (easier) |

And in `ppo_agent.py`:

| Knob | What changes |
|---|---|
| `TOTAL_TIMESTEPS` | How long to train (default 300k = ~70s) |
| `NUM_STEPS` | Rollout size. Bigger = lower-variance updates, fewer updates per session |
| `LR` | Learning rate. 3e-4 is the standard; try 1e-4 if training is unstable |
| `ENT_COEF` | Exploration pressure. Raise to 0.02 if entropy collapses too fast |

See **[GLOSSARY.md](GLOSSARY.md)** for what each of these means in detail.

---

## Reading the training log

While training, you'll see lines like:

```
ep    47 | step    9821 | survived  226 | rollout 5/146
ep    48 | step   10103 | survived  282 | rollout 5/146
=== upd    5/146 | step  10240 | mean_ep_len  219.4 | mean_ep_rew  219.4 | pg_loss -0.057 | v_loss 113.230 | ent 2.143 | clipfrac 0.092 | sps 4419 ===
```

- **`ep N`** lines: every time an episode finishes. `survived` is how long it lasted.
- **`=== upd N/M ===`** lines: every PPO update (~once a second). The key columns:
  - `mean_ep_len` — average of last 50 episodes. **This should climb.**
  - `pg_loss`, `v_loss`, `ent`, `clipfrac` — training-health diagnostics. See GLOSSARY.md.
  - `sps` — steps per second (throughput).

---

## "How do I know learning is saved?"

- Every 10 updates and on exit, `ppo_agent.py` writes `ppo_dodge.pt`.
- `ppo_dodge.pt` stores the model weights, the optimizer state, and the step count.
- `python3 watch_agent.py` loads `ppo_dodge.pt` by default — so the agent you see
  is whatever was last saved.
- **Running `python3 ppo_agent.py` without `--resume` starts fresh** and will
  overwrite `ppo_dodge.pt` when training finishes. If you want to keep an old run,
  copy `ppo_dodge.pt` to another name before training again, e.g. `cp ppo_dodge.pt run1.pt`.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Window won't close, script hangs | Old behavior (pre-fix) — the env wasn't checking for QUIT events. | Pull current `dodge_env.py` (now handles X-button and ESC). |
| `mean_ep_len` stuck near 225 | Agent isn't learning, or env got harder after a knob change. | Train longer, lower difficulty knobs, or check `entropy` (if 0, raise `ENT_COEF`). |
| `weights_only` error on load | PyTorch ≥ 2.6 default. | Already handled in `load_agent` (`weights_only=False`). |
| Training is super slow | You probably have `--render` on. | Drop the flag; train headless and view via Terminal 2. |
| Agent just sits in a corner | Was reward-hacking before — projectiles weren't tracking the agent. | Already fixed: projectiles now aim near the agent (`AIM_RADIUS`). |
