# RL Dodge Agent

A small PPO agent that learns to dodge projectiles in a 2D arena.

- **`dodge_env.py`** — the environment (Gymnasium-compatible)
- **`ppo_agent.py`** — PPO trainer + the ActorCritic network
- **`watch_agent.py`** — load a trained checkpoint, watch it / save GIF / eval
- **`random_agent.py`** — the random-action baseline (sanity check)
- **`GLOSSARY.md`** — what every term in the logs means
- **`ppo_dodge.pt`** — saved checkpoint (created/updated when you train)

---

## Project map

Use this section as the "what file do I touch?" map.

### `dodge_env.py` — the game world

This file defines the Gymnasium environment. PPO treats it like a black box:
give it an action, get back the next observation, reward, and done flags.

Key pieces:

1. `DodgeEnv.__init__()` creates the observation/action spaces and stores render state.
2. `reset()` starts one new episode: center agent, clear projectiles, reset step counter.
3. `_sample_projectile_spawn_pos()` chooses where a projectile appears. Current behavior:
   local ring spawn near the agent, which prevents corner-camping.
4. `_spawn_projectile()` aims that projectile near the current agent position.
5. `_get_obs()` builds the vector the neural net sees: agent state plus closest projectiles.
6. `step(action)` advances one physics frame and returns reward/done info.
7. `render()` and `poll_close_event()` draw the window and let X/ESC stop cleanly.

Be careful:

- Changing `MAX_PROJECTILES` changes the observation size. Old checkpoints will not load.
- `MAX_EPISODE_STEPS` is only the max length of one episode, not how long training runs.
- If the agent learns a weird exploit, inspect spawn/reward design first. RL will find shortcuts.

### `ppo_agent.py` — training and saved learning

This file owns the PPO algorithm and checkpointing.

Key pieces:

1. Hyperparameters at the top control how long/how aggressively the agent trains.
2. `ActorCritic` is the neural net: actor chooses actions, critic estimates value.
3. `save_checkpoint()` writes model weights, optimizer state, and `global_step`.
4. `load_agent()` loads a saved checkpoint for watching/evaluation.
5. `train()` collects rollouts, computes advantages, updates PPO, logs progress, and saves.

Be careful:

- Running without `--resume` starts fresh and can overwrite `ppo_dodge.pt`.
- `TOTAL_TIMESTEPS` controls the whole training session length.
- `NUM_STEPS` controls how many env steps are collected before one PPO update.
- Rendering during training is useful for intuition, but it makes training much slower.

### `watch_agent.py` — viewing, GIFs, and evaluation

This file loads a checkpoint and runs it without learning.

Key pieces:

1. `pick_action()` converts one observation into one action.
2. `run_episode()` plays one full episode and optionally collects frames.
3. `watch_live()` opens the live side window.
4. `evaluate()` runs many headless episodes and prints mean/std/min/max survival.
5. `make_gif()` saves a short visual record to `trained_agent.gif`.

Be careful:

- Watching does not train. It only plays the saved checkpoint.
- If you changed the env shape, old checkpoints will fail with an `obs_dim` mismatch.
- Use deterministic mode for clean evaluation; use `--stochastic` for a more varied watch.

### `random_agent.py` — baseline

This file runs the environment with random actions. It answers: "What score can dumb luck get?"

Key piece:

1. `run_random_agent()` runs N episodes and prints survival stats.

Be careful:

- Re-run this whenever the environment changes. The baseline is part of the problem definition.
- A trained agent should beat random by a clear margin, not by a tiny amount.

### Other project files

| File | Purpose | Update when |
|---|---|---|
| `GLOSSARY.md` | Plain-English meaning of parameters, log columns, and PPO terms. | You add/tune a new knob or forget what a term means. |
| `training.log` | Saved training console output. | You want to preserve a run's learning curve. |
| `ppo_dodge.pt` | Latest saved checkpoint. | Training runs or resumes. |
| `trained_agent.gif` | Latest visual sample of the saved agent. | You refresh the agent or want a quick progress visual. |

---

## Latest training snapshot

This is the README section to update before commits that change training results.
It should tell future-you: "What did the latest agent do, and where can I see it?"

Current environment baseline:

| Metric | Value |
|---|---|
| Random baseline | mean 267.4 steps over 20 episodes |
| Baseline spread | std 191.4, min 55, max 940 |
| Env version note | Local ring spawns, `MAX_PROJECTILES=8`, `SPAWN_PROB=0.03` |

Latest trained-agent result:

| Field | Value |
|---|---|
| Checkpoint | `ppo_dodge.pt` |
| Eval command | `python3 watch_agent.py --mode eval --episodes 30` |
| Eval result | TODO: update after next training run on the current env |
| Training log | `training.log` |
| Visual sample | `trained_agent.gif` |
| Notes | Old 5-projectile checkpoints are stale after the observation-size change. |

Latest visual sample:

![Latest trained agent](trained_agent.gif)

Before committing a new training result, refresh this block:

```bash
python3 ppo_agent.py 2>&1 | tee training.log
python3 watch_agent.py --mode eval --episodes 30
python3 watch_agent.py --mode gif --out trained_agent.gif --seed 7
```

Then update `Eval result`, `Training log`, and any notes above. The README will
only reflect the latest learning curve if this section is kept in sync with the
checkpoint/GIF you commit.

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

Note: this env now observes 8 projectiles instead of 5, so older checkpoints
from the 5-projectile env will not load. Start fresh after this change.

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
Random survives ~267 steps on average in the current local-spawn env
(20 episodes, mean 267.4, std 191.4, min 55, max 940). Whatever you train
must clearly beat that.

---

## Knobs you'll actually want to tune

All in `dodge_env.py` (top of file):

| Knob | What changes | Try |
|---|---|---|
| `SPAWN_PROB` | How often new projectiles appear | 0.03 (easier) → 0.08 (harder) |
| `PROJECTILE_SPEED` | How fast projectiles move | 3.0 (easier) → 6.0 (harder) |
| `AIM_RADIUS` | How precisely projectiles target the agent | 200 (sprays, easier) → 50 (surgical, harder) |
| `SPAWN_DISTANCE_MIN/MAX` | How far from the agent projectiles appear | closer = less reaction time, harder |
| `MAX_PROJECTILES` | How many closest projectiles the agent observes | 8 by default; lower can hide threats |
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
| `mean_ep_len` stuck near 267 | Agent isn't learning, or env got harder after a knob change. | Train longer, lower difficulty knobs, or check `entropy` (if 0, raise `ENT_COEF`). |
| `weights_only` error on load | PyTorch ≥ 2.6 default. | Already handled in `load_agent` (`weights_only=False`). |
| `obs_dim` mismatch when watching/resuming | You changed `MAX_PROJECTILES` or another env observation setting after saving the checkpoint. | Start fresh without `--resume`; old checkpoints are for the old env shape. |
| Training is super slow | You probably have `--render` on. | Drop the flag; train headless and view via Terminal 2. |
| Agent just sits in a corner | Reward-hacking: far edge spawns gave too much reaction time. | Current env uses local ring spawns near the agent (`SPAWN_DISTANCE_MIN/MAX`) and aims near the agent (`AIM_RADIUS`). |
