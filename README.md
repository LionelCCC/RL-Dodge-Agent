# RL Dodge Agent

A small PPO agent that learns to dodge projectiles inside a 3D spherical arena.

The env: agent lives at the origin of a sphere of radius `ARENA_RADIUS`. Red
projectiles spawn on a shell *outside* the sphere and home toward the agent's
current position (with a small AIM_RADIUS jitter so they don't always hit dead-on).
Reward is +1/step; the episode ends on a collision or after `MAX_EPISODE_STEPS`.
The agent picks one of 27 discrete actions every step (each axis in `{-1, 0, +1}`).

```
.
├── dodge_env.py            # the env + physics + perspective renderer
├── ppo_agent.py            # PPO trainer + ActorCritic
├── watch_agent.py          # load checkpoint, watch / eval / GIF
├── random_agent.py         # random-action baseline
├── position_diagnostic.py  # is the agent symmetric? pinned to surface?
├── checkpoints/            # *.pt — saved weights
├── logs/                   # training console output
├── assets/                 # rendered GIFs, screenshots
├── GLOSSARY.md             # what every term in the logs means
└── README.md               # this file
```

---

## Project map

### `dodge_env.py` — the game world

Defines the Gymnasium env. PPO treats it as a black box: action in, (obs, reward, done) out.

Key pieces:
1. `DodgeEnv.__init__` — observation/action spaces, render state.
2. `reset` — agent at origin, no projectiles, step counter at 0.
3. `_sample_projectile_spawn_pos` — uniform on a shell *outside* the arena
   (`ARENA_RADIUS + uniform(SPAWN_DISTANCE_MIN, SPAWN_DISTANCE_MAX)` from origin).
4. `_spawn_projectile` — aim near the agent's *current* position (AIM_RADIUS ball).
5. `_get_obs` — agent state + closest visible projectiles (filtered by `PERCEPTION_RADIUS`).
6. `step` — physics frame; spherical containment is a radial projection.
7. `render` — perspective camera orbiting the origin, draws a wireframe sphere,
   floor grid, depth-sorted spheres with shadows.

Be careful:
- Changing `MAX_PROJECTILES` changes the observation shape. Old checkpoints stop loading.
- `MAX_EPISODE_STEPS` is one episode's cap, not the training budget.
- If the agent learns something weird, inspect spawn/reward design first.

### `ppo_agent.py` — training and saved learning

Owns the PPO algorithm + checkpointing.

Key pieces:
1. Hyperparameters at the top of the file (LR, NUM_STEPS, ENT_COEF, ...).
2. `ActorCritic` — shared MLP trunk, separate policy and value heads.
3. `save_checkpoint` / `load_agent` — model weights, optimizer state, step count.
4. `train` — collect rollouts, GAE advantages, PPO update, log, save latest + best.
5. `auto_eval_and_gif` — runs after training: deterministic eval + a GIF.

Be careful:
- Running without `--resume` starts fresh and overwrites `checkpoints/ppo_dodge.pt` on save.
- Rendering (`--render`) caps the rollout at ~30 FPS — much slower than headless.

### `watch_agent.py` — viewing, GIFs, evaluation

Loads a checkpoint and runs without learning.

Modes:
- `--mode human` — opens a window (drag to orbit, wheel to zoom, ESC to quit).
- `--mode eval` — N deterministic episodes, prints mean/std/min/max.
- `--mode gif` — one episode, writes to `assets/trained_agent.gif`.

### `random_agent.py` — baseline

What does dumb luck get? Whatever you train must beat this clearly.

### `position_diagnostic.py` — sanity check

Aggregates per-step agent positions over N episodes and reports mean ‖pos‖/R,
mean direction vector, % steps near the inner surface, and an ASCII histogram.
Use this whenever you suspect the agent is reward-hacking (camping a quiet
region of the arena instead of dodging).

---

## Setup

```bash
python3 -m pip install --user gymnasium pygame torch pillow numpy
```

---

## How a session works (the mental model)

Two terminals side by side:

| Terminal 1 — training | Terminal 2 — viewing |
|---|---|
| `python3 ppo_agent.py` | `python3 watch_agent.py --mode human` |
| Logs episode + update lines; saves `checkpoints/ppo_dodge.pt` every 10 updates and `checkpoints/ppo_dodge_best.pt` on new peaks. | Plays the best checkpoint by default. Reopen any time to see progress. |

Training is fast and headless; the live window is its own process so it
doesn't slow training down. If you'd rather see training in-loop, use
`python3 ppo_agent.py --render` (slow, capped at 30 FPS).

---

## Common commands

### Train

```bash
# default: 300k steps, headless, ~70s on CPU
python3 ppo_agent.py

# longer run, pipe to a log file
python3 ppo_agent.py --steps 1_000_000 2>&1 | tee logs/training.log

# easier curriculum stage (wider spawn shell)
python3 ppo_agent.py --steps 1_000_000 --curriculum easy

# resume from the canonical latest
python3 ppo_agent.py --resume

# resume from a specific file
python3 ppo_agent.py --resume path/to/file.pt

# train with a live window (slow, 30 FPS cap)
python3 ppo_agent.py --render
```

### Window controls (any `human`-mode render)

| Input | Effect |
|---|---|
| Left-mouse drag | Orbit camera (azimuth + elevation) |
| Arrow keys | Orbit camera (held = continuous) |
| Mouse wheel | Zoom in / out |
| `1`–`5` | Set playback speed to 1× / 2× / 4× / 8× / 16× |
| `+` / `-` | Step speed up / down one level |
| `R` | Reset camera to default |
| `ESC` or window X | Quit cleanly (saves checkpoint if training) |

### Watch / eval / GIF

```bash
# live window, 5 episodes back-to-back, deterministic
python3 watch_agent.py --mode human

# one episode, sampled actions (looks more natural)
python3 watch_agent.py --mode human --episodes 1 --stochastic

# headless eval, 30 episodes
python3 watch_agent.py --mode eval --episodes 30

# save a GIF
python3 watch_agent.py --mode gif --seed 7
```

### Sanity-check the env

```bash
# random baseline — what dumb luck gets
python3 random_agent.py --episodes 30

# is the agent symmetric / not pinned to the surface?
python3 position_diagnostic.py --policy random --episodes 30
python3 position_diagnostic.py --policy ppo    --episodes 30
```

---

## Latest training snapshot

Current env defaults (in `dodge_env.py`):

| Knob | Value |
|---|---|
| `ARENA_RADIUS` | 200 |
| `SPAWN_PROB` | 0.06 |
| `SPAWN_DISTANCE_MIN..MAX` (shell, outside R) | 50..200 |
| `PERCEPTION_RADIUS` | 200 |
| `AGENT_SPEED` / `PROJECTILE_SPEED` | 5.0 / 4.0 |
| `MAX_PROJECTILES` (obs slots) | 8 |
| `AIM_RADIUS` | 100 |

Random baseline (30 episodes):

| Metric | Value |
|---|---|
| Mean | ~334–424 (varies seed-to-seed, std ~250) |
| Position distribution | symmetric, unimodal, mean ‖pos‖/R ≈ 0.35, ~1% near surface |

Latest PPO result (1M steps, default hyperparameters, no curriculum):

| Field | Value |
|---|---|
| Headline checkpoint | `checkpoints/ppo_dodge_best.pt` |
| Eval (20 ep, deterministic) | mean **413.8**, std 224.0, min 130, max 803 |
| Visual sample | `assets/trained_agent.gif` |
| Training log | `logs/training.log` |
| Notes | Tied with random within noise. PPO did not learn meaningfully better than random on this difficulty setting. Open hyperparameter / env-design problem; first thing to try is curriculum (`--curriculum easy 1M` then `--resume --curriculum target 500k`). |

---

## Knobs you'll actually want to tune

In `dodge_env.py`:

| Knob | What changes | Try |
|---|---|---|
| `SPAWN_PROB` | How often projectiles appear | 0.03 (easier) → 0.10 (harder) |
| `PROJECTILE_SPEED` | How fast they move | 3.0 (easier) → 6.0 (harder) |
| `AIM_RADIUS` | How precisely they target the agent | 200 (sprays) → 50 (surgical) |
| `SPAWN_DISTANCE_MIN/MAX` | Shell thickness outside R | closer shell = less reaction time |
| `ARENA_RADIUS` | Size of the playable sphere | smaller = less room to maneuver |
| `PERCEPTION_RADIUS` | What the agent can see | smaller = harder; obs slots zero out |
| `MAX_PROJECTILES` | How many slots in the obs | changes obs shape — restart from scratch |
| `AGENT_SPEED` | How fast the agent moves | 4.0 (harder) → 7.0 (easier) |

In `ppo_agent.py`:

| Knob | What changes |
|---|---|
| `TOTAL_TIMESTEPS` | How long to train (default 300k ≈ 70s) |
| `NUM_STEPS` | Rollout size before each update |
| `LR` | Learning rate (3e-4 is standard) |
| `ENT_COEF` | Exploration pressure |

See **[GLOSSARY.md](GLOSSARY.md)** for what each means in detail.

---

## Reading the training log

```
ep    47 | step    9821 | survived  226 | rollout 5/146
=== upd    5/146 | step  10240 | mean_ep_len 219.4 | mean_ep_rew 219.4 | pg_loss -0.057 | v_loss 113.230 | ent 2.143 | clipfrac 0.092 | sps 4419 ===
```

- **`ep N`** — every time an episode finishes. `survived` = how long it lasted.
- **`=== upd N/M ===`** — every PPO update.
  - `mean_ep_len` — average over the last 50 episodes. **This should climb.**
  - `pg_loss`, `v_loss`, `ent`, `clipfrac` — training-health diagnostics (see GLOSSARY).
  - `sps` — steps per second (throughput).

---

## How learning is saved

- Every 10 updates and on exit, `ppo_agent.py` writes `checkpoints/ppo_dodge.pt`.
- Whenever rolling `mean_ep_len` improves, it writes `checkpoints/ppo_dodge_best.pt`.
- Both store model weights, optimizer state, step count, and the spawn settings.
- `watch_agent.py` loads the `_best` file by default, falling back to the latest.
- Starting a new training command auto-clears `ppo_dodge_best.pt` *after* loading
  `--resume`, so the best file stays scoped to the current run. If you want to
  preserve an old run, `cp` it to a stable name first.

---

## Specification gaming: stories we've learned from

The agent is not lazy or clever — it's a search process. It will exploit
whatever your specification doesn't lock down.

### Story 1 — "The lucky corner" (resolved before sphere refactor)

**What the agent did.** In an earlier rectangular-arena version, projectiles
aimed at a random point in the *central region* and spawned from arena edges.
The trained agent drifted to a corner and stayed there. Survival times climbed —
not from dodging but because random projectiles rarely intersected that corner.

**Why the loophole existed.** Reward was +1/step (correct) but the threat
distribution was *position-independent*: where the agent stood had no influence
on where projectiles went. The optimal policy was "find the lowest static threat
density and never move."

**The fixes.**
1. Projectile aim now tracks the agent (`_spawn_projectile` aims at
   `agent_pos + offset(radius=AIM_RADIUS)`). Moving away doesn't help.
2. Projectile spawns are on a shell *outside* the arena, not at random
   internal positions. The agent can no longer find a "quiet" spawn region.
3. The arena is now a sphere with no corners — there is nowhere to camp.

**The lesson.** When you write a reward function, picture an adversary with
infinite tries trying to beat your spec without doing what you wanted. Each
degree of freedom in the env is something the agent will probe. Fix the spec,
not the agent. Spec-fix > reward-shaping hacks > algorithm tweaks.

**Diagnostic you can reuse.** `position_diagnostic.py` was built for this
class of problem. If the policy's mean position is off-center, or its
‖pos‖/R distribution has a sharp peak somewhere unexpected, suspect spec
gaming before celebrating climbing `mean_ep_len`.

---

## Curriculum learning — the recipe when flat training plateaus

If a single difficulty setting plateaus and you're sure the env isn't leaking:
*don't* immediately reach for bigger nets or more steps. Try curriculum first.

```bash
# Stage 1 — easy stage
python3 ppo_agent.py --steps 1_000_000 --curriculum easy 2>&1 | tee logs/training_easy.log

# Archive the easy-stage best before stage 2 auto-clears it
cp checkpoints/ppo_dodge_best.pt checkpoints/ppo_dodge_easy_best.pt

# Stage 2 — transfer
python3 ppo_agent.py --steps 500_000 --curriculum target \
    --resume checkpoints/ppo_dodge_easy_best.pt 2>&1 | tee logs/training_target.log
```

Mechanism: the easy env has a stronger reward gradient (more actions actually
save you), so the policy commits faster (entropy falls). The pre-committed
policy is a much better starting point for target-stage PPO than random init.

A wide gap between deterministic-eval and sampled-training mean (>15%) usually
means entropy is too high — try a polish stage with lower `ENT_COEF`.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `mean_ep_len` stuck near random | Env may be too hard, or hyperparameters need work. | Try `--curriculum easy`, lower `PROJECTILE_SPEED`, or longer training. |
| Argmax eval << sampled training mean | Policy entropy too high; argmax discards useful exploration. | Lower `ENT_COEF` (0.005 → 0.003) and `--resume` for a polish stage. |
| `checkpoints/ppo_dodge_best.pt` got overwritten | Each training session auto-clears the best file at start. | Always `cp` to a stable archive name after a good run. |
| `weights_only` error on load | PyTorch ≥ 2.6 default. | Handled in `load_agent` (`weights_only=False`). |
| `obs_dim` mismatch when watching/resuming | You changed `MAX_PROJECTILES` (or similar) after saving. | Start fresh without `--resume`. |
| Training is super slow | `--render` is on. | Drop it; train headless and view via Terminal 2. |
| Agent never moves in the window | Loaded the wrong checkpoint, or a stochastic-policy issue. | Confirm the checkpoint path; try `--stochastic`. |
