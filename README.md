# RL Dodge Agent

A small PPO agent that learns to dodge projectiles. Two envs ship in this repo:

- **`dodge_env.py`** — 2D arena (9 discrete actions)
- **`dodge_env_3d.py`** — 3D box (27 discrete actions, z-axis added)

Every script picks the env with `--env {2d, 3d}` (default `2d`). The two envs
keep separate checkpoint files so 2D and 3D runs don't clobber each other.

Files:

- **`dodge_env.py`** / **`dodge_env_3d.py`** — environments
- **`env_factory.py`** — `make_env(env_name)` and per-env defaults (single switchboard)
- **`ppo_agent.py`** — PPO trainer + the ActorCritic network
- **`watch_agent.py`** — load a trained checkpoint, watch it / save GIF / eval
- **`random_agent.py`** — the random-action baseline (sanity check)
- **`GLOSSARY.md`** — what every term in the logs means
- **`ppo_dodge.pt`** / **`ppo_dodge_3d.pt`** — latest weights, per env
- **`ppo_dodge_best.pt`** / **`ppo_dodge_3d_best.pt`** — best-by-mean checkpoint, per env

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
5. `train()` collects rollouts, computes advantages, updates PPO, logs progress, saves latest, and saves best-so-far.

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
| `env_factory.py` | `make_env(env_name)` + per-env defaults (spawn distance, curriculum presets, checkpoint paths). | You add a new env. |
| `dodge_env_3d.py` | 3D version of the env (27 actions, z-axis, top-down render). | You tune 3D-specific knobs. |
| `GLOSSARY.md` | Plain-English meaning of parameters, log columns, and PPO terms. | You add/tune a new knob or forget what a term means. |
| `training.log`, `training_easy.log`, `training_target.log`, `training_3d.log` | Saved training console output, one per run/stage. | You want to preserve a run's learning curve. |
| `ppo_dodge.pt` / `ppo_dodge_3d.pt` | Latest saved checkpoint, per env. | Training runs or resumes. |
| `ppo_dodge_best.pt` / `ppo_dodge_3d_best.pt` | Best checkpoint from the run, chosen by rolling `mean_ep_len`, per env. | PPO finds a good policy and later regresses. |
| `trained_agent.gif` / `trained_agent_3d.gif` | Latest visual sample of the saved agent, per env. | You refresh the agent or want a quick progress visual. |

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

Latest trained-agent result (curriculum: easy 1M → target 500k transfer):

| Field | Value |
|---|---|
| Headline checkpoint | `ppo_dodge_best.pt` (target-stage best from the transfer run) |
| Latest checkpoint | `ppo_dodge.pt` |
| Archived easy-stage best | `ppo_dodge_easy_best.pt` (kept as the curriculum source) |
| Eval command | `python3 watch_agent.py --mode eval --episodes 30 --curriculum target` |
| **Eval result (target, 30 eps, deterministic)** | **mean 444.5, std 310.3, min 95, max 1000** |
| Training logs | `training_easy.log` (stage 1) + `training_target.log` (stage 2) |
| Visual sample | `trained_agent.gif` |
| Notes | Curriculum lift: **288.7 → 444.5 (+54% over no-curriculum baseline)**. Peak rolling `mean_ep_len` during target transfer = 468.8. Eval std 310 is wide — policy hits 1000-step perfect runs but also has fast deaths. The argmax/sampled gap (~16%) remains: lower `ENT_COEF` to 0.003 or another 1M target-stage extension are the obvious next experiments. |

Comparison block (all evals are deterministic argmax, on **target** difficulty unless noted):

**2D env:**

| Configuration | Mean | Std | Min | Max | Notes |
|---|---|---|---|---|---|
| Random baseline | 267.4 | 191.4 | 55 | 940 | 20 eps |
| No-curriculum 1M target | 288.7 | 103.5 | 126 | 529 | prior run, kept for reference |
| Easy-stage best evaluated on **easy** | 397.9 | 213.1 | 175 | 919 | auto-eval at end of stage 1 |
| Easy-stage best evaluated on target (raw transfer, no finetune) | 309.8 | 182.9 | 39 | 1000 | shows transfer alone helps a little |
| **Curriculum: easy 1M + target 500k** | **444.5** | 310.3 | 95 | 1000 | headline 2D result, 30 eps |

**3D env (new):**

| Configuration | Mean | Std | Min | Max | Notes |
|---|---|---|---|---|---|
| Random baseline | 499.2 | 315.3 | 62 | 1000 | 20 eps; 3D random is *much* higher than 2D random (see "What we learned from the 3D port") |
| PPO no-curriculum 1M target | **671.0** | 299.7 | 186 | 1000 | auto-eval at end of training, 20 eps |
| Peak training rolling-mean | 692.9 | — | — | — | best checkpoint = upd ~485 (very late in training; no large regression) |

**3D headline:** PPO beats 3D random by +34% (499.2 → 671.0) with the *same*
hyperparameters as 2D. Auto-eval source: `ppo_dodge_3d_best.pt`. Sample GIF:
`trained_agent_3d.gif` (top-down view, z encoded by circle size + 1D z-strip).

Latest visual sample:

![Latest trained agent](trained_agent.gif)

Before committing a new training result, refresh this block. The curriculum
recipe is what produced the current headline number:

```bash
# Stage 1 — easy stage (1M steps, ~4 min on CPU)
python3 ppo_agent.py --steps 1_000_000 --curriculum easy 2>&1 | tee training_easy.log

# Archive easy-stage best so it survives stage 2's auto-cleanup
cp ppo_dodge_best.pt ppo_dodge_easy_best.pt

# Optional sanity check: how does raw transfer do?
python3 watch_agent.py --mode eval --episodes 30 --curriculum target \
    --checkpoint ppo_dodge_easy_best.pt

# Stage 2 — transfer to target (500k steps, ~2 min)
python3 ppo_agent.py --steps 500_000 --curriculum target \
    --resume ppo_dodge_easy_best.pt 2>&1 | tee training_target.log

# Headline eval + GIF
python3 watch_agent.py --mode eval --episodes 30 --curriculum target
python3 watch_agent.py --mode gif --out trained_agent.gif --seed 7 --curriculum target
```

Then update `Eval result`, `Training logs`, and the comparison table above.
The README will only reflect the latest learning curve if this section is kept
in sync with the checkpoint/GIF you commit.

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
| Prints `ep` and `=== upd` lines as it learns. Saves `ppo_dodge.pt` every 10 updates and `ppo_dodge_best.pt` on new peaks. | Opens a window and plays the best saved checkpoint by default. Reopen this any time to see how the agent has improved. |

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
# 2D (default), 300k steps, headless, ~70s on CPU.
python3 ppo_agent.py

# 3D — same trainer, just pick the env.
python3 ppo_agent.py --env 3d --steps 1_000_000

# Longer training (2D):
python3 ppo_agent.py --steps 1_000_000

# Easier curriculum stage: wider spawn distance, longer reaction time.
python3 ppo_agent.py --steps 1_000_000 --curriculum easy

# Optional: archive the easy-stage source before target transfer.
cp ppo_dodge_best.pt ppo_dodge_easy_best.pt          # 2D
cp ppo_dodge_3d_best.pt ppo_dodge_3d_easy_best.pt    # 3D, when running curriculum there

# Transfer from easy -> target difficulty. This replaces ppo_dodge_best.pt
# with the best target-stage checkpoint once target training improves.
python3 ppo_agent.py --steps 500_000 --curriculum target --resume ppo_dodge_best.pt

# Resume from the env's default latest checkpoint:
python3 ppo_agent.py --resume               # loads ppo_dodge.pt (2D default)
python3 ppo_agent.py --env 3d --resume      # loads ppo_dodge_3d.pt

# Resume from a specific file:
python3 ppo_agent.py --resume my_old_run.pt

# Train with a live window (slow — 30 fps cap). Close the window or hit ESC
# (or Ctrl-C) to save & stop. 3D rendering is now a true perspective view
# (orbit camera, floor grid, depth-sorted spheres + shadows).
python3 ppo_agent.py --render
python3 ppo_agent.py --env 3d --render
```

**3D window controls** (active whenever the 3D env is rendering in human mode —
during `--render` training or `watch_agent.py --env 3d --mode human`):

| Input | Effect |
|---|---|
| Left-mouse drag | Orbit the camera (azimuth + elevation) |
| Arrow keys | Orbit the camera (held = continuous) |
| Mouse wheel | Zoom in / out (`camera_distance`) |
| `1` `2` `3` `4` `5` | Set playback speed directly to 1x / 2x / 4x / 8x / 16x |
| `+` / `-` (or `=` / `-`) | Step playback speed up / down one level |
| `R` | Reset the camera to the default "whole arena visible" view |
| `ESC` or window X | Quit cleanly (saves checkpoint if training) |

The default 3D camera is pulled back (`DEFAULT_DISTANCE = 1300`) so the entire
arena is always in frame. The speed multiplier just raises the `clock.tick`
FPS cap — there's no memory cost, just more CPU. At 16x the simulation
finishes a 1000-step episode in ~2 seconds on a modern laptop.

### Stop training cleanly

- **Ctrl-C** in the training terminal → saves checkpoint, exits.
- If `--render` is on: **close the window** or **press ESC** → same thing.
- The checkpoint also auto-saves every 10 updates, so a crash loses at most ~20s of work.

### Watch the trained agent

```bash
# 2D, live window, 5 episodes back-to-back, deterministic policy:
python3 watch_agent.py --mode human

# 3D — same UI, top-down view + z-strip:
python3 watch_agent.py --env 3d --mode human

# Single episode, sampling actions instead of argmax (looks more natural):
python3 watch_agent.py --mode human --episodes 1 --stochastic

# Different scenarios via seeds:
python3 watch_agent.py --mode human --seed 42

# Evaluate or watch on the easier curriculum stage:
python3 watch_agent.py --mode eval --episodes 30 --curriculum easy

# Close the window or press ESC to stop early.
```

### Save a GIF

```bash
# 2D writes trained_agent.gif by default; 3D writes trained_agent_3d.gif.
python3 watch_agent.py --mode gif --seed 7 --curriculum target
python3 watch_agent.py --env 3d --mode gif --seed 7 --curriculum target
```

### Headless evaluation (just numbers)

```bash
# Mean / std / min / max episode length over 30 episodes:
python3 watch_agent.py --mode eval --episodes 30 --curriculum target
python3 watch_agent.py --env 3d --mode eval --episodes 30 --curriculum target
```

### Sanity-check the env with the random baseline

```bash
python3 random_agent.py            # 2D random ≈ 267 mean
python3 random_agent.py --env 3d   # 3D random ≈ 499 mean (way higher — see "What we learned")
```
2D random survives ~267 steps on average in the current local-spawn env
(20 episodes, mean 267.4, std 191.4, min 55, max 940). 3D random survives
~499 steps over the same number of episodes — the 3D ball aim misses much
more often than the 2D disk aim. Whatever you train must clearly beat its
own env's random baseline.

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
| `--curriculum easy/target` | Use preset spawn distances without editing code | easy = 280..400, target = 220..320 |

And in `ppo_agent.py`:

| Knob | What changes |
|---|---|
| `TOTAL_TIMESTEPS` | How long to train (default 300k = ~70s) |
| `NUM_STEPS` | Rollout size. Bigger = lower-variance updates, fewer updates per session |
| `LR` | Learning rate. 3e-4 is the standard; try 1e-4 if training is unstable |
| `ENT_COEF` | Exploration pressure. Current experiment uses 0.005; lower values make the policy commit faster |

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

- Every 10 updates and on exit, `ppo_agent.py` writes latest weights to `ppo_dodge.pt`.
- Whenever rolling `mean_ep_len` improves, `ppo_agent.py` writes best-so-far weights to `ppo_dodge_best.pt`.
- Both checkpoints store model weights, optimizer state, the step count, and the spawn-distance setting used for that run.
- `python3 watch_agent.py` loads `ppo_dodge_best.pt` by default when it exists, otherwise `ppo_dodge.pt`.
- Starting any new training command clears stale `ppo_dodge_best.pt` after loading `--resume`, so `ppo_dodge_best.pt` stays scoped to the current run/stage.
- **Running `python3 ppo_agent.py` without `--resume` starts fresh** and will
  overwrite `ppo_dodge.pt` when training finishes. If you want to keep an old run,
  copy `ppo_dodge.pt` to another name before training again, e.g. `cp ppo_dodge.pt run1.pt`.

---

## Specification gaming: stories we've learned from

This section is the project's compounding asset. Every time the agent finds
an unintended shortcut, add the story here. Future-you will save days by
re-reading these instead of relearning each lesson.

The pattern is always the same:
> The agent is not lazy or clever — it's a search process. It will exploit
> whatever your specification doesn't lock down. The cheapest path to reward
> wins, even if that path is silly.

### Story 1 — "The lucky corner" (resolved)

**What the agent did.** With the original env, projectiles aimed at a random
point in the *central region* of the arena and spawned from arena edges. The
trained agent learned to drift to a corner of the arena and stay there, doing
nothing. Survival times climbed but not because it was dodging — it had found
a low-traffic zone that random projectiles rarely intersected. From a distance
this looked like learning. It wasn't.

**Why the loophole existed.** The reward (+1/step) was correct, but the env
gave the agent a *position-independent* threat distribution. The agent's
position had no influence on where projectiles would go, so the optimal policy
was "find the spot with the lowest static threat density and never move."

**The fixes.**
1. `_spawn_projectile()` now aims at `agent_pos + offset(radius=AIM_RADIUS)`.
   Projectiles track the agent, so moving to a corner doesn't help.
2. `_sample_projectile_spawn_pos()` now spawns projectiles in a ring of
   `SPAWN_DISTANCE_MIN..MAX` pixels around the agent — not from arena edges.
   This closes a second loophole: far-edge spawns gave a corner-camping agent
   long reaction windows and a clear sightline. Local spawns force the agent
   to react to threats from anywhere, including directly behind it.

**The lesson.** When you write a reward function, picture an adversary with
*infinite* tries trying to beat your spec without doing what you actually
wanted. Spawn position, target location, episode termination conditions —
each one is a degree of freedom the agent will exploit if you let it. Fix the
spec, not the agent. Spec-fix > reward-shaping hacks > algorithm tweaks.

**Diagnostic you can re-use.** If `mean_ep_len` climbs but the agent looks
boring in the live window (stationary, not reacting), suspect spec gaming
before celebrating. A trained agent that doesn't *visibly* dodge usually isn't.

---

## What we learned from the 3D port

The point of porting to 3D wasn't the 3D itself — it was finding out what
actually transfers from 2D work. Findings, with their honest surprises:

**What transferred cleanly:**
- All PPO hyperparameters worked as-is. No tuning. Same `LR=3e-4`,
  `ENT_COEF=0.005`, `NUM_STEPS=2048`, etc.
- The best-checkpoint pattern caught the same kind of late-training noise
  it did in 2D.
- The entropy / clipfrac / v_loss diagnostics meant the exact same things
  and pointed at the same conclusions.
- Curriculum API (`CURRICULUM_PRESETS`, `--curriculum`) compiled clean —
  same code, just different defaults per env.

**What broke / required new thinking:**
- **Action space went from 9 to 27.** Initial entropy is `ln(27) ≈ 3.30`
  (vs `ln(9) ≈ 2.20`). Entropy still dropped to ~1.7 by end-of-training,
  but the per-action probability mass starts much smaller.
- **`obs_dim` is now 54** (6 agent + 8 projectiles × 6). Old 2D checkpoints
  can't load — the safety check in `load_agent` catches this.
- **Separate checkpoint files** were necessary. Mixing 2D and 3D into the
  same `ppo_dodge.pt` would have silently broken on the next run when the
  obs/action shapes flipped.

**What surprised me (the prediction was: "I don't know"):**
- **The 3D *random* baseline is much higher (499.2) than 2D's (267.4).**
  Reason: `AIM_RADIUS=100` defines an offset *ball* in 3D vs an offset
  *disk* in 2D. The agent occupies a fixed sphere of radius 15. The
  ratio "agent volume / aim-target volume" is ~0.3% in 3D vs ~2.2% in
  2D — projectiles miss far more often in 3D, even when aimed. This means
  "beat random" is a much higher bar in 3D, and the env may be *too*
  forgiving at the current `AIM_RADIUS`. If you want the 3D env to be as
  hard relative to random as the 2D env is, drop `AIM_RADIUS` in
  `dodge_env_3d.py` to ~40-50.
- **3D training was *less* regressive than 2D.** Peak rolling-mean
  (692.9) was at update ~485 of 488 — almost no drift. My guess: 27
  actions means each PPO update is averaging gradient signal across a
  wider distribution, so single-step value-function poisoning has less
  bite. Worth confirming with a longer run.
- **The argmax/sampled gap stayed (~10%)**, just like in 2D. Same fix
  applies: a lower-`ENT_COEF` polish stage would close it.

**Open question I'd want to answer next.** Whether **continuous actions**
(swap `Categorical` for `Normal` in the actor head) would be cleanly
better than 27-way discrete. Argument for: continuous is a more natural
fit for "3D unit vector velocity." Argument against: discrete-with-many-
actions has been working fine and continuous needs a separate codepath
(reparam trick, no clip on logits, etc).

---

## Curriculum learning — what we learned

The flat target-difficulty training plateaued at mean_ep_len ≈ 290 (deterministic
eval). A two-stage curriculum (easy 1M → target 500k transfer) reached 444.5 on
the same eval — a **+54% lift** with the same total compute budget.

The mechanism: in the easy env, dodging is easier *and* the reward gradient is
stronger because more actions actually save you. The policy commits faster
(entropy drops from 2.19 → ~1.5). When you `--resume` into the target stage,
that pre-committed policy gives PPO a much better starting point than random
init does. Within ~50k target-stage steps the agent recovers above the
no-curriculum 1M-step peak.

**Practical takeaways for new env knobs:**
- Whenever a single difficulty setting plateaus, *don't* immediately reach for
  bigger networks or more steps. Add a curriculum preset to `CURRICULUM_PRESETS`
  first and try transfer.
- The new `BEST_CHECKPOINT_PATH` is per-session, so always `cp` your stage-1 best
  to a stable name *before* starting stage 2. The training script auto-deletes
  the old best at the start of each session.
- Pay attention to deterministic-eval vs sampled-training-mean. A wide gap
  (>15%) means entropy is too high — try a polish stage with lower `ENT_COEF`.

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Window won't close, script hangs | Old behavior (pre-fix) — the env wasn't checking for QUIT events. | Pull current `dodge_env.py` (now handles X-button and ESC). |
| `mean_ep_len` stuck near 267 | Agent isn't learning, or env got harder after a knob change. | Train longer, lower difficulty knobs, or check `entropy` (if 0, raise `ENT_COEF`). Try `--curriculum easy` first. |
| Argmax eval << sampled training mean | Policy entropy too high; argmax discards useful exploration. | Lower `ENT_COEF` (0.005 → 0.003), or run a polish stage with `--resume` and the lower entropy weight. |
| `ppo_dodge_best.pt` got overwritten by a worse run | Each training session auto-clears the best file. | Always `cp ppo_dodge_best.pt <archive>.pt` after a good run. The curriculum recipe in "Latest training snapshot" shows this. |
| `weights_only` error on load | PyTorch ≥ 2.6 default. | Already handled in `load_agent` (`weights_only=False`). |
| `obs_dim` mismatch when watching/resuming | You changed `MAX_PROJECTILES` or another env observation setting after saving the checkpoint. | Start fresh without `--resume`; old checkpoints are for the old env shape. |
| Training is super slow | You probably have `--render` on. | Drop the flag; train headless and view via Terminal 2. |
| Agent just sits in a corner | Reward-hacking: far edge spawns gave too much reaction time. | Current env uses local ring spawns near the agent (`SPAWN_DISTANCE_MIN/MAX`) and aims near the agent (`AIM_RADIUS`). |
