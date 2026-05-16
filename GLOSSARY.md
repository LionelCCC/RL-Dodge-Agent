# Glossary — what every term means in *this* project

A quick-reference for the words that show up in the training log and the
hyperparameter block. Grouped by "what kind of thing it is."

---

## 0. Parameter tracker — the knobs in plain English

This table is the quick "what does this control?" map. If you tune something,
start here.

### Training/session knobs (`ppo_agent.py`)

| Name | Plain English | In this project | Be careful |
|---|---|---|---|
| `TOTAL_TIMESTEPS` | Total number of env steps for the entire training session. | Default `300_000`. The big "train more" knob. | Not episodes. If episodes average 300 steps, then 300k steps is roughly 1000 episodes. |
| `NUM_STEPS` | How many steps the agent collects before PPO learns from them. | Default `2048` per rollout/update. | Bigger means fewer updates but more data per update. |
| `NUM_MINIBATCHES` | How many chunks each rollout is split into during learning. | Default `32`. With 2048 steps, each minibatch has 64 samples. | Too many tiny minibatches can make updates noisy. |
| `UPDATE_EPOCHS` | How many times PPO reuses the same rollout during one update. | Default `10`. | Too high can overfit to the latest rollout. |
| `LR` | Learning rate: how big each optimizer step is. | Default `3e-4`. | If training is unstable, try `1e-4`. |
| `ANNEAL_LR` | Whether learning rate fades down during training. | Default `False`. | Leaving it off makes short runs and resumed runs less likely to freeze at near-zero LR. |
| `GAMMA` | How much the agent cares about future survival. | Default `0.99`. | Lower values make the agent more short-sighted. |
| `GAE_LAMBDA` | Smooths advantage estimates. | Default `0.95`. | Standard PPO value; don't tune first. |
| `CLIP_COEF` | Limits how much the policy can change in one update. | Default `0.2`. | If `clipfrac` is always huge, reduce LR before touching this. |
| `ENT_COEF` | Exploration pressure. | Default `0.005`. Higher = more random/exploratory. | If argmax eval trails sampled-training mean by >15%, a polish stage with `ENT_COEF=0.003` is the canonical next experiment. |
| `VF_COEF` | How much critic/value loss matters in total loss. | Default `0.5`. | Standard PPO value; don't tune first. |
| `MAX_GRAD_NORM` | Caps gradient size. | Default `0.5`. | Safety rail against unstable updates. |
| `HIDDEN_DIM` | Neural net hidden layer width. | Default `64`. | Bigger isn't automatically better for this small env. |
| `CHECKPOINT_PATH` | Where learning is saved. | `checkpoints/ppo_dodge.pt`. | Running fresh training can overwrite this. |
| `BEST_CHECKPOINT_PATH` | Where the best rolling-mean policy for the current run is saved. | `checkpoints/ppo_dodge_best.pt`. | Use this for eval/watch because PPO can peak and then regress. Archive before transfer if you want to keep the easy-stage source. |
| `CHECKPOINT_INTERVAL` | How often training auto-saves. | Default every `10` updates. | More frequent saves are safer but slightly noisier on disk. |

### Environment/game knobs (`dodge_env.py`)

| Name | Plain English | In this project | Be careful |
|---|---|---|---|
| `ARENA_RADIUS` | Radius of the spherical arena, centered at origin. | Default `200`. | Smaller = less room to maneuver. |
| `AGENT_RADIUS` | Size of the player sphere. | `15` world units. | Bigger agent is easier to hit. |
| `PROJECTILE_RADIUS` | Size of each projectile. | `5`. | Bigger projectile is harder. |
| `AGENT_SPEED` | How far the agent moves per step. | `5.0`. | Faster agent is easier. |
| `PROJECTILE_SPEED` | How far projectiles move per step. | `4.0`. | Faster projectiles are harder. |
| `MAX_PROJECTILES` | How many *closest visible* projectiles the obs includes. | `8`, so obs = `6 + 8*6 = 54` floats. | Changing this breaks old checkpoints because obs size changes. |
| `MAX_EPISODE_STEPS` | Max length of one episode. | `1000`. | Not total training length. Only caps one play. |
| `SPAWN_PROB` | Chance of spawning one projectile each step. | Default `0.06` (~60 attempts per 1000-step episode). | Higher = denser danger. |
| `AIM_RADIUS` | How close to the agent each projectile aims. | `100`. | Smaller = more surgical and harder; larger = more spray-like and easier. |
| `SPAWN_DISTANCE_MIN/MAX` | Shell thickness *outside* `ARENA_RADIUS` where projectiles spawn. | Target: `50..200`. Easy: `100..300`. | The closer the shell, the less reaction time. |
| `PERCEPTION_RADIUS` | Beyond this distance, projectiles are zeroed out of the obs. | Default `200`. | Smaller = harder (agent flies blind on distant threats). |
| `CURRICULUM_PRESETS` | Named difficulty presets used by `--curriculum`. | `target = 50..200`, `easy = 100..300`. | Prefer this over hand-editing constants. |
| `ACTIONS` | The 27 choices the policy can output. | every (dx, dy, dz) with each axis in `{-1, 0, +1}`. Index 13 = stay still. | Diagonals are normalized so they aren't faster than axial moves. |

### Command-line knobs

| Command option | Meaning | Example |
|---|---|---|
| `ppo_agent.py --steps N` | Train for N total env steps this session. | `python3 ppo_agent.py --steps 1_000_000` |
| `ppo_agent.py --resume` | Continue from `checkpoints/ppo_dodge.pt`. | `python3 ppo_agent.py --resume` |
| `ppo_agent.py --resume PATH` | Continue from a specific file. | `python3 ppo_agent.py --resume checkpoints/ppo_dodge_easy_best.pt` |
| `ppo_agent.py --render` | Watch the training loop live, slower. | `python3 ppo_agent.py --render` |
| `ppo_agent.py --curriculum easy` | Train with the easier spawn shell. | `python3 ppo_agent.py --steps 1_000_000 --curriculum easy` |
| `ppo_agent.py --curriculum target` | Train/eval on the real target difficulty. | (default) |
| `watch_agent.py --mode human` | Open live window using the best saved checkpoint. | `python3 watch_agent.py --mode human` |
| `watch_agent.py --mode eval` | Headless stats over N episodes. | `python3 watch_agent.py --mode eval --episodes 30` |
| `watch_agent.py --mode gif` | Save a visual sample GIF. | `python3 watch_agent.py --mode gif --seed 7` |
| `watch_agent.py --stochastic` | Sample actions instead of argmax. | `python3 watch_agent.py --mode human --stochastic` |
| `random_agent.py --episodes N` | Random-action baseline over N episodes. | `python3 random_agent.py --episodes 30` |
| `position_diagnostic.py --policy {random,ppo}` | Check whether the agent's position distribution is symmetric / pinned. | `python3 position_diagnostic.py --policy ppo` |

---

## 1. Time units — how big is each thing?

These four words get mixed up constantly. Sort them once and reading the
log gets much easier.

| Term | What it is | In this project |
|---|---|---|
| **step** | One call to `env.step(action)`. The smallest unit of time in RL. | One physics frame: agent moves, projectiles move, collision is checked. |
| **episode** | One full play from `env.reset()` until the agent dies (terminated) or hits the timeout (truncated). Made of many steps. | Starts with agent at the origin, ends when a projectile hits or `MAX_EPISODE_STEPS=1000` is reached. |
| **rollout** | A *batch* of `NUM_STEPS` consecutive steps the agent collects before each learning update. One rollout can span many episodes (or part of one). | We collect 2048 steps, then learn from them, then collect another 2048. |
| **update** | One PPO learning update: `UPDATE_EPOCHS=10` passes over the rollout, each pass split into `NUM_MINIBATCHES=32` minibatches. | After every rollout, we run ~320 gradient steps. |
| **total_timesteps** | Total number of `env.step` calls for the WHOLE training session. The big "how long to train" knob. | Default 300,000 → `300_000 / NUM_STEPS = ~146` updates. |
| **global_step** | A running counter of env steps so far in this session. Printed in every log line. | Starts at 0, ends at `total_timesteps`. |

**Mental model:**
```
total_timesteps  ⊇  many updates  ⊇  one rollout (NUM_STEPS env interactions)
                                  ⊇  several episodes  ⊇  many steps
```

`MAX_EPISODE_STEPS` (in `dodge_env.py`) caps *one* episode at 1000 steps.
`TOTAL_TIMESTEPS` (in `ppo_agent.py`) caps the *whole training session*.

---

## 2. Reward & return — what the agent is optimizing

| Term | Meaning |
|---|---|
| **reward** | The signal the env hands back at each step. In DodgeEnv it's `+1` for surviving the step. (Death gives nothing — there's no penalty term.) |
| **return** | Sum of rewards over an episode. Since reward is `+1`/step, **return == episode length**. So in our log, `mean_ep_rew` and `mean_ep_len` are the same number. |
| **discount factor / `GAMMA`** | How much future rewards count *now*. `gamma=0.99` means a reward 100 steps away is worth `0.99^100 ≈ 0.37` of a reward right now. Lower gamma → more short-sighted. |

---

## 3. The value function and advantages

PPO is *actor-critic*: there's a **policy** (chooses actions) and a **value function** (predicts how good a state is).

| Term | Meaning |
|---|---|
| **value (V(s))** | The critic's estimate of "starting from state `s`, how much return will I get?" If `s` looks safe, V(s) is high; if it's about to get hit, V(s) is low. |
| **advantage (A(s, a))** | "How much *better* than expected was taking action `a` in state `s`?" Positive → push the policy toward it. Negative → push away. |
| **GAE (Generalized Advantage Estimation)** | The specific recipe we use to *estimate* advantages. Trades off bias and variance via `GAE_LAMBDA`. |
| **`GAE_LAMBDA`** | 0 = lowest variance, highest bias. 1 = unbiased, very noisy. 0.95 is the standard sweet spot. |

---

## 4. The PPO losses (what the log columns are)

Each PPO update computes three losses and adds them together:

| Term | What it measures | What "good" looks like |
|---|---|---|
| **`pg_loss`** | The PPO clipped objective for the policy. Pushes the policy toward higher-advantage actions. | Typically small in magnitude, can be positive or negative. Big jumps mean the policy is changing a lot. |
| **`v_loss`** | Mean squared error between predicted `V(s)` and the actual return. The critic's "how wrong was I?" | Should generally *decrease* over training as the critic gets better. Starts high (untrained critic). |
| **`ent_loss` / entropy** | How *uncertain* the policy is. `ln(num_actions)` is the maximum (uniform). | Starts near `ln(27) ≈ 3.296` (uniform), drifts down as the policy commits. Too low too fast = stopped exploring. |
| **`ENT_COEF`** | Weight on the entropy bonus in the total loss. | Default `0.005`. |
| **`VF_COEF`** | Weight on the value loss. | `0.5`, standard. |

### How the three are combined
```
total_loss = pg_loss  -  ENT_COEF * entropy  +  VF_COEF * v_loss
```
(Subtract entropy because we *want* it high — gradient descent on `-entropy` *increases* entropy.)

---

## 5. The PPO clip mechanism

PPO's defining trick: don't let the policy change too much in one update.

| Term | Meaning |
|---|---|
| **ratio** | `new_policy(a|s) / old_policy(a|s)`. >1 means the new policy likes that action more than the old one did. |
| **`CLIP_COEF`** | Trust-region size. Clip the ratio into `[1 - CLIP_COEF, 1 + CLIP_COEF] = [0.8, 1.2]`. |
| **`clipfrac`** | Fraction of minibatch samples whose ratio was outside the clip range. **High clipfrac (>0.3)** = trying to take big steps, getting clamped. **Low (~0)** = small steps. Both are normal at different stages. |

---

## 6. Throughput & misc

| Term | Meaning |
|---|---|
| **`sps`** | Steps Per Second. ~4000 sps headless, ~30 sps with `--render`. |
| **`mean_ep_len`** | Mean episode length over the last 50 finished episodes. The number to watch — it should climb. |
| **`MAX_GRAD_NORM`** | Global gradient norm clip. Prevents a single bad update from blowing up the network. |
| **`HIDDEN_DIM`** | Width of the MLP hidden layers. 64 is plenty for the current 54-dim observation. |
| **`NUM_MINIBATCHES`** | How many minibatches we split each rollout into. Bigger minibatches = lower variance, fewer updates. |
| **`UPDATE_EPOCHS`** | How many times we pass over each rollout. Too many → overfit to that rollout, policy moves too far. |

---

## 7. Curriculum workflow — the canonical recipe

Use this when a single difficulty setting plateaus.

```bash
# Stage 1 — easy
python3 ppo_agent.py --steps 1_000_000 --curriculum easy

# Archive — the next session auto-clears ppo_dodge_best.pt
cp checkpoints/ppo_dodge_best.pt checkpoints/ppo_dodge_easy_best.pt

# Stage 2 — transfer to target
python3 ppo_agent.py --steps 500_000 --curriculum target \
    --resume checkpoints/ppo_dodge_easy_best.pt
```

Two non-obvious bits:
- `BEST_CHECKPOINT_PATH` is *per-session*. Stage 2 deletes the stage-1 best at
  start (after loading it via `--resume`), then writes its own.
- Auto-eval uses the `--curriculum` you passed to `ppo_agent.py`. So
  stage-1's auto-eval reports easy-difficulty numbers; the apples-to-apples
  comparison is `watch_agent.py --mode eval --curriculum target`.

---

## 8. What "learning" looks like in the logs

Healthy training has all of these moving in the right direction:

- `mean_ep_len` **going up** (the headline number)
- `v_loss` **going down** (critic getting more accurate)
- `entropy` slowly **going down** (policy committing to good actions)
- `clipfrac` settling somewhere in `[0.05, 0.25]`
- `sps` roughly constant

**Warning signs:**
- `mean_ep_len` flat at random-baseline level for many updates → policy isn't learning. Bug, env too hard, or LR too low.
- Entropy crashes to 0 fast → policy committed too early, won't explore further.
- `v_loss` exploding → critic diverging. Lower LR or check reward magnitudes.
- `clipfrac` > 0.5 for sustained periods → updates getting heavily clipped. Reduce LR or `UPDATE_EPOCHS`.
