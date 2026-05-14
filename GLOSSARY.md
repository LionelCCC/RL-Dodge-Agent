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
| `TOTAL_TIMESTEPS` | Total numer of steps of the entire training--how long the whole training session runs. | Default `300_000` calls to `env.step()`. This is the big "train more" knob. | This is not episodes. If episodes average 300 steps, then 300k steps is roughly 1000 episodes. |
| `NUM_STEPS` | How many steps the agent collects before PPO learns from them. | Default `2048` steps per rollout/update. | Bigger means fewer updates but more data per update. |
| `NUM_MINIBATCHES` | How many chunks each rollout is split into during learning. | Default `32`. With 2048 steps, each minibatch has 64 samples. | Too many tiny minibatches can make updates noisy. |
| `UPDATE_EPOCHS` | How many times PPO reuses the same rollout during one update. | Default `10`. | Too high can overfit to the latest rollout. |
| `LR` | Learning rate: how big each optimizer step is. | Default `3e-4`. | If training is unstable, try `1e-4`. |
| `ANNEAL_LR` | Whether learning rate fades down during training. | Default `True`. | Usually keep this on for PPO. |
| `GAMMA` | How much the agent cares about future survival. | Default `0.99`. | Lower values make the agent more short-sighted. |
| `GAE_LAMBDA` | Smooths advantage estimates. | Default `0.95`. | Standard PPO value; don't tune first. |
| `CLIP_COEF` | Limits how much the policy can change in one update. | Default `0.2`. | If `clipfrac` is always huge, reduce LR before touching this. |
| `ENT_COEF` | Exploration pressure. | Default `0.01`. Higher means more random/exploratory. | If entropy crashes early, raise this a bit. |
| `VF_COEF` | How much critic/value loss matters in total loss. | Default `0.5`. | Standard PPO value; don't tune first. |
| `MAX_GRAD_NORM` | Caps gradient size. | Default `0.5`. | Safety rail against unstable updates. |
| `HIDDEN_DIM` | Neural net hidden layer width. | Default `64`. | Bigger isn't automatically better for this small env. |
| `CHECKPOINT_PATH` | Where learning is saved. | Default `ppo_dodge.pt`. | Running fresh training can overwrite this. |
| `CHECKPOINT_INTERVAL` | How often training auto-saves. | Default every `10` updates. | More frequent saves are safer but slightly noisier on disk. |

### Environment/game knobs (`dodge_env.py`)

| Name | Plain English | In this project | Be careful |
|---|---|---|---|
| `ARENA_W`, `ARENA_H` | Width and height of the playfield. | `800 x 600`. | Bigger arena usually makes dodging easier. |
| `AGENT_RADIUS` | Size of the player circle. | `15` pixels. | Bigger agent is easier to hit. |
| `PROJECTILE_RADIUS` | Size of projectile circles. | `5` pixels. | Bigger projectile is harder. |
| `AGENT_SPEED` | How far the agent moves per step. | `5.0` pixels/frame. | Faster agent is easier. |
| `PROJECTILE_SPEED` | How far projectiles move per step. | `4.0` pixels/frame. | Faster projectiles are harder. |
| `MAX_PROJECTILES` | How many closest projectiles the observation includes. | `8`, so observation has `4 + 8*4 = 36` floats. | Changing this breaks old checkpoints because obs size changes. |
| `MAX_EPISODE_STEPS` | Max length of one episode. | `1000` steps. | This is not total training length. It only caps one run/play. |
| `SPAWN_PROB` | Chance of spawning one projectile each step. | `0.03`, about 30 spawn attempts per 1000-step episode. | Higher means denser danger. |
| `AIM_RADIUS` | How close to the agent each projectile aims. | `100` pixels. | Smaller is more precise and harder; larger is more spray-like and easier. |
| `SPAWN_DISTANCE_MIN/MAX` | How far from the agent projectiles appear. | `220` to `320` pixels away. | Too far gives camping reaction time; too close can feel unfair/impossible. |
| `ACTIONS` | The 9 choices the policy can output. | stay, N, NE, E, SE, S, SW, W, NW. | Diagonals are normalized so they are not faster than straight movement. |

### Command-line knobs

| Command option | Meaning | Example |
|---|---|---|
| `ppo_agent.py --steps N` | Train for N total env steps this session. | `python3 ppo_agent.py --steps 1_000_000` |
| `ppo_agent.py --resume` | Continue from `ppo_dodge.pt`. | `python3 ppo_agent.py --resume` |
| `ppo_agent.py --render` | Watch the training loop live, slower. | `python3 ppo_agent.py --render` |
| `watch_agent.py --mode human` | Open live side window using saved checkpoint. | `python3 watch_agent.py --mode human` |
| `watch_agent.py --mode eval` | Run saved checkpoint headlessly and print stats. | `python3 watch_agent.py --mode eval --episodes 30` |
| `watch_agent.py --mode gif` | Save a visual sample GIF. | `python3 watch_agent.py --mode gif --out trained_agent.gif` |
| `watch_agent.py --stochastic` | Sample actions instead of using best action. | `python3 watch_agent.py --mode human --stochastic` |
| `watch_agent.py --episodes N` | Number of episodes to watch/evaluate. | `python3 watch_agent.py --mode eval --episodes 50` |
| `watch_agent.py --seed N` | Pick a reproducible scenario. | `python3 watch_agent.py --mode gif --seed 7` |

---

## 1. Time units — how big is each thing?

These four words get mixed up constantly. Sort them once and reading the
log gets much easier.

| Term | What it is | In this project |
|---|---|---|
| **step** | One call to `env.step(action)`. The smallest unit of time in RL. | One physics frame: agent moves, projectiles move, collision is checked. |
| **episode** | One full play from `env.reset()` until the agent dies (terminated) or hits the timeout (truncated). Made of many steps. | Starts with agent in the center of the arena, ends when a projectile hits it or `MAX_EPISODE_STEPS=1000` is reached. |
| **rollout** | A *batch* of `NUM_STEPS` consecutive steps the agent collects before each learning update. One rollout can span many episodes (or part of one). | We collect 2048 steps, then learn from them, then collect another 2048. |
| **update** | One PPO learning update: `UPDATE_EPOCHS=10` passes over the rollout, each pass split into `NUM_MINIBATCHES=32` minibatches. | After every rollout, we run ~320 gradient steps. |
| **total_timesteps** | Total number of `env.step` calls for the WHOLE training session. The big knob for "how long to train." | Default 300,000. So `300_000 / NUM_STEPS = ~146` updates. |
| **global_step** | A running counter of env steps so far in this session. Printed in every log line. | Starts at 0, ends at `total_timesteps`. |

**Mental model:**
```
total_timesteps  ⊇  many updates  ⊇  one rollout (NUM_STEPS env interactions)
                                  ⊇  several episodes  ⊇  many steps
```

`MAX_EPISODE_STEPS` (in `dodge_env.py`) caps *one* episode at 1000 steps.
`TOTAL_TIMESTEPS` (in `ppo_agent.py`) caps the *whole training session*.
These are different things — don't confuse them.

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
| **advantage (A(s, a))** | "How much *better* than expected was taking action `a` in state `s`?" Positive → that action was better than the value function predicted, push the policy toward it. Negative → worse than expected, push away. |
| **GAE (Generalized Advantage Estimation)** | The specific recipe we use to *estimate* advantages. Trades off bias and variance via `GAE_LAMBDA`. |
| **`GAE_LAMBDA`** | 0 = lowest variance, highest bias. 1 = unbiased, very noisy. 0.95 is the standard sweet spot. |

---

## 4. The PPO losses (what the log columns are)

Each PPO update computes three losses and adds them together:

| Term | What it measures | What "good" looks like |
|---|---|---|
| **`pg_loss`** (policy gradient loss) | The PPO clipped objective for the policy. The thing that pushes the policy toward higher-advantage actions. | Typically small in magnitude, can be positive or negative. Big jumps mean the policy is changing a lot. |
| **`v_loss`** (value loss) | Mean squared error between predicted `V(s)` and the actual return that came after. The critic's "how wrong was I?" | Should generally *decrease* over training as the critic gets better. Starts high (untrained critic). |
| **`ent_loss` / entropy** | How *uncertain* the policy is. `ln(num_actions)` is the maximum (uniform distribution). | Starts near `ln(9) ≈ 2.197` (uniform), drifts down as the policy commits to good actions. Too low too fast = stopped exploring. |
| **`ENT_COEF`** | Weight on the entropy bonus in the total loss. Higher → more exploration pressure. | 0.01 is standard. |
| **`VF_COEF`** | Weight on the value loss. | 0.5 is standard. |

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
| **`CLIP_COEF`** | The "trust region" size. We clip the ratio into `[1 - CLIP_COEF, 1 + CLIP_COEF] = [0.8, 1.2]`. Updates that would push the ratio outside this band are clipped. |
| **`clipfrac`** | Fraction of minibatch samples whose ratio was outside the clip range. **High clipfrac (>0.3)** = trying to take big steps, getting clamped. **Low clipfrac (~0)** = small steps. Both are normal at different stages. |

---

## 6. Throughput & misc

| Term | Meaning |
|---|---|
| **`sps`** | Steps Per Second. Training throughput. ~4000 sps headless, ~30 sps with `--render`. |
| **`mean_ep_len`** | Mean episode length over the last 50 finished episodes. This is the number to watch — it should climb as the agent learns. |
| **`MAX_GRAD_NORM`** | Global gradient norm clip. Prevents a single bad update from blowing up the network. 0.5 is standard. |
| **`LR` / learning rate** | Adam's step size. `3e-4` is the PPO default. We linearly anneal it to 0 by the end (`ANNEAL_LR=True`). |
| **`HIDDEN_DIM`** | Width of the MLP hidden layers. 64 is plenty for the current 36-dim observation. |
| **`NUM_MINIBATCHES`** | How many minibatches we split each rollout into. Bigger minibatches = lower variance, fewer updates. |
| **`UPDATE_EPOCHS`** | How many times we pass over each rollout. Too many → overfit to that rollout, policy moves too far. |

---

## 7. Environment-side knobs (in `dodge_env.py`)

| Constant | What it controls | Effect of raising it |
|---|---|---|
| `ARENA_W`, `ARENA_H` | Arena size in pixels | Bigger room, more dodging space — easier. |
| `AGENT_SPEED` | Pixels/step the agent moves | Faster agent — easier. |
| `PROJECTILE_SPEED` | Pixels/step projectiles move | Faster projectiles — harder. |
| `SPAWN_PROB` | Probability of spawning a projectile per step | More projectiles — harder. |
| `MAX_PROJECTILES` | How many *closest* projectiles the agent's observation includes | More info — easier (more compute too). |
| `AIM_RADIUS` | Projectiles aim at `agent_pos + offset(radius=AIM_RADIUS)` | **Bigger = easier** (sprays more, easier to dodge). **Smaller = harder** (more surgical). 100 px is a moderate setting. |
| `SPAWN_DISTANCE_MIN/MAX` | Projectiles spawn on a local ring this far from the agent, instead of from arena edges. | Smaller = less reaction time, harder. Larger = more reaction time, easier. |
| `MAX_EPISODE_STEPS` | Per-episode timeout | Higher ceiling for "perfect" runs but doesn't change difficulty. |

---

## 8. What "learning" looks like in the logs

Healthy training has all of these moving in the right direction:

- `mean_ep_len` **going up** (this is the headline number)
- `v_loss` **going down** (critic getting more accurate)
- `entropy` slowly **going down** (policy committing to good actions)
- `clipfrac` settling somewhere in `[0.05, 0.25]` (updates are non-trivial but not crazy)
- `sps` roughly constant (throughput stable)

**Warning signs**
- `mean_ep_len` flat at random-baseline level for many updates → policy isn't learning. Bug, or env is too hard, or LR too low.
- Entropy crashes to 0 fast → policy committed too early, won't explore further.
- `v_loss` exploding → critic diverging. Lower LR or check for reward magnitudes.
- `clipfrac` > 0.5 for sustained periods → updates getting heavily clipped. Reduce LR or `UPDATE_EPOCHS`.
