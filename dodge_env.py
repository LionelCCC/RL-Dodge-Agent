"""
DodgeEnv: a 2D arena where an agent dodges straight-line projectiles.
Conforms to the Gymnasium API so any RL algorithm can train on it.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


# ---- Constants. Put them at the top so they're easy to tweak. ----
ARENA_W = 800           # arena width in pixels/units
ARENA_H = 600           # arena height
AGENT_RADIUS = 15
PROJECTILE_RADIUS = 5
AGENT_SPEED = 5.0       # units per step
PROJECTILE_SPEED = 4.0  # units per step
MAX_PROJECTILES = 8     # the K in "K closest projectiles the agent sees"
MAX_EPISODE_STEPS = 1000  # max length of ONE episode (not the total training amount!)
SPAWN_PROB = 0.03       # chance per step that a new projectile spawns (tunable)
AIM_RADIUS = 100        # projectiles aim within this many pixels of the agent.
                        # Small = surgical (hard). Large = spray (easy).
SPAWN_DISTANCE_MIN = 220  # target difficulty: local spawns this far from the agent
SPAWN_DISTANCE_MAX = 320  # wider values are easier because reaction time is longer

CURRICULUM_PRESETS = {
    "target": (SPAWN_DISTANCE_MIN, SPAWN_DISTANCE_MAX),
    "easy": (280, 400),
}

# The 9 discrete actions, as (dx, dy) unit vectors.
# Index 0 = stay still, 1-8 = the 8 compass directions.
# We use 1/sqrt(2) ~= 0.707 for diagonals so speed is the same in all directions.
_INV_SQRT2 = 1.0 / np.sqrt(2.0)
ACTIONS = np.array([
    [ 0,  0],            # 0: stay
    [ 0, -1],            # 1: N (up)
    [ _INV_SQRT2, -_INV_SQRT2],  # 2: NE
    [ 1,  0],            # 3: E
    [ _INV_SQRT2,  _INV_SQRT2],  # 4: SE
    [ 0,  1],            # 5: S (down)
    [-_INV_SQRT2,  _INV_SQRT2],  # 6: SW
    [-1,  0],            # 7: W
    [-_INV_SQRT2, -_INV_SQRT2],  # 8: NW
], dtype=np.float32)


class DodgeEnv(gym.Env):
    """The dodging environment, Gymnasium-compatible."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        render_mode=None,
        spawn_distance_min=SPAWN_DISTANCE_MIN,
        spawn_distance_max=SPAWN_DISTANCE_MAX,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.spawn_distance_min = float(spawn_distance_min)
        self.spawn_distance_max = float(spawn_distance_max)
        if self.spawn_distance_min <= 0:
            raise ValueError("spawn_distance_min must be positive")
        if self.spawn_distance_max < self.spawn_distance_min:
            raise ValueError("spawn_distance_max must be >= spawn_distance_min")

        # Observation: 4 (agent) + MAX_PROJECTILES * 4 (each projectile) floats
        obs_dim = 4 + MAX_PROJECTILES * 4
        # We bound observations roughly by arena size. Not strictly enforced,
        # but it tells downstream code the rough scale.
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # 9 discrete actions: stay still + 8 compass directions
        self.action_space = spaces.Discrete(9)

        # Internal state, populated in reset()
        self.agent_pos = None      # np.array([x, y])
        self.agent_vel = None      # np.array([vx, vy]) — last frame's velocity
        self.projectiles = None    # list of dicts with pos and vel
        self.steps = 0

        # For rendering (we'll fill this in later)
        self._screen = None
        self._clock = None

        # Flag set by render() when the user closes the window or hits ESC.
        # The training/watch loop should poll this and exit cleanly.
        self.closed_by_user = False

    # ------------------------------------------------------------------
    # Gymnasium API: reset()
    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        """Start a new episode. Returns (initial_observation, info_dict)."""
        super().reset(seed=seed)  # seeds self.np_random for reproducibility

        # Agent starts in the center of the arena, at rest.
        self.agent_pos = np.array([ARENA_W / 2, ARENA_H / 2], dtype=np.float32)
        self.agent_vel = np.array([0.0, 0.0], dtype=np.float32)

        # Start with no projectiles. They'll spawn over time inside step().
        self.projectiles = []
        self.steps = 0

        return self._get_obs(), {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _sample_projectile_spawn_pos(self):
        """Sample a spawn point near the agent but still inside the arena.

        Edge spawns are exploitable: a corner-camping agent gets too much
        warning time from far-side projectiles. Local ring spawns make every
        spawn relevant while still leaving a readable reaction window.
        """
        low = PROJECTILE_RADIUS
        high_x = ARENA_W - PROJECTILE_RADIUS
        high_y = ARENA_H - PROJECTILE_RADIUS

        for _ in range(100):
            angle = self.np_random.uniform(0.0, 2.0 * np.pi)
            dist = self.np_random.uniform(self.spawn_distance_min, self.spawn_distance_max)
            pos = self.agent_pos + dist * np.array([np.cos(angle), np.sin(angle)])
            if low <= pos[0] <= high_x and low <= pos[1] <= high_y:
                return pos.astype(np.float32)

        # Fallback for extreme corner cases: sample anywhere in the arena
        # until the projectile is far enough to be dodgeable.
        for _ in range(100):
            pos = np.array([
                self.np_random.uniform(low, high_x),
                self.np_random.uniform(low, high_y),
            ], dtype=np.float32)
            dist = np.linalg.norm(pos - self.agent_pos)
            if self.spawn_distance_min <= dist <= self.spawn_distance_max:
                return pos

        # Last-resort fallback should almost never run, but keeps reset/step
        # total and avoids crashing if constants are changed aggressively.
        direction = np.array([ARENA_W / 2, ARENA_H / 2], dtype=np.float32) - self.agent_pos
        norm = np.linalg.norm(direction) + 1e-8
        pos = self.agent_pos + direction / norm * self.spawn_distance_min
        pos[0] = np.clip(pos[0], low, high_x)
        pos[1] = np.clip(pos[1], low, high_y)
        return pos.astype(np.float32)

    def _spawn_projectile(self):
        """Spawn one local projectile, aimed near the agent.

        Aim offset (radius AIM_RADIUS) is the precision difficulty dial:
          - small radius  -> projectiles aim right at you (hard)
          - large radius  -> "spray" pattern (easy)
        """
        spawn_pos = self._sample_projectile_spawn_pos()
        x, y = spawn_pos

        # Aim at a random point inside a disk of radius AIM_RADIUS around
        # the agent's CURRENT position. sqrt() on the radius makes the
        # samples uniformly distributed over the disk's area (otherwise
        # they cluster near the center).
        angle = self.np_random.uniform(0.0, 2.0 * np.pi)
        r = AIM_RADIUS * np.sqrt(self.np_random.random())
        target_x = self.agent_pos[0] + r * np.cos(angle)
        target_y = self.agent_pos[1] + r * np.sin(angle)

        dx = target_x - x
        dy = target_y - y
        norm = np.sqrt(dx * dx + dy * dy) + 1e-8  # avoid divide-by-zero
        vx = (dx / norm) * PROJECTILE_SPEED
        vy = (dy / norm) * PROJECTILE_SPEED

        self.projectiles.append({
            "pos": spawn_pos,
            "vel": np.array([vx, vy], dtype=np.float32),
        })

    def _get_obs(self):
        """Build the observation vector the agent sees."""
        obs = np.zeros(4 + MAX_PROJECTILES * 4, dtype=np.float32)

        # Agent state: position normalized to roughly [-1, 1], velocity normalized.
        # Normalization makes the neural net's life MUCH easier.
        obs[0] = (self.agent_pos[0] - ARENA_W / 2) / (ARENA_W / 2)
        obs[1] = (self.agent_pos[1] - ARENA_H / 2) / (ARENA_H / 2)
        obs[2] = self.agent_vel[0] / AGENT_SPEED
        obs[3] = self.agent_vel[1] / AGENT_SPEED

        # Sort projectiles by distance to agent (closest first).
        if len(self.projectiles) > 0:
            dists = [
                np.linalg.norm(p["pos"] - self.agent_pos)
                for p in self.projectiles
            ]
            order = np.argsort(dists)
            # Take up to MAX_PROJECTILES of the closest ones
            for slot, idx in enumerate(order[:MAX_PROJECTILES]):
                p = self.projectiles[idx]
                # Egocentric: positions are RELATIVE to the agent.
                rel = p["pos"] - self.agent_pos
                offset = 4 + slot * 4
                obs[offset + 0] = rel[0] / (ARENA_W / 2)  # normalized relative x
                obs[offset + 1] = rel[1] / (ARENA_H / 2)  # normalized relative y
                obs[offset + 2] = p["vel"][0] / PROJECTILE_SPEED  # normalized vx
                obs[offset + 3] = p["vel"][1] / PROJECTILE_SPEED  # normalized vy
            # Empty slots stay as zeros. The agent learns "zeros = no projectile here."

        return obs

    # ------------------------------------------------------------------
    # Gymnasium API: step()
    # ------------------------------------------------------------------
    def step(self, action):
        """
        Advance the world by one timestep given the agent's chosen action.
        Returns (next_obs, reward, terminated, truncated, info).
        """
        self.steps += 1

        # --- 1. Move the agent according to its chosen action ---
        direction = ACTIONS[action]                     # unit vector for this action
        self.agent_vel = direction * AGENT_SPEED        # remembered for the next obs
        self.agent_pos += self.agent_vel

        # Keep the agent inside the arena (clip to walls).
        self.agent_pos[0] = np.clip(self.agent_pos[0], AGENT_RADIUS, ARENA_W - AGENT_RADIUS)
        self.agent_pos[1] = np.clip(self.agent_pos[1], AGENT_RADIUS, ARENA_H - AGENT_RADIUS)

        # --- 2. Move every projectile forward ---
        for p in self.projectiles:
            p["pos"] += p["vel"]

        # --- 3. Remove projectiles that have left the arena ---
        # (with a bit of margin so they fully exit before disappearing)
        margin = 50
        self.projectiles = [
            p for p in self.projectiles
            if -margin <= p["pos"][0] <= ARENA_W + margin
            and -margin <= p["pos"][1] <= ARENA_H + margin
        ]

        # --- 4. Maybe spawn a new projectile ---
        if self.np_random.random() < SPAWN_PROB:
            self._spawn_projectile()

        # --- 5. Check for collisions (agent vs any projectile) ---
        hit = False
        collision_radius = AGENT_RADIUS + PROJECTILE_RADIUS
        for p in self.projectiles:
            d = np.linalg.norm(p["pos"] - self.agent_pos)
            if d < collision_radius:
                hit = True
                break

        # --- 6. Compute reward, terminated, truncated ---
        # Reward = +1 per step survived. Death gives 0 (clean, no death penalty).
        reward = 1.0
        terminated = hit                       # natural end (agent died)
        truncated = self.steps >= MAX_EPISODE_STEPS  # external time limit

        return self._get_obs(), reward, terminated, truncated, {}

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def poll_close_event(self):
        """Return True if the user requested the render window to close."""
        if self.render_mode != "human" or self._screen is None:
            return self.closed_by_user

        import pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.closed_by_user = True
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.closed_by_user = True
        return self.closed_by_user

    def render(self):
        """Draw the arena with pygame. Lazy-init the window on first call."""
        if self.render_mode is None:
            return None

        import pygame
        if self._screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self._screen = pygame.display.set_mode((ARENA_W, ARENA_H))
                pygame.display.set_caption("DodgeEnv")
            else:  # rgb_array
                self._screen = pygame.Surface((ARENA_W, ARENA_H))
            self._clock = pygame.time.Clock()

        # Background
        self._screen.fill((20, 20, 30))  # near-black with a hint of blue

        # Agent (cyan circle)
        pygame.draw.circle(
            self._screen, (0, 200, 255),
            (int(self.agent_pos[0]), int(self.agent_pos[1])),
            AGENT_RADIUS
        )

        # Projectiles (red circles)
        for p in self.projectiles:
            pygame.draw.circle(
                self._screen, (255, 80, 80),
                (int(p["pos"][0]), int(p["pos"][1])),
                PROJECTILE_RADIUS
            )

        if self.render_mode == "human":
            # Drain events. If the user closed the window or pressed ESC,
            # set self.closed_by_user so the caller can break out of its
            # loop cleanly (instead of the script hanging).
            self.poll_close_event()
            pygame.display.flip()
            self._clock.tick(self.metadata["render_fps"])
            return None
        else:
            # rgb_array: return a numpy array of the rendered frame
            arr = pygame.surfarray.array3d(self._screen)
            return np.transpose(arr, (1, 0, 2))

    def close(self):
        if self._screen is not None:
            import pygame
            if self.render_mode == "human":
                pygame.display.quit()
            pygame.quit()
            self._screen = None
