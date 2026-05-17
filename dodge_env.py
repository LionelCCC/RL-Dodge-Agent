"""
DodgeEnv: 3D dodge-the-projectiles env with a spherical arena.

  - Arena is a sphere of radius ARENA_RADIUS, centered at the origin.
    Agent and projectiles live in centered coords (origin = arena center).
  - Projectiles spawn on a shell *outside* the arena (distance from origin
    = ARENA_RADIUS + uniform(SPAWN_DISTANCE_MIN, SPAWN_DISTANCE_MAX)).
    Always valid by construction — no rejection, no fallback.
  - The agent observes only projectiles within PERCEPTION_RADIUS. Beyond
    that, the slot is zero, so obs noise doesn't grow with projectile count.
  - Agent containment is `||pos|| <= ARENA_RADIUS - AGENT_RADIUS`. No corners,
    no walls — only "near the surface" states.
  - Reward = +1/step, episode caps at MAX_EPISODE_STEPS.

Action space: 27 discrete (every (dx, dy, dz) with each in {-1, 0, +1}).
Observation: 6 (agent xyzvxvyvz) + MAX_PROJECTILES * 6 (closest visible).

**Per-episode frame randomization.** Env dynamics are rotationally symmetric
(uniform spawn shell, isotropic aim), but absolute-axis observations and a
world-aligned action grid let a policy learn "+x is safe" anyway — which is
how the first trained policy ended up sliding along the +x+y+z octant
boundary. At each `reset()` we sample a uniform rotation matrix R_ep:
  - observations are rotated WORLD -> AGENT frame via R_ep.T,
  - actions are rotated AGENT -> WORLD frame via R_ep before applying.
The agent therefore operates in a randomly-oriented egocentric frame each
episode. World axes carry no signal; only "where the threats are right now
relative to me" does. Disable with `randomize_frame=False` for ablations.

Note on `spawn_distance_min/max`: these are SHELL distance (distance OUTSIDE
the sphere wall), not absolute distance from the agent.
"""

import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces


# ---- Physics constants -------------------------------------------------------
ARENA_RADIUS = 200
AGENT_RADIUS = 15
PROJECTILE_RADIUS = 5
AGENT_SPEED = 5.0
PROJECTILE_SPEED = 6.0
MAX_PROJECTILES = 8
MAX_EPISODE_STEPS = 1000
SPAWN_PROB = 0.06
AIM_RADIUS = 100
SPAWN_DISTANCE_MIN = 50           # shell thickness — distance OUTSIDE the sphere
SPAWN_DISTANCE_MAX = 200
PERCEPTION_RADIUS = 200           # agent sees projectiles within this radius

CURRICULUM_PRESETS = {
    "target": (SPAWN_DISTANCE_MIN, SPAWN_DISTANCE_MAX),
    "easy": (100, 300),
}

# Per-episode frame randomization. See module docstring.
RANDOMIZE_FRAME = True

# ---- Output-path conventions -------------------------------------------------
# Centralizing these here keeps caller scripts from each owning their own
# string. Each script joins the project root to these names; we don't use
# absolute paths so the repo stays portable.
CHECKPOINT_DIR  = "checkpoints"
LOG_DIR         = "logs"
ASSETS_DIR      = "assets"
CHECKPOINT_PATH       = os.path.join(CHECKPOINT_DIR, "ppo_dodge.pt")
BEST_CHECKPOINT_PATH  = os.path.join(CHECKPOINT_DIR, "ppo_dodge_best.pt")
DEFAULT_GIF_PATH      = os.path.join(ASSETS_DIR, "trained_agent.gif")


def _build_3d_actions():
    """27 actions: every (dx, dy, dz) with each axis in {-1, 0, +1}.
    Each non-zero vector is normalized so diagonal speed equals axis speed."""
    actions = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                v = np.array([dx, dy, dz], dtype=np.float32)
                n = np.linalg.norm(v)
                if n > 0:
                    v = v / n
                actions.append(v)
    return np.array(actions, dtype=np.float32)


ACTIONS = _build_3d_actions()


class DodgeEnv(gym.Env):
    """Gymnasium-compatible 3D dodge env with a spherical arena."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        render_mode=None,
        arena_radius=ARENA_RADIUS,
        spawn_distance_min=SPAWN_DISTANCE_MIN,
        spawn_distance_max=SPAWN_DISTANCE_MAX,
        perception_radius=PERCEPTION_RADIUS,
        randomize_frame=RANDOMIZE_FRAME,
        predictive_aim=True,
        projectile_speed=None,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.arena_radius = float(arena_radius)
        self.spawn_shell_min = float(spawn_distance_min)
        self.spawn_shell_max = float(spawn_distance_max)
        self.perception_radius = float(perception_radius)
        self.randomize_frame = bool(randomize_frame)
        self.predictive_aim = bool(predictive_aim)
        self.projectile_speed = (
            float(PROJECTILE_SPEED) if projectile_speed is None
            else float(projectile_speed)
        )
        if self.arena_radius <= AGENT_RADIUS:
            raise ValueError("arena_radius must be larger than AGENT_RADIUS")
        if self.spawn_shell_min < 0:
            raise ValueError("spawn_distance_min (shell) must be >= 0")
        if self.spawn_shell_max < self.spawn_shell_min:
            raise ValueError("spawn_distance_max must be >= spawn_distance_min")
        if self.perception_radius <= 0:
            raise ValueError("perception_radius must be positive")

        obs_dim = 6 + MAX_PROJECTILES * 6
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(27)

        self.agent_pos = None
        self.agent_vel = None
        self.projectiles = None
        self.steps = 0
        # frame_rotation: WORLD <- AGENT. To put a world vector into the
        # agent frame, multiply by frame_rotation.T. To send an agent-frame
        # action out into the world, multiply by frame_rotation.
        self.frame_rotation = np.eye(3, dtype=np.float32)

        self._screen = None
        self._clock = None
        self.closed_by_user = False

        self.camera_azimuth   = self.DEFAULT_AZIMUTH
        self.camera_elevation = self.DEFAULT_ELEVATION
        self.camera_distance  = self.DEFAULT_DISTANCE
        self._speed_index = 0

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.agent_vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.projectiles = []
        self.steps = 0
        if self.randomize_frame:
            self.frame_rotation = self._sample_uniform_rotation()
        else:
            self.frame_rotation = np.eye(3, dtype=np.float32)
        return self._get_obs(), {}

    def _sample_uniform_rotation(self):
        """Uniform random rotation matrix (Haar measure on SO(3)).

        Recipe: QR-decompose a random Gaussian 3x3 matrix; the orthogonal
        factor Q is uniform on O(3). Fix the sign so det(Q)=+1, giving SO(3).
        """
        m = self.np_random.standard_normal((3, 3)).astype(np.float32)
        q, r = np.linalg.qr(m)
        # Multiply each column of Q by sign of the matching diagonal of R
        # so that R has positive diagonal — this is the Mezzadri (2007)
        # trick to make Q uniformly distributed.
        d = np.sign(np.diag(r))
        d[d == 0] = 1.0
        q = q * d
        # Ensure proper rotation, not reflection.
        if np.linalg.det(q) < 0:
            q[:, 0] = -q[:, 0]
        return q.astype(np.float32)

    def step(self, action):
        self.steps += 1

        # Action vector is in the AGENT frame; rotate it to WORLD before
        # applying. frame_rotation maps WORLD <- AGENT, so frame_rotation @ v.
        agent_frame_dir = ACTIONS[action]
        world_dir = self.frame_rotation @ agent_frame_dir
        self.agent_vel = (world_dir * AGENT_SPEED).astype(np.float32)
        new_pos = self.agent_pos + self.agent_vel

        # Spherical containment: project back onto the inner sphere if outside.
        max_r = self.arena_radius - AGENT_RADIUS
        norm = float(np.linalg.norm(new_pos))
        if norm > max_r:
            new_pos = new_pos * (max_r / norm)
        self.agent_pos = new_pos.astype(np.float32)

        for p in self.projectiles:
            p["pos"] += p["vel"]

        # Remove projectiles that have left the spawn shell and are still
        # heading outward (so they can't loop back). Dot product with position
        # (relative to origin) is positive when moving away from center.
        keep_outer = self.arena_radius + self.spawn_shell_max
        kept = []
        for p in self.projectiles:
            center_dist = float(np.linalg.norm(p["pos"]))
            outward = float(np.dot(p["pos"], p["vel"])) > 0.0
            if center_dist > keep_outer and outward:
                continue
            kept.append(p)
        self.projectiles = kept

        if self.np_random.random() < SPAWN_PROB:
            self._spawn_projectile()

        # Collision check.
        hit = False
        collision_radius = AGENT_RADIUS + PROJECTILE_RADIUS
        for p in self.projectiles:
            if np.linalg.norm(p["pos"] - self.agent_pos) < collision_radius:
                hit = True
                break

        reward = 1.0
        terminated = hit
        truncated = self.steps >= MAX_EPISODE_STEPS

        return self._get_obs(), reward, terminated, truncated, {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _sample_sphere_direction(self):
        """Uniform-direction unit vector on the unit sphere (arccos trick)."""
        theta = self.np_random.uniform(0.0, 2.0 * np.pi)
        cos_phi = 2.0 * self.np_random.random() - 1.0
        sin_phi = np.sqrt(max(0.0, 1.0 - cos_phi * cos_phi))
        return np.array([
            sin_phi * np.cos(theta),
            sin_phi * np.sin(theta),
            cos_phi,
        ], dtype=np.float32)

    def _sample_projectile_spawn_pos(self):
        """Uniform on a spherical shell outside the arena. Always valid."""
        direction = self._sample_sphere_direction()
        shell = self.np_random.uniform(self.spawn_shell_min, self.spawn_shell_max)
        radius = self.arena_radius + shell
        return (radius * direction).astype(np.float32)

    def _predict_agent_pos_at_impact(self, spawn_pos, speed):
        """Where will the agent be when a projectile fired from `spawn_pos`
        at `speed` catches up, assuming the agent keeps its current velocity?

        Solves `||spawn - (pos + vel*t)|| = speed * t` for the positive root.
        Since PROJECTILE_SPEED > AGENT_SPEED in this env, a real positive
        intercept always exists. Falls back to current pos in the degenerate
        cases (no roots, both roots non-positive).
        """
        s = spawn_pos
        p = self.agent_pos
        v = self.agent_vel
        vv = float(np.dot(v, v))
        diff_sp = p - s                                 # (p - s)
        A = speed * speed - vv                          # > 0 when speed > ||v||
        B = 2.0 * float(np.dot(s - p, v))
        C = -float(np.dot(diff_sp, diff_sp))            # < 0 unless spawn==pos
        if A <= 1e-8:
            return p
        disc = B * B - 4.0 * A * C
        if disc < 0:
            return p
        sqrt_disc = float(np.sqrt(disc))
        t = (-B + sqrt_disc) / (2.0 * A)                # larger root
        if t <= 0:
            t_alt = (-B - sqrt_disc) / (2.0 * A)
            if t_alt <= 0:
                return p
            t = t_alt
        return (p + v * t).astype(np.float32)

    def _spawn_projectile(self):
        """Spawn one projectile on the outer shell, aimed near the agent.

        Two aim modes:
          - predictive (default): solve for the agent's position at impact
            time assuming constant velocity, then clip to the containment
            sphere so a boundary-sliding agent doesn't get a free pass.
          - ballistic (predictive_aim=False): aim at the agent's CURRENT
            position. This is the legacy mode that produced boundary
            camping; it's available so the journey GIF can show it.

        In both modes, AIM_RADIUS jitter is preserved.
        """
        spawn_pos = self._sample_projectile_spawn_pos()
        if self.predictive_aim:
            aim_center = self._predict_agent_pos_at_impact(spawn_pos, self.projectile_speed)
            max_r = self.arena_radius - AGENT_RADIUS
            ac_norm = float(np.linalg.norm(aim_center))
            if ac_norm > max_r:
                aim_center = aim_center * (max_r / ac_norm)
        else:
            aim_center = self.agent_pos

        direction = self._sample_sphere_direction()
        r = AIM_RADIUS * (self.np_random.random() ** (1.0 / 3.0))
        target = aim_center + r * direction

        delta = target - spawn_pos
        norm = float(np.linalg.norm(delta)) + 1e-8
        vel = (delta / norm) * self.projectile_speed

        self.projectiles.append({
            "pos": spawn_pos,
            "vel": vel.astype(np.float32),
        })

    def _get_obs(self):
        """Egocentric obs: agent state + K closest *visible* projectiles.

        All vectors are rotated from world frame into the agent's per-episode
        frame via frame_rotation.T. With randomize_frame=True the agent
        therefore never sees the absolute world axes.
        """
        obs = np.zeros(6 + MAX_PROJECTILES * 6, dtype=np.float32)

        R = self.arena_radius
        RT = self.frame_rotation.T
        pos_agent = RT @ self.agent_pos
        vel_agent = RT @ self.agent_vel
        obs[0] = pos_agent[0] / R
        obs[1] = pos_agent[1] / R
        obs[2] = pos_agent[2] / R
        obs[3] = vel_agent[0] / AGENT_SPEED
        obs[4] = vel_agent[1] / AGENT_SPEED
        obs[5] = vel_agent[2] / AGENT_SPEED

        if self.projectiles:
            visible = []
            for p in self.projectiles:
                d = float(np.linalg.norm(p["pos"] - self.agent_pos))
                if d <= self.perception_radius:
                    visible.append((d, p))
            visible.sort(key=lambda dp: dp[0])
            for slot, (_d, p) in enumerate(visible[:MAX_PROJECTILES]):
                rel_world = p["pos"] - self.agent_pos
                rel_agent = RT @ rel_world
                vel_agent_p = RT @ p["vel"]
                offset = 6 + slot * 6
                obs[offset + 0] = rel_agent[0] / R
                obs[offset + 1] = rel_agent[1] / R
                obs[offset + 2] = rel_agent[2] / R
                obs[offset + 3] = vel_agent_p[0] / PROJECTILE_SPEED
                obs[offset + 4] = vel_agent_p[1] / PROJECTILE_SPEED
                obs[offset + 5] = vel_agent_p[2] / PROJECTILE_SPEED

        return obs

    # ------------------------------------------------------------------
    # Rendering — perspective camera orbiting the origin, drawing a sphere
    # wireframe arena with a floor grid for depth cues.
    # ------------------------------------------------------------------
    SCREEN_W = 800
    SCREEN_H = 600
    FOV_DEG = 55.0

    DEFAULT_AZIMUTH   = 0.27
    DEFAULT_ELEVATION = 0.30
    DEFAULT_DISTANCE  = 1000

    KEY_AZ_SPEED    = 0.025
    KEY_EL_SPEED    = 0.020
    MOUSE_AZ_SENS   = 0.006
    MOUSE_EL_SENS   = 0.006
    WHEEL_ZOOM_IN   = 0.90
    WHEEL_ZOOM_OUT  = 1.10
    CAMERA_DIST_MIN = 200
    CAMERA_DIST_MAX = 4000
    CAMERA_EL_MIN   = -1.4
    CAMERA_EL_MAX   =  1.4

    SPEED_LEVELS = (1, 2, 4, 8, 16)

    def poll_close_event(self):
        if self.render_mode != "human" or self._screen is None:
            return self.closed_by_user
        import pygame

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.closed_by_user = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.closed_by_user = True
                elif event.key == pygame.K_r:
                    self._reset_camera()
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                    self._speed_index = min(
                        self._speed_index + 1, len(self.SPEED_LEVELS) - 1
                    )
                elif event.key == pygame.K_MINUS:
                    self._speed_index = max(self._speed_index - 1, 0)
                elif pygame.K_1 <= event.key <= pygame.K_5:
                    self._speed_index = min(
                        event.key - pygame.K_1, len(self.SPEED_LEVELS) - 1
                    )
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 4:
                    self.camera_distance *= self.WHEEL_ZOOM_IN
                elif event.button == 5:
                    self.camera_distance *= self.WHEEL_ZOOM_OUT
            elif event.type == pygame.MOUSEMOTION:
                if event.buttons[0]:
                    dx, dy = event.rel
                    self.camera_azimuth   -= dx * self.MOUSE_AZ_SENS
                    self.camera_elevation += dy * self.MOUSE_EL_SENS

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:  self.camera_azimuth   -= self.KEY_AZ_SPEED
        if keys[pygame.K_RIGHT]: self.camera_azimuth   += self.KEY_AZ_SPEED
        if keys[pygame.K_UP]:    self.camera_elevation += self.KEY_EL_SPEED
        if keys[pygame.K_DOWN]:  self.camera_elevation -= self.KEY_EL_SPEED

        self.camera_elevation = max(
            self.CAMERA_EL_MIN, min(self.CAMERA_EL_MAX, self.camera_elevation)
        )
        self.camera_distance = max(
            self.CAMERA_DIST_MIN, min(self.CAMERA_DIST_MAX, self.camera_distance)
        )
        return self.closed_by_user

    def _reset_camera(self):
        self.camera_azimuth   = self.DEFAULT_AZIMUTH
        self.camera_elevation = self.DEFAULT_ELEVATION
        self.camera_distance  = self.DEFAULT_DISTANCE

    def _compute_camera_basis(self):
        """Build the orthonormal camera basis + position from the current
        orbit state. Cheap — runs every render frame, since the user may be
        orbiting interactively.

        Spherical convention: azimuth=0 looks along +z, azimuth>0 rotates
        toward +x; elevation=0 is level, elevation>0 lifts the camera above
        the scene (looking down).
        """
        center = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        az = float(self.camera_azimuth)
        el = float(self.camera_elevation)
        d  = float(self.camera_distance)
        r_h = d * np.cos(el)
        offset = np.array([
             r_h * np.sin(az),
            -d   * np.sin(el),
            -r_h * np.cos(az),
        ], dtype=np.float32)
        self.camera_pos = center + offset

        forward = -offset / (np.linalg.norm(offset) + 1e-8)
        world_up = np.array([0.0, -1.0, 0.0], dtype=np.float32)
        right = np.cross(forward, world_up)
        right = right / (np.linalg.norm(right) + 1e-8)
        up = np.cross(right, forward)
        up = up / (np.linalg.norm(up) + 1e-8)
        self.camera_right = right
        self.camera_up = up
        self.camera_forward = forward
        self.focal_length = (self.SCREEN_W / 2.0) / np.tan(
            np.radians(self.FOV_DEG) / 2.0
        )
        self.near_plane = 1.0

    def _project(self, world_point):
        v = np.asarray(world_point, dtype=np.float32) - self.camera_pos
        x_view = float(np.dot(v, self.camera_right))
        y_view = float(np.dot(v, self.camera_up))
        z_view = float(np.dot(v, self.camera_forward))
        if z_view <= self.near_plane:
            return None
        sx = (x_view * self.focal_length / z_view) + self.SCREEN_W / 2.0
        sy = -(y_view * self.focal_length / z_view) + self.SCREEN_H / 2.0
        return sx, sy, z_view

    def _project_radius(self, depth, world_radius):
        return max(1, int(world_radius * self.focal_length / max(depth, 1.0)))

    @property
    def _floor_y(self):
        """Floor is the equatorial plane *just below* the sphere (y=+R since
        in this convention world y is "down")."""
        return float(self.arena_radius)

    def _draw_floor_grid(self, divisions=8):
        """Subtle gridlines on the y=+R plane so depth is readable."""
        import pygame
        col = (40, 45, 70)
        half = self.arena_radius
        for i in range(divisions + 1):
            z = -half + 2 * half * i / divisions
            a = self._project(np.array([-half, self._floor_y, z], dtype=np.float32))
            b = self._project(np.array([ half, self._floor_y, z], dtype=np.float32))
            if a and b:
                pygame.draw.aaline(self._screen, col, (a[0], a[1]), (b[0], b[1]))
        for i in range(divisions + 1):
            x = -half + 2 * half * i / divisions
            a = self._project(np.array([x, self._floor_y, -half], dtype=np.float32))
            b = self._project(np.array([x, self._floor_y,  half], dtype=np.float32))
            if a and b:
                pygame.draw.aaline(self._screen, col, (a[0], a[1]), (b[0], b[1]))

    def _draw_circle_3d(self, basis_u, basis_v, segments=32):
        """Draw a great-circle of radius=arena_radius in the plane spanned by
        (basis_u, basis_v), centered at origin. Depth-fade like the box edges."""
        import pygame
        R = self.arena_radius
        prev_proj = None
        prev_depth = None
        # Close the loop by going one past 2π.
        for i in range(segments + 1):
            t = 2.0 * np.pi * i / segments
            pt = (R * np.cos(t)) * basis_u + (R * np.sin(t)) * basis_v
            proj = self._project(pt)
            if prev_proj is not None and proj is not None:
                avg = (prev_depth + proj[2]) * 0.5
                # Tune fade to the smaller scene size.
                t_fade = max(0.0, min(1.0, (avg - 300.0) / 1200.0))
                bright = int(160 - 100 * t_fade)
                col = (bright, bright, min(255, bright + 25))
                pygame.draw.aaline(
                    self._screen, col, (prev_proj[0], prev_proj[1]),
                    (proj[0], proj[1]),
                )
            prev_proj = proj
            prev_depth = proj[2] if proj is not None else None

    def _draw_circle_3d_at_height(self, y, r, segments=32):
        """Draw a horizontal (constant-y) circle of radius r at height y."""
        import pygame
        prev_proj = None
        prev_depth = None
        for i in range(segments + 1):
            t = 2.0 * np.pi * i / segments
            pt = np.array([r * np.cos(t), y, r * np.sin(t)], dtype=np.float32)
            proj = self._project(pt)
            if prev_proj is not None and proj is not None:
                avg = (prev_depth + proj[2]) * 0.5
                t_fade = max(0.0, min(1.0, (avg - 300.0) / 1200.0))
                bright = int(160 - 100 * t_fade)
                col = (bright, bright, min(255, bright + 25))
                pygame.draw.aaline(
                    self._screen, col, (prev_proj[0], prev_proj[1]),
                    (proj[0], proj[1]),
                )
            prev_proj = proj
            prev_depth = proj[2] if proj is not None else None

    def _draw_arena_sphere(self, latitudes=5, longitudes=6, segments=40):
        """Wireframe sphere: a few latitude rings + a few great circles
        through the y-axis. Depth-fade so far edges are dimmer than near ones."""
        R = self.arena_radius
        # Latitude rings (excluding poles).
        for i in range(1, latitudes):
            y = -R + 2.0 * R * i / latitudes
            r_at_y = np.sqrt(max(0.0, R * R - y * y))
            self._draw_circle_3d_at_height(y, r_at_y, segments=segments)
        # Great circles through the vertical (y) axis.
        for i in range(longitudes):
            phi = np.pi * i / longitudes  # 0..π is enough (the other half is the same circle)
            u = np.array([np.cos(phi), 0.0, np.sin(phi)], dtype=np.float32)
            v = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            self._draw_circle_3d(u, v, segments=segments)

    def _draw_objects(self):
        import pygame
        try:
            import pygame.gfxdraw as gfx
            have_gfx = True
        except ImportError:
            have_gfx = False

        raw = [(self.agent_pos, AGENT_RADIUS, (60, 220, 255), (200, 245, 255))]
        for p in self.projectiles:
            raw.append((p["pos"], PROJECTILE_RADIUS, (255, 80, 80), (255, 190, 190)))

        items = []
        for pos, wr, fill, outline in raw:
            proj = self._project(pos)
            if proj is None:
                continue
            items.append((proj, pos, wr, fill, outline))

        items.sort(key=lambda it: -it[0][2])

        # Floor shadows.
        for (sx, sy, d), pos, wr, _fill, _outline in items:
            shadow_world = np.array(
                [pos[0], self._floor_y, pos[2]], dtype=np.float32
            )
            sp = self._project(shadow_world)
            if sp is None:
                continue
            ssx, ssy, sd = sp
            height_above = max(0.0, self._floor_y - pos[1])
            atten = max(0.25, 1.0 - height_above / (2.0 * self.arena_radius))
            sr = max(2, int(wr * 0.9 * self.focal_length * atten / sd))
            color = (10, 12, 22)
            if have_gfx:
                gfx.filled_circle(self._screen, int(ssx), int(ssy), sr, color)
            else:
                pygame.draw.circle(self._screen, color, (int(ssx), int(ssy)), sr)

        for (sx, sy, d), _pos, wr, fill, outline in items:
            r = self._project_radius(d, wr)
            if have_gfx:
                gfx.filled_circle(self._screen, int(sx), int(sy), r, fill)
                gfx.aacircle(self._screen, int(sx), int(sy), r, outline)
            else:
                pygame.draw.circle(self._screen, fill, (int(sx), int(sy)), r)
                pygame.draw.circle(self._screen, outline, (int(sx), int(sy)), r, 1)

    def _draw_hud(self):
        import pygame
        if not hasattr(self, "_hud_font") or self._hud_font is None:
            pygame.font.init()
            self._hud_font  = pygame.font.SysFont(None, 22)
            self._help_font = pygame.font.SysFont(None, 18)

        speed = self.SPEED_LEVELS[self._speed_index]
        status = self._hud_font.render(
            f"step {self.steps}/{MAX_EPISODE_STEPS}   "
            f"proj {len(self.projectiles)}   "
            f"speed {speed}x",
            True, (180, 200, 220),
        )
        self._screen.blit(status, (10, 8))

        if self.render_mode == "human":
            help_line = self._help_font.render(
                "drag = orbit   wheel = zoom   arrows = orbit   "
                "1-5 = speed   +/- = step speed   R = reset view   ESC = quit",
                True, (110, 130, 160),
            )
            self._screen.blit(help_line, (10, self.SCREEN_H - 22))

    def render(self):
        if self.render_mode is None:
            return None

        import pygame

        if self._screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self._screen = pygame.display.set_mode((self.SCREEN_W, self.SCREEN_H))
                pygame.display.set_caption("DodgeEnv (perspective)")
            else:
                self._screen = pygame.Surface((self.SCREEN_W, self.SCREEN_H))
            self._clock = pygame.time.Clock()
            self._hud_font = None

        self._compute_camera_basis()
        self._screen.fill((14, 16, 28))
        self._draw_floor_grid()
        self._draw_arena_sphere()
        self._draw_objects()
        self._draw_hud()

        if self.render_mode == "human":
            self.poll_close_event()
            pygame.display.flip()
            target_fps = self.metadata["render_fps"] * self.SPEED_LEVELS[self._speed_index]
            self._clock.tick(target_fps)
            return None
        else:
            arr = pygame.surfarray.array3d(self._screen)
            return np.transpose(arr, (1, 0, 2))

    def close(self):
        if self._screen is not None:
            import pygame
            if self.render_mode == "human":
                pygame.display.quit()
            pygame.quit()
            self._screen = None
            self._hud_font = None
