"""
DodgeEnv3D: 3D port of DodgeEnv. Near-clone with a z-axis.

Differences from the 2D env:
  - Arena is ARENA_W x ARENA_H x ARENA_D (a box).
  - Agent pos / vel are 3-vectors.
  - Action space is 27 = 3^3: each axis picks {-1, 0, +1}.
  - Observation grows to 6 (agent) + MAX_PROJECTILES * 6 (each projectile).
  - Rendering is a top-down view with z encoded as circle size, plus a
    1D z-strip at the bottom that shows depth positions.

Everything else (reward = +1/step, episode timeout, sorting projectiles by
distance, local ring spawns, AIM_RADIUS targeting) follows the 2D recipe.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


# ---- Constants. Keep this block in sync with dodge_env.py where possible. ----
ARENA_W = 800            # x
ARENA_H = 600            # y
ARENA_D = 600            # z (depth). Same scale as height.
AGENT_RADIUS = 15
PROJECTILE_RADIUS = 5
AGENT_SPEED = 5.0
PROJECTILE_SPEED = 4.0
MAX_PROJECTILES = 8
MAX_EPISODE_STEPS = 1000
SPAWN_PROB = 0.03
AIM_RADIUS = 100
SPAWN_DISTANCE_MIN = 220   # same defaults as 2D — easier to compare transfer.
SPAWN_DISTANCE_MAX = 320

CURRICULUM_PRESETS = {
    "target": (SPAWN_DISTANCE_MIN, SPAWN_DISTANCE_MAX),
    "easy": (280, 400),
}


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


ACTIONS = _build_3d_actions()   # shape (27, 3). Index 13 = stay still.


class DodgeEnv3D(gym.Env):
    """3D version of DodgeEnv, Gymnasium-compatible."""

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

        # Observation: 6 (agent: xyz + vxvyvz) + MAX_PROJECTILES * 6
        obs_dim = 6 + MAX_PROJECTILES * 6
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(27)

        self.agent_pos = None     # np.array([x, y, z])
        self.agent_vel = None
        self.projectiles = None
        self.steps = 0

        self._screen = None
        self._clock = None
        self.closed_by_user = False

        # Camera orbit state — adjustable at runtime via mouse / arrows / wheel.
        # Stored on the instance (not the class) so multiple envs in one process
        # don't share orbits.
        self.camera_azimuth   = self.DEFAULT_AZIMUTH
        self.camera_elevation = self.DEFAULT_ELEVATION
        self.camera_distance  = self.DEFAULT_DISTANCE

        # Render speed multiplier — index into SPEED_LEVELS. Default 1x.
        self._speed_index = 0

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = np.array(
            [ARENA_W / 2, ARENA_H / 2, ARENA_D / 2], dtype=np.float32
        )
        self.agent_vel = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.projectiles = []
        self.steps = 0
        return self._get_obs(), {}

    def step(self, action):
        self.steps += 1

        direction = ACTIONS[action]
        self.agent_vel = direction * AGENT_SPEED
        self.agent_pos += self.agent_vel

        # Clip to arena box.
        self.agent_pos[0] = np.clip(self.agent_pos[0], AGENT_RADIUS, ARENA_W - AGENT_RADIUS)
        self.agent_pos[1] = np.clip(self.agent_pos[1], AGENT_RADIUS, ARENA_H - AGENT_RADIUS)
        self.agent_pos[2] = np.clip(self.agent_pos[2], AGENT_RADIUS, ARENA_D - AGENT_RADIUS)

        for p in self.projectiles:
            p["pos"] += p["vel"]

        # Remove projectiles that left the box (with margin so they fully exit).
        margin = 50
        self.projectiles = [
            p for p in self.projectiles
            if (-margin <= p["pos"][0] <= ARENA_W + margin
                and -margin <= p["pos"][1] <= ARENA_H + margin
                and -margin <= p["pos"][2] <= ARENA_D + margin)
        ]

        if self.np_random.random() < SPAWN_PROB:
            self._spawn_projectile()

        # Collision check — 3D distance.
        hit = False
        collision_radius = AGENT_RADIUS + PROJECTILE_RADIUS
        for p in self.projectiles:
            d = np.linalg.norm(p["pos"] - self.agent_pos)
            if d < collision_radius:
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
        """Uniform-direction unit vector on the unit sphere (no clustering at poles)."""
        # arccos trick avoids the standard (theta, phi) pole-clustering.
        theta = self.np_random.uniform(0.0, 2.0 * np.pi)
        cos_phi = 2.0 * self.np_random.random() - 1.0  # uniform in [-1, 1]
        sin_phi = np.sqrt(max(0.0, 1.0 - cos_phi * cos_phi))
        return np.array([
            sin_phi * np.cos(theta),
            sin_phi * np.sin(theta),
            cos_phi,
        ], dtype=np.float32)

    def _sample_projectile_spawn_pos(self):
        """Spawn point on a spherical shell around the agent, clipped to the box."""
        low = PROJECTILE_RADIUS
        high_x = ARENA_W - PROJECTILE_RADIUS
        high_y = ARENA_H - PROJECTILE_RADIUS
        high_z = ARENA_D - PROJECTILE_RADIUS

        for _ in range(100):
            direction = self._sample_sphere_direction()
            dist = self.np_random.uniform(self.spawn_distance_min, self.spawn_distance_max)
            pos = self.agent_pos + dist * direction
            if (low <= pos[0] <= high_x
                    and low <= pos[1] <= high_y
                    and low <= pos[2] <= high_z):
                return pos.astype(np.float32)

        # Fallback 1: sample anywhere in the box until distance is in range.
        for _ in range(100):
            pos = np.array([
                self.np_random.uniform(low, high_x),
                self.np_random.uniform(low, high_y),
                self.np_random.uniform(low, high_z),
            ], dtype=np.float32)
            dist = np.linalg.norm(pos - self.agent_pos)
            if self.spawn_distance_min <= dist <= self.spawn_distance_max:
                return pos

        # Last resort: aim toward arena center, push out by min distance.
        center = np.array([ARENA_W / 2, ARENA_H / 2, ARENA_D / 2], dtype=np.float32)
        direction = center - self.agent_pos
        norm = np.linalg.norm(direction) + 1e-8
        pos = self.agent_pos + direction / norm * self.spawn_distance_min
        pos[0] = np.clip(pos[0], low, high_x)
        pos[1] = np.clip(pos[1], low, high_y)
        pos[2] = np.clip(pos[2], low, high_z)
        return pos.astype(np.float32)

    def _spawn_projectile(self):
        """Spawn one local projectile aimed near the agent (3D ball offset)."""
        spawn_pos = self._sample_projectile_spawn_pos()

        # Aim at a random point inside a *ball* of radius AIM_RADIUS around
        # the agent. Uniform-in-ball: direction on sphere * r * cbrt(uniform).
        direction = self._sample_sphere_direction()
        r = AIM_RADIUS * (self.np_random.random() ** (1.0 / 3.0))
        target = self.agent_pos + r * direction

        delta = target - spawn_pos
        norm = float(np.linalg.norm(delta)) + 1e-8
        vel = (delta / norm) * PROJECTILE_SPEED

        self.projectiles.append({
            "pos": spawn_pos,
            "vel": vel.astype(np.float32),
        })

    def _get_obs(self):
        """Egocentric observation: agent's own state + K closest projectiles."""
        obs = np.zeros(6 + MAX_PROJECTILES * 6, dtype=np.float32)

        # Agent state normalized to ~[-1, 1] per axis.
        obs[0] = (self.agent_pos[0] - ARENA_W / 2) / (ARENA_W / 2)
        obs[1] = (self.agent_pos[1] - ARENA_H / 2) / (ARENA_H / 2)
        obs[2] = (self.agent_pos[2] - ARENA_D / 2) / (ARENA_D / 2)
        obs[3] = self.agent_vel[0] / AGENT_SPEED
        obs[4] = self.agent_vel[1] / AGENT_SPEED
        obs[5] = self.agent_vel[2] / AGENT_SPEED

        if len(self.projectiles) > 0:
            dists = [
                float(np.linalg.norm(p["pos"] - self.agent_pos))
                for p in self.projectiles
            ]
            order = np.argsort(dists)
            for slot, idx in enumerate(order[:MAX_PROJECTILES]):
                p = self.projectiles[idx]
                rel = p["pos"] - self.agent_pos
                offset = 6 + slot * 6
                obs[offset + 0] = rel[0] / (ARENA_W / 2)
                obs[offset + 1] = rel[1] / (ARENA_H / 2)
                obs[offset + 2] = rel[2] / (ARENA_D / 2)
                obs[offset + 3] = p["vel"][0] / PROJECTILE_SPEED
                obs[offset + 4] = p["vel"][1] / PROJECTILE_SPEED
                obs[offset + 5] = p["vel"][2] / PROJECTILE_SPEED

        return obs

    # ------------------------------------------------------------------
    # Rendering — true 3D perspective projection
    # ------------------------------------------------------------------
    # The camera sits outside the arena and looks toward its center. Every
    # world point goes through a real perspective projection (translate ->
    # rotate into camera frame -> perspective divide), so depth is genuinely
    # visible: closer things are larger, the arena box has vanishing-point
    # edges, and a floor grid anchors the eye.
    #
    # No external 3D library — just pygame + a little linear algebra.
    # All of this only runs when render_mode != None; headless training
    # never touches any of these code paths.

    SCREEN_W = 800
    SCREEN_H = 600
    FOV_DEG = 55.0

    # Camera orbit defaults — chosen so the whole arena box is visible from
    # the start, even when the agent drifts to a corner. The user can
    # interactively rotate/zoom from these defaults via mouse + arrow keys
    # (see poll_close_event). Pulled further back than the original fixed
    # camera so the arena never goes off-screen.
    DEFAULT_AZIMUTH   = 0.27    # radians; horizontal angle around the scene
    DEFAULT_ELEVATION = 0.30    # radians; vertical tilt (+ve = looking down)
    DEFAULT_DISTANCE  = 1300    # world units from the scene center

    # Input sensitivities. Tune if the controls feel too fast or slow.
    KEY_AZ_SPEED    = 0.025     # rad/frame while arrow held
    KEY_EL_SPEED    = 0.020
    MOUSE_AZ_SENS   = 0.006     # rad / pixel of left-drag
    MOUSE_EL_SENS   = 0.006
    WHEEL_ZOOM_IN   = 0.90      # camera_distance *= this on wheel up
    WHEEL_ZOOM_OUT  = 1.10
    CAMERA_DIST_MIN = 300
    CAMERA_DIST_MAX = 4500
    CAMERA_EL_MIN   = -1.4      # ~ -80°; stay away from the gimbal-lock pole
    CAMERA_EL_MAX   =  1.4

    # Speed multipliers on the base render FPS (1x = 30 FPS, 16x = 480 FPS).
    # Memory cost: zero. CPU cost: rendering loop spins faster but the env
    # itself is so cheap it doesn't matter on a 16 GB machine.
    SPEED_LEVELS = (1, 2, 4, 8, 16)

    # Floor plane in world coords (y=ARENA_H is "down" — agent stands on it).
    _FLOOR_Y = float(ARENA_H)

    _BOX_EDGES = (
        (0, 1), (1, 2), (2, 3), (3, 0),      # back face (z=0)
        (4, 5), (5, 6), (6, 7), (7, 4),      # front face (z=ARENA_D)
        (0, 4), (1, 5), (2, 6), (3, 7),      # connecting edges
    )

    def poll_close_event(self):
        """Drain pygame events. Updates orbit-camera state, speed multiplier,
        and the closed-by-user flag.

        Keybindings (active in `render_mode == "human"`):
          - Arrow keys / left-mouse drag : orbit (azimuth + elevation)
          - Mouse wheel                  : zoom (camera distance)
          - 1 .. 5                       : set speed to {1, 2, 4, 8, 16}x
          - +/- (or =/-)                 : step speed up/down one level
          - R                            : reset camera to defaults
          - ESC / window close X         : quit cleanly
        """
        if self.render_mode != "human" or self._screen is None:
            return self.closed_by_user
        import pygame

        # Discrete events.
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
                if event.button == 4:        # wheel up = zoom in
                    self.camera_distance *= self.WHEEL_ZOOM_IN
                elif event.button == 5:      # wheel down = zoom out
                    self.camera_distance *= self.WHEEL_ZOOM_OUT
            elif event.type == pygame.MOUSEMOTION:
                if event.buttons[0]:         # left button held -> orbit
                    dx, dy = event.rel
                    self.camera_azimuth   -= dx * self.MOUSE_AZ_SENS
                    self.camera_elevation += dy * self.MOUSE_EL_SENS

        # Continuous keys (arrows held).
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:  self.camera_azimuth   -= self.KEY_AZ_SPEED
        if keys[pygame.K_RIGHT]: self.camera_azimuth   += self.KEY_AZ_SPEED
        if keys[pygame.K_UP]:    self.camera_elevation += self.KEY_EL_SPEED
        if keys[pygame.K_DOWN]:  self.camera_elevation -= self.KEY_EL_SPEED

        # Clamp ranges so we can't end up inside the arena or at the pole.
        self.camera_elevation = max(
            self.CAMERA_EL_MIN, min(self.CAMERA_EL_MAX, self.camera_elevation)
        )
        self.camera_distance = max(
            self.CAMERA_DIST_MIN, min(self.CAMERA_DIST_MAX, self.camera_distance)
        )
        return self.closed_by_user

    def _reset_camera(self):
        """R key: snap orbit + zoom back to the 'whole arena visible' default."""
        self.camera_azimuth   = self.DEFAULT_AZIMUTH
        self.camera_elevation = self.DEFAULT_ELEVATION
        self.camera_distance  = self.DEFAULT_DISTANCE

    def _compute_camera_basis(self):
        """(Re)build the orthonormal camera basis + position from current orbit
        state. Cheap — runs every render frame, since the user may be orbiting.

        Spherical convention:
          - azimuth=0  -> camera looks along +z (default 'front' of the box)
          - azimuth>0  -> orbit toward +x
          - elevation=0-> level
          - elevation>0-> camera lifted above the scene (looking down)
        """
        center = np.array(
            [ARENA_W / 2, ARENA_H / 2, ARENA_D / 2], dtype=np.float32
        )
        az = float(self.camera_azimuth)
        el = float(self.camera_elevation)
        d  = float(self.camera_distance)
        r_h = d * np.cos(el)
        offset = np.array([
             r_h * np.sin(az),
            -d   * np.sin(el),      # negative because world y goes DOWN
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
        # FOV is constant, so focal length is constant too.
        self.focal_length = (self.SCREEN_W / 2.0) / np.tan(
            np.radians(self.FOV_DEG) / 2.0
        )
        self.near_plane = 1.0

    def _project(self, world_point):
        """3D world point -> (screen_x, screen_y, depth) or None if clipped."""
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
        """How big does a world-space radius look at this camera depth?"""
        return max(1, int(world_radius * self.focal_length / max(depth, 1.0)))

    def _arena_corners(self):
        """The 8 corners of the arena box, in world coords."""
        return [
            np.array([0,       0,       0      ], dtype=np.float32),
            np.array([ARENA_W, 0,       0      ], dtype=np.float32),
            np.array([ARENA_W, ARENA_H, 0      ], dtype=np.float32),
            np.array([0,       ARENA_H, 0      ], dtype=np.float32),
            np.array([0,       0,       ARENA_D], dtype=np.float32),
            np.array([ARENA_W, 0,       ARENA_D], dtype=np.float32),
            np.array([ARENA_W, ARENA_H, ARENA_D], dtype=np.float32),
            np.array([0,       ARENA_H, ARENA_D], dtype=np.float32),
        ]

    def _draw_floor_grid(self, divisions=8):
        """Subtle gridlines on the y=ARENA_H plane so depth is readable."""
        import pygame
        col = (40, 45, 70)
        # Lines parallel to the x axis (at fixed z).
        for i in range(divisions + 1):
            z = (i / divisions) * ARENA_D
            a = self._project(np.array([0.0, self._FLOOR_Y, z], dtype=np.float32))
            b = self._project(np.array([ARENA_W, self._FLOOR_Y, z], dtype=np.float32))
            if a and b:
                pygame.draw.aaline(self._screen, col, (a[0], a[1]), (b[0], b[1]))
        # Lines parallel to the z axis (at fixed x).
        for i in range(divisions + 1):
            x = (i / divisions) * ARENA_W
            a = self._project(np.array([x, self._FLOOR_Y, 0.0], dtype=np.float32))
            b = self._project(np.array([x, self._FLOOR_Y, ARENA_D], dtype=np.float32))
            if a and b:
                pygame.draw.aaline(self._screen, col, (a[0], a[1]), (b[0], b[1]))

    def _draw_arena_box(self):
        """Wireframe of the arena. Near edges are brighter than far ones so the
        viewer perceives depth even before objects move."""
        import pygame
        projected = [self._project(c) for c in self._arena_corners()]
        for a, b in self._BOX_EDGES:
            if projected[a] is None or projected[b] is None:
                continue
            ax, ay, da = projected[a]
            bx, by, db = projected[b]
            # Linear depth-fade. Tunes brightness to a useful range for the
            # current camera offset.
            avg = (da + db) * 0.5
            t = max(0.0, min(1.0, (avg - 400.0) / 1500.0))
            bright = int(180 - 110 * t)
            col = (bright, bright, min(255, bright + 25))
            pygame.draw.aaline(self._screen, col, (ax, ay), (bx, by))

    def _draw_objects(self):
        """Draw agent + projectiles as 3D spheres-as-discs with floor shadows.
        Depth-sort so near objects overlay far ones."""
        import pygame
        try:
            import pygame.gfxdraw as gfx
            have_gfx = True
        except ImportError:
            have_gfx = False

        # Each item: (world_pos, world_radius, fill_color, outline_color)
        raw = [(self.agent_pos, AGENT_RADIUS, (60, 220, 255), (200, 245, 255))]
        for p in self.projectiles:
            raw.append((p["pos"], PROJECTILE_RADIUS, (255, 80, 80), (255, 190, 190)))

        # Project everything once.
        items = []
        for pos, wr, fill, outline in raw:
            proj = self._project(pos)
            if proj is None:
                continue
            items.append((proj, pos, wr, fill, outline))

        # Sort far -> near so near objects overlay far ones.
        items.sort(key=lambda it: -it[0][2])

        # Pass 1: floor shadows under every object (drawn before bodies so a
        # near object's shadow doesn't sit on top of its own sphere).
        for (sx, sy, d), pos, wr, _fill, _outline in items:
            shadow_world = np.array(
                [pos[0], self._FLOOR_Y, pos[2]], dtype=np.float32
            )
            sp = self._project(shadow_world)
            if sp is None:
                continue
            ssx, ssy, sd = sp
            # Higher-up objects (smaller y) cast fainter, slightly smaller shadows.
            height_above = max(0.0, self._FLOOR_Y - pos[1])
            atten = max(0.25, 1.0 - height_above / self._FLOOR_Y)
            sr = max(2, int(wr * 0.9 * self.focal_length * atten / sd))
            color = (10, 12, 22)
            if have_gfx:
                gfx.filled_circle(self._screen, int(ssx), int(ssy), sr, color)
            else:
                pygame.draw.circle(self._screen, color, (int(ssx), int(ssy)), sr)

        # Pass 2: the bodies themselves (with AA outline).
        for (sx, sy, d), _pos, wr, fill, outline in items:
            r = self._project_radius(d, wr)
            if have_gfx:
                gfx.filled_circle(self._screen, int(sx), int(sy), r, fill)
                gfx.aacircle(self._screen, int(sx), int(sy), r, outline)
            else:
                pygame.draw.circle(self._screen, fill, (int(sx), int(sy)), r)
                pygame.draw.circle(self._screen, outline, (int(sx), int(sy)), r, 1)

    def _draw_hud(self):
        """Status text (step, projectile count, speed) + control hints."""
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

        # Control hint only in interactive (human) mode; GIFs don't need it.
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
                pygame.display.set_caption("DodgeEnv3D (perspective)")
            else:
                self._screen = pygame.Surface((self.SCREEN_W, self.SCREEN_H))
            self._clock = pygame.time.Clock()
            self._hud_font = None

        # Camera state may have changed since the last frame (user orbited /
        # zoomed), so rebuild the basis every render. Cheap.
        self._compute_camera_basis()

        # 1. Background (dark blue, faintly sky-like).
        self._screen.fill((14, 16, 28))

        # 2. Floor grid — drawn first so everything else sits on top.
        self._draw_floor_grid()

        # 3. Arena wireframe (with near/far brightness gradient).
        self._draw_arena_box()

        # 4. Objects (agent + projectiles) with floor shadows + depth sort.
        self._draw_objects()

        # 5. HUD overlay (step counter, speed, control hints).
        self._draw_hud()

        if self.render_mode == "human":
            self.poll_close_event()
            pygame.display.flip()
            # Speed multiplier scales the FPS cap. At 16x = 480 FPS cap;
            # if the cycle can't sustain that, it just runs as fast as it can.
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
