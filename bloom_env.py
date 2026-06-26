"""
Gymnasium environment for the Bloom Chain puzzle.

The game logic is your original code, refactored to run *headless*:
  - no window / no event loop / no FPS clock
  - renders into an off-screen 64x64 pygame Surface
  - observations are returned as raw RGB pixels (64, 64, 3) uint8

Key RL-relevant design choices (see README):
  - Action space: Discrete(5) -> up, down, left, right, interact
  - Reward: sparse. +1 only when a level is solved ("win"). 0 otherwise.
    (An optional tiny per-target shaping reward is available but defaults OFF.)
  - Each reset() draws a *fresh procedurally generated level* with a new seed,
    so the agent must generalize across layouts instead of memorizing one.
  - The blink animation is frozen so identical game states render identically
    (removes a small "noisy-TV"-style source of pixel noise from the obs).
"""

import os
# Headless unless explicitly told otherwise (watch.py sets BLOOM_HEADLESS=0).
if os.environ.get("BLOOM_HEADLESS", "1") == "1":
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import random
import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces

# pygame must be initialized once for Surfaces / surfarray to work.
pygame.init()
# A 1x1 dummy display makes surfarray reliable across SDL backends.
try:
    pygame.display.set_mode((1, 1))
except pygame.error:
    pass

# ----------------------------------------------------------------------------
# Constants & palette (copied from your file)
# ----------------------------------------------------------------------------
TILE_SIZE = 6
UI_HEIGHT = 16
CANVAS_SIZE = 64

EMPTY = 0
SOIL = 1
ROCK = 2
SEED = 3
TARGET_A = 10
TARGET_B = 11
TARGET_C = 12
TARGET_D = 13

PALETTE = {
    -1: (0, 0, 0),
    0: (20, 20, 25),
    2: (220, 50, 50),
    3: (50, 200, 50),
    4: (240, 220, 50),
    5: (100, 100, 100),
    6: (200, 50, 200),
    7: (240, 140, 50),
    8: (100, 180, 240),
    9: (120, 70, 40),
    11: (50, 200, 200),
    13: (70, 70, 70),
    14: (240, 100, 150),
    15: (255, 255, 255),
}

_LEVEL_CONFIGS = [
    {"name": "1: The Seedling", "w": 5, "h": 5, "targets": 2},
    {"name": "2: Growth",       "w": 6, "h": 6, "targets": 3},
    {"name": "3: Branches",     "w": 6, "h": 6, "targets": 4},
    {"name": "4: Complexity",   "w": 7, "h": 7, "targets": 4},
    {"name": "5: The Weave",    "w": 7, "h": 7, "targets": 5},
    {"name": "6: Overgrowth",   "w": 8, "h": 8, "targets": 5},
    {"name": "7: Labyrinth",    "w": 8, "h": 8, "targets": 6},
    {"name": "8: The Core",     "w": 8, "h": 8, "targets": 7},
]
NUM_LEVELS = len(_LEVEL_CONFIGS)


# ----------------------------------------------------------------------------
# Headless game core (your logic, display code removed; win/lose -> flags)
# ----------------------------------------------------------------------------
class HeadlessBloomChain:
    def __init__(self, seed, level_index=0, scramble=1.0):
        self.canvas = pygame.Surface((CANVAS_SIZE, CANVAS_SIZE))
        self._base_seed = seed
        self._current_level_index = level_index
        self._scramble = scramble  # 0.0 = intact path (easy), 1.0 = full scramble
        self._anim_frame = 0  # frozen -> deterministic rendering
        # terminal flags, read by the gym wrapper
        self._level_solved = False
        self._dead = False
        self.load_level()

    # --- win/lose become flags instead of auto-advancing/resetting ----------
    def next_level(self):
        self._level_solved = True

    def lose(self):
        self._dead = True

    def load_level(self):
        cfg = _LEVEL_CONFIGS[self._current_level_index]
        rng = random.Random(self._base_seed + self._current_level_index)

        self.grid_w = cfg["w"]
        self.grid_h = cfg["h"]
        self.offset_x = (CANVAS_SIZE - (self.grid_w * TILE_SIZE)) // 2
        self.offset_y = ((CANVAS_SIZE - UI_HEIGHT) - (self.grid_h * TILE_SIZE)) // 2

        self._cursor_x, self._cursor_y = self.grid_w // 2, self.grid_h // 2
        self._selected_pos = None
        self._error_pos = None
        self._blooms = set()
        self._activated_coords = set()
        self._seq_idx = 0
        self._level_solved = False
        self._dead = False

        self._grid, self._sequence = self._generate_level(
            self.grid_w, self.grid_h, cfg["targets"], rng, self._scramble)

        for y in range(self.grid_h):
            for x in range(self.grid_w):
                if self._grid[y][x] == SEED:
                    self._blooms.add((x, y))

    # --- procedural generation (unchanged) ----------------------------------
    def _generate_level(self, w, h, num_targets, rng, scramble=1.0):
        target_length = num_targets * 2 + 1
        path_set = set()

        def find_path(x, y, current_path, strict):
            if len(current_path) == target_length:
                return current_path
            dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            rng.shuffle(dirs)
            for dx, dy in dirs:
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h and (nx, ny) not in path_set:
                    if strict:
                        touch_count = sum(1 for adx, ady in [(0, 1), (0, -1), (1, 0), (-1, 0)]
                                          if (nx + adx, ny + ady) in path_set)
                        if touch_count != 1:
                            continue
                    path_set.add((nx, ny))
                    res = find_path(nx, ny, current_path + [(nx, ny)], strict)
                    if res:
                        return res
                    path_set.discard((nx, ny))
            return None

        for attempt in range(200):
            strict = attempt < 100
            grid = [[EMPTY for _ in range(w)] for _ in range(h)]
            sx, sy = rng.randint(0, w - 1), rng.randint(0, h - 1)
            path_set.clear()
            path_set.add((sx, sy))
            path = find_path(sx, sy, [(sx, sy)], strict)
            if not path:
                continue

            grid[sy][sx] = SEED
            for px, py in path[1:]:
                grid[py][px] = SOIL

            available_indices = list(range(1, len(path) - 1))
            if len(available_indices) < num_targets:
                continue
            step = len(available_indices) / num_targets
            chosen_indices = sorted(int(i * step + step / 2) for i in range(num_targets))
            chosen_indices = [available_indices[min(ci, len(available_indices) - 1)]
                              for ci in chosen_indices]

            sequence = []
            pool = [TARGET_A, TARGET_B, TARGET_C, TARGET_D]
            valid_targets = True
            occupied = set()

            for p_idx in chosen_indices:
                px, py = path[p_idx]
                adj_empty = []
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    tx, ty = px + dx, py + dy
                    if 0 <= tx < w and 0 <= ty < h and grid[ty][tx] == EMPTY and (tx, ty) not in occupied:
                        touch = sum(1 for adx, ady in [(0, 1), (0, -1), (1, 0), (-1, 0)]
                                    if 0 <= tx + adx < w and 0 <= ty + ady < h
                                    and grid[ty + ady][tx + adx] in (SOIL, SEED))
                        if touch == 1:
                            adj_empty.append((tx, ty))
                if not adj_empty:
                    valid_targets = False
                    break
                tx, ty = rng.choice(adj_empty)
                t_type = pool[len(sequence) % len(pool)]
                grid[ty][tx] = t_type
                sequence.append(t_type)
                occupied.add((tx, ty))

            if not valid_targets:
                continue

            empty_cells = [(x, y) for y in range(h) for x in range(w) if grid[y][x] == EMPTY]
            num_rocks = min(len(empty_cells) // 3, 6)
            if num_rocks > 0:
                for rx, ry in rng.sample(empty_cells, num_rocks):
                    grid[ry][rx] = ROCK

            soil_cells = list(path[1:])
            current_empty = [(x, y) for y in range(h) for x in range(w) if grid[y][x] == EMPTY]
            for _ in range(int(len(soil_cells) * 2 * scramble)):
                if soil_cells and current_empty:
                    s_idx = rng.randint(0, len(soil_cells) - 1)
                    e_idx = rng.randint(0, len(current_empty) - 1)
                    s_x, s_y = soil_cells[s_idx]
                    e_x, e_y = current_empty[e_idx]
                    grid[s_y][s_x], grid[e_y][e_x] = grid[e_y][e_x], grid[s_y][s_x]
                    soil_cells[s_idx], current_empty[e_idx] = (e_x, e_y), (s_x, s_y)

            return grid, sequence
        return self._generate_fallback(w, h, num_targets)

    def _generate_fallback(self, w, h, num_targets):
        grid = [[EMPTY for _ in range(w)] for _ in range(h)]
        path = []
        for row in range(h):
            cols = range(w) if row % 2 == 0 else range(w - 1, -1, -1)
            for col in cols:
                path.append((col, row))
        path = path[:num_targets * 2 + 1]
        sx, sy = path[0]
        grid[sy][sx] = SEED
        for px, py in path[1:]:
            grid[py][px] = SOIL

        pool = [TARGET_A, TARGET_B, TARGET_C, TARGET_D]
        sequence = []
        step = max(1, len(path) // (num_targets + 1))
        anchor_indices = [min(step * (i + 1), len(path) - 1) for i in range(num_targets)]
        for i, a_idx in enumerate(anchor_indices):
            px, py = path[a_idx]
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                tx, ty = px + dx, py + dy
                if 0 <= tx < w and 0 <= ty < h and grid[ty][tx] == EMPTY:
                    t_type = pool[i % len(pool)]
                    grid[ty][tx] = t_type
                    sequence.append(t_type)
                    break
        return grid, sequence

    # --- simulation (unchanged logic; lose/next_level now set flags) --------
    def _run_simulation_tick(self):
        new_blooms = set()
        targets_hit_this_tick = set()
        for (bx, by) in self._blooms:
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = bx + dx, by + dy
                if 0 <= nx < self.grid_w and 0 <= ny < self.grid_h:
                    cell = self._grid[ny][nx]
                    if cell == SOIL and (nx, ny) not in self._blooms:
                        new_blooms.add((nx, ny))
                    elif cell in (TARGET_A, TARGET_B, TARGET_C, TARGET_D) and (nx, ny) not in self._activated_coords:
                        targets_hit_this_tick.add((cell, nx, ny))
        self._blooms.update(new_blooms)

        if targets_hit_this_tick:
            if len(targets_hit_this_tick) > 1:
                self.lose()
                return
            hit_cell, hit_x, hit_y = list(targets_hit_this_tick)[0]
            expected = self._sequence[self._seq_idx]
            if hit_cell == expected:
                self._seq_idx += 1
                self._activated_coords.add((hit_x, hit_y))
                if self._seq_idx >= len(self._sequence):
                    self.next_level()
            else:
                self.lose()

    def handle_interaction(self):
        curr_entity = self._grid[self._cursor_y][self._cursor_x]
        is_locked = (self._cursor_x, self._cursor_y) in self._blooms
        if curr_entity == SEED:
            self._run_simulation_tick()
            self._selected_pos = None
        elif curr_entity in (EMPTY, SOIL):
            if is_locked:
                self._error_pos = (self._cursor_x, self._cursor_y)
            else:
                if not self._selected_pos:
                    self._selected_pos = (self._cursor_x, self._cursor_y)
                else:
                    sx, sy = self._selected_pos
                    temp = self._grid[sy][sx]
                    self._grid[sy][sx] = self._grid[self._cursor_y][self._cursor_x]
                    self._grid[self._cursor_y][self._cursor_x] = temp
                    self._selected_pos = None

    # --- the single entry point the env uses --------------------------------
    def step_action(self, action):
        """0=up 1=down 2=left 3=right 4=interact"""
        if action == 4:
            self.handle_interaction()
        else:
            dx, dy = 0, 0
            if action == 0:
                dy = -1
            elif action == 1:
                dy = 1
            elif action == 2:
                dx = -1
            elif action == 3:
                dx = 1
            self._cursor_x = max(0, min(self.grid_w - 1, self._cursor_x + dx))
            self._cursor_y = max(0, min(self.grid_h - 1, self._cursor_y + dy))
            self._error_pos = None

    # --- rendering (your render(), minus scaling/blit/flip) -----------------
    def draw_pixels(self, base_x, base_y, pixels_2d):
        for y, row in enumerate(pixels_2d):
            for x, color_id in enumerate(row):
                if color_id != -1:
                    color = PALETTE.get(color_id, (255, 0, 255))
                    self.canvas.set_at((base_x + x, base_y + y), color)

    def _get_target_color(self, entity):
        return {TARGET_A: 3, TARGET_B: 8, TARGET_C: 6, TARGET_D: 11}.get(entity, 15)

    def draw_tile(self, px, py, entity, is_activated):
        pixels = [[-1] * TILE_SIZE for _ in range(TILE_SIZE)]
        if entity == SOIL:
            pixels = [[9] * TILE_SIZE for _ in range(TILE_SIZE)]
            pixels[1][1] = pixels[4][4] = 13
            pixels[1][4] = pixels[4][1] = 0
        elif entity == ROCK:
            pixels = [[5] * TILE_SIZE for _ in range(TILE_SIZE)]
            pixels[2][2] = pixels[3][3] = 13
        elif entity == SEED:
            pixels[1][2] = pixels[1][3] = pixels[4][2] = pixels[4][3] = 4
            pixels[2][1] = pixels[3][1] = pixels[2][4] = pixels[3][4] = 4
            pixels[2][2] = pixels[2][3] = pixels[3][2] = pixels[3][3] = 7
        elif entity in (TARGET_A, TARGET_B, TARGET_C, TARGET_D):
            col = self._get_target_color(entity)
            if is_activated:
                pixels[1][2] = pixels[1][3] = pixels[4][2] = pixels[4][3] = col
                pixels[2][1] = pixels[3][1] = pixels[2][4] = pixels[3][4] = col
                pixels[2][2] = pixels[2][3] = pixels[3][2] = pixels[3][3] = 14
            else:
                pixels[2][2] = pixels[2][3] = pixels[3][2] = pixels[3][3] = col
                pixels[1][2] = pixels[1][3] = 3
        self.draw_pixels(px, py, pixels)

    def render_to_array(self):
        self.canvas.fill(PALETTE[0])

        for y in range(self.grid_h):
            for x in range(self.grid_w):
                entity = self._grid[y][x]
                if entity != EMPTY:
                    px = self.offset_x + x * TILE_SIZE
                    py = self.offset_y + y * TILE_SIZE
                    self.draw_tile(px, py, entity, (x, y) in self._activated_coords)

        for (bx, by) in self._blooms:
            if self._grid[by][bx] == SEED:
                continue
            px = self.offset_x + bx * TILE_SIZE
            py = self.offset_y + by * TILE_SIZE
            pixels = [[-1] * TILE_SIZE for _ in range(TILE_SIZE)]
            for dy in range(1, TILE_SIZE - 1):
                for dx in range(1, TILE_SIZE - 1):
                    pixels[dy][dx] = 14
            pixels[2][2] = pixels[2][3] = pixels[3][2] = pixels[3][3] = 6
            self.draw_pixels(px, py, pixels)

        if self._error_pos:
            ex = self.offset_x + self._error_pos[0] * TILE_SIZE
            ey = self.offset_y + self._error_pos[1] * TILE_SIZE
            err_pix = [[-1] * TILE_SIZE for _ in range(TILE_SIZE)]
            for i in range(TILE_SIZE):
                err_pix[0][i] = err_pix[TILE_SIZE - 1][i] = err_pix[i][0] = err_pix[i][TILE_SIZE - 1] = 2
            self.draw_pixels(ex, ey, err_pix)

        if self._selected_pos:
            sx = self.offset_x + self._selected_pos[0] * TILE_SIZE
            sy = self.offset_y + self._selected_pos[1] * TILE_SIZE
            sel_col = 7  # frozen (no blink)
            sel_pix = [[-1] * TILE_SIZE for _ in range(TILE_SIZE)]
            for i in range(TILE_SIZE):
                sel_pix[0][i] = sel_pix[TILE_SIZE - 1][i] = sel_pix[i][0] = sel_pix[i][TILE_SIZE - 1] = sel_col
            self.draw_pixels(sx, sy, sel_pix)

        cx = self.offset_x + self._cursor_x * TILE_SIZE
        cy = self.offset_y + self._cursor_y * TILE_SIZE
        c_col = 7 if self._selected_pos else 15
        cur_pix = [[-1] * TILE_SIZE for _ in range(TILE_SIZE)]
        for i in range(TILE_SIZE):
            cur_pix[0][i] = cur_pix[TILE_SIZE - 1][i] = cur_pix[i][0] = cur_pix[i][TILE_SIZE - 1] = c_col
        if self._selected_pos:
            cur_pix[2][2] = cur_pix[3][3] = cur_pix[2][3] = cur_pix[3][2] = c_col
        self.draw_pixels(cx, cy, cur_pix)

        ui_y = CANVAS_SIZE - UI_HEIGHT
        pygame.draw.rect(self.canvas, PALETTE[5], (0, ui_y, CANVAS_SIZE, 1))
        spacing = CANVAS_SIZE // max(1, len(self._sequence))
        for i, target_id in enumerate(self._sequence):
            col = self._get_target_color(target_id)
            ix = (spacing * i) + (spacing // 2) - (TILE_SIZE // 2)
            iy = ui_y + 5
            draw_col = 14 if i < self._seq_idx else col
            tgt_pix = [[-1] * TILE_SIZE for _ in range(TILE_SIZE)]
            for j in range(1, TILE_SIZE - 1):
                tgt_pix[1][j] = tgt_pix[TILE_SIZE - 2][j] = tgt_pix[j][1] = tgt_pix[j][TILE_SIZE - 2] = draw_col
            self.draw_pixels(ix, iy, tgt_pix)
            if i < len(self._sequence) - 1:
                pygame.draw.line(self.canvas, PALETTE[5],
                                 (ix + TILE_SIZE + 1, iy + TILE_SIZE // 2),
                                 (ix + spacing - 2, iy + TILE_SIZE // 2))

        arr = pygame.surfarray.array3d(self.canvas)   # (W, H, 3), x-major
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)  # (H, W, 3)


# ----------------------------------------------------------------------------
# Gymnasium environment
# ----------------------------------------------------------------------------
class BloomChainEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, level_index=0, randomize_seed=True,
                 max_steps=300, reward_per_target=0.0, render_mode=None, scramble=1.0,
                 step_penalty=0.0, nav_shaping=0.0):
        super().__init__()
        self.level_index = level_index
        self.randomize_seed = randomize_seed
        self.max_steps = max_steps
        self.reward_per_target = reward_per_target
        self.scramble = scramble
        self.step_penalty = step_penalty
        self.nav_shaping = nav_shaping
        self.render_mode = render_mode

    def __init__(self, level_index=0, randomize_seed=True,
                 max_steps=300, reward_per_target=0.0, render_mode=None, scramble=1.0,
                 step_penalty=0.0, nav_shaping=0.0, cursor_channel=False,
                 start_radius=None):
        super().__init__()
        self.level_index = level_index
        self.randomize_seed = randomize_seed
        self.max_steps = max_steps
        self.reward_per_target = reward_per_target
        self.scramble = scramble
        self.step_penalty = step_penalty
        self.nav_shaping = nav_shaping
        self.cursor_channel = cursor_channel
        self.start_radius = start_radius  # None=center; int=spawn within radius of seed
        self.render_mode = render_mode

        ch = 4 if cursor_channel else 3
        self.observation_space = spaces.Box(0, 255, (CANVAS_SIZE, CANVAS_SIZE, ch), dtype=np.uint8)
        self.action_space = spaces.Discrete(5)
        self.game = None
        self.steps = 0

    def _make_obs(self):
        """RGB frame; optional 4th channel marking the cursor's tile (see
        --cursor-channel). Actions are unchanged either way (5 human buttons)."""
        rgb = self.game.render_to_array()  # (64,64,3)
        if not self.cursor_channel:
            return rgb
        cur = np.zeros((CANVAS_SIZE, CANVAS_SIZE, 1), dtype=np.uint8)
        px = self.game.offset_x + self.game._cursor_x * TILE_SIZE
        py = self.game.offset_y + self.game._cursor_y * TILE_SIZE
        cur[py:py + TILE_SIZE, px:px + TILE_SIZE, 0] = 255
        return np.concatenate([rgb, cur], axis=2)  # (64,64,4)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self.randomize_seed:
            game_seed = int(self.np_random.integers(1, 1_000_000))
        else:
            game_seed = (seed if seed is not None else 12345)
        lvl = self.level_index
        if options and "level_index" in options:
            lvl = options["level_index"]
        self.game = HeadlessBloomChain(seed=game_seed, level_index=lvl, scramble=self.scramble)
        self.steps = 0
        # locate the seed (target of navigation shaping / curriculum spawn)
        self._seed_pos = None
        for y in range(self.game.grid_h):
            for x in range(self.game.grid_w):
                if self.game._grid[y][x] == SEED:
                    self._seed_pos = (x, y)
        # reverse curriculum: spawn the cursor within start_radius of the seed.
        # radius 0 = cursor ON the seed (just press interact); widen as it learns.
        if self.start_radius is not None and self._seed_pos is not None:
            sx, sy = self._seed_pos
            r = self.start_radius
            cands = [(x, y)
                     for y in range(self.game.grid_h)
                     for x in range(self.game.grid_w)
                     if abs(x - sx) + abs(y - sy) <= r]
            self.game._cursor_x, self.game._cursor_y = cands[int(self.np_random.integers(len(cands)))]
        return self._make_obs(), {"seq_idx": 0, "solved": False}

    def _dist_to_seed(self):
        if self._seed_pos is None:
            return 0
        sx, sy = self._seed_pos
        return abs(self.game._cursor_x - sx) + abs(self.game._cursor_y - sy)

    def step(self, action):
        prev_seq = self.game._seq_idx
        d0 = self._dist_to_seed()
        self.game.step_action(int(action))
        self.steps += 1

        reward = -self.step_penalty
        # potential-based shaping: reward moving the cursor toward the seed.
        # telescopes out, so it doesn't change the optimal policy; it just gives
        # purposeful navigation a per-step gradient instead of a long flat stretch.
        if self.nav_shaping > 0.0:
            reward += self.nav_shaping * (d0 - self._dist_to_seed())
        if self.reward_per_target > 0.0 and self.game._seq_idx > prev_seq:
            reward += self.reward_per_target * (self.game._seq_idx - prev_seq)

        terminated = False
        if self.game._level_solved:
            reward += 1.0
            terminated = True
        elif self.game._dead:
            terminated = True  # loss: no reward (sparse)

        truncated = self.steps >= self.max_steps
        obs = self._make_obs()
        info = {"seq_idx": self.game._seq_idx,
                "solved": bool(self.game._level_solved),
                "dead": bool(self.game._dead)}
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.game is None:
            return None
        return self.game.render_to_array()


class SyncVecBloom:
    """Minimal synchronous vector env with SAME-STEP autoreset.

    When an env terminates/truncates it is reset immediately and the *fresh*
    observation is returned on the same step. This matches the done-masking
    convention used in the PPO loop (and old-gym semantics), avoiding the
    next-step autoreset behavior introduced in Gymnasium 1.x.
    """
    def __init__(self, fns):
        self.envs = [f() for f in fns]
        self.num_envs = len(self.envs)
        self.single_observation_space = self.envs[0].observation_space
        self.single_action_space = self.envs[0].action_space

    def reset(self, seed=None):
        obs = []
        for i, e in enumerate(self.envs):
            o, _ = e.reset(seed=None if seed is None else seed + i)
            obs.append(o)
        return np.stack(obs), {}

    def step(self, actions):
        obs, rews, terms, truncs, solved = [], [], [], [], []
        for e, a in zip(self.envs, actions):
            o, r, term, trunc, info = e.step(a)
            sol = info.get("solved", False)
            if term or trunc:
                o, _ = e.reset()  # immediate reset, return fresh obs
            obs.append(o); rews.append(r)
            terms.append(term); truncs.append(trunc); solved.append(sol)
        return (np.stack(obs),
                np.array(rews, dtype=np.float32),
                np.array(terms), np.array(truncs),
                {"solved": np.array(solved)})

    def close(self):
        for e in self.envs:
            e.close()


def make_env(level_index=0, randomize_seed=True, max_steps=300, reward_per_target=0.0, scramble=1.0, step_penalty=0.0, nav_shaping=0.0, cursor_channel=False, start_radius=None):
    """Thunk for gymnasium vector envs."""
    def thunk():
        return BloomChainEnv(level_index=level_index,
                             randomize_seed=randomize_seed,
                             max_steps=max_steps,
                             reward_per_target=reward_per_target,
                             scramble=scramble,
                             step_penalty=step_penalty,
                             nav_shaping=nav_shaping,
                             cursor_channel=cursor_channel,
                             start_radius=start_radius)
    return thunk
