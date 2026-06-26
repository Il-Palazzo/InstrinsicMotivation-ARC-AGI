"""
bloom_click_env.py — Bloom Chain with a "mouse-click" action space.

Instead of moving a cursor with WASD, the agent clicks tiles directly. This
collapses a swap from an ~8-step keyboard chain (navigate, select, navigate,
swap) down to TWO actions, which is the change that gives PPO a real chance at
the swap skill the keyboard agent could never learn.

Action space : Discrete(64)  -> click cell (r, c) on a fixed max 8x8 grid.
               Boards smaller than 8x8 mask the out-of-range / non-clickable
               cells. info["action_mask"] (and env.action_mask()) gives the
               boolean mask the policy must apply.

Two-click protocol (your spec):
  - first click selects a tile; second click resolves:
      tile + tile  -> swap them
      seed + seed  -> tick the simulation (advance blooms)
      seed + other -> nothing (selection clears)
      other + seed -> nothing (selection clears)
  - rocks / targets / already-bloomed tiles are NOT clickable (masked out).

Reward : sparse +1 on win, 0 otherwise (no hand-tuned shaping by default).
Obs    : the same 64x64x3 pixels as the keyboard version (cursor drawn at the
         last click = a mouse pointer; selected tile highlighted).
"""

import os
if os.environ.get("BLOOM_HEADLESS", "1") == "1":
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from bloom_env import HeadlessBloomChain, SOIL, EMPTY, SEED, CANVAS_SIZE

MAX_GRID = 8
N_CLICK_ACTIONS = MAX_GRID * MAX_GRID  # 64


class BloomClickEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, level_index=0, scramble=1.0, max_steps=60,
                 randomize_seed=True, reward_per_target=0.0):
        super().__init__()
        self.level_index = level_index
        self.scramble = scramble
        self.max_steps = max_steps
        self.randomize_seed = randomize_seed
        self.reward_per_target = reward_per_target

        self.observation_space = spaces.Box(0, 255, (CANVAS_SIZE, CANVAS_SIZE, 3), np.uint8)
        self.action_space = spaces.Discrete(N_CLICK_ACTIONS)
        self.game = None
        self.sel = None
        self.steps = 0

    # --- which cells can be clicked right now ---
    def action_mask(self):
        m = np.zeros(N_CLICK_ACTIONS, dtype=bool)
        g = self.game
        for r in range(g.grid_h):
            for c in range(g.grid_w):
                cell = g._grid[r][c]
                clickable = (cell == SEED) or (
                    cell in (SOIL, EMPTY) and (c, r) not in g._blooms)
                if clickable:
                    m[r * MAX_GRID + c] = True
        return m

    def _obs(self):
        return self.game.render_to_array()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        gseed = int(self.np_random.integers(1, 2**31)) if self.randomize_seed else (seed or 12345)
        lvl = self.level_index
        if options and "level_index" in options:
            lvl = options["level_index"]
        self.game = HeadlessBloomChain(seed=gseed, level_index=lvl, scramble=self.scramble)
        self.game._selected_pos = None
        self.sel = None
        self.steps = 0
        return self._obs(), {"solved": False, "action_mask": self.action_mask()}

    def step(self, action):
        g = self.game
        a = int(action)
        r, c = divmod(a, MAX_GRID)
        prev_seq = g._seq_idx
        in_grid = (r < g.grid_h and c < g.grid_w)
        cell = g._grid[r][c] if in_grid else None

        if in_grid:                       # draw the "mouse pointer" at the click
            g._cursor_x, g._cursor_y = c, r

        if self.sel is None:
            # first click: select if it's a meaningful tile
            if cell == SEED:
                self.sel = ("seed", r, c); g._selected_pos = (c, r)
            elif in_grid and cell in (SOIL, EMPTY) and (c, r) not in g._blooms:
                self.sel = ("tile", r, c); g._selected_pos = (c, r)
            # else: no-op (masked anyway)
        else:
            typ, sr, sc = self.sel
            if typ == "seed":
                if cell == SEED:
                    g._run_simulation_tick()          # seed + seed -> tick
                # seed + other -> nothing
            else:  # selected a tile
                if in_grid and cell in (SOIL, EMPTY) and (c, r) not in g._blooms:
                    g._grid[sr][sc], g._grid[r][c] = g._grid[r][c], g._grid[sr][sc]
                # tile + seed / invalid -> nothing
            self.sel = None
            g._selected_pos = None

        self.steps += 1
        reward = 0.0
        if self.reward_per_target > 0.0 and g._seq_idx > prev_seq:
            reward += self.reward_per_target * (g._seq_idx - prev_seq)
        terminated = False
        if g._level_solved:
            reward += 1.0; terminated = True
        elif g._dead:
            terminated = True
        truncated = self.steps >= self.max_steps

        info = {"solved": bool(g._level_solved), "dead": bool(g._dead),
                "seq_idx": g._seq_idx, "action_mask": self.action_mask()}
        return self._obs(), reward, terminated, truncated, info

    def render(self):
        return None if self.game is None else self._obs()


class ClickVec:
    """Synchronous vectorizer with same-step reset that also returns action masks."""
    def __init__(self, fns):
        self.envs = [f() for f in fns]
        self.num_envs = len(self.envs)
        self.single_observation_space = self.envs[0].observation_space
        self.single_action_space = self.envs[0].action_space

    def reset(self, seed=None):
        obs, masks = [], []
        for i, e in enumerate(self.envs):
            o, info = e.reset(seed=None if seed is None else seed + i)
            obs.append(o); masks.append(info["action_mask"])
        return np.stack(obs), np.stack(masks)

    def step(self, actions):
        obs, rews, terms, truncs, solved, masks = [], [], [], [], [], []
        for e, a in zip(self.envs, actions):
            o, rr, term, trunc, info = e.step(a)
            sol = info["solved"]
            if term or trunc:
                o, info = e.reset()
            obs.append(o); rews.append(rr); terms.append(term)
            truncs.append(trunc); solved.append(sol); masks.append(info["action_mask"])
        return (np.stack(obs), np.array(rews, np.float32), np.array(terms),
                np.array(truncs), {"solved": np.array(solved)}, np.stack(masks))

    def set_scramble(self, s):
        for e in self.envs:
            e.scramble = s

    def close(self):
        for e in self.envs:
            e.close()


def make_click_env(level_index=0, scramble=1.0, max_steps=60,
                   randomize_seed=True, reward_per_target=0.0):
    def thunk():
        return BloomClickEnv(level_index, scramble, max_steps,
                             randomize_seed, reward_per_target)
    return thunk
