"""
play_live.py — watch a trained agent play click-Bloom-Chain in real time.

    python play_live.py                       # latest LPM checkpoint in results/
    python play_live.py --ckpt results/ckpts/lpm_u30000.pt
    python play_live.py --scramble 0.2 --fps 3 --greedy

Opens a window and plays fresh (randomly generated) boards live, showing the
agent's clicks (mouse pointer + selected tile), the step count, the current
scramble difficulty, and a running win tally. Close the window or press Q/Esc
to quit.

By default it SAMPLES from the policy (a touch of variety, like the training
distribution). Use --greedy for the deterministic best-action policy.
"""

import os, sys, glob, time, argparse
os.environ["BLOOM_HEADLESS"] = "0"          # we want a real window, not the dummy driver
import numpy as np
import torch
import pygame

from bloom_click_env import BloomClickEnv, N_CLICK_ACTIONS
from agent import ActorCritic


def latest_ckpt(outdir, method="lpm"):
    cks = sorted(glob.glob(os.path.join(outdir, "ckpts", f"{method}_u*.pt")))
    return cks[-1] if cks else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None, help="checkpoint .pt (default: latest lpm in --outdir/ckpts)")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--method", default="lpm", help="which method's latest ckpt to auto-pick")
    ap.add_argument("--level", type=int, default=0)
    ap.add_argument("--scramble", type=float, default=None,
                    help="board difficulty (default: the checkpoint's curriculum scramble)")
    ap.add_argument("--max-steps", type=int, default=80)
    ap.add_argument("--fps", type=float, default=4.0, help="clicks per second (watchable speed)")
    ap.add_argument("--scale", type=int, default=9, help="pixel upscale (window = 64*scale)")
    ap.add_argument("--episodes", type=int, default=0, help="0 = run until you close the window")
    ap.add_argument("--greedy", action="store_true", help="argmax instead of sampling")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else torch.device(args.device))

    ckpt = args.ckpt or latest_ckpt(args.outdir, args.method)
    if not ckpt or not os.path.exists(ckpt):
        print(f"No checkpoint found in {args.outdir}/ckpts. Pass --ckpt path/to/{args.method}_uXXXXX.pt")
        sys.exit(1)
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    meta = ck.get("meta", {})
    sc = args.scramble if args.scramble is not None else float(meta.get("scramble", 1.0))
    print(f"loaded {os.path.basename(ckpt)}  (trained to step {meta.get('step','?')}, "
          f"scramble {meta.get('scramble','?')}, solve_rate {meta.get('solve_rate','?')})")

    agent = ActorCritic(in_ch=3, n_actions=N_CLICK_ACTIONS, hw=64).to(device)
    agent.load_state_dict(ck["agent"]); agent.eval()
    env = BloomClickEnv(level_index=args.level, scramble=sc,
                        max_steps=args.max_steps, randomize_seed=True)

    pygame.init()
    S = args.scale; W = 64 * S; BAR = 76
    screen = pygame.display.set_mode((W, W + BAR))
    pygame.display.set_caption(f"Bloom Chain — {args.method.upper()} agent  (scramble {sc:.2f})")
    try:
        font = pygame.font.SysFont("menlo,consolas,monospace", 18)
        big = pygame.font.SysFont("menlo,consolas,monospace", 30, bold=True)
    except Exception:
        font = pygame.font.Font(None, 22); big = pygame.font.Font(None, 34)
    clock = pygame.time.Clock()

    def policy(obs, mask):
        x = (torch.as_tensor(obs, dtype=torch.float32, device=device) / 255.0).unsqueeze(0).permute(0, 3, 1, 2)
        m = torch.as_tensor(mask, device=device).unsqueeze(0)
        with torch.no_grad():
            logits = agent.actor(agent._features(x)).masked_fill(~m, -1e8)
            if args.greedy:
                return int(logits.argmax(-1))
            return int(torch.distributions.Categorical(logits=logits).sample())

    def draw(obs, ep, step, status, wins, plays):
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(pygame.transform.scale(surf, (W, W)), (0, 0))
        pygame.draw.rect(screen, (18, 18, 22), (0, W, W, BAR))
        info = f"ep {ep}   step {step:2d}   scramble {sc:.2f}   wins {wins}/{plays}"
        screen.blit(font.render(info, True, (225, 225, 225)), (12, W + 10))
        col = (90, 220, 120) if status == "SOLVED" else (
            (235, 120, 120) if status == "failed" else (180, 180, 190))
        screen.blit(big.render(status, True, col), (12, W + 34))
        pygame.display.flip()

    def pump():
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                return False
            if e.type == pygame.KEYDOWN and e.key in (pygame.K_ESCAPE, pygame.K_q):
                return False
        return True

    running = True; ep = wins = plays = 0
    while running and (args.episodes == 0 or ep < args.episodes):
        ep += 1
        obs, ifo = env.reset(); status = "playing"; step = 0; done = False
        draw(obs, ep, step, status, wins, plays)
        while not done and running:
            running = pump()
            if not running:
                break
            a = policy(obs, ifo["action_mask"])
            obs, r, term, trunc, ifo = env.step(a)
            step += 1; done = term or trunc
            status = "SOLVED" if ifo["solved"] else ("failed" if done else "playing")
            draw(obs, ep, step, status, wins, plays)
            clock.tick(args.fps)
        if not running:
            break
        plays += 1; wins += int(status == "SOLVED")
        draw(obs, ep, step, status, wins, plays)
        t0 = time.time()                       # hold the final frame so the result is visible
        while time.time() - t0 < 1.3:
            if not pump():
                running = False; break
            clock.tick(30)

    pygame.quit()
    print(f"done — solved {wins}/{plays}")


if __name__ == "__main__":
    main()
