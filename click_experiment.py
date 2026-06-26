"""
click_experiment.py — one-command experiment on click-based Bloom Chain.

    python click_experiment.py                      # full run (RND vs LPM)
    python click_experiment.py --timesteps 200000   # shorter
    python click_experiment.py --intrinsics rnd     # just one method

What it does, end to end:
  1. Trains a masked PPO agent on the click env, once per intrinsic method
     (default: rnd AND lpm), with an automatic scramble curriculum.
  2. Logs per-update metrics to results/<intrinsic>.csv.
  3. Saves checkpoints every --ckpt-every updates to results/ckpts/ (for replay).
  4. Generates comparison graphs -> results/plots/*.png.
  5. Renders replay GIFs of saved checkpoints -> results/replays/*.gif,
     so you can watch how the agent learned to win.

Everything lands in --outdir (default ./results). Just run it.
"""

import os, csv, time, glob, argparse
os.environ.setdefault("BLOOM_HEADLESS", "1")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from bloom_click_env import make_click_env, ClickVec, N_CLICK_ACTIONS, MAX_GRID
from agent import ActorCritic, RNDModel, RunningMeanStd


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--intrinsics", type=str, default="rnd,lpm",
                   help="Which exploration methods to train & compare.")
    p.add_argument("--timesteps", type=int, default=2_000_000)
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument("--num-steps", type=int, default=128)
    p.add_argument("--level", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=60, help="Max clicks per episode.")
    # scramble curriculum
    p.add_argument("--scramble-start", type=float, default=0.0)
    p.add_argument("--sc-step", type=float, default=0.1)
    p.add_argument("--sc-max", type=float, default=1.0)
    p.add_argument("--rc-threshold", type=float, default=0.6)
    p.add_argument("--rc-window", type=int, default=150)
    # ppo / rnd
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--gamma", type=float, default=0.999)
    p.add_argument("--int-gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--num-minibatches", type=int, default=4)
    p.add_argument("--update-epochs", type=int, default=4)
    p.add_argument("--clip-coef", type=float, default=0.1)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--ext-coef", type=float, default=2.0)
    p.add_argument("--int-coef", type=float, default=1.0)
    p.add_argument("--lpm-alpha", type=float, default=0.1)
    p.add_argument("--rnd-update-prop", type=float, default=0.25)
    p.add_argument("--obs-norm-clip", type=float, default=5.0)
    p.add_argument("--init-norm-steps", type=int, default=50)
    p.add_argument("--reward-per-target", type=float, default=0.0)
    # io
    p.add_argument("--outdir", type=str, default="results")
    p.add_argument("--ckpt-every", type=int, default=40, help="Checkpoint every N updates.")
    p.add_argument("--replay-episodes", type=int, default=30)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--no-replay", action="store_true")
    return p.parse_args()


def device_of(a):
    return (torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if a.device == "auto" else torch.device(a.device))


# ---------------------------------------------------------------------------
# Training (masked PPO + RND/LPM, scramble curriculum, CSV + checkpoints)
# ---------------------------------------------------------------------------
def train_one(intrinsic, args, device):
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    ckpt_dir = os.path.join(args.outdir, "ckpts"); os.makedirs(ckpt_dir, exist_ok=True)
    csv_path = os.path.join(args.outdir, f"{intrinsic}.csv")
    csv_f = open(csv_path, "w", newline=""); writer = csv.writer(csv_f)
    writer.writerow(["update", "step", "solve_rate", "scramble", "ext_ret",
                     "int_rew", "pg_loss", "v_loss", "rnd_loss", "sps"])

    envs = ClickVec([make_click_env(args.level, args.scramble_start, args.max_steps,
                                    True, args.reward_per_target)
                     for _ in range(args.num_envs)])
    n_act = N_CLICK_ACTIONS
    agent = ActorCritic(in_ch=3, n_actions=n_act, hw=64).to(device)
    rnd = RNDModel(in_ch=3, hw=64, lpm=(intrinsic == "lpm"), n_actions=n_act).to(device)
    rnd_params = list(rnd.predictor.parameters())
    if intrinsic == "lpm":
        rnd_params += list(rnd.error_predictor.parameters())
    opt = optim.Adam(list(agent.parameters()) + rnd_params, lr=args.lr, eps=1e-5)

    obs_rms = RunningMeanStd(shape=(1, 3, 64, 64)); ret_rms = RunningMeanStd(shape=())
    disc_int = np.zeros(args.num_envs)

    def to_chw(o):
        return (torch.as_tensor(o, dtype=torch.float32, device=device) / 255.0).permute(0, 3, 1, 2).contiguous()

    def norm_rnd(x):
        mean = torch.as_tensor(obs_rms.mean, dtype=torch.float32, device=device)
        std = torch.as_tensor(np.sqrt(obs_rms.var), dtype=torch.float32, device=device)
        return torch.clamp((x - mean) / (std + 1e-8), -args.obs_norm_clip, args.obs_norm_clip)

    N, E = args.num_steps, args.num_envs
    obs_b = torch.zeros((N, E, 3, 64, 64), device=device)
    nobs_b = torch.zeros((N, E, 3, 64, 64), device=device)
    act_b = torch.zeros((N, E), dtype=torch.long, device=device)
    mask_b = torch.zeros((N, E, n_act), dtype=torch.bool, device=device)
    logp_b = torch.zeros((N, E), device=device)
    er_b = torch.zeros((N, E), device=device); ir_b = torch.zeros((N, E), device=device)
    done_b = torch.zeros((N, E), device=device)
    ev_b = torch.zeros((N, E), device=device); iv_b = torch.zeros((N, E), device=device)

    # seed obs-norm
    nobs_np, masks_np = envs.reset(seed=args.seed)
    frames = []
    for _ in range(args.init_norm_steps):
        a = np.array([np.random.choice(np.flatnonzero(m)) for m in masks_np])
        nobs_np, _, _, _, _, masks_np = envs.step(a)
        frames.append(to_chw(nobs_np).cpu().numpy())
    obs_rms.update(np.concatenate(frames, 0))
    nobs_np, masks_np = envs.reset(seed=args.seed)
    next_obs = to_chw(nobs_np)
    next_mask = torch.as_tensor(masks_np, device=device)
    next_done = torch.zeros(E, device=device)

    ep_solved, rc_solves = [], []
    cur_sc = args.scramble_start
    gstep = 0; updates = args.timesteps // (N * E); t0 = time.time()

    for upd in range(1, updates + 1):
        for s in range(N):
            gstep += E
            obs_b[s] = next_obs; done_b[s] = next_done; mask_b[s] = next_mask
            with torch.no_grad():
                act, lp, _, ve, vi = agent.get_action_and_value(next_obs, mask=next_mask)
            act_b[s] = act; logp_b[s] = lp; ev_b[s] = ve; iv_b[s] = vi
            nobs_np, rew, term, trunc, info, masks_np = envs.step(act.cpu().numpy())
            done = np.logical_or(term, trunc)
            er_b[s] = torch.as_tensor(rew, dtype=torch.float32, device=device)
            next_obs = to_chw(nobs_np)
            next_mask = torch.as_tensor(masks_np, device=device)
            next_done = torch.as_tensor(done, dtype=torch.float32, device=device)
            nobs_b[s] = next_obs
            with torch.no_grad():
                if intrinsic == "lpm":
                    a_oh = F.one_hot(act, n_act).float()
                    ir_b[s] = rnd.lpm_curiosity(norm_rnd(next_obs), a_oh, args.lpm_alpha)
                else:
                    ir_b[s] = rnd.intrinsic_reward(norm_rnd(next_obs))
            for i in range(E):
                if info["solved"][i]:
                    ep_solved.append(1.0); rc_solves.append(1.0)
                elif done[i]:
                    ep_solved.append(0.0); rc_solves.append(0.0)

        # normalize intrinsic by std of discounted intrinsic returns
        ir_np = ir_b.cpu().numpy(); per = []
        for t in range(N):
            disc_int = disc_int * args.int_gamma + ir_np[t]; per.append(disc_int.copy())
        ret_rms.update(np.array(per).reshape(-1))
        ir_b = ir_b / float(np.sqrt(ret_rms.var) + 1e-8)

        with torch.no_grad():
            nve, nvi = agent.get_value(next_obs)
        eadv = torch.zeros_like(er_b); iadv = torch.zeros_like(ir_b); el = il = 0
        for t in reversed(range(N)):
            env_ = nve if t == N - 1 else ev_b[t + 1]
            inv_ = nvi if t == N - 1 else iv_b[t + 1]
            nonterm = 1.0 - (next_done if t == N - 1 else done_b[t + 1])
            ed = er_b[t] + args.gamma * env_ * nonterm - ev_b[t]
            el = ed + args.gamma * args.gae_lambda * nonterm * el; eadv[t] = el
            idd = ir_b[t] + args.int_gamma * inv_ - iv_b[t]
            il = idd + args.int_gamma * args.gae_lambda * il; iadv[t] = il
        eret = eadv + ev_b; iret = iadv + iv_b
        obs_rms.update(nobs_b.reshape(-1, 3, 64, 64).cpu().numpy())

        b_obs = obs_b.reshape(-1, 3, 64, 64); b_nobs = nobs_b.reshape(-1, 3, 64, 64)
        b_act = act_b.reshape(-1); b_mask = mask_b.reshape(-1, n_act)
        b_lp = logp_b.reshape(-1)
        b_adv = (args.ext_coef * eadv + args.int_coef * iadv).reshape(-1)
        b_eret = eret.reshape(-1); b_iret = iret.reshape(-1)
        b_rnd = norm_rnd(b_nobs)
        bs = N * E; mb = bs // args.num_minibatches; idx = np.arange(bs)
        pg = vl = rl = 0.0
        for _ in range(args.update_epochs):
            np.random.shuffle(idx)
            for st in range(0, bs, mb):
                j = idx[st:st + mb]
                _, nlp, ent, nve2, nvi2 = agent.get_action_and_value(b_obs[j], b_act[j], mask=b_mask[j])
                ratio = (nlp - b_lp[j]).exp()
                adv = b_adv[j]; adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                pg_loss = torch.max(-adv * ratio,
                                    -adv * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)).mean()
                v_loss = 0.5 * ((nve2 - b_eret[j]) ** 2).mean() + 0.5 * ((nvi2 - b_iret[j]) ** 2).mean()
                pred, tgt = rnd(b_rnd[j])
                rper = ((pred - tgt.detach()) ** 2).mean(1)
                m = (torch.rand(len(rper), device=device) < args.rnd_update_prop).float()
                rnd_loss = (rper * m).sum() / torch.clamp(m.sum(), min=1.0)
                if intrinsic == "lpm":
                    rnd_loss = rnd_loss + rnd.error_predictor_loss(b_rnd[j], F.one_hot(b_act[j], n_act).float())
                loss = pg_loss - args.ent_coef * ent.mean() + args.vf_coef * v_loss + rnd_loss
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(list(agent.parameters()) + rnd_params, 0.5)
                opt.step()
                pg, vl, rl = pg_loss.item(), v_loss.item(), rnd_loss.item()

        # scramble curriculum
        if args.sc_step > 0 and len(rc_solves) >= args.rc_window:
            rate = float(np.mean(rc_solves[-args.rc_window:]))
            if rate >= args.rc_threshold and cur_sc < args.sc_max - 1e-9:
                cur_sc = min(args.sc_max, round(cur_sc + args.sc_step, 4))
                envs.set_scramble(cur_sc); rc_solves.clear()
                print(f"  [{intrinsic}] scramble -> {cur_sc:.2f} (rate {rate:.2f})")

        sr = float(np.mean(ep_solved[-200:])) if ep_solved else 0.0
        sps = int(gstep / (time.time() - t0))
        writer.writerow([upd, gstep, f"{sr:.4f}", f"{cur_sc:.3f}", f"{b_eret.mean().item():.4f}",
                         f"{ir_b.mean().item():.4f}", f"{pg:.4f}", f"{vl:.4f}", f"{rl:.4f}", sps])
        if upd % 10 == 0 or upd == updates:
            csv_f.flush()
            print(f"[{intrinsic}] upd {upd}/{updates} step {gstep} | solve {sr:.3f} | sc {cur_sc:.2f} | int {ir_b.mean().item():.3f} | {sps} sps")
        if upd % args.ckpt_every == 0 or upd == updates:
            torch.save({"agent": agent.state_dict(), "obs_rms_mean": obs_rms.mean,
                        "obs_rms_var": obs_rms.var, "meta": {"intrinsic": intrinsic,
                        "update": upd, "step": gstep, "scramble": cur_sc, "solve_rate": sr}},
                       os.path.join(ckpt_dir, f"{intrinsic}_u{upd:05d}.pt"))

    csv_f.close(); envs.close()
    return csv_path


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def make_plots(args, intrinsics):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pdir = os.path.join(args.outdir, "plots"); os.makedirs(pdir, exist_ok=True)
    data = {}
    for it in intrinsics:
        p = os.path.join(args.outdir, f"{it}.csv")
        if os.path.exists(p):
            data[it] = np.genfromtxt(p, delimiter=",", names=True)
    if not data:
        return
    colors = {"rnd": "#1f77b4", "lpm": "#d62728"}

    # 1) headline comparison: solve rate vs steps
    plt.figure(figsize=(8, 5))
    for it, d in data.items():
        plt.plot(d["step"], d["solve_rate"], label=it.upper(), color=colors.get(it), lw=2)
    plt.xlabel("environment steps"); plt.ylabel("solve rate (last 200 eps)")
    plt.title("Solve rate: RND vs LPM (click Bloom Chain)"); plt.legend(); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(pdir, "solve_rate_comparison.png"), dpi=130); plt.close()

    # 2) 2x2 diagnostics
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    panels = [("solve_rate", "solve rate"), ("scramble", "curriculum scramble"),
              ("int_rew", "intrinsic reward"), ("ext_ret", "extrinsic return")]
    for a, (k, t) in zip(ax.flat, panels):
        for it, d in data.items():
            a.plot(d["step"], d[k], label=it.upper(), color=colors.get(it))
        a.set_title(t); a.set_xlabel("steps"); a.grid(alpha=.3); a.legend()
    plt.tight_layout(); plt.savefig(os.path.join(pdir, "diagnostics.png"), dpi=130); plt.close()
    print(f"plots -> {pdir}/")


# ---------------------------------------------------------------------------
# Replay: checkpoint -> GIF of the agent playing (prefer a win)
# ---------------------------------------------------------------------------
def replay_checkpoint(ckpt_path, args, device, scale=6):
    from PIL import Image
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    meta = ck.get("meta", {})
    agent = ActorCritic(in_ch=3, n_actions=N_CLICK_ACTIONS, hw=64).to(device)
    agent.load_state_dict(ck["agent"]); agent.eval()
    env = make_click_env(args.level, meta.get("scramble", 1.0), args.max_steps, True,
                         args.reward_per_target)()

    best_frames, best_solved = None, False
    for _ in range(args.replay_episodes):
        o, info = env.reset(); frames = []; done = False; solved = False
        while not done:
            frames.append(o.copy())
            x = (torch.as_tensor(o, dtype=torch.float32, device=device) / 255.0).unsqueeze(0).permute(0, 3, 1, 2)
            mask = torch.as_tensor(info["action_mask"], device=device).unsqueeze(0)
            with torch.no_grad():
                logits = agent.actor(agent._features(x)).masked_fill(~mask, -1e8)
                a = int(logits.argmax(-1))
            o, r, term, trunc, info = env.step(a); done = term or trunc
            solved = info["solved"]
        frames.append(o.copy())
        if solved and not best_solved:
            best_frames, best_solved = frames, True; break
        if best_frames is None or len(frames) > len(best_frames):
            best_frames = frames
    env.close()
    imgs = [Image.fromarray(f).resize((64 * scale, 64 * scale), Image.NEAREST) for f in best_frames]
    rdir = os.path.join(args.outdir, "replays"); os.makedirs(rdir, exist_ok=True)
    tag = f"{meta.get('intrinsic','?')}_u{meta.get('update',0):05d}_sc{meta.get('scramble',0):.2f}_{'WIN' if best_solved else 'try'}"
    out = os.path.join(rdir, f"{tag}.gif")
    imgs[0].save(out, save_all=True, append_images=imgs[1:], duration=400, loop=0)
    return out, best_solved


def main():
    args = get_args()
    device = device_of(args); print("device:", device)
    os.makedirs(args.outdir, exist_ok=True)
    intrinsics = [s.strip() for s in args.intrinsics.split(",") if s.strip()]

    for it in intrinsics:
        print(f"\n===== training: {it} =====")
        train_one(it, args, device)

    if not args.no_plots:
        make_plots(args, intrinsics)

    if not args.no_replay:
        # replay a progression of checkpoints for the LAST method (how it learned)
        last = intrinsics[-1]
        cks = sorted(glob.glob(os.path.join(args.outdir, "ckpts", f"{last}_u*.pt")))
        picks = cks[:: max(1, len(cks) // 4)][:4] + ([cks[-1]] if cks else [])
        for ck in dict.fromkeys(picks):           # dedup, keep order
            out, won = replay_checkpoint(ck, args, device)
            print(f"replay {'WIN ' if won else 'try '}-> {out}")

    print(f"\nDone. Everything in {args.outdir}/  (csv, plots/, ckpts/, replays/)")


if __name__ == "__main__":
    main()
