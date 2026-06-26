"""
make_graphs.py — build comparison graphs from existing result CSVs.

Run anytime after training (no training, no torch needed — just matplotlib):

    python make_graphs.py                 # reads ./results/*.csv
    python make_graphs.py --outdir runs   # a different results folder
    python make_graphs.py --smooth 25     # rolling-mean smoothing window

It picks up whichever methods are present (none / rnd / lpm) and writes:
    <outdir>/plots/solve_rate_comparison.png   headline curve(s)
    <outdir>/plots/diagnostics.png             solve / scramble / int / ext
    <outdir>/plots/summary.txt                 final + best numbers per method
"""

import os, glob, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABEL = {"none": "Sparse only (no curiosity)", "rnd": "RND", "lpm": "LPM"}
COLOR = {"none": "#7f7f7f", "rnd": "#1f77b4", "lpm": "#d62728"}
ORDER = ["none", "rnd", "lpm"]


def smooth(y, w):
    if w <= 1 or len(y) < w:
        return y
    k = np.ones(w) / w
    return np.convolve(y, k, mode="same")


def load(outdir):
    data = {}
    for p in sorted(glob.glob(os.path.join(outdir, "*.csv"))):
        name = os.path.splitext(os.path.basename(p))[0]
        try:
            d = np.genfromtxt(p, delimiter=",", names=True)
            if d.size > 1:
                data[name] = d
        except Exception as e:
            print(f"  skip {p}: {e}")
    return {k: data[k] for k in ORDER if k in data} or data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--smooth", type=int, default=15, help="rolling-mean window (1 = off)")
    args = ap.parse_args()

    data = load(args.outdir)
    if not data:
        print(f"No CSVs found in {args.outdir}/. Run training first.")
        return
    print("found:", ", ".join(data.keys()))
    pdir = os.path.join(args.outdir, "plots"); os.makedirs(pdir, exist_ok=True)

    def col(d, name):
        return d[name] if name in d.dtype.names else np.full(len(d["step"]), np.nan)

    # 1) headline: solve rate vs steps
    plt.figure(figsize=(9, 5.5))
    for k, d in data.items():
        plt.plot(d["step"], smooth(d["solve_rate"], args.smooth),
                 label=LABEL.get(k, k), color=COLOR.get(k), lw=2)
    plt.xlabel("environment steps"); plt.ylabel("solve rate (rolling 200 episodes)")
    plt.title("Solve rate on click-based Bloom Chain")
    plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig(os.path.join(pdir, "solve_rate_comparison.png"), dpi=140); plt.close()

    # 2) diagnostics 2x2
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    panels = [("solve_rate", "Solve rate"), ("scramble", "Curriculum scramble (difficulty)"),
              ("int_rew", "Intrinsic reward"), ("ext_ret", "Extrinsic return")]
    for a, (key, title) in zip(ax.flat, panels):
        for k, d in data.items():
            y = col(d, key)
            a.plot(d["step"], smooth(y, args.smooth) if key != "scramble" else y,
                   label=LABEL.get(k, k), color=COLOR.get(k))
        a.set_title(title); a.set_xlabel("steps"); a.grid(alpha=.3); a.legend()
    plt.tight_layout(); plt.savefig(os.path.join(pdir, "diagnostics.png"), dpi=140); plt.close()

    # 3) text summary
    lines = ["method                         final_solve  best_solve  max_scramble  final_step"]
    for k, d in data.items():
        sr = d["solve_rate"]; sc = col(d, "scramble")
        lines.append(f"{LABEL.get(k,k):30s} {sr[-1]:11.3f} {np.nanmax(sr):11.3f} "
                     f"{np.nanmax(sc):13.2f} {int(d['step'][-1]):11d}")
    summary = "\n".join(lines)
    with open(os.path.join(pdir, "summary.txt"), "w") as f:
        f.write(summary + "\n")
    print("\n" + summary)
    print(f"\nplots -> {pdir}/  (solve_rate_comparison.png, diagnostics.png, summary.txt)")


if __name__ == "__main__":
    main()
