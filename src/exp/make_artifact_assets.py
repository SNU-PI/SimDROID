import os
"""Build result plots + stats JSON for the Bundle A artifact from analysis outputs."""
import csv, json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(os.environ.get("VWM_ARTIFACTS", "artifacts")) / "bundle_a"
OUT = ROOT / "analysis" / "artifact_assets"
OUT.mkdir(parents=True, exist_ok=True)

INK, SUB, LINE = "#21252C", "#626B78", "#E1E4EA"
ACCENT, BLUE, OK, WARN = "#C4402C", "#3E68A8", "#256E46", "#8A5A14"
FAM_KO = {"hill_roll": "A1 rolling hill", "two_ball": "A2 two-ball", "pendulum_rod": "A3 rod pendulum"}
plt.rcParams.update({"font.size": 10.5, "axes.edgecolor": LINE, "axes.labelcolor": INK,
                     "xtick.color": SUB, "ytick.color": SUB, "text.color": INK,
                     "axes.titlecolor": INK, "figure.facecolor": "white", "axes.facecolor": "white"})

def wilson(k, n, z=1.96):
    if n == 0: return (np.nan, np.nan)
    p = k / n; d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return c-h, c+h

def fit_logistic(S, k, n):
    """MLE grid fit of p=1/(1+exp(-b(S-m))). Returns (m=PSE, b=slope) or None."""
    S, k, n = np.asarray(S, float), np.asarray(k, float), np.asarray(n, float)
    use = n > 0
    S, k, n = S[use], k[use], n[use]
    if len(S) < 3 or k.sum() == 0 or (n-k).sum() == 0:
        return None
    best, arg = -1e18, None
    for m in np.arange(0.40, 2.05, 0.01):
        for b in list(np.arange(0.5, 30.5, 0.25)):
            p = 1/(1+np.exp(-b*(S-m)))
            p = np.clip(p, 1e-6, 1-1e-6)
            ll = float((k*np.log(p) + (n-k)*np.log(1-p)).sum())
            if ll > best: best, arg = ll, (float(m), float(b))
    return arg

def main(seeds):
    summary = json.loads((ROOT/"analysis/summary.json").read_text())
    rows = list(csv.DictReader(open(ROOT/"analysis/samples.csv")))
    for r in rows:
        for f in ("S","valid_frac"): r[f] = float(r[f]) if r[f] not in ("", None) else None
        for f in ("prediction","outcome","valid","correct"): r[f] = int(r[f])
        r["boundary"] = r["boundary"] in ("True","1")
        r["law_slope"] = float(r["law_slope"]) if r["law_slope"] else None
        r["momentum_ratio"] = float(r["momentum_ratio"]) if r["momentum_ratio"] else None

    stats = {"seeds": seeds}
    # ---- figure 1: P(경계통과 예측 | S) 3-panel
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.4), dpi=170, sharey=True)
    for ax, fam in zip(axes, ("hill_roll","two_ball","pendulum_rod")):
        curve = summary[fam]["curve"]
        Ss = [c["S"] for c in curve]
        ax.plot([0.44,1,1,2.06],[0,0,1,1], color=BLUE, lw=1.7, label="ground truth", zorder=2)
        ax.axhline(1.0, color=SUB, ls=(0,(4,3)), lw=1.1, label="encoder baseline", zorder=1)
        ax.axvline(1.0, color=LINE, lw=1)
        ks = []; ns_ = []
        for c in curve:
            nd = c["n"] - c["n_undecided"]
            k = (c["p_success"] * nd) if nd and not np.isnan(c["p_success"]) else 0
            ks.append(round(k)); ns_.append(nd)
            if nd:
                lo, hi = wilson(k, nd)
                ax.plot([c["S"]]*2, [lo,hi], color=ACCENT, lw=1.2, alpha=.55, zorder=3)
        pm = [k/n if n else np.nan for k,n in zip(ks,ns_)]
        ax.plot(Ss, pm, "-o", color=ACCENT, lw=1.8, ms=4.5, label="Cosmos V2W", zorder=4)
        for S,c in zip(Ss,curve):
            if c["n_undecided"]:
                ax.bar(S, c["n_undecided"]/c["n"]*0.16, width=0.028, bottom=-0.24,
                       color=SUB, alpha=.45, zorder=1)
        fit = fit_logistic(Ss, ks, ns_)
        if fit and 0.45 < fit[0] < 2.0 and fit[1] > 1.0:
            xs = np.linspace(0.45, 2.05, 200)
            ax.plot(xs, 1/(1+np.exp(-fit[1]*(xs-fit[0]))), color=ACCENT, lw=1, ls=":", alpha=.8)
            ax.annotate(f"PSE≈{fit[0]:.2f}", (fit[0], 0.5), textcoords="offset points",
                        xytext=(6,-2), fontsize=9, color=ACCENT)
        stats[fam+"_fit"] = {"PSE": fit[0], "slope": fit[1]} if fit else None
        ax.set_title(FAM_KO[fam], fontsize=11)
        ax.set_xlim(0.44, 2.06); ax.set_ylim(-0.26, 1.06)
        ax.set_xticks([0.5,1.0,1.5,2.0]); ax.set_yticks([0,0.5,1.0])
        ax.set_xlabel("S (saturation)")
        ax.text(0.46,-0.215,"undecided", fontsize=8, color=SUB)
    axes[0].set_ylabel("P(predict pass | decided)")
    axes[2].legend(loc="lower right", fontsize=8.5, frameon=False)
    fig.tight_layout(); fig.savefig(OUT/"curves3.png", bbox_inches="tight"); plt.close(fig)

    # ---- figure 2: law metrics
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.2), dpi=170)
    ax = axes[0]  # hill slopes
    hs = [r["law_slope"] for r in rows if r["family"]=="hill_roll" and r["role"]=="grid" and r["law_slope"] is not None]
    hs = [s for s in hs if -80 < s < 80]
    ax.hist(hs, bins=np.arange(-60,62,6), color=ACCENT, alpha=.75)
    for v, lab, col in [(-14.0,"rolling −(10/7)g", OK), (-19.6,"sliding −2g", BLUE),
                        (summary["hill_roll"]["gt_law_slope_mean"], "GT measured", SUB)]:
        if v is not None: ax.axvline(v, color=col, lw=1.4, ls="--")
    ax.set_title("A1 energy-law slope d(v²)/dy"); ax.set_xlabel("slope (m/s² eq.)"); ax.set_ylabel("rollouts")
    stats["hill_gen_slopes"] = {"n": len(hs), "mean": float(np.mean(hs)) if hs else None,
                                "median": float(np.median(hs)) if hs else None}
    ax = axes[1]  # two-ball momentum by S
    for r in rows:
        if r["family"]=="two_ball" and r["role"]=="grid" and r["momentum_ratio"] is not None:
            ax.plot(r["S"], max(min(r["momentum_ratio"],2.2),-1.6), "o", ms=3.6, color=ACCENT, alpha=.6)
    ax.axhline(1.0, color=OK, lw=1.4, ls="--"); ax.axhline(0.0, color=LINE, lw=1)
    ax.text(2.0, 1.03, "GT 0.98–1.00", fontsize=8.5, color=OK, ha="right")
    ax.set_title("A2 momentum ratio Σmv(after)/Σmv(before)"); ax.set_xlabel("S")
    ax.set_ylim(-1.65, 2.3)
    mr = [r["momentum_ratio"] for r in rows if r["family"]=="two_ball" and r["role"]=="grid" and r["momentum_ratio"] is not None]
    stats["two_ball_momentum"] = {"n": len(mr), "mean": float(np.mean(mr)), "median": float(np.median(mr))}
    ax = axes[2]  # pendulum slopes
    ps = [r["law_slope"] for r in rows if r["family"]=="pendulum_rod" and r["role"]=="grid" and r["law_slope"] is not None]
    ps = [s for s in ps if -80 < s < 80]
    ax.hist(ps, bins=np.arange(-60,62,6), color=ACCENT, alpha=.75)
    if summary["pendulum_rod"]["gt_law_slope_mean"] is not None:
        ax.axvline(summary["pendulum_rod"]["gt_law_slope_mean"], color=SUB, lw=1.4, ls="--")
    ax.set_title("A3 energy-law slope"); ax.set_xlabel("slope")
    stats["pend_gen_slopes"] = {"n": len(ps), "mean": float(np.mean(ps)) if ps else None,
                                "median": float(np.median(ps)) if ps else None}
    fig.tight_layout(); fig.savefig(OUT/"laws3.png", bbox_inches="tight"); plt.close(fig)

    # ---- per-family stats for map table
    for fam in ("hill_roll","two_ball","pendulum_rod"):
        fr = [r for r in rows if r["family"]==fam and r["role"]=="grid"]
        sm = summary[fam]
        stats[fam] = {
            "n": len(fr),
            "valid_rate_strict": sm["valid_rate"],
            "valid_frac_mean": float(np.mean([r["valid_frac"] for r in fr])),
            "undecided_rate": sm["undecided_rate"],
            "acc_nonboundary": sm["accuracy_nonboundary"],
            "encoder_acc": sm["encoder_accuracy_nonboundary"],
            "gt_slope": sm["gt_law_slope_mean"], "gen_slope": sm["gen_law_slope_mean"],
            "momentum_mean": sm["momentum_ratio_mean"],
            "always_pass_frac": float(np.mean([r["prediction"]==1 for r in fr])),
        }
    stats["prechecks"] = summary["prechecks"]

    # ---- downscaled result sheets for embedding
    import imageio.v2 as imageio
    for fam in ("hill_roll","two_ball","pendulum_rod"):
        sp = ROOT/"analysis/sheets"/f"{fam}.jpg"
        if sp.exists():
            im = imageio.imread(sp)[::2, ::2]
            imageio.imwrite(OUT/f"sheet_{fam}.jpg", im, quality=72)
    (OUT/"stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print(json.dumps({k:v for k,v in stats.items() if not isinstance(v, dict) or len(str(v))<400},
                     indent=2, ensure_ascii=False)[:1800])
    print("assets in", OUT)

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv)>1 else "1-3")
