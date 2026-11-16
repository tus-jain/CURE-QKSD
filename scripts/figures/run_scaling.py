"""Fig. 9 and the numbers of Sec. III E and Appendix F, from data/scaling."""

import numpy as np
import matplotlib.pyplot as plt

from cure_qksd import scaling as sc
from cure_qksd import style as _style
from cure_qksd.paths import DATA

_style.apply()
plt.rcParams["axes.formatter.use_mathtext"] = True

syn = dict(np.load(DATA / "scaling" / "synthetic.npz"))
n2 = dict(np.load(DATA / "scaling" / "n2.npz"))
G = np.arange(1, sc.G_MAX + 1)
COLORS = {5: _style.NAVY, 10: _style.TEAL, 20: _style.OCHRE}
rng = np.random.default_rng(0)


def p_est(R, g):
    return 1 - (1 - sc.eps_g(g) / np.pi) ** R


def model(N1, R):
    out = [N1]
    for g in range(2, len(R) + 1):
        out.append(out[-1] * p_est(R[g - 1], g))
    return np.array(out)


def mean_ci(x, B=2000):
    idx = rng.integers(0, len(x), (B, len(x)))
    boot = x[idx].mean(1)
    return x.mean(0), np.percentile(boot, 2.5, axis=0), np.percentile(boot, 97.5, axis=0)


def images(d, tag):
    return (d[tag + "__images"] - d[tag + "__on_true"]).astype(float)


def plot_counts(ax, x, R, color, marker, label, mfc, ls):
    m, lo, hi = mean_ci(x)
    keep = m > 0
    ax.errorbar(G[keep], m[keep], yerr=[m[keep] - lo[keep], hi[keep] - m[keep]], fmt=marker, color=color,
                mfc=mfc, ms=4, elinewidth=0.7, capsize=1.5, label=label)
    ax.plot(G, model(m[0], R), ls, color=color, lw=0.9, alpha=0.85)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.7))
for R, c in COLORS.items():
    tag = f"R{R}_Nb10_s0.005"
    plot_counts(ax1, images(syn, tag), np.full(sc.G_MAX, R), c, "o", f"$R={R}$", c, "-")
    ax2.plot(G, syn[tag + "__correct"].all(2).mean(0), "o-", color=c, ms=3.5, lw=0.9)
tag = "n2_Nb30"
plot_counts(ax1, images(n2, tag), n2[tag + "__roots"].mean(0), _style.INK, "s", r"N$_2$", "white", "--")
ax2.plot(G, n2[tag + "__correct"].all(2).mean(0), "s--", color=_style.INK, mfc="white", ms=4.5, lw=0.8)
ax1.set_yscale("log")
ax1.set_xlabel("number of runs $G$")
ax1.set_ylabel("phantom peaks per calculation")
ax1.set_ylim(3e-4, 3e3)
ax1.legend(ncol=2, loc="upper right")
ax2.set_xlabel("number of runs $G$")
ax2.set_ylabel("fraction with correct winding numbers")
ax2.set_ylim(-0.03, 1.03)
for ax, label in [(ax1, "(a)"), (ax2, "(b)")]:
    ax.set_xticks(G)
    ax.set_title(label, loc="left", fontsize=9.5)
fig.tight_layout()
fig.savefig(_style.figure_path("fig09_run_scaling.pdf"))
print("wrote fig09_run_scaling.pdf")


def ratios(x):
    return [x[:, g].sum() / x[:, g - 1].sum() if x[:, g - 1].sum() >= 20 else np.nan for g in range(1, x.shape[1])]


def no_removal_stop(n_peaks):
    return np.array([next((g for g in range(1, len(r)) if r[g] >= r[g - 1]), len(r) - 1) for r in n_peaks])


print(f"\neps_g: g=2 {sc.eps_g(2):.3f} rad, g=6 {sc.eps_g(6):.3f} rad; p for R=10: {p_est(10, 2):.3f}")
print("\nSynthetic survival per run, measured / estimate (g = 2..6, at least 20 images before the run):")
for Nb in (10, 1, 100):
    for R in COLORS:
        r = ratios(images(syn, f"R{R}_Nb{Nb}_s0.005"))
        print(f"  Nb={Nb:3d} R={R:2d}: " + " ".join("  -  " if np.isnan(v) else f"{v / p_est(R, g + 2):.2f}"
                                                  for g, v in enumerate(r)))
print("\nN2 survival at run 2, measured / estimate:")
for Nb in (3, 10, 30):
    tag = f"n2_Nb{Nb}"
    print(f"  Nb={Nb:2d}: {ratios(images(n2, tag))[0] / p_est(n2[tag + '__roots'].mean(0)[1], 2):.2f}")
m = images(n2, "n2_Nb30").mean(0)
print(f"\nN2, window 30 periods: images per calculation {m[0]:.0f} after one run, {m[-1]:.3f} after six")
print("N2 E0 and E1 correct from run 2 on, fraction per window:",
      [float(n2[f'n2_Nb{Nb}__correct'][:, 1:].all(2).mean()) for Nb in (3, 10, 30)])
print(f"N2 images on true energies: lowest {np.min([n2[f'n2_Nb{Nb}__min_E_on_true'].min() for Nb in (3, 10, 30)]):.2f} Ha above E1")

print("\nStopping rule (calculations that do not stop count as six runs):")
for R in COLORS:
    runs, right = [], []
    for Nb in (1, 10, 100):
        tag = f"R{R}_Nb{Nb}_s0.005"
        gs = syn[tag + "__g_stop"]
        runs.append(gs.mean() + 1)
        right.append(syn[tag + "__correct"].all(2)[np.arange(len(gs)), gs].mean())
    print(f"  R={R:2d}: mean runs {', '.join(f'{v:.1f}' for v in runs)} for Nb = 1, 10, 100; "
          f"all targets correct {', '.join(f'{100 * v:.0f}%' for v in right)}")
print("  R=10, Nb=10, all targets correct at the stopping point versus phase noise:")
for s in (0.0, 0.005, 0.01):
    tag = f"R10_Nb10_s{s}"
    gs = syn[tag + "__g_stop"]
    print(f"    {s} rad: {100 * syn[tag + '__correct'].all(2)[np.arange(len(gs)), gs].mean():.1f}%")

print("\nImages on true energies, synthetic: measured / (R-1) n_W 2 eps_phi / pi, n_W = 3")
for R in COLORS:
    est = (R - 1) * 3 * 2 * sc.eps_g(2) / np.pi
    print(f"  R={R:2d}: " + ", ".join(f"{syn[f'R{R}_Nb{Nb}_s0.005__on_true'].mean() / est:.2f}"
                                       for Nb in (1, 10, 100)) + " for Nb = 1, 10, 100")

pe = n2["n6_Nb10__phase_err"]
worst = np.median(pe[:, :2], axis=0).max(0)
print(f"\nN2 E2-E5, largest median phase error in runs 1 and 2: {np.round(worst[2:], 3).tolist()} rad")

print("\nStopping when a new run removes no peak:")
for Nb in (3, 10, 30):
    tag = f"n2_Nb{Nb}"
    gs = no_removal_stop(n2[tag + "__n_peaks"])
    ok = n2[tag + "__correct"].all(2)[np.arange(len(gs)), gs].mean()
    print(f"  N2 Nb={Nb:2d}: mean runs {gs.mean() + 1:.1f}, both targets correct {100 * ok:.0f}%")
for R in COLORS:
    extra = []
    for Nb in (1, 10, 100):
        tag = f"R{R}_Nb{Nb}_s0.005"
        extra.append(no_removal_stop(syn[tag + "__n_peaks"]).mean() - syn[tag + "__g_stop"].mean())
    print(f"  synthetic R={R:2d}: extra runs versus the stopping rule {', '.join(f'{v:.1f}' for v in extra)}")
