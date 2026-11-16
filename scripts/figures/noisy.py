import numpy as np
import matplotlib.pyplot as plt
from cure_qksd import style as _style
from cure_qksd.plot_labels import PLOT_TITLE
from cure_qksd.open_data import load_sweeps

_style.apply()

systems, results = load_sweeps()

LAYOUT = [["LiH", "H6"], ["H2O_R1.4", "HF"], ["BeH2", "N2"]]
STATE_NAME = {0: "ground state", 1: "1st excited state"}
BASE_ONLY = "#7f8c8d"
N2_MAX_FACTOR = 86.0

fig, axes = plt.subplots(3, 2, figsize=(7.0, 7.2))
any_missing = False

for row, keys in enumerate(LAYOUT):
    for col, key in enumerate(keys):
        ax = axes[row, col]
        r = results[key]["tau_noisy"]
        k = int(r.get("state", 0))
        tau = r["tau"]
        x = r["factors"]
        shown = np.ones(len(x), dtype=bool)
        if key == "N2":
            shown = r["factors"] <= N2_MAX_FACTOR

        ax.fill_between(x[shown], r["plain_q25"][shown], r["plain_q75"][shown], color=_style.BRICK, alpha=0.15)
        ax.semilogy(x[shown], np.maximum(r["plain_median"][shown], 1e-16), color=_style.BRICK,
                    marker="x", markersize=3.2, linewidth=1.1)
        ax.semilogy(x[shown], np.maximum(r["cure1_median"][shown], 1e-16), color=BASE_ONLY,
                    linestyle="--", linewidth=1.0, marker="s", markersize=2.0, zorder=2)
        ax.fill_between(x[shown], r["cure_q25"][shown], r["cure_q75"][shown], color=_style.NAVY, alpha=0.15)
        ax.semilogy(x[shown], np.maximum(r["cure_median"][shown], 1e-16), color=_style.NAVY,
                    marker="o", markersize=2.6, linewidth=1.3, zorder=3)

        returned = np.array([np.sum(np.isfinite(e)) for e in r["cure_errors"]])
        missing = (returned < 0.5 * len(r["cure_errors"][0])) & shown
        if np.any(missing):
            any_missing = True
            ytop = 4.0 * np.nanmax(r["plain_median"])
            ax.semilogy(x[missing], np.full(missing.sum(), ytop), linestyle="none", marker="o",
                        markersize=3.2, markerfacecolor="white", markeredgecolor=_style.NAVY, zorder=4)
        ax.axvspan(1.0, x[shown].max() * 1.02, color=_style.SLATE, alpha=0.08, zorder=0)
        ax.axhline(1.6e-3, color=_style.INK, linewidth=0.6, linestyle=":", alpha=0.6)
        M_noisy = results[key].get("M_noisy", results[key]["M"])
        ax.set_title(rf"{PLOT_TITLE[key]}, {STATE_NAME[k]}, $M={M_noisy}$", loc="left", fontsize=7.2)
        ax.set_yscale("log")

handles = [
    plt.Line2D([0], [0], color=_style.BRICK, marker="x", markersize=4, linewidth=1.1),
    plt.Line2D([0], [0], color=_style.NAVY, marker="o", markersize=3, linewidth=1.3),
    plt.Line2D([0], [0], color=BASE_ONLY, marker="s", markersize=2.5, linewidth=1.0, linestyle="--"),
    plt.Line2D([0], [0], color=_style.NAVY, marker="o", markersize=3.5, markerfacecolor="white", linestyle="none"),
]
labels = ["plain UQKSD (principal branch)", "CURE-QKSD", "CURE-QKSD, base run only",
          "state not returned by CURE-QKSD"]
if not any_missing:
    handles, labels = handles[:3], labels[:3]
fig.legend(handles, labels,
           loc="upper center", ncol=3 if not any_missing else 2, fontsize=8.0, frameon=False,
           bbox_to_anchor=(0.5, 1.03 if not any_missing else 1.06), title=r"$\sigma=10^{-4}$")

for ax in axes[:, 0]:
    ax.set_ylabel(r"median $|E_{\mathrm{est}}-E_{\mathrm{exact}}|$ (Ha)")
for ax in axes[-1, :]:
    ax.set_xlabel(r"$\tau/\tau_{\mathrm{alias}}$")

fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(_style.figure_path("fig08_noisy.pdf"), bbox_inches="tight")
print("wrote fig08_noisy.pdf")

for keys in LAYOUT:
    for key in keys:
        r = results[key]["tau_noisy"]
        k = int(r.get("state", 0))
        factors = r["factors"]
        shown = np.ones(len(factors), dtype=bool)
        if key == "N2":
            shown = r["factors"] <= N2_MAX_FACTOR
        n_trials = r["cure_errors"].shape[1]
        print(f"\n{key} state {k}: tau_alias={results[key]['tau_alias']:.4f}, M={results[key]['M_noisy']}")
        print("   tau/tau_alias        plain_med    CURE_med  [q25,q75]          base-only_med  returned(CURE)  wrong>1e-2 (CURE, base-only)  grids")
        for i, f in enumerate(factors):
            if not shown[i]:
                continue
            ce, c1 = r["cure_errors"][i], r["cure1_errors"][i]
            print(f"   {f:12.2f}  {r['plain_median'][i]:10.2e}  {r['cure_median'][i]:10.2e} [{r['cure_q25'][i]:.1e},{r['cure_q75'][i]:.1e}]"
                  f"  {r['cure1_median'][i]:10.2e}  {np.sum(np.isfinite(ce))}/{n_trials}  {int(np.sum(~(ce < 1e-2)))}, {int(np.sum(~(c1 < 1e-2)))}"
                  f"  {np.bincount(np.asarray(r['n_grids'][i], dtype=int), minlength=3)[1:]}")
