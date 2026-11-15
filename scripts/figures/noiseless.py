import numpy as np
import matplotlib.pyplot as plt
from cure_qksd import style as _style
from cure_qksd.plot_labels import PLOT_TITLE
from cure_qksd.open_data import load_sweeps

_style.apply()

systems, results = load_sweeps()

ORDER = ["H6", "HF", "N2", "LiH", "H2O_R1.4", "BeH2"]

STATE_COLORS = [_style.NAVY, _style.OCHRE, "#239b56", "#2e86c1", "#6e2c00", "#6a1b9a"]
STATE_MARKERS = ["o", "s", "^", "D", "v", "P"]

fig, axes = plt.subplots(3, 2, figsize=(7.0, 7.2))
axes_flat = axes.flatten()

for ax, key in zip(axes_flat, ORDER):
    r = results[key]["tau_noiseless"]
    tau = r["tau"]
    err = r["err"]
    n_states = err.shape[1]
    ta_state = results[key].get("tau_alias_per_state", np.full(n_states, r["tau_alias"]))
    for s in range(n_states):
        ax.axvspan(ta_state[s], tau.max() * 1.02, color=STATE_COLORS[s], alpha=0.03,
                   zorder=0, ymin=0.0, ymax=1.0)
    for s in range(n_states):
        ax.semilogy(tau, np.maximum(err[:, s], 1e-16), color=STATE_COLORS[s],
                    marker=STATE_MARKERS[s], markersize=2.4, linewidth=1.1)
        ax.axvline(ta_state[s], color=STATE_COLORS[s], linewidth=0.7, linestyle=":", alpha=0.7)
    ax.axhline(1.6e-3, color=_style.INK, linewidth=0.6, linestyle=":", alpha=0.6)
    ax.set_title(rf"{PLOT_TITLE[key]}, $M={results[key]['M']}$", loc="left", fontsize=7.6)
    ax.set_yscale("log")
    plotted = np.maximum(err[np.isfinite(err)], 1e-16)
    ymin = 10.0 ** (np.floor(np.log10(plotted.min())) - 1.0)
    ax.set_ylim(ymin, 1e1)

handles = [plt.Line2D([0], [0], color=c, marker=m, markersize=3, linewidth=1.1)
           for c, m in zip(STATE_COLORS, STATE_MARKERS)]
fig.legend(handles, ["ground state", "1st excited", "2nd excited", "3rd excited", "4th excited",
                    "5th excited"],
           loc="upper center", ncol=6, fontsize=7.6, frameon=False,
           bbox_to_anchor=(0.5, 1.03))

for ax in axes[:, 0]:
    ax.set_ylabel(r"$|E_{\mathrm{est}}-E_{\mathrm{exact}}|$ (Ha)")
for ax in axes[-1, :]:
    ax.set_xlabel(r"base $\tau$")

fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig(_style.figure_path("fig07_noiseless.pdf"), bbox_inches="tight")
print("wrote fig07_noiseless.pdf")

for key in ORDER:
    r = results[key]["tau_noiseless"]
    f = np.asarray(r["factors"], dtype=float)
    print(f"\n{key}: tau_alias={r['tau_alias']:.4f}, M={results[key]['M']}, grids used max={int(r['n_grids'].max())}")
    for i in range(len(f)):
        print(f"   {f[i]:6.2f} tau_alias  " + " ".join(f"{e:8.1e}" for e in r["err"][i]))
