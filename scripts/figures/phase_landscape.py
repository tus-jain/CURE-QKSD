import numpy as np
import matplotlib.pyplot as plt
from cure_qksd import style as _style

_style.apply()

E0, E1, E2 = -2.0, -1.1, -0.4
E_HF = -1.7
tau = 2.0

theta = [-E0 * tau, -E1 * tau, -E2 * tau]
E_grid = np.linspace(-2.6, 0.6, 2000)
curves = [np.cos(th + E_grid * tau) for th in theta]
landscape = np.maximum.reduce(curves)

fig, (top, bot) = plt.subplots(
    2, 1, figsize=(3.4, 3.6), sharex=True,
    gridspec_kw={"height_ratios": [1.05, 1], "hspace": 0.12},
)

colors = [_style.NAVY, _style.TEAL, _style.OCHRE]
labels = [r"$\cos(\theta_0 + E\tau)$", r"$\cos(\theta_1 + E\tau)$", r"$\cos(\theta_2 + E\tau)$"]
styles = ["-", "--", "-."]
for c, ls, curve, lab in zip(colors, styles, curves, labels):
    top.plot(E_grid, curve, color=c, linestyle=ls, linewidth=1.15, label=lab)

top.axvline(E_HF, color=_style.INK, linewidth=0.8, alpha=0.6)
top.text(E_HF - 0.06, 1.28, r"$E_{\mathrm{HF}}$", ha="right", va="top", fontsize=7.6)
top.set_ylabel("phase-consistency curves")
top.set_ylim(-1.15, 1.35)
top.legend(loc="upper right", fontsize=6.6, ncol=1)
top.set_title(r"single time step $\tau$", loc="left")

bot.plot(E_grid, landscape, color=_style.INK, linewidth=1.3)
bot.fill_between(E_grid, -1.15, landscape, color=_style.INK, alpha=0.06)
peak_E = [E0, E1, E2]
peak_labels = [r"$E_0$", r"$E_1$", r"$E_2$"]
for c, E_peak in zip(colors, peak_E):
    bot.plot([E_peak], [1.0], marker="o", markersize=3.3, color=c, zorder=5)
    bot.axvline(E_peak, ymin=0, ymax=0.93, color=c, linestyle=":", linewidth=0.9, alpha=0.85)
bot.axvline(E_HF, color=_style.INK, linewidth=0.8, alpha=0.6)
bot.set_ylabel(r"$L_1(E)$")
bot.set_xlabel(r"energy $E$")
bot.set_ylim(-1.15, 1.35)

bot.set_xticks(peak_E)
bot.set_xticklabels(peak_labels)
for tick_label, c in zip(bot.get_xticklabels(), colors):
    tick_label.set_color(c)
    tick_label.set_fontsize(9.0)

fig.tight_layout()
fig.savefig(_style.figure_path("fig05_phase_landscape.pdf"), bbox_inches="tight")
print("wrote fig05_phase_landscape.pdf")
