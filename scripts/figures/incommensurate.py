import numpy as np
import matplotlib.pyplot as plt
from cure_qksd import style as _style

_style.apply()

E0, E1, E2 = -2.0, -1.1, -0.4
energies = [E0, E1, E2]

phi = (1.0 + np.sqrt(5.0)) / 2.0
tau1 = 2.0
tau2 = tau1 / phi

E_grid = np.linspace(-2.6, 4.6, 4000)


def landscape(tau, E_grid, energies):
    theta = [-E * tau for E in energies]
    curves = [np.cos(th + E_grid * tau) for th in theta]
    return np.maximum.reduce(curves)


L1 = landscape(tau1, E_grid, energies)
L2 = landscape(tau2, E_grid, energies)
L_total = 0.5 * (L1 + L2)

period1 = 2 * np.pi / tau1
phantom_E = E0 + period1
i_true = np.argmin(np.abs(E_grid - E0))
i_phantom = np.argmin(np.abs(E_grid - phantom_E))
print(f"period 2*pi/tau1 = {period1:.4f}")
print(f"true ground state E0 = {E0:.4f}: L1={L1[i_true]:.4f}, L2={L2[i_true]:.4f}, "
      f"L_total={L_total[i_true]:.4f}")
print(f"phantom image E0+2pi/tau1 = {phantom_E:.4f}: L1={L1[i_phantom]:.4f}, "
      f"L2={L2[i_phantom]:.4f}, L_total={L_total[i_phantom]:.4f}")

fig, (ax1, ax2, ax3) = plt.subplots(
    3, 1, figsize=(3.4, 4.6), sharex=True,
    gridspec_kw={"hspace": 0.15},
)

for ax, L, tau, lab in [(ax1, L1, tau1, r"$L_1(E)$  ($\tau_1$)"),
                         (ax2, L2, tau2, r"$L_2(E)$  ($\tau_2=\tau_1/\varphi$)")]:
    ax.plot(E_grid, L, color=_style.SLATE, linewidth=1.0)
    ax.fill_between(E_grid, -1.15, L, color=_style.SLATE, alpha=0.08)
    ax.set_ylabel(lab, fontsize=8.2)
    ax.set_ylim(-1.15, 1.3)

ax3.plot(E_grid, L_total, color=_style.INK, linewidth=1.25)
ax3.fill_between(E_grid, -1.15, L_total, color=_style.INK, alpha=0.08)
ax3.set_ylabel(r"$L_{\mathrm{total}}=\frac{1}{2}(L_1+L_2)$", fontsize=8.2)
ax3.set_ylim(-1.15, 1.3)
ax3.set_xlabel(r"energy $E$")

for ax in (ax2, ax3):
    for c, E_peak in zip([_style.NAVY, _style.TEAL, _style.OCHRE], energies):
        ax.axvline(E_peak, color=c, linestyle=":", linewidth=0.9, alpha=0.9)


for c, E_peak in zip([_style.NAVY, _style.TEAL, _style.OCHRE], energies):
    ax1.plot([E_peak, E_peak], [-1.15, 1.15], color=c, linestyle=":", linewidth=0.9, alpha=0.9)

ax1.set_ylim(-1.15, 1.95)
ax1.text(E0, 1.5, r"$E_0$", color=_style.NAVY, ha="center", va="bottom", fontsize=7.4)
ax1.text(E1, 1.5, r"$E_1$", color=_style.TEAL, ha="center", va="bottom", fontsize=7.4)
ax1.text(E2, 1.5, r"$E_2$", color=_style.OCHRE, ha="center", va="bottom", fontsize=7.4)

ax1.annotate("phantom\n(periodic image)", xy=(phantom_E, 1.0),
             xytext=(phantom_E, 1.32), va="bottom", ha="center",
             fontsize=6.4, color=_style.BRICK,
             arrowprops=dict(arrowstyle="-", color=_style.BRICK, linewidth=0.7, alpha=0.8))
ax3.set_ylim(-1.15, 1.45)
ax3.annotate("suppressed", xy=(phantom_E, L_total[i_phantom]),
             xytext=(phantom_E, 1.05), va="bottom", ha="center",
             fontsize=6.4, color=_style.BRICK,
             arrowprops=dict(arrowstyle="-", color=_style.BRICK, linewidth=0.7, alpha=0.8))

ax1.set_title(r"$\tau_1=%.3g$, $\tau_2=\tau_1/\varphi=%.3g$" % (tau1, tau2), loc="left", fontsize=8.6)

fig.tight_layout()
fig.savefig(_style.figure_path("fig06_incommensurate.pdf"), bbox_inches="tight")
print("wrote fig06_incommensurate.pdf")
