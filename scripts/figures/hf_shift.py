import zlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from cure_qksd import engine as qe
from cure_qksd import style as _style
from scripts.data.run_sweeps import RNG_SEED, SIGMA_MAIN, M_NOISY, tau_alias_of
from cure_qksd.open_data import load_molecules

_style.apply()

KEY = "H6"
FACTORS = [1.3, 2.2]
C_VAL = 1e-3

_, molecules = load_molecules()
mol = molecules[KEY]
M = M_NOISY[KEY]
seed = RNG_SEED + zlib.crc32(KEY.encode()) % 1000
E_ex, V = np.linalg.eigh(mol.H.toarray())
w_ex = np.abs(V.conj().T @ np.asarray(mol.ref_state).ravel()) ** 2
E_ref = E_ex[w_ex > 1e-10]
E0, E1 = E_ref[0], E_ref[1]

fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1), gridspec_kw={"wspace": 0.22})
for ax, f, lab in zip(axes, FACTORS, "ab"):
    tau = f * tau_alias_of(mol)
    P = 2 * np.pi / tau
    ac = qe.add_autocorr_noise(qe.autocorrelation_trajectory(mol.H, mol.ref_state, M, tau), SIGMA_MAIN,
                               np.random.default_rng(seed))
    S, F = qe.build_S_F(ac, M)
    roots, _ = qe.solve_uqksd(F, S, qe.noise_adaptive_eps(SIGMA_MAIN, M))
    phases, _ = qe.validate_roots(roots, ac, M, weight_rel_cutoff=C_VAL)
    Ea = -np.asarray(phases) / tau

    for e in E_ref:
        ax.axhline(e, color=_style.SLATE, lw=0.3, alpha=0.35)
    for e in (E0, E1):
        ax.axhline(e, color=_style.OCHRE, lw=0.8, ls="--")
    for a in Ea:
        for n in range(int(np.floor((2 * E0 - P - a) / P)), int(np.ceil((P - a) / P)) + 1):
            v = a + n * P
            ax.plot([v + P / 2, v - P / 2], [v, v], color=_style.NAVY, lw=0.9, solid_capstyle="butt")
    lo = mol.E_hf - P / 2
    at_hf = np.sort(lo + np.mod(Ea - lo, P))
    ax.axvline(mol.E_hf, color=_style.INK, lw=0.6, ls=":")
    ax.text(mol.E_hf, P / 2, r" $E_{\mathrm{HF}}$", va="top", ha="left", fontsize=7.4)
    ax.plot([mol.E_hf] * 2, at_hf[:2], "o", ms=3.2, mfc="white", mec=_style.BRICK, mew=0.9, zorder=5)
    ax.set_xlim(0, 2 * E0)
    ax.set_ylim(2 * E0 - P / 2, P / 2)
    ax.set_xlabel(r"shift $E_s$ (Ha)")
    ax.set_title(rf"({lab}) $\tau={f:g}\,\tau_{{\mathrm{{alias}}}}$, $2\pi/\tau={P:.2f}$ Ha", loc="left", fontsize=8.2)

    phys = np.sort(E0 - 0.05 + np.mod(Ea - (E0 - 0.05), P))
    Emin, Emax = phys[0], phys[-1]
    a_lo, a_hi = Emax - P / 2, Emin + P / 2
    below = at_hf[at_hf < E0 - qe.CHEMICAL_ACCURACY]
    print(f"{KEY} {f} tau_alias: tau={tau:.3f}, 2pi/tau={P:.3f} Ha, {len(Ea)} validated roots")
    print(f"  E_min={Emin:.4f} (E_0={E0:.4f}), E_max={Emax:.4f} ({Emax - mol.E_hf:.3f} above E_HF), "
          f"E_max-E_min={Emax - Emin:.3f} Ha")
    print(f"  all validated energies correct for E_s in ({a_lo:.3f}, {a_hi:.3f}], width {a_hi - a_lo:.3f} Ha; "
          f"E_HF={mol.E_hf:.4f} inside: {a_lo < mol.E_hf <= a_hi}")
    print(f"  shift by E_HF: reported E_0, E_1 = {at_hf[0]:.4f}, {at_hf[1]:.4f} "
          f"(errors {abs(at_hf[0] - E0):.1e}, {abs(at_hf[1] - E1):.1e} Ha)")
    print(f"  images below E_0 at E_s=E_HF: {len(below)}, of roots "
          f"{np.round(np.sort(below + P) - mol.E_hf, 3)} Ha above E_HF")
axes[0].set_ylabel("returned energy (Ha)")
handles = [Line2D([], [], color=_style.NAVY, lw=0.9),
           Line2D([], [], color=_style.SLATE, lw=0.8, alpha=0.6),
           Line2D([], [], color=_style.OCHRE, lw=0.8, ls="--"),
           Line2D([], [], ls="none", marker="o", ms=3.2, mfc="white", mec=_style.BRICK, mew=0.9)]
labels = ["returned energy", "exact energies", r"exact $E_0$, $E_1$",
          r"reported $E_0$, $E_1$"]
axes[0].legend(handles, labels, loc="lower left", fontsize=6.8)
fig.savefig(_style.figure_path("fig10_hf_shift.pdf"), bbox_inches="tight")
