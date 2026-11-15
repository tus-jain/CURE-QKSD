import time
import zlib
import numpy as np
import matplotlib.pyplot as plt
from cure_qksd import style as _style
from cure_qksd import engine as qe
from scripts.data.run_sweeps import window_for, tau_alias_of
from cure_qksd.plot_labels import PLOT_TITLE
from cure_qksd.open_data import load_molecules
from cure_qksd.paths import DATA

_style.apply()

_, molecules = load_molecules()

SIGMA = 1e-4
TRIALS = 8
COLORS = [_style.BRICK, _style.OCHRE, _style.TEAL, _style.NAVY]

MAX_EXTRA_GRIDS = 1

CASES = [
    ("BeH2", [18, 20, 26, 34, 44, 56, 70, 85, 100], [0.5, 1.0, 2.0, 4.0]),
    ("LiH", [8, 12, 16, 20, 26, 32, 40, 50], [0.5, 1.0, 2.0, 4.0]),
    ("H2O_R1.4", [10, 15, 20, 25, 30, 40, 50, 60, 80, 100], [0.5, 1.0, 2.0, 4.0]),
]
CACHE_FILE = DATA / "m_sweep.npz"


def cache_key_for(key, factor):
    return f"{key}__{factor:g}"


def sweep_M_noisy_gs(mol, M_values, window, tau, sigma, trials, seed0):
    med = np.empty(len(M_values))
    for i, M in enumerate(M_values):
        errs = np.empty(trials)
        for t in range(trials):
            rng = np.random.default_rng(seed0 + t)
            res = qe.run_cure_qksd(mol, M=M, tau1=tau, num_states=1, sigma=sigma, rng=rng,
                                    window=window, max_extra_grids=MAX_EXTRA_GRIDS)
            Eg = res.E_unwrapped[0] if len(res.E_unwrapped) >= 1 else np.nan
            errs[t] = abs(Eg - mol.exact_energies[0]) if np.isfinite(Eg) else np.nan
        med[i] = np.nanmedian(errs)
    return med


if CACHE_FILE.exists():
    with np.load(CACHE_FILE, allow_pickle=False) as stored:
        cache = {key: stored[key].copy() for key in stored.files}
else:
    cache = {}

fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))

for ax, (key, M_values, tau_factors) in zip(axes, CASES):
    M_values = np.array(M_values)
    needed_keys = [cache_key_for(key, factor) for factor in tau_factors]
    missing = any(cache_key not in cache for cache_key in needed_keys)
    if missing:
        mol = molecules[key]
        ta = tau_alias_of(mol)
        window = window_for(mol, 1)
    t0 = time.time()
    for factor, color in zip(tau_factors, COLORS):
        cache_key = cache_key_for(key, factor)
        if cache_key in cache:
            med = cache[cache_key]
        else:
            tau = factor * ta

            seed0 = 30000 + int(1000 * factor) + zlib.crc32(key.encode()) % 1000
            med = sweep_M_noisy_gs(mol, M_values, window, tau, SIGMA, TRIALS, seed0)
            cache[cache_key] = med
            np.savez_compressed(CACHE_FILE, **cache)
        style = "--" if factor <= 1.0 else "-"
        label = rf"${factor:g}\,\tau_{{\mathrm{{alias}}}}$"
        ax.semilogy(M_values, np.maximum(med, 1e-16), style, color=color, marker="o",
                    markersize=2.8, linewidth=1.2, label=label)
    print(f"  {key}: {time.time()-t0:.1f}s")
    ax.axhline(qe.CHEMICAL_ACCURACY, color=_style.INK, linewidth=0.6, linestyle=":", alpha=0.6)
    ax.set_xlabel(r"Krylov dimension $M$")
    ax.set_title(PLOT_TITLE[key], loc="left", fontsize=7.6)
    ax.set_yscale("log")

axes[0].set_ylabel(r"median $|E_{\mathrm{est}}-E_0|$ (Ha), $\sigma=10^{-4}$")

handles, labels = axes[-1].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=7.0, frameon=False,
           bbox_to_anchor=(0.5, 1.12), title="dashed: unaliased ($\\tau\\leq\\tau_{\\mathrm{alias}}$), solid: aliased",
           title_fontsize=6.5)

fig.tight_layout()
fig.savefig(_style.figure_path("fig10_m_sweep.pdf"), bbox_inches="tight")
print("wrote fig10_m_sweep.pdf")
for key, M_values, tau_factors in CASES:
    print(f"\n{key}")
    for factor in tau_factors:
        values = cache[cache_key_for(key, factor)]
        pairs = " ".join(f"M={size}:{error:.3e}" for size, error in zip(M_values, values))
        print(f"  {factor:g} tau_alias: {pairs}")
