import numpy as np
from scipy.linalg import eigh, eigvalsh
import matplotlib.pyplot as plt
from cure_qksd import style as _style
from cure_qksd.conditioning import (
    SIGMA,
    add_autocorr_noise,
    autocorrelation,
    build_spectrum,
    median_noise_norm,
    principal_ground_energy,
    s_from_autocorr,
    sf_from_autocorr,
    thresholded_roots,
)

_style.apply()
GOLDEN = (1.0 + np.sqrt(5.0)) / 2.0
FAINT = "#b9c3cf"
FAINT_ORANGE = "#f2b47e"

CHECK_LEVEL = 2 * 0.9999 - 1


def cure_ground_energy(base_roots, check_roots, tau, energy_window=(-1.05, 1.05)):
    base = base_roots[(np.abs(base_roots) > 0.7) & (np.abs(base_roots) < 1.3)]
    check = check_roots[(np.abs(check_roots) > 0.7) & (np.abs(check_roots) < 1.3)]
    if len(base) == 0 or len(check) == 0:
        return np.nan
    tau_check = tau / GOLDEN
    check_phases = np.angle(check)
    accepted = []
    for phase in np.angle(base):
        k_min = int(np.ceil((energy_window[0] * tau + phase) / (2.0 * np.pi)))
        k_max = int(np.floor((energy_window[1] * tau + phase) / (2.0 * np.pi)))
        for winding in range(k_min, k_max + 1):
            energy = (-phase + 2.0 * np.pi * winding) / tau
            if np.max(np.cos(check_phases + energy * tau_check)) >= CHECK_LEVEL:
                accepted.append(energy)
    return np.min(accepted) if accepted else np.nan


def median_error_curve(energies, weights, M, basis_time, thresholds, trials, seed0):
    tau = basis_time / (M - 1)
    autocorr_clean = autocorrelation(energies, weights, M, tau)
    aliased = tau * np.max(np.abs(energies)) > np.pi
    if aliased:
        tau_check = tau / GOLDEN
        autocorr_check_clean = autocorrelation(energies, weights, M, tau_check)
    all_errors = np.empty((trials, len(thresholds)))
    for trial in range(trials):
        rng = np.random.default_rng(seed0 + trial)
        F_noisy, S_noisy = sf_from_autocorr(
            add_autocorr_noise(autocorr_clean, SIGMA, rng), M
        )
        s_evals, s_evecs = eigh(S_noisy)
        if aliased:
            F_check, S_check = sf_from_autocorr(
                add_autocorr_noise(autocorr_check_clean, SIGMA, rng), M
            )
            check_evals, check_evecs = eigh(S_check)
        for i, epsilon in enumerate(thresholds):
            base_roots = thresholded_roots(F_noisy, s_evals, s_evecs, epsilon)
            if aliased:
                check_roots = thresholded_roots(
                    F_check, check_evals, check_evecs, epsilon
                )
                estimate = cure_ground_energy(base_roots, check_roots, tau)
            else:
                estimate = principal_ground_energy(base_roots, tau)
            all_errors[trial, i] = (
                estimate - energies[0] if np.isfinite(estimate) else np.nan
            )
    return np.nanmedian(np.abs(all_errors), axis=0), np.nanmedian(all_errors, axis=0)


def count_saturation_M(energies, weights, basis_time, M_values=np.arange(4, 26)):
    """First M at which the number of exact eigenvalues of S above the median
    ||Delta S||_2 reaches its final value, with the definition and M grid of
    Fig. 2(b)."""
    counts = np.array([
        np.count_nonzero(
            eigvalsh(s_from_autocorr(
                autocorrelation(energies, weights, M - 1, basis_time / (M - 1))
            )) > median_noise_norm(M)
        )
        for M in M_values
    ])
    return int(M_values[np.flatnonzero(counts == counts[-1])[0]])


def scan_M(energies, weights, basis_time, M_values, thresholds, trials, seed0):
    """Median |error| and median signed error for every scanned threshold
    (rows: M, columns: epsilon)."""
    medians = [
        median_error_curve(
            energies, weights, M, basis_time, thresholds, trials, seed0 + 100 * M
        )
        for M in M_values
    ]
    return np.array([m[0] for m in medians]), np.array([m[1] for m in medians])


energies, weights = build_spectrum()
thresholds = np.logspace(-6, -1, 51)
trials = 31
T_values = [10.0, 20.0, 40.0]
M_min = 3
M_max = 28

fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6), sharey=True)
for ax, basis_time in zip(axes, T_values):
    M_values = np.arange(M_min, M_max + 1)
    all_medians, signed_medians = scan_M(
        energies, weights, basis_time, M_values, thresholds, trials, seed0=40000 + int(basis_time)
    )
    optimal = np.nanmin(all_medians, axis=1)

    M_grid = np.broadcast_to(M_values[:, None], all_medians.shape)
    below = signed_medians < 0
    for mask, color, zorder, label in (
        (~below, FAINT, 1, r"median $E_{\mathrm{est}}>E_0$"),
        (below, FAINT_ORANGE, 2, r"median $E_{\mathrm{est}}<E_0$"),
    ):
        ax.semilogy(M_grid[mask], all_medians[mask], ls="none", marker="o", markersize=1.8,
                    markerfacecolor=color, markeredgecolor="none", zorder=zorder,
                    label=label)
    print(f"  T_basis={basis_time:g}: {np.sum(~below & np.isfinite(all_medians))} gray, "
          f"{np.sum(below)} orange scanned points")

    M_saturation = count_saturation_M(energies, weights, basis_time)
    ax.axvline(M_saturation, color=_style.SLATE, linewidth=0.8, linestyle=":", alpha=0.85,
               zorder=0.5)
    print(f"  T_basis={basis_time:g}: Fig. 2(b) count saturates at M={M_saturation}")
    valid = np.isfinite(optimal)
    ax.semilogy(M_values[valid], optimal[valid], color=_style.NAVY, marker="o", markersize=2.6,
                zorder=3, label=r"$\epsilon_{\mathrm{optimal}}$")
    ax.set_title(rf"$T_{{\mathrm{{basis}}}}={basis_time:g}$", loc="left")
    ax.set_xlabel(r"Krylov dimension $M$")
    ax.set_xlim(M_min - 0.5, M_max + 0.5)
    ax.set_xticks([3, 5, 10, 15, 20, 25, 28])
    ax.set_ylim(1.0e-3, 10.0)
    ax.grid(True, which="both")

    tail_median = np.nanmedian(optimal[-5:])
    within_ten_percent = np.abs(optimal / tail_median - 1.0) <= 0.10
    plateau_start = M_values[-1]
    for i, M in enumerate(M_values):
        if np.all(within_ten_percent[i:]):
            plateau_start = M
            break

    big_drops = M_values[1:][optimal[1:] < 0.9 * optimal[:-1]]
    last_big_drop = int(big_drops[-1]) if len(big_drops) else None
    print(f"T_basis={basis_time:g}: M_min={M_values[0]}, "
          f"optimal(M_min)={optimal[0]:.6e}, optimal-tail-median={tail_median:.6e}, "
          f"10%-neighborhood from M={plateau_start}, last >10% drop at M={last_big_drop}, "
          f"optimal(M={M_max})={optimal[-1]:.6e}")
    print("  early-M optimal: " + ", ".join(
        f"M={M}: {optimal[np.where(M_values == M)[0][0]]:.6e}"
        for M in (3, 4, 5, 6)
    ))
    print(f"  largest median error over scanned thresholds: {np.nanmax(all_medians):.3e}")

axes[0].set_ylabel(r"median $|E_{\mathrm{est}}-E_0|$")

axes[1].legend(loc="center right", fontsize=6.8, markerscale=1.8)
fig.tight_layout()
output_name = "fig03_threshold_trap.pdf"
fig.savefig(_style.figure_path(output_name), bbox_inches="tight")
print(f"wrote {output_name}")
