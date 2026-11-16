import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt
from cure_qksd import style as _style
from cure_qksd.conditioning import (
    SIGMA,
    T_BASIS,
    add_autocorr_noise,
    autocorrelation,
    build_spectrum,
    median_noise_norm,
    principal_ground_energy,
    sf_from_autocorr,
    thresholded_roots,
)

_style.apply()


def errors_vs_threshold(energies, weights, M, thresholds, trials, seed0):
    """Return signed errors E_est - E_0 versus the threshold epsilon at one M."""
    tau = T_BASIS / (M - 1)
    autocorr_clean = autocorrelation(energies, weights, M, tau)
    errors = np.empty((trials, len(thresholds)))
    for trial in range(trials):
        rng = np.random.default_rng(seed0 + trial)
        F_noisy, S_noisy = sf_from_autocorr(
            add_autocorr_noise(autocorr_clean, SIGMA, rng), M
        )
        s_evals, s_evecs = eigh(S_noisy)
        for i, epsilon in enumerate(thresholds):
            roots = thresholded_roots(F_noisy, s_evals, s_evecs, epsilon)
            estimate = principal_ground_energy(roots, tau)
            errors[trial, i] = (
                estimate - energies[0] if np.isfinite(estimate) else np.nan
            )
    return errors


energies, weights = build_spectrum()
relative_band = (0.5, 2.0)

fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.4, 3.2))

M_values = np.array([4, 5, 6, 8, 10, 12, 16, 20, 24, 30, 36, 48, 60, 80, 100])
counts = np.empty(len(M_values))
for i, M in enumerate(M_values):
    tau = T_BASIS / (M - 1)
    autocorr_clean = autocorrelation(energies, weights, M, tau)
    noise_norm = median_noise_norm(M)
    rng = np.random.default_rng(1000 + M)
    trial_counts = np.empty(100)
    for trial in range(len(trial_counts)):
        _, S_noisy = sf_from_autocorr(
            add_autocorr_noise(autocorr_clean, SIGMA, rng), M
        )
        s_evals = eigh(S_noisy, eigvals_only=True)
        trial_counts[trial] = np.count_nonzero(
            (s_evals > relative_band[0] * noise_norm)
            & (s_evals < relative_band[1] * noise_norm)
        )
    counts[i] = np.median(trial_counts)

axa.plot(M_values, counts, color=_style.BRICK, marker="o", markersize=3.2, linewidth=1.3)
for M, color in zip([6, 36, 80], [_style.NAVY, _style.OCHRE, _style.BRICK]):
    axa.axvline(M, color=color, linewidth=0.8, linestyle=":", alpha=0.85)
axa.set_xlabel(r"Krylov dimension $M$")
axa.set_ylabel("number of eigenvalues in\n" + r"$[0.5,2]\times$ median $\|\Delta\mathbf{S}\|_2$")
axa.set_title(r"(a) eigenvalues of $\widetilde{\mathbf{S}}$ near median $\|\Delta\mathbf{S}\|_2$, $T_{\mathrm{basis}}=10$",
              loc="left", fontsize=7.6)
axa.grid(True, alpha=0.3)

thresholds = np.logspace(-5, -1, 141)
M_demo = [6, 36, 80]
colors = [_style.NAVY, _style.OCHRE, _style.BRICK]
styles = ["-", "--", "-."]
n_trials = 100
error_level = 1.0e-2

for M, color, style in zip(M_demo, colors, styles):
    seed0 = 20000 + 100 * M
    signed = errors_vs_threshold(energies, weights, M, thresholds, n_trials, seed0)
    median = np.nanmedian(np.abs(signed), axis=0)
    axb.loglog(thresholds, np.maximum(median, 1e-16), color=color, linestyle=style,
               linewidth=1.55, label=rf"$M={M}$")

    noise_norm = median_noise_norm(M)
    best_idx = int(np.nanargmin(median))
    print(f"M={M:3d}: near-norm count={counts[np.where(M_values == M)[0][0]]:.0f}, "
          f"median ||Delta S||_2={noise_norm:.3e}, scan range {np.nanmin(median):.3e}"
          f"..{np.nanmax(median):.3e} (ratio {np.nanmax(median) / np.nanmin(median):.2f}), "
          f"best at epsilon={thresholds[best_idx]:.3e}, error at 1e-1={median[-1]:.3e}")
    good = median < error_level
    if not good[0] and good.any():
        first = int(np.argmax(good))
        print(f"       error<{error_level:g} for every epsilon>={thresholds[first]:.3e} "
              f"({thresholds[first] / noise_norm:.2f}x median): {bool(good[first:].all())}; "
              f"next lower epsilon {thresholds[first - 1]:.3e} "
              f"({100 * (1 - thresholds[first - 1] / thresholds[first]):.1f}% lower) "
              f"gives {median[first - 1]:.3g}")
    probes = np.array([0.5, 1.0, 2.0]) * noise_norm
    probe = errors_vs_threshold(energies, weights, M, probes, n_trials, seed0)
    for factor, column in zip([0.5, 1.0, 2.0], probe.T):
        failed = np.abs(column) > error_level
        print(f"       epsilon={factor:g}x median: median error {np.nanmedian(np.abs(column)):.3g}, "
              f"{int(failed.sum())}/{n_trials} realizations above {error_level:g}"
              + (f" (all below E_0: {bool(np.all(column[failed] < 0))})" if failed.any() else ""))

axb.set_xlabel(r"threshold $\epsilon_{\mathrm{thresh}}$")
axb.set_ylabel(r"median $|E_{\mathrm{est}}-E_0|$")
axb.set_title(r"(b) threshold scan for three values of $M$, $T_{\mathrm{basis}}=10$", loc="left", fontsize=7.6)
axb.legend(fontsize=6.4, loc="upper right", bbox_to_anchor=(0.99, 0.99))
axb.set_xlim(thresholds[0], thresholds[-1])
axb.grid(True, which="both", alpha=0.25)

fig.tight_layout()
output_path = _style.figure_path("fig04_threshold_scan.pdf")
fig.savefig(output_path, bbox_inches="tight")
print("wrote fig04_threshold_scan.pdf")
print(f"shared model: rank={len(energies)}, T_basis={T_BASIS:g}, sigma={SIGMA:.1e}")
for M, count in zip(M_values, counts):
    print(f"M={M:3d}: median noisy-eigenvalue count in "
          f"[{relative_band[0]}, {relative_band[1]}] x median ||Delta S||_2 = {count:.0f}")
