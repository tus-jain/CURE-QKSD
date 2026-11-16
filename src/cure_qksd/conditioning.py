import numpy as np
from scipy.linalg import eig, eigvalsh, toeplitz

SPECTRUM_SIZE = 48
SIGMA = 1.0e-4
T_BASIS = 10.0


def build_spectrum():
    grid = np.linspace(0.0, 1.0, SPECTRUM_SIZE)
    energies = -1.0 + 2.0 * grid ** 2
    ell = np.arange(SPECTRUM_SIZE)
    enhancement = 1.0 + SPECTRUM_SIZE / (ell + 1.0) ** 2
    weights = enhancement * np.exp(-(energies - energies[0]) / 0.5)
    weights /= np.sum(weights)
    return energies, weights


def autocorrelation(energies, weights, max_lag, tau):
    """c(k) for k=0,...,max_lag."""
    k = np.arange(max_lag + 1)
    return np.sum(
        weights[:, None] * np.exp(-1j * energies[:, None] * tau * k[None, :]),
        axis=0,
    )


def s_from_autocorr(autocorr):
    """Overlap matrix from c(0),...,c(M-1)."""
    return toeplitz(np.conj(autocorr), autocorr)


def sf_from_autocorr(autocorr, M):
    """UQKSD F and S from c(0),...,c(M)."""
    S = s_from_autocorr(autocorr[:M])
    F_row = autocorr[1:M + 1]
    F_col = np.zeros(M, dtype=complex)
    F_col[0] = F_row[0]
    if M > 1:
        F_col[1] = autocorr[0]
    if M > 2:
        F_col[2:] = np.conj(F_row[:-2])
    return toeplitz(F_col, F_row), S


def add_autocorr_noise(autocorr, sigma, rng):
    noisy = autocorr.copy()
    n = len(noisy) - 1
    if n:
        noisy[1:] += (
            rng.normal(0.0, sigma, n) + 1j * rng.normal(0.0, sigma, n)
        ) / np.sqrt(2.0)
    return noisy


def median_noise_norm(M, sigma=SIGMA, trials=100, seed0=9000):
    rng = np.random.default_rng(seed0 + M)
    norms = np.empty(trials)
    for trial in range(trials):
        noise = add_autocorr_noise(np.zeros(M, dtype=complex), sigma, rng)
        norms[trial] = np.max(np.abs(eigvalsh(s_from_autocorr(noise))))
    return np.median(norms)


def thresholded_roots(F, s_evals, s_evecs, threshold):
    """Solve the canonically orthogonalized UQKSD problem above threshold."""
    keep = s_evals > threshold
    if not np.any(keep):
        return np.array([], dtype=complex)
    V = s_evecs[:, keep]
    s_kept = s_evals[keep]
    reduced_F = V.conj().T @ F @ V
    inv_sqrt = 1.0 / np.sqrt(s_kept)
    standard_F = reduced_F * inv_sqrt[:, None] * inv_sqrt[None, :]
    return eig(standard_F, right=False)


def principal_ground_energy(roots, tau):
    near_unit = roots[(np.abs(roots) > 0.7) & (np.abs(roots) < 1.3)]
    if len(near_unit) == 0:
        return np.nan
    return np.min(-np.angle(near_unit) / tau)
