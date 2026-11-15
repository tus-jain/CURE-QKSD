import numpy as np
from scipy.linalg import eigvalsh
import matplotlib.pyplot as plt
from cure_qksd import style as _style
from cure_qksd.conditioning import (
    autocorrelation,
    build_spectrum,
    median_noise_norm,
    s_from_autocorr,
)

_style.apply()


def overlap_eigenvalues(energies, weights, M, basis_time):
    tau = basis_time / (M - 1)
    S = s_from_autocorr(autocorrelation(energies, weights, M - 1, tau))
    return np.sort(eigvalsh(S))[::-1]


def signal_surviving_rank(energies, weights, M_values, basis_time):
    ranks = np.empty(len(M_values))
    for i, M in enumerate(M_values):
        ranks[i] = np.count_nonzero(
            overlap_eigenvalues(energies, weights, M, basis_time) > median_noise_norm(M)
        )
    return ranks


energies, weights = build_spectrum()
panel_a_time = 20.0

M_a = np.arange(2, 26)
lambda_1 = np.empty(len(M_a))
lambda_M = np.empty(len(M_a))
for i, M in enumerate(M_a):
    eigenvalues = overlap_eigenvalues(energies, weights, M, panel_a_time)
    lambda_1[i] = eigenvalues[0]
    lambda_M[i] = max(eigenvalues[-1], 0.0)
observed_a = np.array([median_noise_norm(M) for M in M_a])

T_values = [10.0, 20.0, 40.0]
M_b = np.arange(4, 26)
ranks = {
    T: signal_surviving_rank(energies, weights, M_b, T)
    for T in T_values
}

fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.0, 2.75))
positive = lambda_M > 0.0
axa.semilogy(M_a, lambda_1, color=_style.NAVY, label=r"$\lambda_1(\mathbf{S})$")
axa.semilogy(M_a[positive], lambda_M[positive], color=_style.BRICK,
             label=r"$\lambda_M(\mathbf{S})$")
axa.semilogy(M_a, observed_a, color=_style.OCHRE, linewidth=1.0,
             label=r"median $\|\Delta\mathbf{S}\|_2$")
axa.set_xlabel(r"Krylov dimension $M$  (fixed $T_{\mathrm{basis}}$, $\tau=T_{\mathrm{basis}}/(M-1)$)")
axa.set_ylabel(r"eigenvalue of $\mathbf{S}$")
axa.legend(loc="lower left", fontsize=6.6)
axa.set_title(r"(a) $T_{\mathrm{basis}}=20$, $\sigma=10^{-4}$", loc="left")
axa.set_xlim(2, 25)

for T, color in zip(T_values, [_style.NAVY, _style.TEAL, _style.OCHRE]):
    axb.plot(M_b, ranks[T], color=color, label=rf"$T_{{\mathrm{{basis}}}}={T:g}$")
axb.set_xlabel(r"Krylov dimension $M$")
axb.set_ylabel("number of eigenvalues of " + r"$\mathbf{S}$" + "\n"
               + r"above median $\|\Delta\mathbf{S}\|_2$")
axb.legend(loc="upper left", fontsize=6.8)
axb.set_title(r"(b)", loc="left")
axb.set_xlim(4, 25)
axb.set_ylim(3, 18)

fig.tight_layout()
output_path = _style.figure_path("fig02_conditioning.pdf")
fig.savefig(output_path, bbox_inches="tight")
print("wrote fig02_conditioning.pdf")
w_ground, w_next7 = weights[0], weights[1:8].sum()
print(f"reference weights: ground={w_ground:.4f}, next seven={w_next7:.4f}, "
      f"remaining {len(weights) - 8} levels={1.0 - w_ground - w_next7:.4f}")
for M in [7, 17]:
    i = int(np.where(M_a == M)[0][0])
    condition = np.inf if lambda_M[i] == 0.0 else lambda_1[i] / lambda_M[i]
    print(f"M={M:3d}: lambda_1={lambda_1[i]:.3e}, lambda_M={lambda_M[i]:.3e}, "
          f"kappa={condition:.3e}, median_noise={observed_a[i]:.3e}")
below = M_a[lambda_M < observed_a]
print(f"T_basis={panel_a_time:g}: lambda_M below median ||Delta S||_2 from M={below[0]}")
for T in T_values:
    final_rank = int(ranks[T][-1])
    first_saturated = int(M_b[np.flatnonzero(ranks[T] == final_rank)[0]])
    print(f"T_basis={T:g}: signal-surviving rank saturates at {final_rank} "
          f"from M={first_saturated}")
