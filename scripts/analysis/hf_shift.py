import zlib
import numpy as np
from cure_qksd import engine as qe
from scripts.data.run_sweeps import (RNG_SEED, SIGMA_MAIN, MAX_EXTRA_GRIDS, M_NOISY, NOISY_STATE,
                             FACTORS_NOISY, window_for, tau_alias_of)
from cure_qksd.open_data import load_molecules

N_REAL = 8

EXTRA_FACTORS = {"HF": [40.0, 50.0, 60.0]}
WRONG = 0.1
C_VAL = 1e-3

_, molecules = load_molecules()


def exact_spectrum(mol):
    E, V = np.linalg.eigh(mol.H.toarray())
    return E, np.abs(V.conj().T @ np.asarray(mol.ref_state).ravel()) ** 2


def nearest_image(E_exact, w_exact, energy, tau):
    period = 2 * np.pi / tau
    dist = np.abs(np.mod(energy - E_exact + period / 2, period) - period / 2)
    dist[w_exact < 1e-10] = np.inf
    j = int(np.argmin(dist))
    return j, dist[j]


def shifted_states(mol, M, tau, rng, n):
    ac = qe.add_autocorr_noise(qe.autocorrelation_trajectory(mol.H, mol.ref_state, M, tau), SIGMA_MAIN, rng)
    S, F = qe.build_S_F(ac, M)
    roots, _ = qe.solve_uqksd(F, S, qe.noise_adaptive_eps(SIGMA_MAIN, M))
    phases, _ = qe.validate_roots(roots, ac, M, weight_rel_cutoff=C_VAL)
    period = 2 * np.pi / tau
    E = np.sort(mol.E_hf + np.mod(-np.asarray(phases) / tau - mol.E_hf + period / 2, period) - period / 2)
    out = np.full(n, np.nan)
    out[:min(n, len(E))] = E[:n]
    return out


for key in ["H6", "HF", "N2"]:
    mol = molecules[key]
    M, k = M_NOISY[key], NOISY_STATE[key]
    n = k + 1
    ta = tau_alias_of(mol)
    E_exact, w_exact = exact_spectrum(mol)
    exact = mol.exact_energies[:n]
    top = E_exact[w_exact >= C_VAL * w_exact.max()].max() - mol.E_hf
    seed0 = RNG_SEED + zlib.crc32(key.encode()) % 1000
    window = window_for(mol, n)
    print(f"\n{key}: M={M}, states 0..{k}, E_top - E_HF = {top:.2f} Ha, predicted onset pi/(E_top-E_HF) = "
          f"{np.pi / top:.3f} a.u. = {np.pi / top / ta:.1f} tau_alias; E_0 - E_HF = {exact[0] - mol.E_hf:.4f} Ha")
    print("   tau/ta   tau(a.u.) | shifted E0: median, wrong | shifted E1: median, wrong | CURE E0, E1 median")
    reported_first = False
    for f in list(FACTORS_NOISY[key]) + EXTRA_FACTORS.get(key, []):
        tau = f * ta
        sh, cu = [], []
        for t in range(N_REAL):
            sh.append(np.abs(shifted_states(mol, M, tau, np.random.default_rng(seed0 + t), n) - exact))
            res = qe.run_cure_qksd(mol, M=M, tau1=tau, num_states=n, sigma=SIGMA_MAIN,
                                   rng=np.random.default_rng(seed0 + t), window=window,
                                   max_extra_grids=MAX_EXTRA_GRIDS, weight_rel_cutoff=C_VAL)
            got = np.full(n, np.nan); got[:len(res.E_unwrapped)] = np.sort(res.E_unwrapped)
            cu.append(np.abs(got - exact))
        sh, cu = np.array(sh), np.array(cu)
        wrong = np.sum(~(sh < WRONG), axis=0)
        cwrong = np.sum(~(cu < WRONG), axis=0)
        print(f"   {f:6.1f}  {tau:8.3f}  | {np.nanmedian(sh[:,0]):9.2e}, {wrong[0]}/{N_REAL}      | "
              f"{np.nanmedian(sh[:,k]):9.2e}, {wrong[k]}/{N_REAL}      | {np.nanmedian(cu[:,0]):.1e}, {np.nanmedian(cu[:,k]):.1e}"
              f"  (CURE wrong {cwrong[0]}/{N_REAL}, {cwrong[k]}/{N_REAL})")
        aliased_k = tau > np.pi / abs(exact[k])
        if not reported_first and aliased_k and wrong.max() == N_REAL and cwrong.max() == 0:
            reported_first = True
            t = 0
            Es = shifted_states(mol, M, tau, np.random.default_rng(seed0 + t), n)
            for s in range(n):
                j, d = nearest_image(E_exact, w_exact, Es[s], tau)
                print(f"      realization 0 at {f:g} tau_alias: shifted 'E{s}' = {Es[s]:.4f} Ha is the image of the state "
                      f"{E_exact[j] - mol.E_hf:+.2f} Ha from E_HF (weight {w_exact[j]:.1e}); exact E{s} = {exact[s]:.4f} Ha")
