"""Images and winding numbers versus the number of runs (Sec. III E, Fig. 9, Appendix F).

Synthetic phases: R energies, three targets inside a window of Nb periods, the others within
20 periods outside it. N2: UQKSD phases with M = 20 and sigma = 1e-4 in windows widened to
Nb periods. Runs one and two use tau1 and tau1/phi, later runs tau_g drawn from [0.55, 0.95] tau1.
Writes data/scaling/synthetic.npz and data/scaling/n2.npz.
"""

import time
import zlib

import numpy as np

from cure_qksd import engine as qe
from cure_qksd import scaling as sc
from cure_qksd.open_data import load_molecules
from cure_qksd.paths import DATA

OUT = DATA / "scaling"
G = sc.G_MAX
N_TARGET = 3
OUTSIDE_SPAN = 20
SYNTHETIC = ([(R, Nb, 0.005) for R in (5, 10, 20) for Nb in (1, 10, 100)]
             + [(10, 10, s) for s in (0.0, 0.01)])

N2_M = 20
N2_SIGMA = 1e-4
N2_TAU1 = 3.0
N2_BUFFER = 0.10
N2_WEIGHT = 1e-3
N2_ID_TOL = 0.02
N2_TRIALS = 200
N2_CASES = [(2, Nb) for Nb in (3, 10, 30)] + [(6, 10)]


def multipliers(rng):
    return np.r_[1.0, 1 / qe.GOLDEN, rng.uniform(0.55, 0.95, G - 2)]


def synthetic_trial(R, Nb, s, rng):
    P = 2 * np.pi
    window = (0.0, Nb * P)
    n_out = R - N_TARGET
    side = rng.random(n_out) < 0.5
    off = rng.uniform(0, OUTSIDE_SPAN * P, n_out)
    outside = np.where(side, window[0] - off, window[1] + off)
    E = np.concatenate([np.sort(rng.uniform(*window, N_TARGET)), outside])
    mults = multipliers(rng)
    phase_sets = [sc.wrap(-E * t + rng.normal(0.0, s, R)) for t in mults]
    res = sc.evaluate(phase_sets, mults, window, E, E[:N_TARGET], np.arange(N_TARGET), E)
    return res, window


def synthetic(R, Nb, s):
    n = int(np.clip(2e6 / (R * R * Nb), 150, 1000))
    rng = np.random.default_rng(zlib.crc32(repr((R, Nb, s, "random")).encode()))
    d = {k: np.zeros((n, G), int) for k in ("images", "on_true", "n_peaks")}
    correct = np.zeros((n, G, N_TARGET), bool)
    g_stop = np.zeros(n, int)
    for t in range(n):
        res, window = synthetic_trial(R, Nb, s, rng)
        for g, r in enumerate(res):
            d["images"][t, g], d["on_true"][t, g], d["n_peaks"][t, g] = r["n_images"], r["n_on_true"], r["n_peaks"]
            correct[t, g] = r["correct"]
        g_stop[t] = sc.stop_index(d["n_peaks"][t], N_TARGET, window)
    d.update(correct=correct, g_stop=g_stop)
    return d


def n2_phases(mol, tau, rng):
    c = qe.add_autocorr_noise(qe.autocorrelation_trajectory(mol.H, mol.ref_state, N2_M, tau), N2_SIGMA, rng)
    S, F = qe.build_S_F(c, N2_M)
    roots, _ = qe.solve_uqksd(F, S, qe.noise_adaptive_eps(N2_SIGMA, N2_M))
    phases, _ = qe.validate_roots(roots, c, N2_M)
    return phases


def n2(mol, E_all, w_all, n_states, Nb):
    tau1 = N2_TAU1
    lo = (mol.E_ccsd - N2_BUFFER) * tau1
    window = (lo, lo + Nb * 2 * np.pi)
    states = E_all[w_all >= N2_WEIGHT * w_all.max()] * tau1
    targets = np.asarray(mol.exact_energies[:n_states]) * tau1
    rng = np.random.default_rng(zlib.crc32(repr(("N2", n_states, tau1, Nb)).encode()))
    d = {k: np.zeros((N2_TRIALS, G), int) for k in ("images", "on_true", "n_peaks", "roots")}
    d["min_E_on_true"] = np.full((N2_TRIALS, G), np.inf)
    correct = np.zeros((N2_TRIALS, G, n_states), bool)
    phase_err = np.zeros((N2_TRIALS, G, n_states))
    for t in range(N2_TRIALS):
        mults = multipliers(rng)
        phase_sets = [n2_phases(mol, tau1 * m, rng) for m in mults]
        d["roots"][t] = [len(p) for p in phase_sets]
        for g, (ph, m) in enumerate(zip(phase_sets, mults)):
            phase_err[t, g] = np.abs(sc.wrap(ph[:, None] + targets[None, :] * m)).min(0) if len(ph) else np.pi
        base = phase_sets[0]
        dist = np.abs(sc.wrap(base[:, None] + states[None, :]))
        j = dist.argmin(1)
        root_E = np.where(dist[np.arange(len(j)), j] <= N2_ID_TOL, states[j], np.nan)
        d_t = np.abs(sc.wrap(base[:, None] + targets[None, :]))
        t_root = np.where(d_t.min(0) <= N2_ID_TOL, d_t.argmin(0), -1)
        for g, r in enumerate(sc.evaluate(phase_sets, mults, window, root_E, targets, t_root, states)):
            d["images"][t, g], d["on_true"][t, g], d["n_peaks"][t, g] = r["n_images"], r["n_on_true"], r["n_peaks"]
            d["min_E_on_true"][t, g] = (r["min_E_on_true"] - targets[-1]) / tau1
            correct[t, g] = r["correct"]
    d.update(correct=correct, phase_err=phase_err)
    return d


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    store = {}
    for R, Nb, s in SYNTHETIC:
        for k, v in synthetic(R, Nb, s).items():
            store[f"R{R}_Nb{Nb}_s{s}__{k}"] = v
        print(f"synthetic R={R} Nb={Nb} s={s}: {time.time()-t0:.0f}s", flush=True)
    np.savez_compressed(OUT / "synthetic.npz", **store)
    _, molecules = load_molecules()
    mol = molecules["N2"]
    E_all, V = np.linalg.eigh(mol.H.toarray())
    w_all = np.abs(V.conj().T @ mol.ref_state) ** 2
    store = {}
    for n_states, Nb in N2_CASES:
        for k, v in n2(mol, E_all, w_all, n_states, Nb).items():
            store[f"n{n_states}_Nb{Nb}__{k}"] = v
        print(f"N2 targets={n_states} Nb={Nb}: {time.time()-t0:.0f}s", flush=True)
    np.savez_compressed(OUT / "n2.npz", **store)
