"""CURE-QKSD over many runs (Sec. III E and Appendix F).

Energies are in units of 1/tau1, so one period 2 pi/tau1 is 2 pi and run g uses the time
step tau_g/tau1. Landscapes, peaks, root assignment, and unwrapping are those of engine.py.
"""

import numpy as np

from cure_qksd import engine as qe

LEVEL = qe.ACCEPT_LEVEL
DPHI = 5e-4
G_MAX = 6


def eps_g(g):
    """Largest phase misfit of one run allowed by L_total >= LEVEL with g runs."""
    return float(np.arccos(1 - g * (1 - LEVEL)))


HALF = 1.5 * eps_g(G_MAX)


def wrap(x):
    return np.angle(np.exp(1j * x))


def candidates(base_phases, window):
    """Energies -theta_i + 2 pi k inside the window, with their root i and winding number k."""
    lo, hi = window
    k = np.arange(np.floor((lo - np.pi) / (2 * np.pi)) - 1, np.ceil((hi + np.pi) / (2 * np.pi)) + 2)
    E = -base_phases[:, None] + 2 * np.pi * k[None, :]
    root = np.broadcast_to(np.arange(len(base_phases))[:, None], E.shape)
    wind = np.broadcast_to(k[None, :], E.shape)
    inside = (E >= lo) & (E <= hi)
    order = np.argsort(E[inside])
    return E[inside][order], root[inside][order], wind[inside][order].astype(int)


def energy_grid(cand, window):
    """Points of a grid with spacing DPHI over the window that lie within HALF of a candidate,
    with NaN between separate stretches. Every accepted peak lies within eps_g < HALF of a
    candidate, so this grid finds the same peaks as the full grid."""
    lo = window[0]
    full = np.arange(window[0], window[1], DPHI)
    mark = np.zeros(len(full) + 1, int)
    a = np.clip(np.ceil((cand - HALF - lo) / DPHI).astype(int), 0, len(full))
    b = np.clip(np.floor((cand + HALF - lo) / DPHI).astype(int) + 1, 0, len(full))
    np.add.at(mark, a, 1)
    np.add.at(mark, b, -1)
    idx = np.flatnonzero(np.cumsum(mark)[:-1] > 0)
    return np.insert(full[idx], np.flatnonzero(np.diff(idx) > 1) + 1, np.nan)


def single_run_landscapes(phase_sets, mults, grid):
    finite = np.isfinite(grid)
    out = []
    for ph, t in zip(phase_sets, mults):
        L = np.full(grid.shape, -2.0)
        if len(ph):
            L[finite], _ = qe.phase_landscape([ph], [t], grid[finite])
        out.append(L)
    return np.array(out)


def accepted_peaks(grid, L):
    return qe.find_accepted_peaks(np.where(np.isfinite(grid), grid, 0.0), L, level=LEVEL)


def report(peaks, base_phases, window, num_states):
    """Reported energies as in run_cure_qksd, as (energy, base root, winding number),
    sorted by energy."""
    res = qe.CureResult()
    lo, hi = window
    if len(peaks) == 0:
        qe._fill_missing_targets(res, base_phases, set(), 1.0, lo, hi, num_states)
    else:
        mE, mi = qe.match_peaks_to_base_roots(peaks, base_phases, 1.0)
        mE, mi = np.asarray(mE), np.asarray(mi, dtype=int)
        for j in np.argsort(mE)[:num_states]:
            E, k = qe.unwrap_energy(mE[j], base_phases[mi[j]], 1.0)
            res.E_unwrapped.append(E)
            res.winding.append(k)
        if len(res.E_unwrapped) < num_states:
            qe._fill_missing_targets(res, base_phases, set(int(i) for i in mi), 1.0, lo, hi, num_states)
    out = []
    for E, k in zip(res.E_unwrapped, res.winding):
        i = int(np.argmin(np.abs(-base_phases + 2 * np.pi * k - E)))
        out.append((float(E), i, int(k)))
    return sorted(out)


def evaluate(phase_sets, mults, window, root_state_E, targets, target_roots, true_E):
    """Results after G = 1, ..., len(phase_sets) runs.

    root_state_E[i] is the energy of the state that base-run root i belongs to (NaN if none),
    target_roots[j] the base-run root of target j (-1 if none), and true_E the energies of
    the states with weight in the reference state. For each G:
      n_images       accepted candidates whose winding number is wrong for their root,
      n_on_true      those of them within 2 eps_g of a true energy,
      min_E_on_true  lowest energy among the latter (inf if none),
      n_peaks        number of accepted peaks,
      correct        per target: the j-th lowest reported energy comes from root
                     target_roots[j] with the correct winding number.
    """
    base = phase_sets[0]
    cand, root, wind = candidates(base, window)
    k_true = np.round((root_state_E[root] + base[root]) / (2 * np.pi))
    right = np.isfinite(k_true) & (wind == np.nan_to_num(k_true, nan=1e9))
    grid = energy_grid(cand, window)
    Ls = single_run_landscapes(phase_sets, mults, grid)
    tr = np.asarray(target_roots, int)
    k_target = np.where(tr >= 0, np.round((targets + base[np.maximum(tr, 0)]) / (2 * np.pi)), np.nan)
    out = []
    for G in range(1, len(phase_sets) + 1):
        L = np.where(np.isfinite(grid), Ls[:G].mean(0), -2.0)
        peaks = accepted_peaks(grid, L)
        r = eps_g(G) + 2 * DPHI
        if len(peaks):
            accepted = (np.abs(cand[:, None] - peaks[None, :]) <= r).any(1)
            image = accepted & ~right
            on_true = image & (np.abs(cand[:, None] - true_E[None, :]) <= 2 * r).any(1)
        else:
            image = on_true = np.zeros(len(cand), bool)
        correct = np.zeros(len(targets), bool)
        for j, (_, i, k) in enumerate(report(peaks, base, window, len(targets))[:len(targets)]):
            correct[j] = tr[j] >= 0 and i == tr[j] and k == k_target[j]
        out.append({"n_images": int(image.sum()), "n_on_true": int(on_true.sum()),
                    "min_E_on_true": float(cand[on_true].min()) if on_true.any() else np.inf,
                    "n_peaks": len(peaks), "correct": correct})
    return out


def stop_index(n_peaks, num_states, window):
    """Index of the run at which run_cure_qksd stops, or of the last run."""
    first = 1 if window[1] - window[0] > 2 * np.pi else 0
    for g in range(first, len(n_peaks)):
        if n_peaks[g] == num_states:
            return g
    return len(n_peaks) - 1
