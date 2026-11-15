import sys
import time
import zlib

import numpy as np

from cure_qksd import engine as qe
from cure_qksd.open_data import load_molecules, load_sweeps, save_sweeps

RNG_SEED = 20260914
SIGMA_MAIN = 1e-4
MAX_EXTRA_GRIDS = 1
WEIGHT_CUTOFF = 1e-3
WINDOW_BUFFER = 0.10
WINDOW_RULE = "ccsd_hf_gap_buffer_0.10"

M_PER_SYSTEM = {"H6": 44, "H2O_R1.4": 67, "N2": 80, "HF": 65, "LiH": 38, "BeH2": 80}
M_NOISY = {"LiH": 20, "HF": 65, "H2O_R1.4": 50, "N2": 20, "H6": 44, "BeH2": 44}
NUM_STATES = {"H6": 3, "LiH": 4, "HF": 4, "BeH2": 6, "N2": 6, "H2O_R1.4": 5}
NOISY_STATE = {"H6": 1, "HF": 1, "N2": 1}


def window_for(mol, n_states):
    lo = mol.E_ccsd - mol.E_hf - WINDOW_BUFFER
    hi = 0.0 if n_states == 1 else mol.hf_gap + WINDOW_BUFFER
    return (lo, hi)


FACTOR_GRID = np.array([0.2, 0.3, 0.5, 0.8, 1.0, 1.3, 1.7, 2.2, 2.8, 3.5, 4.3, 5.2, 6.2, 7.3, 8.5, 10.0,
                        11.7, 13.7, 16.0, 18.7, 21.9, 25.0, 29.2, 34.1, 39.9, 46.6, 54.5, 63.7, 74.4,
                        87.0, 101.7])
GRID_END = {"H6": 1.7, "LiH": 2.2, "BeH2": 5.2, "N2": 34.1, "HF": 10.0, "H2O_R1.4": 25.0}
FACTORS_NOISELESS = {k: [float(f) for f in FACTOR_GRID if f <= end] for k, end in GRID_END.items()}
MAX_FACTOR = {"H6": 3.0}
DEFAULT_MAX_FACTOR = 7.0
FACTORS_NOISY = {
    "H6": [0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.3, 1.7, 2.2],
    "HF": [0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 13.7, 16.0, 20.0, 25.0],
    "N2": [0.5, 1, 2, 5, 10, 17, 26, 34, 43, 51, 60, 68, 77, 86, 94, 103, 120],
    "H2O_R1.4": [0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.3, 1.7, 2.2, 2.8, 3.5, 4.3, 5.2, 6.2, 7.3, 8.5, 10.0],
}


def tau_alias_of(mol):
    return np.pi / abs(mol.exact_energies[0])


def sweep_tau_noiseless(mol, M, window, factors, num_states, weight_rel_cutoff=1e-3):
    ta = tau_alias_of(mol)
    out = {"factors": factors, "tau_alias": ta, "tau": [], "E": [], "err": [],
           "n_grids": [], "cond_base": []}
    for f in factors:
        tau1 = f * ta
        res = qe.run_cure_qksd(mol, M=M, tau1=tau1, num_states=num_states, sigma=0.0,
                                window=window, max_extra_grids=MAX_EXTRA_GRIDS, weight_rel_cutoff=weight_rel_cutoff)
        got = np.full(num_states, np.nan)
        got[: len(res.E_unwrapped)] = res.E_unwrapped
        err = np.abs(np.sort(got) - mol.exact_energies[:num_states])
        out["tau"].append(tau1)
        out["E"].append(got)
        out["err"].append(err)
        out["n_grids"].append(res.n_grids_used)
        out["cond_base"].append(res.cond_base)
    for k in ("tau", "E", "err", "n_grids", "cond_base"):
        out[k] = np.array(out[k])
    return out


def sweep_M_noiseless(mol, M_values, window, tau_factor, num_states, weight_rel_cutoff=1e-3):
    ta = tau_alias_of(mol)
    tau1 = tau_factor * ta
    out = {"M": np.array(M_values), "tau1": tau1, "tau_alias": ta, "err": [], "n_grids": []}
    for M in M_values:
        res = qe.run_cure_qksd(mol, M=M, tau1=tau1, num_states=num_states, sigma=0.0,
                                window=window, max_extra_grids=MAX_EXTRA_GRIDS, weight_rel_cutoff=weight_rel_cutoff)
        got = np.full(num_states, np.nan)
        got[: len(res.E_unwrapped)] = res.E_unwrapped
        err = np.abs(np.sort(got) - mol.exact_energies[:num_states])
        out["err"].append(err)
        out["n_grids"].append(res.n_grids_used)
    out["err"] = np.array(out["err"])
    out["n_grids"] = np.array(out["n_grids"])
    return out


def sweep_tau_noisy(mol, M, window, factors, sigma, trials, seed0, weight_rel_cutoff=1e-3, state=0):
    ta = tau_alias_of(mol)
    n = state + 1
    exact = mol.exact_energies[state]
    keys = ("plain", "cure", "cure1")
    out = {"factors": factors, "tau_alias": ta, "state": state, "tau": [], "n_grids": []}
    for k in keys:
        for q in ("median", "q25", "q75"):
            out[f"{k}_{q}"] = []
        out[f"{k}_errors"] = []
    for f in factors:
        tau1 = f * ta
        errs = {k: [] for k in keys}
        grids = []
        for t in range(trials):
            rng = np.random.default_rng(seed0 + t)
            if state == 0:
                E_plain = qe.run_uqksd_principal_branch(mol, M=M, tau=tau1, sigma=sigma, rng=rng,
                                                          weight_rel_cutoff=weight_rel_cutoff)
            else:
                E_plain = qe.run_uqksd_principal_branch_states(mol, M=M, tau=tau1, num_states=n, sigma=sigma,
                                                                 rng=rng, weight_rel_cutoff=weight_rel_cutoff)[state]
            errs["plain"].append(abs(E_plain - exact) if np.isfinite(E_plain) else np.nan)
            for k, extra in (("cure", MAX_EXTRA_GRIDS), ("cure1", 0)):
                res = qe.run_cure_qksd(mol, M=M, tau1=tau1, num_states=n, sigma=sigma,
                                        rng=np.random.default_rng(seed0 + t), window=window,
                                        max_extra_grids=extra, weight_rel_cutoff=weight_rel_cutoff)
                got = np.sort(res.E_unwrapped)
                errs[k].append(abs(got[state] - exact) if len(got) > state else np.nan)
                if k == "cure":
                    grids.append(res.n_grids_used)
        out["tau"].append(tau1)
        out["n_grids"].append(grids)
        for k in keys:
            e = np.array(errs[k])
            out[f"{k}_errors"].append(e)
            if np.all(np.isnan(e)):
                out[f"{k}_median"].append(np.nan); out[f"{k}_q25"].append(np.nan); out[f"{k}_q75"].append(np.nan)
            else:
                out[f"{k}_median"].append(np.nanmedian(e))
                out[f"{k}_q25"].append(np.nanpercentile(e, 25))
                out[f"{k}_q75"].append(np.nanpercentile(e, 75))
    for k in list(out.keys()):
        if k not in ("factors", "tau_alias", "state"):
            out[k] = np.array(out[k])
    return out


def _cache_key_for(key, mol):
    """Parameters of one system's sweeps, stored with its results and checked before reuse."""
    M = M_PER_SYSTEM[key]
    M_noisy = M_NOISY[key]
    n_states = NUM_STATES[key]
    wcut = WEIGHT_CUTOFF
    max_factor = MAX_FACTOR.get(key, DEFAULT_MAX_FACTOR)
    factors_noiseless = list(FACTORS_NOISELESS[key])
    base_grid = np.array([0.5, 0.8, 1.0, 1.3, 1.7, 2.2, 2.8, 3.5, 4.3, 5.2, 6.2, 7.3, 8.5])
    if key in FACTORS_NOISY:
        factors_noisy = list(FACTORS_NOISY[key])
    else:
        fn = np.array(factors_noiseless)
        factors_noisy = (base_grid[base_grid <= max_factor][:-1] if len(fn) > 1 else fn).tolist()
    return [
        M, M_noisy, n_states, wcut, factors_noiseless, factors_noisy,
        SIGMA_MAIN, MAX_EXTRA_GRIDS, NOISY_STATE.get(key, 0), qe.ACCEPT_RULE, qe.FILL_RULE,
        "noisy_sweep",
        WINDOW_RULE,
        list(window_for(mol, n_states)), list(window_for(mol, 1)),
        qe.NOISE_MODEL, qe.THRESHOLD_RULE,
    ]


def main():
    requested = set(sys.argv[1:])

    systems, molecules = load_molecules()

    try:
        _, prior_results = load_sweeps()
    except FileNotFoundError:
        prior_results = {}

    results = {}
    for key, mol in molecules.items():
        cache_key = _cache_key_for(key, mol)

        if requested and key not in requested:
            prior = prior_results.get(key)
            if prior is None or prior.get("_cache_key") != cache_key:
                raise RuntimeError(
                    f"{key}: no stored result with the current parameters. "
                    f"Add {key} to the command line or run without arguments."
                )
            results[key] = prior
            print(f"=== {key}: stored result reused ===")
            save_sweeps(systems, results)
            continue

        M = M_PER_SYSTEM[key]
        M_noisy = M_NOISY[key]
        ta = tau_alias_of(mol)
        n_states = NUM_STATES[key]
        window_multi = window_for(mol, n_states)
        window_gs = window_for(mol, 1)
        wcut = WEIGHT_CUTOFF
        max_factor = MAX_FACTOR.get(key, DEFAULT_MAX_FACTOR)
        print(f"=== {key} (M={M}, M_noisy={M_noisy}, tau_alias={ta:.4f}, E_hf={mol.E_hf:.6f}, "
              f"E_ccsd={mol.E_ccsd:.6f}, hf_gap={mol.hf_gap:.6f}, "
              f"window_multi={window_multi}, window_gs={window_gs}, "
              f"weight_cutoff={wcut:.1e}, max_factor={max_factor:.1f}) ===")

        lo_multi, hi_multi = mol.E_hf + window_multi[0], mol.E_hf + window_multi[1]
        target = mol.exact_energies[:n_states]
        if np.any(target < lo_multi) or np.any(target > hi_multi):
            raise RuntimeError(
                f"{key}: the energy window {lo_multi, hi_multi} "
                f"does not contain the target energies {target}."
            )

        base_grid = np.array([0.5, 0.8, 1.0, 1.3, 1.7, 2.2, 2.8, 3.5, 4.3, 5.2, 6.2, 7.3, 8.5])
        factors_noiseless = np.array(FACTORS_NOISELESS[key])
        t0 = time.time()
        r_tau_noiseless = sweep_tau_noiseless(mol, M, window_multi, factors_noiseless, n_states,
                                               weight_rel_cutoff=wcut)
        print(f"  tau-sweep (noiseless): {time.time()-t0:.1f}s, "
              f"max grids used={r_tau_noiseless['n_grids'].max()}, "
              f"median err(state0)={np.nanmedian(r_tau_noiseless['err'][:,0]):.2e}")

        M_values = sorted(set([max(6, M // 3), max(8, M // 2), M, int(M * 1.5), M * 2]))
        t0 = time.time()
        r_M_noiseless = sweep_M_noiseless(mol, M_values, window_multi, tau_factor=2.0, num_states=n_states,
                                           weight_rel_cutoff=wcut)
        print(f"  M-sweep (noiseless): {time.time()-t0:.1f}s")

        if key in FACTORS_NOISY:
            factors_noisy = np.array(FACTORS_NOISY[key])
        else:
            factors_noisy = base_grid[base_grid <= max_factor][:-1] if len(factors_noiseless) > 1 else factors_noiseless
        t0 = time.time()
        state = NOISY_STATE.get(key, 0)
        window_noisy = window_for(mol, state + 1)
        r_tau_noisy = sweep_tau_noisy(mol, M_noisy, window_noisy, factors_noisy, SIGMA_MAIN, trials=8,
                                       seed0=RNG_SEED + zlib.crc32(key.encode()) % 1000, weight_rel_cutoff=wcut,
                                       state=state)
        print(f"  tau-sweep (noisy, sigma={SIGMA_MAIN:.0e}): {time.time()-t0:.1f}s")

        results[key] = {
            "M": M, "M_noisy": M_noisy, "window": window_multi, "window_gs": window_gs,
            "window_rule": WINDOW_RULE,
            "tau_alias": ta, "E_hf": mol.E_hf, "E_ccsd": mol.E_ccsd, "hf_gap": mol.hf_gap,
            "weight_cutoff": wcut, "max_factor_tested": max_factor,
            "exact_energies": mol.exact_energies[:n_states],
            "tau_alias_per_state": np.pi / np.abs(mol.exact_energies[:n_states]),
            "tau_noiseless": r_tau_noiseless, "M_noiseless": r_M_noiseless,
            "tau_noisy": r_tau_noisy, "noisy_state": NOISY_STATE.get(key, 0), "window_noisy": window_for(mol, NOISY_STATE.get(key, 0) + 1),
            "accept_level": qe.ACCEPT_LEVEL,
            "_cache_key": cache_key,
        }

        save_sweeps(systems, results)

    print("wrote data/sweeps")


if __name__ == "__main__":
    main()
