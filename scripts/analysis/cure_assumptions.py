import zlib
import numpy as np
from scipy.signal import find_peaks
from cure_qksd import engine as qe
from scripts.data.run_sweeps import (RNG_SEED, SIGMA_MAIN, NOISE_TRIALS, MAX_EXTRA_GRIDS, NUM_STATES, NOISY_STATE, window_for, tau_alias_of)
from cure_qksd.open_data import load_molecules, load_sweeps

SYSTEMS = ["H6", "HF", "N2", "LiH", "H2O_R1.4", "BeH2"]
ACCEPT = qe.ACCEPT_LEVEL
NONZERO = 1e-10
DEGENERATE = 1e-8
N_REAL = NOISE_TRIALS

_, molecules = load_molecules()
_, results = load_sweeps()


def n_targets(key):
    return NUM_STATES[key]


def exact_spectrum(mol):
    """All sector eigenvalues and their squared overlaps with |phi_0>."""
    E, V = np.linalg.eigh(mol.H.toarray())
    return E, np.abs(V.conj().T @ np.asarray(mol.ref_state).ravel()) ** 2


def target_weights(mol, n):
    """Squared overlap of each of the n lowest targets with |phi_0>, summing degenerate partners."""
    E, w = np.asarray(mol.exact_energies), np.asarray(mol.exact_overlaps)
    return np.array([w[np.abs(E - e) < DEGENERATE].sum() for e in E[:n]])


def cure_landscape(mol, M, tau1, sigma, rng, c_val, window, n):
    """The steps of qe.run_cure_qksd, keeping the final landscape, the accepted
    peaks, and the guide energies that are reported."""
    E_grid = np.linspace(mol.E_hf + window[0], mol.E_hf + window[1], 4000)
    multipliers = [1.0, 1.0 / qe.GOLDEN, 1.0 / np.sqrt(3.0), 1.0 / np.pi][: 1 + MAX_EXTRA_GRIDS]
    min_grids = 2 if window[1] - window[0] > 2 * np.pi / tau1 else 1
    phase_sets, taus = [], []
    for step, mult in enumerate(multipliers):
        tau = tau1 * mult
        autocorr = qe.autocorrelation_trajectory(mol.H, mol.ref_state, M, tau)
        if sigma > 0.0:
            autocorr = qe.add_autocorr_noise(autocorr, sigma, rng)
        S, F = qe.build_S_F(autocorr, M)
        roots, _ = qe.solve_uqksd(F, S, qe.noise_adaptive_eps(sigma, M))
        phases, _ = qe.validate_roots(roots, autocorr, M, weight_rel_cutoff=c_val)
        phase_sets.append(np.asarray(phases))
        taus.append(tau)
        landscape, _ = qe.phase_landscape(phase_sets, taus, E_grid)
        peaks = qe.find_accepted_peaks(E_grid, landscape, level=ACCEPT)
        if step + 1 >= min_grids and len(peaks) == n:
            break
    reported, unwrapped = np.array([]), np.array([])
    used = set()
    if len(phase_sets[0]) and len(peaks):
        matched_E, matched_idx = qe.match_peaks_to_base_roots(peaks, phase_sets[0], tau1)
        used = set(int(i) for i in matched_idx)
        order = np.argsort(matched_E)[:n]
        reported = np.array([matched_E[j] for j in order])
        unwrapped = np.array([qe.unwrap_energy(matched_E[j], phase_sets[0][matched_idx[j]], tau1)[0]
                              for j in order])
    if len(phase_sets[0]) and unwrapped.size < n:
        fill = qe.CureResult()
        fill.E_unwrapped = list(unwrapped)
        qe._fill_missing_targets(fill, phase_sets[0], used, tau1, E_grid[0], E_grid[-1], n)
        unwrapped = np.array(fill.E_unwrapped)
        reported = np.concatenate([reported, unwrapped[reported.size:]])
    return E_grid, landscape, taus, peaks, reported, unwrapped


def is_image(energy, E_pop, taus):
    d_true = np.min(np.abs(energy - E_pop))
    d_image = np.inf
    for tau in taus:
        period = 2 * np.pi / tau
        offset = np.mod(energy - E_pop + period / 2, period) - period / 2
        k = np.round((energy - E_pop - offset) / period)
        if np.any(k != 0):
            d_image = min(d_image, np.min(np.abs(offset[k != 0])))
    return d_image < 0.5 * d_true


print("1. Pre-solution window")
ccsd_err, upper_margins = [], []
for key in SYSTEMS:
    mol = molecules[key]
    n = n_targets(key)
    E0 = mol.exact_energies[0]
    lo, hi = window_for(mol, n)
    ccsd_err.append(abs(mol.E_ccsd - E0))
    line = (f"   {key:9s} E_CCSD - E_0 = {1e3 * (mol.E_ccsd - E0):+6.1f} mHa, "
            f"lower edge {1e3 * (E0 - mol.E_hf - lo):5.1f} mHa below E_0")
    if n > 1:
        margin = hi - (mol.exact_energies[n - 1] - mol.E_hf)
        upper_margins.append(margin)
        line += f"; highest target {1e3 * margin:5.1f} mHa below the upper edge of W_multi"
    print(line)
print(f"   largest |E_CCSD - E_0| = {1e3 * max(ccsd_err):.1f} mHa; "
      f"smallest margin below the upper edge of W_multi = {1e3 * min(upper_margins):.1f} mHa")


print("\n2. Golden-ratio time steps")
hits = []
for k1 in range(1, 401):
    x = np.linspace(k1 - 0.05, k1 + 0.05, 20001)
    L_one_state = 0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * x / qe.GOLDEN))
    if L_one_state.max() >= ACCEPT:
        hits.append(x[np.argmax(L_one_state)])
first = hits[0]
print(f"   the images of one state reach L_total >= {ACCEPT} at {np.round(hits[:4], 2)} periods 2 pi/tau_1")
for key in SYSTEMS:
    mol, r = molecules[key], results[key]
    E, w = exact_spectrum(mol)
    span = E[w > NONZERO].max() - E[w > NONZERO].min()
    tau_max = max(r["tau_noiseless"]["tau"].max(), r["tau_noisy"]["tau"].max())
    reach = first * 2 * np.pi / tau_max
    print(f"   {key:9s} largest tau_1 in Figs. 7-8 = {tau_max / tau_alias_of(mol):4.1f} tau_alias: "
          f"{first:.2f} periods = {reach:6.1f} Ha; energy range of states with nonzero weight = {span:5.1f} Ha"
          f" ({'farther' if reach > span else 'NOT farther'})")


print("\n3. Second run at tau_2 = tau_1/phi: same M, same measurements per value")
print(f"   total evolution time of both runs / base run alone = 1 + 1/phi = {1 + 1 / qe.GOLDEN:.3f}")

print("\n4. Squared overlaps with |phi_0> for the target sets of Fig. 7")
print("   system    targets  smallest  others   ratio | ground state alone: (1-p0)/p0")
for key in SYSTEMS:
    mol = molecules[key]
    n = n_targets(key)
    w = target_weights(mol, n)
    p_min, others = w.min(), 1.0 - w.sum()
    p0 = target_weights(mol, 1)[0]
    print(f"   {key:9s} {n:6d}  {p_min:8.4f}  {others:6.3f}  {others / p_min:6.2f} | {(1 - p0) / p0:5.2f}")


print(f"\n5. Image peaks in the calculations of Figs. 7 and 8 (acceptance level {ACCEPT})")
for label, sigma, n_real in [("Fig. 8", SIGMA_MAIN, N_REAL), ("Fig. 7", 0.0, 1)]:
    for key in SYSTEMS:
        mol, r = molecules[key], results[key]
        E, w = exact_spectrum(mol)
        E_pop = E[w > NONZERO]
        k_state = NOISY_STATE.get(key, 0) if sigma > 0 else 0
        n = k_state + 1 if sigma > 0 else n_targets(key)
        sweep = r["tau_noisy"] if sigma > 0 else r["tau_noiseless"]
        M = r["M_noisy"] if sigma > 0 else r["M"]
        window = window_for(mol, n)
        seed0 = RNG_SEED + zlib.crc32(key.encode()) % 1000
        runs = accepted = 0
        reported_at, highest, deviation, deviation_at = [], -np.inf, 0.0, None
        for i, f in enumerate(sweep["factors"]):
            tau1 = f * tau_alias_of(mol)
            errs = []
            for t in range(n_real):
                rng = np.random.default_rng(seed0 + t) if sigma > 0 else None
                E_grid, L_total, taus, peaks, reported, unwrapped = cure_landscape(
                    mol, M, tau1, sigma, rng, r["weight_cutoff"], window, n)
                runs += 1
                maxima, _ = find_peaks(L_total)
                for j in maxima:
                    if is_image(E_grid[j], E_pop, taus):
                        highest = max(highest, L_total[j])
                accepted += sum(is_image(p, E_pop, taus) for p in peaks)
                reported_at += [f for p in reported if is_image(p, E_pop, taus)]
                if sigma > 0:
                    srt = np.sort(unwrapped)
                    errs.append(abs(srt[k_state] - mol.exact_energies[k_state]) if srt.size > k_state else np.nan)
                else:
                    got = np.full(n, np.nan)
                    got[:unwrapped.size] = unwrapped
                    errs = np.abs(np.sort(got) - mol.exact_energies[:n])
            stored = sweep["cure_median"][i] if sigma > 0 else sweep["err"][i]
            mine = np.nanmedian(errs) if sigma > 0 else errs
            d = np.nanmax(np.abs(np.atleast_1d(mine) - np.atleast_1d(stored)))
            if d > deviation:
                deviation, deviation_at = d, f
        top = f"{highest:.3f}" if np.isfinite(highest) else "none"
        where = f" (at {', '.join(f'{f:g}' for f in reported_at)} tau_alias)" if reported_at else ""
        dev = f"{deviation:.1e} Ha" + (f" at {deviation_at:g} tau_alias" if deviation > 1e-9 else "")
        print(f"   {label} {key:9s} {runs:3d} calculations: accepted image peaks {accepted}, "
              f"reported as a target {len(reported_at)}{where}; highest image peak L_total = {top}; "
              f"largest deviation from stored errors {dev}")


print("\n6. Base run alone vs. CURE-QKSD (second run allowed): calculations where a target error exceeds 1e-2 Ha")
for label, sigma in [("Fig. 7", 0.0), ("Fig. 8", SIGMA_MAIN)]:
    for key in SYSTEMS:
        mol, r = molecules[key], results[key]
        if sigma > 0:
            k_state = NOISY_STATE.get(key, 0)
            tn = r["tau_noisy"]
            base_wrong = [f for f, e in zip(tn["factors"], tn["cure1_errors"]) for x in e if not x < 1e-2]
            both_wrong = [f for f, e in zip(tn["factors"], tn["cure_errors"]) for x in e if not x < 1e-2]
            base_only = sorted(set(base_wrong) - set(both_wrong))
            n_trials = tn["cure_errors"].shape[1]
            print(f"   {label} {key:9s} (state {k_state}): base run alone wrong where CURE-QKSD is right at "
                  f"{base_only if base_only else 'no tau'}"
                  + (f"; wrong realizations there: " + ", ".join(f"{f:g}: {base_wrong.count(f)}/{n_trials}" for f in base_only) if base_only else ""))
            continue
        n = n_targets(key)
        window = window_for(mol, n)
        bad = []
        for i, f in enumerate(r["tau_noiseless"]["factors"]):
            res = qe.run_cure_qksd(mol, M=r["M"], tau1=f * tau_alias_of(mol), num_states=n, sigma=0.0,
                                   window=window, max_extra_grids=0, weight_rel_cutoff=r["weight_cutoff"])
            got = np.full(n, np.nan)
            got[:len(res.E_unwrapped)] = np.sort(res.E_unwrapped)
            e_base = np.abs(got - mol.exact_energies[:n])
            e_two = r["tau_noiseless"]["err"][i]
            if np.any(~(e_base < 1e-2) & (e_two < 1e-2)):
                bad.append((f, float(np.nanmax(e_base))))
        print(f"   {label} {key:9s}: base run alone wrong where CURE-QKSD is right at "
              + (", ".join(f"{f:g} tau_alias (largest error {e:.2f} Ha)" for f, e in bad) if bad else "no tau"))
