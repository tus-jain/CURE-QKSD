"""Recompute the Ritz-weight counts quoted in Appendix E."""

import numpy as np
from scipy.linalg import eig, eigh

from cure_qksd import engine as qe
from cure_qksd.open_data import load_molecules, load_sweeps

ORDER = ["H6", "HF", "N2", "LiH", "H2O_R1.4", "BeH2"]


def exact_measure(molecule):
    """Return all sector energies, eigenvectors, and reference amplitudes."""
    matrix = molecule.H.tocoo()
    if np.all(matrix.row == matrix.col):
        energies = np.asarray(molecule.H.diagonal()).real
        vectors = np.eye(len(energies))
    else:
        energies, vectors = eigh(molecule.H.toarray())
    amplitudes = vectors.conj().T @ molecule.ref_state
    accessible = np.abs(amplitudes) ** 2 > 1e-10
    return energies[accessible], vectors[:, accessible], amplitudes[accessible]


def validated_ritz_vectors(molecule, size, tau):
    """Return validated base-run Ritz coefficients and exact-state weights."""
    autocorrelation = qe.autocorrelation_trajectory(molecule.H, molecule.ref_state, size, tau)
    overlap, propagator = qe.build_S_F(autocorrelation, size)
    overlap_values, overlap_vectors = eigh(overlap)
    keep = overlap_values > qe.noise_adaptive_eps(0.0, size)
    basis = overlap_vectors[:, keep]
    inverse = 1.0 / np.sqrt(overlap_values[keep])
    reduced = basis.conj().T @ propagator @ basis
    reduced = reduced * inverse[:, None] * inverse[None, :]
    roots, vectors = eig(reduced)
    coefficients = basis @ (inverse[:, None] * vectors)
    near_circle = np.abs(np.abs(roots) - 1.0) < 0.1
    roots = roots[near_circle]
    coefficients = coefficients[:, near_circle]
    vandermonde = np.vander(roots, N=size, increasing=True).T
    fitted, *_ = np.linalg.lstsq(vandermonde, autocorrelation[:size], rcond=None)
    relative = np.abs(fitted) / np.max(np.abs(fitted))
    roots = roots[relative > 1e-3]
    coefficients = coefficients[:, relative > 1e-3]
    energies, _, amplitudes = exact_measure(molecule)
    time_indices = np.arange(size)
    ritz_amplitudes = amplitudes[:, None] * (
        np.exp(-1j * np.outer(energies * tau, time_indices)) @ coefficients
    )
    norms = np.einsum("ik,ij,jk->k", coefficients.conj(), overlap, coefficients).real
    weights = np.abs(ritz_amplitudes) ** 2 / norms[None, :]
    return np.angle(roots), weights


_, molecules = load_molecules()
_, results = load_sweeps()
total = accurate = unresolved = shared = displaced = inaccurate_resolved = missing = 0
minimum_accurate_weight = 1.0
resolved_roots = 0
maximum_resolved_error = 0.0
for key in ORDER:
    molecule = molecules[key]
    result = results[key]
    sweep = result["tau_noiseless"]
    size = int(result["M"])
    accessible_energies, _, _ = exact_measure(molecule)
    for point, tau in enumerate(sweep["tau"]):
        estimates = np.sort(np.asarray(sweep["E"][point], dtype=float))
        phases, weights = validated_ritz_vectors(molecule, size, float(tau))
        available = set(range(len(phases)))
        assignments = []
        for estimate in estimates:
            if not np.isfinite(estimate) or not available:
                assignments.append(None)
                continue
            index = min(available, key=lambda item: abs(np.angle(np.exp(1j * (phases[item] + tau * estimate)))))
            assignments.append(index)
            available.remove(index)
        largest_root = np.argmax(weights, axis=1)

        def shares_root(state):
            neighbors = [neighbor for neighbor in (state - 1, state + 1) if 0 <= neighbor < len(estimates)]
            return any(largest_root[state] == largest_root[neighbor] for neighbor in neighbors)

        for state, error in enumerate(sweep["err"][point]):
            total += 1
            assignment = assignments[state] if state < len(assignments) else None
            if assignment is None or not np.isfinite(error):
                missing += 1
                unresolved += 1
                shared += shares_root(state)
                continue
            assigned_weight = float(weights[state, assignment])
            state_resolved = bool(np.max(weights[state]) >= 0.9)
            if np.max(weights[:, assignment]) >= 0.9:
                resolved_roots += 1
                root_state = int(np.argmax(weights[:, assignment]))
                root_error = abs(estimates[state] - accessible_energies[root_state])
                maximum_resolved_error = max(maximum_resolved_error, float(root_error))
            if error <= qe.CHEMICAL_ACCURACY:
                accurate += 1
                minimum_accurate_weight = min(minimum_accurate_weight, assigned_weight)
            elif not state_resolved:
                unresolved += 1
                shared += shares_root(state)
            elif assigned_weight < 0.9:
                displaced += 1
            else:
                inaccurate_resolved += 1

print(f"points: {total}")
print(f"within chemical accuracy: {accurate}")
print(f"minimum correct-state weight among accurate points: {minimum_accurate_weight:.3f}")
print(f"unresolved state: {unresolved} (shared with a neighboring target: {shared})")
print(f"resolved state displaced by a lower unresolved state: {displaced}")
print(f"correct resolved root above chemical accuracy: {inaccurate_resolved}")
print(f"missing estimates: {missing}")
print(f"reported roots with at least 90% weight: {resolved_roots}")
print(f"largest error of a reported resolved root: {maximum_resolved_error:.3e} Ha")
