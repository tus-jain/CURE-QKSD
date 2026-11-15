"""Print Table I from the stored molecular data and sweep settings."""

import numpy as np

from cure_qksd.open_data import load_molecules, load_sweeps

ORDER = ["H6", "HF", "N2", "LiH", "H2O_R1.4", "BeH2"]
NAMES = {
    "H6": "H6 chain",
    "HF": "HF",
    "N2": "N2",
    "LiH": "LiH",
    "H2O_R1.4": "H2O",
    "BeH2": "BeH2",
}
DISTANCES = {
    "H6": 1.5,
    "HF": 1.5,
    "N2": 1.5,
    "LiH": 3.595,
    "H2O_R1.4": 1.4,
    "BeH2": 2.5,
}


def reference_count(molecule):
    """Return the count of eigenstates with reference weight above 1e-10."""
    diagonal = molecule.H.tocoo()
    if np.all(diagonal.row == diagonal.col):
        weights = np.abs(molecule.ref_state) ** 2
    else:
        _, vectors = np.linalg.eigh(molecule.H.toarray())
        weights = np.abs(vectors.conj().T @ molecule.ref_state) ** 2
    return int(np.count_nonzero(weights > 1e-10))


_, molecules = load_molecules()
_, results = load_sweeps()
header = (
    "System | d (A) | n_q | n_ref | |gamma_0|^2 | E_gs (Ha) | tau_alias | "
    "W_gs (Ha) | W_multi (Ha) | M | Targets | tau_max | M noisy | State"
)
print(header)
print("---|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---")
for key in ORDER:
    molecule = molecules[key]
    result = results[key]
    ground_window = result["window_gs"]
    multi_window = result["window"]
    state = int(result["noisy_state"])
    print(
        f"{NAMES[key]} | {DISTANCES[key]:g} | {molecule.n_qubits} | "
        f"{reference_count(molecule)} | {molecule.exact_overlaps[0]:.3f} | "
        f"{molecule.exact_energies[0]:.3f} | {result['tau_alias']:.3f} | "
        f"[{ground_window[0]:.3f},{ground_window[1]:.0f}] | "
        f"[{multi_window[0]:.3f},{multi_window[1]:.3f}] | {result['M']} | "
        f"{len(result['exact_energies'])} | {result['tau_noiseless']['tau'][-1]:.2f} | "
        f"{result['M_noisy']} | E_{state}"
    )
