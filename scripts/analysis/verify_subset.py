"""Regenerate one molecule, one noiseless sweep, and three noisy medians."""

import time
import zlib

import numpy as np

from cure_qksd import engine as qe
from cure_qksd.open_data import load_molecules, load_sweeps
from scripts.data.build_molecules import N_TARGET_STATES, build_systems
from scripts.data.run_sweeps import (
    FACTORS_NOISELESS,
    M_NOISY,
    M_PER_SYSTEM,
    NOISY_STATE,
    NUM_STATES,
    RNG_SEED,
    SIGMA_MAIN,
    sweep_tau_noiseless,
    sweep_tau_noisy,
    window_for,
)

systems, stored_molecules = load_molecules()
_, stored_results = load_sweeps()
key = "LiH"
spec = build_systems()[key]
start = time.perf_counter()
rebuilt = qe.build_molecule(
    key,
    spec["geometry"],
    spec["basis"],
    multiplicity=spec["multiplicity"],
    n_target_states=N_TARGET_STATES[key],
    k_search=80,
    overlap_floor=1e-10,
)
reference = stored_molecules[key]
energy_difference = float(np.max(np.abs(rebuilt.exact_energies - reference.exact_energies)))
overlap_difference = float(np.max(np.abs(rebuilt.exact_overlaps - reference.exact_overlaps)))
print(
    f"LiH molecule build: {time.perf_counter() - start:.1f}s; "
    f"max energy difference {energy_difference:.3e} Ha; "
    f"max overlap difference {overlap_difference:.3e}"
)

key = "H6"
molecule = stored_molecules[key]
states = NUM_STATES[key]
start = time.perf_counter()
noiseless = sweep_tau_noiseless(
    molecule,
    M_PER_SYSTEM[key],
    window_for(molecule, states),
    np.asarray(FACTORS_NOISELESS[key]),
    states,
)
stored_noiseless = stored_results[key]["tau_noiseless"]
noiseless_difference = float(np.nanmax(np.abs(noiseless["err"] - stored_noiseless["err"])))
print(f"H6 noiseless sweep: {time.perf_counter() - start:.1f}s; max error-array difference {noiseless_difference:.3e} Ha")

cases = [("H2O_R1.4", 2.2), ("BeH2", 1.7), ("N2", 86.0)]
for key, factor in cases:
    molecule = stored_molecules[key]
    state = NOISY_STATE.get(key, 0)
    seed = RNG_SEED + zlib.crc32(key.encode()) % 1000
    start = time.perf_counter()
    noisy = sweep_tau_noisy(
        molecule,
        M_NOISY[key],
        window_for(molecule, state + 1),
        np.asarray([factor]),
        SIGMA_MAIN,
        8,
        seed,
        state=state,
    )
    stored = stored_results[key]["tau_noisy"]
    index = int(np.where(np.isclose(np.asarray(stored["factors"], dtype=float), factor))[0][0])
    difference = abs(float(noisy["cure_median"][0]) - float(stored["cure_median"][index]))
    print(
        f"{key} E{state}, {factor:g} tau_alias: median {noisy['cure_median'][0]:.3e} Ha; "
        f"stored difference {difference:.3e} Ha; {time.perf_counter() - start:.1f}s"
    )
