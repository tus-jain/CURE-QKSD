"""Read and write the repository's JSON and NumPy data files."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

from .engine import Molecule

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DATA_DIRECTORY = REPOSITORY_ROOT / "data"


def _json_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    return value


def save_molecules(systems, molecules, directory=DATA_DIRECTORY / "molecules"):
    """Write molecular Hamiltonians, reference states, and metadata."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "systems.json").write_text(
        json.dumps(_json_value(systems), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {}
    for key, molecule in molecules.items():
        matrix = molecule.H.tocsr()
        filename = f"{key}.npz"
        np.savez_compressed(
            directory / filename,
            h_data=matrix.data,
            h_indices=matrix.indices,
            h_indptr=matrix.indptr,
            h_shape=np.asarray(matrix.shape, dtype=np.int64),
            ref_state=molecule.ref_state,
            exact_energies=molecule.exact_energies,
            exact_overlaps=molecule.exact_overlaps,
        )
        manifest[key] = {
            "file": filename,
            "label": molecule.label,
            "n_qubits": molecule.n_qubits,
            "E_hf": molecule.E_hf,
            "E_ccsd": molecule.E_ccsd,
            "hf_gap": molecule.hf_gap,
        }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_molecules(directory=DATA_DIRECTORY / "molecules"):
    """Load molecular Hamiltonians, reference states, and metadata."""
    directory = Path(directory)
    systems = json.loads((directory / "systems.json").read_text(encoding="utf-8"))
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    molecules = {}
    for key, metadata in manifest.items():
        with np.load(directory / metadata["file"], allow_pickle=False) as arrays:
            shape = tuple(int(value) for value in arrays["h_shape"])
            matrix = csr_matrix(
                (arrays["h_data"], arrays["h_indices"], arrays["h_indptr"]),
                shape=shape,
            )
            molecules[key] = Molecule(
                label=metadata["label"],
                H=matrix,
                ref_state=arrays["ref_state"].copy(),
                n_qubits=int(metadata["n_qubits"]),
                E_hf=float(metadata["E_hf"]),
                E_ccsd=float(metadata["E_ccsd"]),
                hf_gap=float(metadata["hf_gap"]),
                exact_energies=arrays["exact_energies"].copy(),
                exact_overlaps=arrays["exact_overlaps"].copy(),
            )
    return systems, molecules


def _pack_mapping(mapping):
    arrays = {}

    def pack(value, path):
        if isinstance(value, dict):
            return {key: pack(item, path + (key,)) for key, item in value.items()}
        if isinstance(value, np.ndarray):
            array_key = "array_" + str(len(arrays))
            arrays[array_key] = value
            return {"__array__": array_key}
        return _json_value(value)

    return pack(mapping, ()), arrays


def save_sweeps(systems, results, directory=DATA_DIRECTORY / "sweeps"):
    """Write sweep arrays to NPZ and their structure to JSON."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    structure, arrays = _pack_mapping({"systems": systems, "results": results})
    np.savez_compressed(directory / "results.npz", **arrays)
    (directory / "results.json").write_text(
        json.dumps(structure, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_sweeps(directory=DATA_DIRECTORY / "sweeps"):
    """Load sweep arrays from NPZ and their structure from JSON."""
    directory = Path(directory)
    structure = json.loads((directory / "results.json").read_text(encoding="utf-8"))
    with np.load(directory / "results.npz", allow_pickle=False) as arrays:
        def unpack(value):
            if isinstance(value, dict) and set(value) == {"__array__"}:
                return arrays[value["__array__"]].copy()
            if isinstance(value, dict):
                return {key: unpack(item) for key, item in value.items()}
            return value

        data = unpack(structure)
    return data["systems"], data["results"]
