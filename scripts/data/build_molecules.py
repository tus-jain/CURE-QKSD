import time

import numpy as np

from cure_qksd import engine as qe
from cure_qksd.open_data import save_molecules


def build_systems():
    systems = {}
    systems["H6"] = dict(
        geometry=qe.linear_chain_geometry("H", 1.5, 6), basis="sto-3g",
        multiplicity=1, pretty=r"Stretched H$_6$ chain ($r_{\mathrm{HH}}=1.5\,\mathrm{\AA}$)",
    )
    systems["H2O_R1.4"] = dict(
        geometry=qe.stretched_water_geometry(1.4), basis="sto-3g", multiplicity=1,
        pretty=r"Stretched H$_2$O ($R=1.4\,\mathrm{\AA}$)",
    )
    systems["N2"] = dict(
        geometry=[("N", (0, 0, 0)), ("N", (0, 0, 1.5))], basis="sto-3g", multiplicity=1,
        pretty=r"Stretched N$_2$ ($d=1.5\,\mathrm{\AA}$)", builder="sector",
    )
    systems["HF"] = dict(
        geometry=[("F", (0, 0, 0)), ("H", (0, 0, 1.5))], basis="sto-3g", multiplicity=1,
        pretty=r"Stretched HF ($d=1.5\,\mathrm{\AA}$)",
    )
    systems["LiH"] = dict(
        geometry=[("Li", (0, 0, 0)), ("H", (0, 0, 3.595))], basis="sto-3g", multiplicity=1,
        pretty=r"Stretched LiH ($d=3.595\,\mathrm{\AA}$)",
    )
    systems["BeH2"] = dict(
        geometry=qe.linear_beh2_geometry(2.5), basis="sto-3g", multiplicity=1,
        pretty=r"Stretched BeH$_2$ ($d=2.5\,\mathrm{\AA}$)",
    )
    return systems


N_TARGET_STATES = {"H6": 3, "H2O_R1.4": 5, "LiH": 4, "HF": 4, "BeH2": 6, "N2": 6}


def main():
    systems = build_systems()
    molecules = {}
    for key, spec in systems.items():
        t0 = time.time()
        n_target = N_TARGET_STATES[key]
        if spec.get("builder") == "sector":
            mol, info = qe.build_molecule_sector(key, spec["geometry"], spec["basis"],
                                                 n_target_states=n_target, overlap_floor=1e-10)
            print(f"{key:10s} sector: {info}")
        else:
            mol = qe.build_molecule(key, spec["geometry"], spec["basis"],
                                     multiplicity=spec["multiplicity"], n_target_states=n_target,
                                     k_search=80, overlap_floor=1e-10)
        dt = time.time() - t0
        molecules[key] = mol
        print(f"{key:10s} n_qubits={mol.n_qubits:2d} sector_dim={mol.H.shape[0]:4d}  "
              f"E_hf={mol.E_hf:14.8f}  E_ccsd={mol.E_ccsd:14.8f}  "
              f"hf_gap={mol.hf_gap:.8f}  "
              f"E_exact={np.array2string(mol.exact_energies, precision=8)}  "
              f"overlaps={np.array2string(mol.exact_overlaps, precision=5)}  "
              f"[{dt:.1f}s]")
        save_molecules(systems, molecules)

    print("wrote data/molecules")


if __name__ == "__main__":
    main()
