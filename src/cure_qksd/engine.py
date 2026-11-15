from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from itertools import combinations
from scipy.linalg import eig, eigh, toeplitz
from scipy.sparse.linalg import expm_multiply, eigsh
from scipy.signal import find_peaks

CHEMICAL_ACCURACY = 1.6e-3
GOLDEN = (1.0 + np.sqrt(5.0)) / 2.0
NOISE_MODEL = "autocorrelation_lag_toeplitz"
THRESHOLD_RULE = "1.5_sigma_sqrt_M_empirical"

ACCEPT_LEVEL = 0.9999
ACCEPT_RULE = "L_total_ge_0.9999"

FILL_RULE = "fill_nearest_no_passthrough"


@dataclass
class Molecule:
    """Sector Hamiltonian, reference state, and reference data of one molecule (energies in Ha)."""

    label: str
    H: "scipy.sparse.spmatrix"
    ref_state: np.ndarray
    n_qubits: int
    E_hf: float
    E_ccsd: float
    hf_gap: float
    exact_energies: np.ndarray
    exact_overlaps: np.ndarray


def hartree_fock_ref_statevector(n_qubits: int, n_alpha: int, n_beta: int) -> np.ndarray:
    """Hartree-Fock determinant as a statevector, with interleaved alpha/beta spin orbitals."""
    if n_qubits % 2 != 0:
        raise ValueError("n_qubits must be even (interleaved alpha/beta spin-orbitals).")
    occ = np.zeros(n_qubits, dtype=int)
    for a in range(n_alpha):
        occ[2 * a] = 1
    for b in range(n_beta):
        occ[2 * b + 1] = 1
    idx = int(np.dot(occ, 2 ** (np.arange(n_qubits)[::-1])))
    state = np.zeros(2 ** n_qubits, dtype=complex)
    state[idx] = 1.0
    return state


def fixed_spin_sector_indices(n_qubits: int, n_alpha: int, n_beta: int) -> np.ndarray:
    """Basis-state indices with n_alpha alpha and n_beta beta electrons."""
    n_orbitals = n_qubits // 2
    alpha_weights = [1 << (n_qubits - 1 - 2 * p) for p in range(n_orbitals)]
    beta_weights = [1 << (n_qubits - 2 - 2 * p) for p in range(n_orbitals)]
    indices = [
        sum(alpha_weights[p] for p in occ_a) + sum(beta_weights[p] for p in occ_b)
        for occ_a in combinations(range(n_orbitals), n_alpha)
        for occ_b in combinations(range(n_orbitals), n_beta)
    ]
    return np.asarray(sorted(indices), dtype=int)


def build_molecule(label, geometry, basis, multiplicity=1, charge=0,
                    n_target_states=6, k_search=60, overlap_floor=1e-12):
    """Jordan-Wigner Hamiltonian in the fixed-spin sector of the Hartree-Fock state, with E_HF,
    E_CCSD, the HOMO-LUMO gap, and the lowest exact states with nonzero reference weight."""
    from openfermion import MolecularData, get_sparse_operator
    from openfermion.transforms import jordan_wigner
    from openfermionpyscf import run_pyscf

    mol = MolecularData(geometry=geometry, basis=basis, multiplicity=multiplicity, charge=charge)

    mol = run_pyscf(mol, run_scf=True, run_fci=False, verbose=False)

    from pyscf import cc as pyscf_cc
    ccsd = pyscf_cc.CCSD(mol._pyscf_data["scf"])
    ccsd.verbose = 0
    ccsd.run()
    fermion_H = mol.get_molecular_hamiltonian()
    qubit_H = jordan_wigner(fermion_H)
    H_full = get_sparse_operator(qubit_H).tocsr()

    n_q = mol.n_orbitals * 2
    n_e, mult = mol.n_electrons, multiplicity
    n_alpha = (n_e + mult - 1) // 2
    n_beta = (n_e - mult + 1) // 2
    hf_state_full = hartree_fock_ref_statevector(n_q, n_alpha, n_beta)

    sector_indices = fixed_spin_sector_indices(n_q, n_alpha, n_beta)
    H = H_full[sector_indices, :][:, sector_indices].tocsr()
    hf_state = hf_state_full[sector_indices]
    if not np.isclose(np.linalg.norm(hf_state), 1.0):
        raise RuntimeError(f"{label}: Hartree--Fock state was lost during symmetry projection.")

    E_hf = float(np.real(np.vdot(hf_state, H @ hf_state)))
    E_ccsd = float(ccsd.e_tot)
    orbital_energies = np.asarray(mol.orbital_energies, dtype=float)
    hf_gap = float(orbital_energies[n_alpha] - orbital_energies[n_alpha - 1])
    if not np.isfinite(E_ccsd) or not np.isfinite(hf_gap) or hf_gap <= 0.0:
        raise RuntimeError(
            f"{label}: invalid pre-solution window data "
            f"(E_ccsd={E_ccsd}, hf_gap={hf_gap})."
        )

    if H.shape[0] <= 128:
        evals, evecs = eigh(H.toarray())
    else:
        k = min(k_search, H.shape[0] - 2)
        evals, evecs = eigsh(H, k=k, which="SA", tol=1e-12)
    order = np.argsort(evals)
    evals, evecs = evals[order], evecs[:, order]
    overlaps = np.abs(hf_state.conj() @ evecs) ** 2

    keep = overlaps > overlap_floor
    accessible_E = evals[keep][:n_target_states]
    accessible_w = overlaps[keep][:n_target_states]
    if len(accessible_E) < n_target_states:
        raise RuntimeError(
            f"{label}: only found {len(accessible_E)} HF-accessible states "
            f"below overlap floor {overlap_floor}; increase k_search."
        )

    return Molecule(
        label=label, H=H, ref_state=hf_state.astype(complex), n_qubits=n_q,
        E_hf=E_hf, E_ccsd=E_ccsd, hf_gap=hf_gap,
        exact_energies=accessible_E, exact_overlaps=accessible_w,
    )


def stretched_water_geometry(bond_length_angstrom, angle_deg=104.5):
    half = np.deg2rad(angle_deg / 2.0)
    x = bond_length_angstrom * np.sin(half)
    y = bond_length_angstrom * np.cos(half)
    return [("O", (0.0, 0.0, 0.0)), ("H", (x, y, 0.0)), ("H", (-x, y, 0.0))]


def linear_chain_geometry(symbol, bond_length_angstrom, n_atoms):
    return [(symbol, (0.0, 0.0, i * bond_length_angstrom)) for i in range(n_atoms)]


def linear_beh2_geometry(bond_length_angstrom):
    d = bond_length_angstrom
    return [("Be", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, d)), ("H", (0.0, 0.0, -d))]


def autocorrelation_trajectory(H, ref_state, M_max, tau):
    """c(k) = <phi_0| exp(-i H k tau) |phi_0> for k = 0, ..., M_max."""
    if _is_diagonal(H):
        E = np.asarray(H.diagonal()).real
        w = np.abs(np.asarray(ref_state).ravel()) ** 2
        k = np.arange(M_max + 1)
        return np.exp(-1j * np.outer(k * tau, E)) @ w
    A = -1j * H

    rng_state = np.random.get_state()
    np.random.seed(0)
    try:
        states = expm_multiply(A, ref_state, start=0.0, stop=M_max * tau, num=M_max + 1, endpoint=True)
    finally:
        np.random.set_state(rng_state)
    return states @ ref_state.conj()


def _is_diagonal(H):
    try:
        coo = H.tocoo()
    except AttributeError:
        return False
    return bool(coo.nnz > 0 and np.all(coo.row == coo.col))


def build_S_F(autocorr, M):
    """Toeplitz S (M x M) and F (M x M, UQKSD) from c(0..M)."""
    S = toeplitz(np.conj(autocorr[:M]), autocorr[:M])
    F_row = autocorr[1:M + 1]
    F_col = np.zeros(M, dtype=complex)
    F_col[0] = F_row[0]
    if M > 1:
        F_col[1] = autocorr[0]
    if M > 2:
        F_col[2:] = np.conj(F_row[:-2])
    F = toeplitz(F_col, F_row)
    return S, F


def add_autocorr_noise(autocorr, sigma, rng):
    """Add complex Gaussian noise with E|z|^2 = sigma^2 to c(k) for k >= 1."""
    noisy = autocorr.copy()
    n = len(autocorr) - 1
    noisy[1:] += (rng.normal(0.0, sigma, n) + 1j * rng.normal(0.0, sigma, n)) / np.sqrt(2.0)
    return noisy


def solve_uqksd(F, S, eps):
    """Threshold S at eps, solve the reduced UQKSD GEVP for eigenvalues only."""
    s_evals, s_evecs = eigh(S)
    keep = s_evals > eps
    if not np.any(keep):
        return np.array([], dtype=complex), 0
    V = s_evecs[:, keep]
    s_kept = s_evals[keep]
    F_reduced = V.conj().T @ F @ V
    inv_sqrt = 1.0 / np.sqrt(s_kept)
    A = F_reduced * inv_sqrt[:, None] * inv_sqrt[None, :]
    return eig(A, right=False), int(np.count_nonzero(keep))


def noise_adaptive_eps(sigma, M, floor=1e-15, prefactor=1.5):
    """Thresholding cutoff 1.5 sigma sqrt(M), or floor for noiseless data."""
    if sigma <= 0.0:
        return floor
    return prefactor * sigma * np.sqrt(M)


def validate_roots(roots, autocorr_row, M, circle_tol=0.1, weight_rel_cutoff=1e-3):
    """Phases and relative weights of the roots within circle_tol of the unit circle whose
    fitted amplitude in c(k) exceeds weight_rel_cutoff times the largest amplitude."""
    if len(roots) == 0:
        return np.array([]), np.array([])
    near_circle = np.abs(np.abs(roots) - 1.0) < circle_tol
    cand = roots[near_circle]
    if len(cand) == 0:
        return np.array([]), np.array([])
    V = np.vander(cand, N=M, increasing=True).T
    amps, *_ = np.linalg.lstsq(V, autocorr_row[:M], rcond=None)
    weights = np.abs(amps)
    if weights.max() <= 0:
        return np.array([]), np.array([])
    weights = weights / weights.max()
    keep = weights > weight_rel_cutoff
    return np.angle(cand[keep]), weights[keep]


def phase_landscape(phase_sets, tau_list, E_grid):
    """Phase-consistency landscape: the average over runs of max_i cos(theta_i + tau E)."""
    total = np.zeros_like(E_grid)
    n_used = 0
    for phases, tau in zip(phase_sets, tau_list):
        if len(phases) == 0:
            continue
        scores = np.cos(phases[:, None] + tau * E_grid[None, :])
        total += np.max(scores, axis=0)
        n_used += 1
    if n_used == 0:
        return np.full_like(E_grid, -1.0), 0
    return total / n_used, n_used


def find_accepted_peaks(E_grid, landscape, level=0.85):
    """Energies of the local maxima of the landscape at or above level."""
    idx, _ = find_peaks(landscape, height=level)
    return np.sort(E_grid[idx])


def match_peaks_to_base_roots(candidate_peaks, base_phases, tau1):
    """Greedy one-to-one match: each candidate peak claims the base-run
    phase it fits best, skipping phases already claimed."""
    matched_E, matched_phase_idx = [], []
    used = set()
    for E_cand in candidate_peaks:
        scores = np.cos(base_phases + tau1 * E_cand)
        order = np.argsort(scores)[::-1]
        for idx in order:
            if idx not in used:
                matched_E.append(E_cand)
                matched_phase_idx.append(idx)
                used.add(idx)
                break
    return matched_E, matched_phase_idx


def unwrap_energy(E_guide, theta_aliased, tau1):
    """Energy (-theta + 2 pi k) / tau1 closest to E_guide, and its winding number k."""
    shift = 2 * np.pi / tau1
    E_aliased = -theta_aliased / tau1
    k = int(np.round((E_guide - E_aliased) / shift))
    return E_aliased + k * shift, k


@dataclass
class CureResult:
    """Output of run_cure_qksd."""

    E_unwrapped: list = field(default_factory=list)
    winding: list = field(default_factory=list)
    n_grids_used: int = 0
    rank_base: int = 0
    cond_base: float = np.nan
    base_autocorr: np.ndarray = None
    F_base: np.ndarray = None
    S_base: np.ndarray = None


def _fill_missing_targets(result, base_phases, used, tau1, E_lo, E_hi, num_states):
    """Report unmatched base-run phases at the winding closest to the window until
    num_states energies are reported."""
    period = 2 * np.pi / tau1
    cands = []
    for i, theta in enumerate(base_phases):
        if i in used:
            continue
        E_al = -theta / tau1
        k_lo = np.ceil((E_lo - E_al) / period)
        k_hi = np.floor((E_hi - E_al) / period)
        if k_lo <= k_hi:
            k, dist = k_lo, 0.0
        else:
            below, above = E_al + k_hi * period, E_al + k_lo * period
            k, dist = (k_hi, E_lo - below) if E_lo - below <= above - E_hi else (k_lo, above - E_hi)
        cands.append((dist, E_al + k * period, int(k)))
    cands.sort()
    while len(result.E_unwrapped) < num_states and cands:
        _, E, k = cands.pop(0)
        result.E_unwrapped.append(float(E))
        result.winding.append(k)


def run_cure_qksd(mol: Molecule, M, tau1, num_states, sigma=0.0, rng=None,
                   window=(-1.0, 2.5), grid_points=4000,
                   max_extra_grids=2, accept_level=ACCEPT_LEVEL, weight_rel_cutoff=1e-3):
    """CURE-QKSD (Sec. II B). UQKSD runs at tau1, tau1/phi, ... (up to max_extra_grids
    additional runs) are added until the landscape on the energy window has num_states peaks
    at or above accept_level. The peaks are matched to base-run phases and unwrapped."""
    if rng is None:
        rng = np.random.default_rng(0)
    multipliers = [1.0, 1.0 / GOLDEN, 1.0 / np.sqrt(3.0), 1.0 / np.pi][: 1 + max_extra_grids]

    E_grid = np.linspace(mol.E_hf + window[0], mol.E_hf + window[1], grid_points)
    phase_sets, tau_list = [], []
    base_phases_valid = None
    base_weights_valid = None
    S_base = F_base = None
    rank_base, cond_base = 0, np.nan

    period1 = 2 * np.pi / tau1
    window_width = window[1] - window[0]
    min_grids_required = 2 if window_width > period1 else 1

    for step, mult in enumerate(multipliers):
        tau = tau1 * mult
        autocorr = autocorrelation_trajectory(mol.H, mol.ref_state, M, tau)
        if sigma > 0.0:
            autocorr = add_autocorr_noise(autocorr, sigma, rng)
        S, F = build_S_F(autocorr, M)
        eps = noise_adaptive_eps(sigma, M)
        roots, rank = solve_uqksd(F, S, eps)
        phases, weights = validate_roots(roots, autocorr, M, weight_rel_cutoff=weight_rel_cutoff)
        phase_sets.append(phases)
        tau_list.append(tau)

        if step == 0:
            base_phases_valid = phases
            base_weights_valid = weights
            S_base, F_base = S, F
            rank_base = rank
            s_evals = eigh(S, eigvals_only=True)
            positive = s_evals[s_evals > 0]
            cond_base = (positive.max() / positive.min()) if len(positive) else np.inf

        landscape, n_used = phase_landscape(phase_sets, tau_list, E_grid)
        peaks = find_accepted_peaks(E_grid, landscape, level=accept_level)
        if (step + 1) >= min_grids_required and len(peaks) == num_states:
            break

    result = CureResult(n_grids_used=step + 1, rank_base=rank_base, cond_base=cond_base,
                         base_autocorr=autocorr if step == 0 else None, F_base=F_base, S_base=S_base)
    if base_phases_valid is None or len(base_phases_valid) == 0:
        return result
    if len(peaks) == 0:
        _fill_missing_targets(result, base_phases_valid, set(), tau1, E_grid[0], E_grid[-1], num_states)
        return result

    matched_E, matched_idx = match_peaks_to_base_roots(peaks, base_phases_valid, tau1)
    matched_E = np.asarray(matched_E)
    matched_idx = np.asarray(matched_idx, dtype=int)

    order = np.argsort(matched_E)[:num_states]
    for j in order:
        E_unwrapped, k = unwrap_energy(matched_E[j], base_phases_valid[matched_idx[j]], tau1)
        result.E_unwrapped.append(E_unwrapped)
        result.winding.append(k)
    if len(result.E_unwrapped) < num_states:
        _fill_missing_targets(result, base_phases_valid, set(int(i) for i in matched_idx), tau1,
                              E_grid[0], E_grid[-1], num_states)
    return result


def run_uqksd_principal_branch(mol: Molecule, M, tau, sigma=0.0, rng=None, weight_rel_cutoff=1e-3):
    """Plain UQKSD: the lowest validated energy -theta/tau on the principal branch."""
    if rng is None:
        rng = np.random.default_rng(0)
    autocorr = autocorrelation_trajectory(mol.H, mol.ref_state, M, tau)
    if sigma > 0.0:
        autocorr = add_autocorr_noise(autocorr, sigma, rng)
    S, F = build_S_F(autocorr, M)
    eps = noise_adaptive_eps(sigma, M)
    roots, rank = solve_uqksd(F, S, eps)
    phases, weights = validate_roots(roots, autocorr, M, weight_rel_cutoff=weight_rel_cutoff)
    if len(phases) == 0:
        return np.nan
    E_candidates = -phases / tau
    return E_candidates[np.argmin(E_candidates)]


def run_uqksd_principal_branch_states(mol: Molecule, M, tau, num_states, sigma=0.0, rng=None,
                                      weight_rel_cutoff=1e-3):
    """Plain principal-branch UQKSD for several states: the num_states
    lowest validated principal-branch energies, ascending (NaN-padded).
    With num_states=1 this is run_uqksd_principal_branch."""
    if rng is None:
        rng = np.random.default_rng(0)
    autocorr = autocorrelation_trajectory(mol.H, mol.ref_state, M, tau)
    if sigma > 0.0:
        autocorr = add_autocorr_noise(autocorr, sigma, rng)
    S, F = build_S_F(autocorr, M)
    roots, rank = solve_uqksd(F, S, noise_adaptive_eps(sigma, M))
    phases, weights = validate_roots(roots, autocorr, M, weight_rel_cutoff=weight_rel_cutoff)
    out = np.full(num_states, np.nan)
    E_sorted = np.sort(-np.asarray(phases) / tau)[:num_states]
    out[:len(E_sorted)] = E_sorted
    return out


def _string_irreps(norb, nel, orbsym):
    from pyscf.fci import cistring
    strs = cistring.make_strings(range(norb), nel)
    out = np.zeros(len(strs), dtype=int)
    for i, string in enumerate(strs):
        x = 0
        for p in range(norb):
            if (int(string) >> p) & 1:
                x ^= int(orbsym[p])
        out[i] = x
    return out


def build_molecule_sector(label, geometry, basis, n_target_states=6, overlap_floor=1e-10,
                          measure_floor=1e-16):
    """FCI Hamiltonian in the spin and point-group sector of the Hartree-Fock determinant,
    stored in its eigenbasis (diagonal H) together with the reference weights."""
    import ctypes
    import scipy.sparse as sp
    from pyscf import gto, scf, cc, mcscf, ao2mo, lib, symm
    from pyscf.fci import cistring
    from pyscf.fci.direct_spin1 import libfci, make_hdiag

    atom = "; ".join(f"{sym} {x} {y} {z}" for sym, (x, y, z) in geometry)
    mol = gto.M(atom=atom, basis=basis, symmetry=True, verbose=0)
    mf = scf.RHF(mol).run()
    ccsd = cc.CCSD(mf).run()
    nocc = mol.nelectron // 2
    hf_gap = float(mf.mo_energy[nocc] - mf.mo_energy[nocc - 1])
    norb = mf.mo_coeff.shape[1]
    nelec = (nocc, nocc)
    mc = mcscf.CASCI(mf, norb, mol.nelectron)
    h1, ecore = mc.get_h1eff()
    h1 = np.ascontiguousarray(h1)
    eri = ao2mo.restore(1, mc.get_h2eff(), norb)
    orbsym = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, mf.mo_coeff)

    na = cistring.num_strings(norb, nelec[0])
    nb = cistring.num_strings(norb, nelec[1])
    ia, ib = _string_irreps(norb, nelec[0], orbsym), _string_irreps(norb, nelec[1], orbsym)
    A, B = np.meshgrid(np.arange(na), np.arange(nb), indexing="ij")
    sel = (ia[A] ^ ib[B]) == (ia[0] ^ ib[0])
    addra, addrb = A[sel], B[sel]
    stra = cistring.addrs2str(norb, nelec[0], addra)
    strb = cistring.addrs2str(norb, nelec[1], addrb)
    n = len(addra)
    h0 = np.zeros((n, n))
    libfci.FCIpspace_h0tril(h0.ctypes.data_as(ctypes.c_void_p), h1.ctypes.data_as(ctypes.c_void_p),
                            eri.ctypes.data_as(ctypes.c_void_p), stra.ctypes.data_as(ctypes.c_void_p),
                            strb.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(norb), ctypes.c_int(n))
    h0 = lib.hermi_triu(h0)
    hdiag = make_hdiag(h1, eri, norb, nelec, compress=False)
    h0[np.arange(n), np.arange(n)] = hdiag[addra * nb + addrb]
    E_hf = float(h0[0, 0] + ecore)
    evals, evecs = eigh(h0)
    weights = evecs[0, :] ** 2
    del evecs
    evals = evals + ecore
    keep = weights > measure_floor
    E, w = evals[keep], weights[keep]
    order = np.argsort(E)
    E, w = E[order], w[order]
    accessible = w > overlap_floor
    return Molecule(
        label=label, H=sp.diags(E).tocsr(), ref_state=np.sqrt(w).astype(complex),
        n_qubits=2 * norb, E_hf=E_hf, E_ccsd=float(ccsd.e_tot), hf_gap=hf_gap,
        exact_energies=E[accessible][:n_target_states], exact_overlaps=w[accessible][:n_target_states],
    ), {"sector_dim_fixed_spin": na * nb, "sector_dim_symmetry": n, "n_states_with_weight": int(accessible.sum())}
