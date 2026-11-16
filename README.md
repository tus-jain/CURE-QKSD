# CURE-QKSD

Code and data for the paper

> T. K. Jain, R. Maitra, and N. J. Mayhall, *Classical phase unwrapping enables large-time-step unitary quantum Krylov subspace diagonalization*, [arXiv:2611.14214](https://arxiv.org/abs/2611.14214).

Classical Unwrapping for Resolving Eigenvalues (CURE-QKSD) recovers the multiple of 2π/τ that is lost when unitary quantum Krylov subspace diagonalization (UQKSD) is run at a large time step τ. A classical energy window limits each phase to a few candidate energies, and UQKSD runs at incommensurate time steps remove the candidates that are not true energies. This repository reproduces every figure, Table I, and the numbers quoted in the text.

## Installation

Python 3.14.2 was used. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Reproduction

```bash
./reproduce.sh
```

This regenerates all figures in `figures/` and prints Table I and the quoted numbers from the stored data in `data/`. The noisy ensembles in Figs. 2--4, 8, and 11 use 100 realizations. Fig. 1 is rendered from its SVG source with headless Chrome or Chromium, which is found automatically or set with `CHROME_BIN=/path/to/chrome`.

To run a single script, use the repository root as the working directory and set the import path:

```bash
export PYTHONPATH=src:.
python scripts/figures/noisy.py
```

## Contents

| Paper item | Script | Output |
|---|---|---|
| Fig. 1, CURE-QKSD overview | `scripts/figures/render_overview.sh` | `figures/fig01_overview.pdf` |
| Fig. 2, eigenvalues of **S** at fixed T<sub>basis</sub> | `scripts/figures/conditioning.py` | `figures/fig02_conditioning.pdf` |
| Fig. 3, best thresholded error at fixed T<sub>basis</sub> | `scripts/figures/threshold_trap.py` | `figures/fig03_threshold_trap.pdf` |
| Fig. 4, threshold scan | `scripts/figures/threshold_scan.py` | `figures/fig04_threshold_scan.pdf` |
| Fig. 5, phase-consistency landscape | `scripts/figures/phase_landscape.py` | `figures/fig05_phase_landscape.pdf` |
| Fig. 6, incommensurate time steps | `scripts/figures/incommensurate.py` | `figures/fig06_incommensurate.pdf` |
| Table I | `scripts/figures/table_i.py` | printed |
| Fig. 7, noiseless molecular results | `scripts/figures/noiseless.py` | `figures/fig07_noiseless.pdf` |
| Fig. 8, noisy molecular results | `scripts/figures/noisy.py` | `figures/fig08_noisy.pdf` |
| Fig. 9, images and winding numbers versus the number of runs | `scripts/figures/run_scaling.py` | `figures/fig09_run_scaling.pdf` |
| Fig. 10, Hartree–Fock shift | `scripts/figures/hf_shift.py` | `figures/fig10_hf_shift.pdf` |
| Fig. 11, error versus Krylov dimension | `scripts/figures/m_sweep.py` | `figures/fig11_m_sweep.pdf` |
| Sec. III F, Hartree–Fock shift for H<sub>6</sub>, HF, and N<sub>2</sub> | `scripts/analysis/hf_shift.py` | printed |
| Sec. III and Appendix D, window margins, reference weights, image peaks, and second runs | `scripts/analysis/cure_assumptions.py` | printed |
| Appendix E, root weights | `scripts/analysis/root_weights.py` | printed |

Each figure script also prints the numbers quoted in the text for that figure.

The method is implemented in `src/cure_qksd/engine.py` (`run_cure_qksd`). The model spectrum of Figs. 2–4 is in `src/cure_qksd/conditioning.py`, and the many-run tests of Sec. III E and Appendix F are in `src/cure_qksd/scaling.py`.

## Data

- `data/molecules/`: for each molecule, the Hamiltonian in a symmetry sector of the Hartree–Fock reference (sparse), the reference state, the exact energies and reference weights of the targeted states, E<sub>HF</sub>, E<sub>CCSD</sub>, and the HOMO–LUMO gap. Geometries and basis sets are in `systems.json`.
- `data/sweeps/`: the noiseless and noisy time-step sweeps of Figs. 7 and 8.
- `data/scaling/`: the synthetic and N<sub>2</sub> many-run tests of Fig. 9 and Appendix F.
- `data/m_sweep.npz`: the medians of Fig. 11.

Arrays are stored as NPZ and metadata as JSON.

To regenerate the data from scratch:

```bash
export PYTHONPATH=src:.
python scripts/data/build_molecules.py      # PySCF and OpenFermion
python scripts/data/run_sweeps.py           # all molecules
python scripts/data/run_sweeps.py N2        # one molecule
python scripts/data/run_scaling.py          # Fig. 9 and Appendix F, about 5 minutes
```

`python scripts/analysis/verify_subset.py` rebuilds LiH, reruns the H<sub>6</sub> noiseless sweep, recomputes three noisy medians, and compares them with the stored data.

## Runtime

With one thread per process, `reproduce.sh` takes about 40 minutes on a laptop, most of it in `scripts/analysis/cure_assumptions.py`, and `verify_subset.py` takes about 3 minutes.

## License and citation

MIT License. Please cite the paper above. Citation metadata are in `CITATION.cff`.
