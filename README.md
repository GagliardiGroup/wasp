# WASP: Weighted Active Space Protocol

**WASP** (Weighted Active Space Protocol) is a Python framework designed to generate molecular orbital (MO) coefficient guesses through interpolation from a reference library of wave function checkpoints.

## 📚 Requirements

To use WASP, you'll need the following Python packages:

- [ASE](https://gitlab.com/ase/ase)
- [PySCF](https://github.com/pyscf/pyscf)
- [PySCF-FORGE](https://github.com/pyscf/pyscf-forge) (for MCPDFT support)
- numpy
- h5py

## ▶️ Quick Start

To run the example and see WASP in action:

```bash
python examples/run_example.py
```

This script will:

- Load a set of checkpoint geometries and their MO coefficient files (`.hdf5`)
- Read a target geometry from an `.xyz` file
- Compute RMSDs between the target and checkpoint geometries
- Interpolate a guess MO coefficient matrix using normalized inverse-distance weights
- Run a MCPDFT calculation on the target geometry using orthonormalized guess MO coefficient matrix

## 📜 Citation

If you use **WASP** in your research or publications, please cite the following:

```bibtex
@misc{seal2025wasp,
      title={Weighted Active Space Protocol for Multireference Machine-Learned Potentials}, 
      author={Aniruddha Seal and Simone Perego and Matthew R. Hennefarth and Umberto Raucci and Luigi Bonati and Andrew L. Ferguson and Michele Parrinello and Laura Gagliardi},
      year={2025},
      eprint={2505.10505},
      archivePrefix={arXiv},
      primaryClass={physics.chem-ph},
      url={https://arxiv.org/abs/2505.10505}, 
}
```
