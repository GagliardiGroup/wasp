import numpy as np
import h5py
from ase.build.rotate import minimize_rotation_and_translation

def extract_mo_coeff(filename):
    """Extract MO coefficients from an HDF5 file."""
    with h5py.File(filename, 'r') as f:
        return f['mo_coeff'][:]  # Extract the 'mo_coeff' dataset

def compute_rmsd(atoms1, atoms2):
    """Compute RMSD between two ASE Atoms objects."""
    minimize_rotation_and_translation(atoms1, atoms2)
    return np.sqrt(np.mean(np.sum((atoms1.get_positions() - atoms2.get_positions())**2, axis=1)))

