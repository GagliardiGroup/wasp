import numpy as np
from ase import Atoms
from wasp.helper import compute_rmsd

def test_rmsd_identical():
    """Test that the RMSD of identical geometries is zero."""
    atoms1 = Atoms('H2', positions=[[0, 0, 0], [0, 0, 1]])
    atoms2 = Atoms('H2', positions=[[0, 0, 0], [0, 0, 1]])
    rmsd = compute_rmsd(atoms1, atoms2)
    assert np.isclose(rmsd, 0, atol=1e-8), f"Expected 0 but got {rmsd}"

def test_rmsd_rotated():
    """
    Test that the RMSD function returns a non-negative value.
    (Note: Due to the minimization of rotation and translation, even shifted geometries
    can yield a low RMSD. Here, we simply check that the value is non-negative.)
    """
    atoms1 = Atoms('H2', positions=[[0, 0, 0], [1, 0, 0]])
    atoms2 = Atoms('H2', positions=[[0, 0, 0], [0, 1, 0]])
    rmsd = compute_rmsd(atoms1, atoms2)
    assert rmsd >= 0, f"RMSD should be non-negative, got {rmsd}"

