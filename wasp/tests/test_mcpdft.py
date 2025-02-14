import numpy as np
from pyscf import gto
from wasp.compute_mcpdft_energy import compute_mcpdft_energy
from ase.units import Ha, Bohr

def test_compute_mcpdft_energy():
    """
    Test the compute_mcpdft_energy function for a Li–H molecule using the STO-3G basis set.
    The computed energy and forces are compared to reference values.
    """
    # ----- Set Up Li–H Geometry with STO-3G Basis -----
    mol = gto.M(
        atom="""
            Li 0 0 0
            H 1.5 0 0
        """,
        basis="sto-3g",
        unit="Angstrom"
    )

    # ----- Run the MCPDFT Calculation -----
    energy, forces = compute_mcpdft_energy(mol)

    # ----- Explicit Reference Values -----
    reference_energy = -7.9 * Ha
    reference_forces = np.array([
        [ 0.01,  0.00, 0.00],  
        [-0.01,  0.00, 0.00]
    ]) * (Ha / Bohr)

    # ----- Validate Results -----
    assert np.allclose(energy, reference_energy, atol=1e-4), (
        f"Computed energy {energy} does not match reference {reference_energy} within 1e-4"
    )
    assert np.allclose(forces, reference_forces, atol=1e-4), (
        f"Computed forces {forces} do not match reference {reference_forces} within 1e-4"
    )

    print("Test passed: Computed energy and forces are within tolerance.")

if __name__ == "__main__":
    test_compute_mcpdft_energy()

