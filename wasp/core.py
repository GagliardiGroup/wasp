import numpy as np
import h5py
from ase.io import read
from ase.build.rotate import minimize_rotation_and_translation
from ase.units import Ha

from pyscf.pbc.tools.pyscf_ase import atoms_from_ase
from pyscf import gto, mcscf, mcpdft


class WASP:
    def __init__(self,
                 delta,
                 checkpoint_xyz,
                 checkpoint_indices,
                 checkpoint_hdf5_folder,
                 xyz,
                 basis="def2-TZVP",
                 ncas=9,
                 nelecas=7,
                 nstates=2,
                 state_idx=0,
                 ot_func="tPBE",
                 spin=1,
                 charge=1,
                 verbose=False):
        """
        Initialize the WASP class.

        Parameters:
        - delta : Threshold distance for selecting nearby geometries.
        - checkpoint_xyz : Path to XYZ trajectory file used for WASP interpolation.
        - checkpoint_indices (list[int]): Indices of geometries in the trajectory used for interpolation.
        - checkpoint_hdf5_folder : Folder containing HDF5 files with MO coefficients.
        - xyz : Path to XYZ file with the target geometry.
        - basis : Basis set.
        - ncas : Number of active orbitals.
        - nelecas : Number of active electrons in CAS.
        - nstates : Number of states for state averaging.
        - state_idx : Index of the state to compute energy and gradients.
        - ot_func : On-top functional (e.g., "tPBE").
        - spin : Spin multiplicity (2S).
        - charge : Molecular charge.
        - verbose : If True, print RMSD and weights during interpolation.
        """
        self.delta = delta
        self.checkpoint_xyz = read(checkpoint_xyz, index=':')
        self.checkpoint_geometries = [self.checkpoint_xyz[i] for i in checkpoint_indices]
        self.checkpoint_hdf5_files = [f"{checkpoint_hdf5_folder}/mcscf_frame_{i}.hdf5" for i in checkpoint_indices]
        self.checkpoint_mo_coeffs = [self.extract_mo_coeff(f) for f in self.checkpoint_hdf5_files]

        self.basis = basis
        self.ncas = ncas
        self.nelecas = nelecas
        self.nstates = nstates
        self.state_idx = state_idx
        self.ot_func = ot_func
        self.spin = spin
        self.charge = charge
        self.target_geometry = read(xyz)
        self.verbose = verbose

    @staticmethod
    def extract_mo_coeff(filename):
        """
        Extract MO coefficients from an HDF5 file.

        Parameters:
        - filename : Path to HDF5 file.

        Returns:
        - np.ndarray: Molecular orbital coefficients.
        """
        with h5py.File(filename, 'r') as f:
            return f['mo_coeff'][:]

    def rmsd(self, atoms1, atoms2):
        """
        Compute RMSD between two ASE geometries after alignment.

        Parameters:
        - atoms1, atoms2 (ASE Atoms): Geometries to compare.

        Returns:
        - float: Root-mean-square deviation.
        """
        minimize_rotation_and_translation(atoms1, atoms2)
        diff = atoms1.get_positions() - atoms2.get_positions()
        return np.sqrt(np.mean(np.sum(diff**2, axis=1)))

    def compute_weights(self, atoms_new):
        """
        Compute weights based on RMSD between the target geometry and checkpoint geometries.

        Parameters:
        - atoms_new (ASE Atoms): Target geometry.

        Returns:
        - np.ndarray: Normalized weights.
        """
        distances = np.array([self.rmsd(atoms_new, chk_geom) for chk_geom in self.checkpoint_geometries])

        nearest_idxs = np.where(distances <= self.delta)[0]
        if len(nearest_idxs) == 0:
            raise ValueError("No geometries within delta distance.")

        weights = np.zeros(len(self.checkpoint_geometries))
        if np.any(distances[nearest_idxs] == 0):
            for idx in nearest_idxs:
                if distances[idx] == 0:
                    weights[idx] = 1
                    break
        else:
            weights[nearest_idxs] = 1 / distances[nearest_idxs]
            weights /= np.sum(weights)

        if self.verbose:
            print("\nWASP Weighting Results:")
            for i, (w, d) in enumerate(zip(weights, distances)):
                print(f"  Geometry index {i}: RMSD = {d:.6f} Å, Weight = {w:.6f}")

        return weights

    def guess_mo_coeff(self, new_geom):
        """
        Generate interpolated MO coefficient guess for the target geometry.

        Parameters:
        - new_geom (ASE Atoms): Target geometry.

        Returns:
        - np.ndarray: Estimated MO coefficient matrix.
        """
        weights = self.compute_weights(new_geom)
        C_new = np.zeros_like(self.checkpoint_mo_coeffs[0])
        for i, weight in enumerate(weights):
            C_new += weight * self.checkpoint_mo_coeffs[i]
        return C_new

    def compute_mcpdft_energy(self, mol, mo_coeff):
        """
        Perform MCPDFT energy and gradient calculation.

        Parameters:
        - mol (pyscf.gto.Mole): PySCF molecule.
        - mo_coeff (np.ndarray): MO coefficients to initialize the calculation.

        Returns:
        - tuple:
            - float: Total MCPDFT energy (Hartree).
            - np.ndarray: Nuclear gradients.
        """
        hf = mol.UHF()
        hf.max_cycle = 0
        hf.run()

        weights = [1.0 / self.nstates] * self.nstates
        mc = mcpdft.CASSCF(hf, self.ot_func, self.ncas, self.nelecas, grids_level=6).state_average_(weights)
        mc.max_cycle = 2000
        mc.conv_tol = 1e-7
        mc.conv_tol_grad = 1e-4

        mc.mo_coeff = mcscf.project_init_guess(mc, mo_coeff, prev_mol=None)
        mc.run()

        scanner = mc.nuc_grad_method(state=self.state_idx).as_scanner()
        scanner.max_cycle = 2000

        etot, grad = scanner(mol)

        if not scanner.converged:
            raise RuntimeError("Gradients did not converge.")

        return etot * Ha, grad

    def evaluate_wasp_pdft(self):
        """
        Evaluate MCPDFT energy and gradients for the target geometry
        with the orthonormalized WASP MO coefficient matrix guess.

        Returns:
        - tuple:
            - float: MCPDFT total energy.
            - np.ndarray: Nuclear gradients.
        """
        mol = gto.M(
            atom=atoms_from_ase(self.target_geometry),
            basis=self.basis,
            verbose=0,
            output=None,
            spin=self.spin,
            unit="Angstrom",
            symmetry=False,
            charge=self.charge
        )

        mo_coeff = self.guess_mo_coeff(self.target_geometry)
        return self.compute_mcpdft_energy(mol, mo_coeff)