import numpy as np
import h5py
from pyscf import gto, mcscf, mcpdft, M
from ase.units import Ha, Bohr

def compute_mcpdft_energy(mol, mo_coeff, chk_folder, frame_index=0, flag=False):
    """
    Perform a SA(2)-MCPDFT calculation and return the energy and forces.
    
    :param mol: A PySCF molecule object.
    :param mo_coeff: Guess MO coefficients.
    :param chk_folder: Folder path to save checkpoint files.
    :param frame_index: Index of the current frame.
    :param flag: If True, save the MO coefficients.
    :return: Tuple (energy, forces)
    """
    hf = mol.UHF()
    hf.max_cycle = 0
    hf.run()

    mc = mcpdft.CASSCF(hf, "tPBE", 9, 7, grids_level=6).state_average_([.5, .5])
    mc.max_cycle = 2000
    mc.conv_tol = 1e-7
    mc.conv_tol_grad = 1e-4

    mc.mo_coeff = mcscf.project_init_guess(mc, mo_coeff, prev_mol=None)
    mc.run()

    scanner = mc.nuc_grad_method(state=0).as_scanner()
    scanner.max_cycle = 2000

    etot, grad = scanner(mol)
    if not scanner.converged:
        raise RuntimeError('Gradients did not converge!')

    if flag:
        save_mo_coeff(mc, chk_folder, frame_index)

    energy_ase = etot * Ha
    forces_ase = -grad * (Ha / Bohr)
    return energy_ase, forces_ase

def save_mo_coeff(mc, chk_folder, frame_index):
    """
    Save the MO coefficient for the current frame.
    
    :param mc: The MCPDFT calculation object.
    :param chk_folder: Folder path to save the file.
    :param frame_index: Frame index for naming.
    """
    chkfile_name = f"{chk_folder}mcscf_frame_{frame_index}.hdf5"
    with h5py.File(chkfile_name, 'w') as f:
        f.create_dataset('mo_coeff', data=mc.mo_coeff)
    print(f"Saved mo_coeff for frame {frame_index} to {chkfile_name}")

