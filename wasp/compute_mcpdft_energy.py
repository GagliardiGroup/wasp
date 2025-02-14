import numpy as np
import h5py
from pyscf import gto, mcscf, mcpdft, M
from ase.units import Ha, Bohr

def compute_mcpdft_energy(mol, mo_coeff=None, t_func="tPBE", ncas=9, nelecas=7, chk_folder=None, frame_index=0, flag=False):
    """
    Perform a SA(2)-MCPDFT calculation and return the energy and forces.
    
    :param mol: A PySCF molecule object.
    :param mo_coeff: Guess MO coefficients. If provided, they are used to project the initial guess.
    :param t_func: The type of translated functional to use (default "tPBE").
    :param ncas: The number of active space orbitals.
    :param nelecas: The number of active space electrons.
    :param chk_folder: Folder path to save checkpoint files.
    :param frame_index: Index of the current frame.
    :param flag: If True, save the MO coefficients.
    :return: Tuple (energy, forces) in ASE units.
    """
    hf = mol.UHF()
    hf.max_cycle = 0
    hf.run()

    mc = mcpdft.CASSCF(hf, t_func, ncas, nelecas, grids_level=6).state_average_([0.5, 0.5])
    mc.max_cycle = 2000
    mc.conv_tol = 1e-7
    mc.conv_tol_grad = 1e-4

    if mo_coeff is not None:
        mc.mo_coeff = mcscf.project_init_guess(mc, mo_coeff, prev_mol=None)

    mc.run()

    scanner = mc.nuc_grad_method(state=0).as_scanner()
    scanner.max_cycle = 2000

    etot, grad = scanner(mol)
    if not scanner.converged:
        raise RuntimeError('Gradients did not converge!')

    # Optionally, save the MO coefficients if requested.
    if flag and chk_folder is not None:
        save_mo_coeff(mc, chk_folder, frame_index)

    # Convert the energy and forces to ASE units.
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

