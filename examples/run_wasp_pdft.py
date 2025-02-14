import numpy as np
import pickle
from ase.io import read, write
from pyscf.pbc.tools.pyscf_ase import atoms_from_ase
from pyscf import gto

from wasp.helper import extract_mo_coeff
from wasp.compute_weights import guess_mo_coeff
from wasp.compute_mcpdft_energy import compute_mcpdft_energy

# ----- Parameters and File Paths (adjust these as needed) -----
epsilon = 3.2
output_pickle_file = "epsilon_8.pkl"

trajectory_path = '/path/to/trajectory/'
trajectory = read(trajectory_path + 'train_filtered.xyz', index=':')

order_trajectory = np.load('order_trajectory_0.npy')

path_chk = '/path/to/chk/'
reference_geo_chks = read(path_chk + 'reference_geom.traj', index=':')

chk_folder = path_chk + 'mo_coeffs/'

def main():
    print("Starting simulation...")
    
    # Load reference geometries and their corresponding MO coefficients
    geometries = [reference_geo_chks[i] for i in range(len(reference_geo_chks))]
    hdf5_files = [f'{chk_folder}mcscf_frame_{i}.hdf5' for i in range(len(reference_geo_chks))]
    mo_coeffs = [extract_mo_coeff(file) for file in hdf5_files]
    
    results = []
    chk_index = len(reference_geo_chks)
    
    for i in order_trajectory:
        print(f'Processing frame {i}')
        frame = atoms_from_ase(trajectory[i])
        mol = gto.M(
            atom=frame,
            basis="def2-TZVP",
            verbose=5,
            output="pyscf.log",
            spin=1,
            unit="Angstrom",
            symmetry=False,
            charge=1)
        
        mo_coeff_frame = guess_mo_coeff(trajectory[i], geometries, mo_coeffs, epsilon)
        energy, forces = compute_mcpdft_energy(mol, mo_coeff_frame, chk_folder, frame_index=chk_index, flag=True)
        
        geometries.append(trajectory[i])
        mo_coeffs.append(extract_mo_coeff(f'{chk_folder}mcscf_frame_{chk_index}.hdf5'))
        chk_index += 1
        
        results.append({
            'frame_index': i,
            'energy': energy,
            'forces': forces,
        })

    with open(output_pickle_file, 'wb') as f:
        pickle.dump(results, f)

    write(path_chk + 'combined_chk_coordinates.traj', geometries)
    print("Finished simulation.")

if __name__ == "__main__":
    main()

