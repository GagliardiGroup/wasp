from ase.io import read, write
from pyscf.pbc.tools.pyscf_ase import atoms_from_ase

from wasp.helper import extract_mo_coeff
from wasp.compute_weights import guess_mo_coeff
from wasp.compute_mcpdft_energy import compute_mcpdft_energy

# ----- Parameters and File Paths (adjust these as needed) -----
epsilon = 0.8

geom = read('frame_neb_5.xyz')

path_chk = 'data/'
reference_geo_chks = read(path_chk + 'reference_geom.traj', index=':')
chk_folder = path_chk + 'mo_coeffs/'

def main():
    
    # ----- Load reference geometries and their corresponding MO coefficients -----
    ref_geometries = [reference_geo_chks[i] for i in range(len(reference_geo_chks))]
    ref_hdf5_files = [f'{chk_folder}mcscf_frame_{i}.hdf5' for i in range(len(reference_geo_chks))]
    ref_mo_coeffs = [extract_mo_coeff(file) for file in ref_hdf5_files]

    # ----- Build the Molecule -----
    frame = atoms_from_ase(geom)
    mol = gto.M(atom=frame,
            basis="def2-TZVP",
            verbose=5,
            output="pyscf.log",
            spin=1,
            unit="Angstrom",
            symmetry=False,
            charge=1)
    
    # ----- Generating Wavefunction guess using WASP -----
    mo_coeff_frame = guess_mo_coeff(geom, ref_geometries, ref_mo_coeffs, epsilon)

    # ----- Compute MCPDFT Energy and Forces -----
    energy, forces = compute_mcpdft_energy(mol, mo_coeff_frame)

if __name__ == "__main__":
    main()

