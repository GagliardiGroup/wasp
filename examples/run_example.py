from src.wasp import WASP

wasp = WASP(
    delta=0.8,
    checkpoint_xyz='examples/data/trajectory.xyz',
    checkpoint_indices=[0, 6, 9, 11],
    checkpoint_hdf5_folder='examples/data/hdf5',
    xyz='examples/data/geometry.xyz',
    basis='def2-TZVP',
    ncas=9,
    nelecas=7,
    nstates=2,
    state_idx=0,
    ot_func='tPBE',
    spin=1,
    charge=1,
    verbose=True  # Enables printing of RMSDs and weights
)

energy, gradients = wasp.evaluate_wasp_pdft()
print(f"\nEnergy: {energy}")
print("Gradients:")
print(gradients)
