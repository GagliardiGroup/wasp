import numpy as np
from wasp.helper import compute_rmsd

def compute_weights(atoms_new, atoms_array, epsilon):
    """
    Compute the weights for geometries based on their distance to the new geometry.
    Uses the nearest three neighbours (with an epsilon cutoff).
    """
    distances = np.zeros(len(atoms_array))
    for i, atoms in enumerate(atoms_array):
        distances[i] = compute_rmsd(atoms_new, atoms)

    nearest_idxs = np.where(distances <= epsilon)[0]
    if len(nearest_idxs) == 0:
        raise ValueError("No geometries within epsilon distance.")

    weights = np.zeros(len(atoms_array))  # Initialize weights

    # Select the top 3 smallest distances
    if len(nearest_idxs) > 3:
        top_3_idxs = nearest_idxs[np.argsort(distances[nearest_idxs])[:3]]
    else:
        top_3_idxs = nearest_idxs

    # Check if any of the selected distances are zero
    if np.any(distances[top_3_idxs] == 0):
        for idx in top_3_idxs:
            if distances[idx] == 0:
                weights[idx] = 1
                break  # Only set one to 1
    else:
        weights[top_3_idxs] = 1 / distances[top_3_idxs]
        weights /= np.sum(weights)  # Normalize weights

    return weights

def guess_mo_coeff(new_geom, geoms, mo_coeffs, epsilon):
    """
    Compute the guess MO coefficient matrix for a new geometry using a weighted average.
    :param new_geom: The new geometry (ASE Atoms object).
    :param geoms: List of reference geometries.
    :param mo_coeffs: List of MO coefficient matrices corresponding to geoms.
    :param epsilon: Cutoff distance.
    :return: A weighted average MO coefficient matrix.
    """
    geoms_weights = compute_weights(new_geom, geoms, epsilon)
    print("Weights:", geoms_weights)
    print(f"Number of MO coefficients: {len(mo_coeffs)}")

    # Compute the weighted sum of MO coefficients
    C_new = np.zeros_like(mo_coeffs[0])
    for i, weight in enumerate(geoms_weights):
        C_new += weight * mo_coeffs[i]

    return C_new