from invert_CM import *
from CM_preparation import *
from ase import Atoms
from ase.visualize import view
from rdkit import Chem
from rdkit.Chem import AllChem
import random

random.seed(123)

def visualize_molecule_from_smiles(smiles):
    """
    Converts a SMILES string to a 3D geometry, visualizes it using ASE, and returns the atomic symbols, positions, and master vector.

    Parameters:
    smiles (str): SMILES string of the molecule.

    Returns:
    tuple: A tuple containing a list of atomic symbols, a numpy array of atomic positions, and a list representing the master vector.
    """
    # Convert SMILES to RDKit molecule
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)

    # Generate 3D coordinates for the molecule
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    AllChem.MMFFOptimizeMolecule(mol)

    # Remove hydrogens
    mol = Chem.RemoveHs(mol)

    # Get the atomic positions and numbers
    positions = mol.GetConformer().GetPositions()
    atoms = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    symbols = [Chem.GetPeriodicTable().GetElementSymbol(num) for num in atoms]

    # Define the molecule using ASE Atoms object
    molecule = Atoms(symbols=symbols, positions=positions)

    # Create the master vector (sorted atomic numbers excluding hydrogens)
    master_vec = sorted(atoms, reverse=True)

    return atoms, symbols, positions, master_vec


def compute_rmsd(original, recovered):
    """
    Compute the RMSD between the original and recovered Cartesian coordinates.

    Parameters:
    original (numpy.ndarray): The original Cartesian coordinates.
    recovered (numpy.ndarray): The recovered Cartesian coordinates.

    Returns:
    float: The RMSD between the original and recovered Cartesian coordinates.
    """
    # Translate the coordinates
    original -= rmsd.centroid(original)
    recovered -= rmsd.centroid(recovered)

    # Rotate the coordinates
    U = rmsd.kabsch(original, recovered)
    original = np.dot(original, U)

    # Compute the RMSD
    rmsd_value = rmsd.rmsd(original, recovered)

    return rmsd_value


if __name__ == "__main__":
    smiles = "CC(=O)OC1=CC=CC=C1C(=O)OC"

    atoms, symbols, positions, master_vec1 = visualize_molecule_from_smiles(smiles)

    print("Master Vector:", master_vec1)

    # Create the ASE Atoms object
    mol = Atoms(symbols=symbols, positions=positions)

    # Generate and standardize the Coulomb matrix

    standardized_CM = get_standardized_CM(atoms, positions, master_vec1)
    print("Standardized CM:\n", standardized_CM)
    errors = []
    # Invert the Coulomb matrix
    noise_levels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0, 5.0]
    for n in noise_levels:
        distance_mat, master_vec2 = recover_distance_mat(
            standardized_CM + np.random.rand(*standardized_CM.shape) * n
        )


        print("Master Vector:", master_vec2)

        # Recover the Cartesian coordinates
        cartesian = cartesian_recovery(distance_mat)
        # truncate last two columns
        cartesian = cartesian[:,0:3]
        # remove imaginary part
        cartesian = np.real(cartesian)

        print("Cartesian coordinates shape:", cartesian.shape)
        print("Cartesian coordinates:\n", cartesian)

        # Create the recovered ASE Atoms object
        rec_ethanol = Atoms(symbols=symbols, positions=cartesian)
        print("Recovered mol:")
        #view(rec_ethanol)
        # Get the RMSD between the original and recovered molecule

        error = compute_rmsd(positions, cartesian)
        errors.append(error)
        print("RMSD:", error)

    errors = np.array(errors)
    print("Errors:", errors)
    #plot the errors as a function of noise level
    import matplotlib.pyplot as plt
    plt.plot(noise_levels, errors)
    plt.xlabel("Noise Level")
    plt.ylabel("RMSD")
    plt.title("RMSD vs Noise Level")
    plt.show()
    