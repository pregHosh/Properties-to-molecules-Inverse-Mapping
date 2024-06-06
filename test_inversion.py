from invert_CM import *
from CM_preparation import *
from ase import Atoms
from ase.visualize import view
from rdkit import Chem
from rdkit.Chem import AllChem
import pdb

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

    # Visualize the molecule
    view(molecule)

    # Create the master vector (sorted atomic numbers excluding hydrogens)
    master_vec = sorted(atoms, reverse=True)

    return atoms, symbols, positions, master_vec


if __name__ == "__main__":
    smiles = "CCCCCCCC"

    atoms, symbols, positions, master_vec1 = visualize_molecule_from_smiles(smiles)

    print("Master Vector:", master_vec1)

    # Create the ASE Atoms object
    mol = Atoms(symbols=symbols, positions=positions)

    # Generate and standardize the Coulomb matrix
    standardized_CM = get_standardized_CM(atoms, positions, master_vec1)
    print("Standardized CM:\n", standardized_CM)

    # Invert the Coulomb matrix
    distance_mat, master_vec2 = recover_distance_mat(standardized_CM)

    # add noise to the distance matrix

    distance_mat = distance_mat #+ np.random.rand(*distance_mat.shape)

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
    view(rec_ethanol)
    # Get the RMSD between the original and recovered molecule
    error = rmsd.rmsd(positions, cartesian)
    print("RMSD:", error)
