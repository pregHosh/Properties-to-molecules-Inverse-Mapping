import numpy as np
from matplotlib import pyplot as plt
import torch
from molecular_generation_utils import *
from invert_CM import *
from Model import Multi_VAE
from torch.distributions.categorical import Categorical
from tqdm import tqdm
import copy
from ase.visualize import view
from invert_CM import *
from CM_preparation import *
from test_inversion import compute_rmsd, CM_2_xyz
import os


def create_directory(mod):
    str_folder = "./str_{}".format(mod)
    if not os.path.exists(str_folder):
        os.mkdir(str_folder)
    return str_folder


def write_xyz_file(master_vector, cartesian_coordinates, output_file):
    """
    Converts the charges to elemental symbols and writes the coordinates to an XYZ file.

    Parameters:
    master_vector (array-like): Array of charges.
    cartesian_coordinates (array-like): Array of cartesian coordinates with shape (N, 3).
    output_file (str): Path to the output XYZ file.

    Returns:
    None
    """
    # Define a mapping of charges to elemental symbols
    charge_to_element = {
        1: "H",
        5: "B",
        6: "C",
        7: "N",
        8: "O",
        9: "F",
        # Add more elements as needed
    }

    # Convert charges to elemental symbols
    elements = [charge_to_element.get(charge, "X") for charge in master_vector]

    # Write to XYZ file
    with open(output_file, "w") as file:
        num_atoms = len(elements)
        file.write(f"{num_atoms}\n")
        file.write("Molecule\n")
        for element, coord in zip(elements, cartesian_coordinates):
            file.write(f"{element} {coord[0]} {coord[1]} {coord[2]}\n")


reproduce_paper = False

if reproduce_paper:
    paper_path = "_paper"
else:
    paper_path = ""


properties = torch.load("./dati/data/properties_total.pt".format(paper_path))
p_means = torch.load("./dati/data/properties_means.pt".format(paper_path))
p_stds = torch.load("./dati/data/properties_stds.pt".format(paper_path))
norm_props = (properties - p_means) / p_stds

properties_list = [
    "eAT",
    "eMBD",
    "eXX",
    "mPOL",
    "eNN",
    "eNE",
    "eEE",
    "eKIN",
    "DIP",
    "HLgap",
    "HOMO_0",
    "LUMO_0",
    "HOMO_1",
    "LUMO_1",
    "HOMO_2",
    "LUMO_2",
    "dimension",
]
p_arr = np.array(properties_list)


# List all subfolders in ./out_models starting with the name "checkpoints_"
subfolders = ["paper", "checkpoints_128", "checkpoints_256"]
dimensions = [21, 128, 256]

for dim, mod in zip(dimensions,subfolders):

    str_folder = create_directory(mod)
    if mod=="paper":
        MODEL_PATH = "./models_saved/masked/epoch=2597-step=145487.ckpt"
    else:
        MODEL_PATH = "./our_models/{}/last.ckpt".format(mod)

    modello = Multi_VAE.load_from_checkpoint(
        MODEL_PATH,
        map_location=torch.device("cpu"),
        structures_dim=len(torch.load("./dati/data/data_val/CMs.pt")[0, :]),
        properties_dim=len(torch.load("./dati/data/data_val/properties.pt")[0, :]),
        latent_size=dim,
        extra_dim=32 - len(torch.load("./dati/data/data_val/properties.pt")[0, :]),
        initial_lr=1e-3,
        properties_means=p_means,
        properties_stds=p_stds,
        beta=4.0,
        alpha=1.0,
        decay=0.01,
    )

    modello.eval()
    modello.freeze()

    gm, labels = props_fit_Gaussian_mix(
        norm_props, min_components=91, max_components=92  # 91  # 92
    )

    # recall that here the target value is the normalized one

    generated = start_generation(
        modello,
        {"mPOL": 2.0, "eMBD": 2.0},
        p_arr,
        11,
        int(5e3),
        gm.means_,
        gm.covariances_,
        cm_diff=5,
        deltaz=6,
        check_new_comp=False,
        verbose=False,
    )

    for ind, CM in enumerate(generated):
        rec_xyz, master_vec = CM_2_xyz(CM)
        print("Master Vector:", master_vec)
        print("Cartesian coordinates shape:", rec_xyz.shape)
        print("Cartesian coordinates:\n", rec_xyz)

        write_xyz_file(master_vec, rec_xyz, str_folder + "/{}.xyz".format(ind))
