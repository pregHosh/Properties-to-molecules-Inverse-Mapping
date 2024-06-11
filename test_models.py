import numpy as np
from matplotlib import pyplot as plt
from molecular_generation_utils import *
from invert_CM import *
import torch
from Model import Multi_VAE
from torch.distributions.categorical import Categorical
from tqdm import tqdm
import copy
from ase.visualize import view

reproduce_paper = False

if reproduce_paper:
    paper_path = '_paper'
else:
    paper_path = ''


properties = torch.load('./dati/data/properties_total.pt'.format(paper_path))
p_means = torch.load("./dati/data/properties_means.pt".format(paper_path))
p_stds = torch.load("./dati/data/properties_stds.pt".format(paper_path))
norm_props = (properties - p_means)/p_stds

properties_list =  ['eAT', 'eMBD', 'eXX', 'mPOL', 'eNN', 'eNE', 'eEE', 'eKIN', 'DIP', 'HLgap', 'HOMO_0', 'LUMO_0', 'HOMO_1', 'LUMO_1', 'HOMO_2', 'LUMO_2', 'dimension']
p_arr = np.array(properties_list)

PATH = "./our_models/checkpoints_128/last.ckpt"
# "./models_saved/masked/epoch=2597-step=145487.ckpt"
#
#'./models_saved/masked/epoch=2597-step=145487.ckpt'

modello = Multi_VAE.load_from_checkpoint(
    PATH,
    map_location=torch.device("cpu"),
    structures_dim=len(torch.load("./dati/data/data_val/CMs.pt")[0, :]),
    properties_dim=len(torch.load("./dati/data/data_val/properties.pt")[0, :]),
    latent_size=21,
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
    norm_props,
    min_components = 91, #91
    max_components = 92 #92
    )



# recall that here the target value is the normalized one

generated = start_generation(
    modello,
    {
        'mPOL': 2.,
        'eMBD': 2.
    },
    p_arr,
    177,
    int(5e3),
    gm.means_,
    gm.covariances_,
    cm_diff = 40,
    deltaz = 6,
    check_new_comp = False,
    verbose = False
)




# load the coulomb matrices in the dataset
CMs = torch.load('./dati/data/CMs_total.pt'.format(paper_path))


CM = generated[4]

from invert_CM import *
from CM_preparation import *

distance_mat, master_vec2 = recover_distance_mat(
    CM
)


print("Master Vector:", master_vec2)

# Recover the Cartesian coordinates
cartesian = cartesian_recovery(distance_mat)
# truncate last two columns
cartesian = cartesian[:,0:3]
# remove imaginary part
cartesian = np.real(cartesian)
print("Cartesian coordinates:", cartesian)

# Create the recovered ASE Atoms object
rec_mol = Atoms(symbols=master_vec2, positions=cartesian)
print("Recovered mol:")
#view(rec_mol)
# Get the RMSD between the original and recovered molecule
#
#
