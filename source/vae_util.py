import os

import rdkit.Chem as rdkit_util
from helpers import xyz_to_smiles

os.system("export KERAS_BACKEND=tensorflow")
from chemvae.vae_utils import VAEUtils


def vae(dir="../data/xyz/"):

    os.system("export KERAS_BACKEND=tensorflow")
    mat_vae = []
    names = []
    vae = VAEUtils(directory='../data/models/zinc_properties')

    temp = xyz_to_smiles(dir)
    print("imported smiles")
    # here we used the nn derived from the zinc dataset

    for j,i in enumerate(temp):
        if (rdkit_util.MolFromSmiles(i) != None):
            try:
                X_1 = vae.smiles_to_hot(i, canonize_smiles=True)
                Z_1 = vae.encode(X_1)
                mat_vae.append(Z_1)
                names.append(i)
            except:
                print("not correctly encoded")
    return names, mat_vae

