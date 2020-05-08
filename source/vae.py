from chemvae.vae_utils import VAEUtils
from chemvae import mol_utils as mu
import numpy as np
import rdkit.Chem as rdkit_util
from helpers import xyz_to_smiles


def vae():

    fid = 0
    temp = xyz_to_smiles()
    # here we used the nn derived from the zinc dataset
    vae = VAEUtils(directory='../data/models/zinc_properties')

    for j,i in enumerate(temp):
        print(j)

        if (rdkit_util.MolFromSmiles(i) != None):

            X_1 = vae.smiles_to_hot(i, canonize_smiles=True)
            z_1 = vae.encode(X_1)
            X_r = vae.decode(z_1)


            print('{:20s} : {}'.format('Input', i))
            reconstruct = vae.hot_to_smiles(X_r, strip=True)[0]
            print('{:20s} : {}'.format('Reconstruction', reconstruct))

            if (i == reconstruct):
                fid += 1

            print('{:20s} : {} with norm {:.3f}'.format('Z representation', z_1.shape, np.linalg.norm(z_1)))
    print("VAE recovery fidelity: " + str(fid / float(len(temp))))

vae()