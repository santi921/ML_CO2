import rdkit.Chem as rdkit_util
from helpers import xyz_to_smiles
from chemvae.vae_utils import *

#add this to execution: export KERAS_BACKEND=tensorflow

def vae(dir="../data/xyz/"):
    print("something")
    temp = xyz_to_smiles(dir)
    print("import")
    # here we used the nn derived from the zinc dataset
    vae = VAEUtils(directory='../data/models/zinc_properties')
    mat_vae = []
    names = []

    for j,i in enumerate(temp):
        print("enters loop")
        if (rdkit_util.MolFromSmiles(i) != None):
            print("enter ")
            X_1 = vae.smiles_to_hot(i, canonize_smiles=True)
            try:
                Z_1 = vae.encode(X_1)
                mat_vae.append(Z_1)
                names.append(i)
                break
            except:
                print("not correctly encoded")
    mat_vae = np.numpy(mat_vae)
    return names, mat_vae

