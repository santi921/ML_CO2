import os
import sys
import pybel

os.system("export KERAS_BACKEND=tensorflow")
from chemvae.vae_utils import VAEUtils
from helpers import merge_dir_and_data

from utils.selfies_util import compare_equality


def vae(dir="../data/xyz/"):
    os.system(
        "export KERAS_BACKEND=tensorflow"
    )  # you might need to run this command commandline
    vae = VAEUtils(directory="../data/models/zinc_properties")

    ret = []
    homo = []
    homo1 = []
    diff = []
    names = []
    smiles = []

    print("..........converting xyz to smiles.......")
    dir_fl_names, list_to_sort = merge_dir_and_data(dir=dir)
    smiles = []
    rm_ind = []

    for j, i in enumerate(dir_fl_names):
        try:
            mol = next(pybel.readfile("xyz", dir + i))
            smi = mol.write(format="smi")
            smiles.append(smi.split()[0].strip())
            sys.stdout.write("\r %s / " % j + str(len(dir_fl_names)))
            sys.stdout.flush()
        except:
            rm_ind.append(j)

    rm_ind.reverse()
    [dir_fl_names.pop(i) for i in rm_ind]
    [list_to_sort.pop(i) for i in rm_ind]
    print("\n\nsmiles length: " + str(len(smiles)) + "\n\n")
    # ---------------------------------------------------------------------------
    for tmp, item in enumerate(smiles):
        try:
            mol = next(
                pybel.readfile("xyz", dir + list_to_sort[tmp].split(":")[0] + ".xyz")
            )
            mol_shift = next(
                pybel.readfile(
                    "xyz", dir + list_to_sort[tmp + 1].split(":")[0] + ".xyz"
                )
            )
            smi = mol.write(format="smi").split()[0].strip()
            smi_shift = mol_shift.write(format="smi").split()[0].strip()
            X_1 = vae.smiles_to_hot(str(item), canonize_smiles=True)
            X_shift = vae.smiles_to_hot(str(smiles[tmp + 1]), canonize_smiles=True)
            Z_1 = vae.encode(X_1).tolist()
            Z_shift = vae.encode(X_shift).tolist()
            if item == smi:
                ret.append(Z_1)
                names.append(item)
                homo_temp = float(list_to_sort[tmp].split(":")[1])
                homo1_temp = float(list_to_sort[tmp].split(":")[2])
                homo.append(homo_temp)
                homo1.append(homo1_temp)
                diff.append(homo_temp - homo1_temp)
            else:
                try:
                    if item == smi_shift:
                        ret.append(Z_shift)
                        names.append(item)
                        homo_temp = float(list_to_sort[tmp + 1].split(":")[1])
                        homo1_temp = float(list_to_sort[tmp + 1].split(":")[2])
                        homo.append(homo_temp)
                        homo1.append(homo1_temp)
                        diff.append(homo_temp - homo1_temp)
                except:
                    # print(list_to_sort[tmp].split(":")[0], item[0:-4])
                    pass
            sys.stdout.write("\r %s /" % tmp + str(len(dir_fl_names)))
            sys.stdout.flush()
        except:
            pass
    print(len(diff))
    return names, ret, homo, homo1, diff
