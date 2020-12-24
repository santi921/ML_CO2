import sys
import pybel
import numpy as np
from selfies import encoder
from helpers import merge_dir_and_data

# worked in python 3
def selfies(dir="../data/xyz/DB/"):
    ret = []
    homo = []
    homo1 = []
    diff = []
    names = []

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
    #---------------------------------------------------------------------------
    for tmp, item in enumerate(smiles):
        try:
            selfies_temp = encoder(item)
            selfies_temp_shift = encoder(smiles[tmp+1])
            selfies_temp_shift_min = encoder(smiles[tmp-1])

            mol = next(pybel.readfile("xyz", dir + list_to_sort[tmp].split(":")[0] + ".xyz"))
            mol_shift = next(pybel.readfile("xyz", dir + list_to_sort[tmp+1].split(":")[0] + ".xyz"))
            mol_shift_min = next(pybel.readfile("xyz", dir + list_to_sort[tmp-1].split(":")[0] + ".xyz"))
            smi = mol.write(format="smi").split()[0].strip()
            smi_shift = mol_shift.write(format="smi").split()[0].strip()
            smi_shift_min = mol_shift_min.write(format="smi").split()[0].strip()

            if (item == smi):
                ret.append(selfies_temp)
                names.append(item)
                homo_temp = float(list_to_sort[tmp].split(":")[1])
                homo1_temp = float(list_to_sort[tmp].split(":")[2])
                homo.append(homo_temp)
                homo1.append(homo1_temp)
                diff.append(homo_temp - homo1_temp)
            else:
                try:
                    if (item == smi_shift):
                        ret.append(selfies_temp_shift)
                        names.append(item)
                        homo_temp = float(list_to_sort[tmp+1].split(":")[1])
                        homo1_temp = float(list_to_sort[tmp+1].split(":")[2])
                        homo.append(homo_temp)
                        homo1.append(homo1_temp)
                        diff.append(homo_temp - homo1_temp)
                    else:
                        pass

                except:
                    try:
                        if ("item" == smi_shift_min):
                            ret.append(selfies_temp_shift_min)
                            names.append(item)
                            homo_temp = float(list_to_sort[tmp + 1].split(":")[1])
                            homo1_temp = float(list_to_sort[tmp + 1].split(":")[2])
                            homo.append(homo_temp)
                            homo1.append(homo1_temp)
                            diff.append(homo_temp - homo1_temp)
                    except:
                        print("failed to match")
                        pass
            sys.stdout.write("\r %s /" % tmp + str(len(dir_fl_names)))
            sys.stdout.flush()
        except:
            pass
    print(len(names))
    print(len(ret))
    print(len(homo))
    print(len(homo1))
    print(len(diff))

    ret = np.array(ret)
    return names, ret, homo, homo1, diff

