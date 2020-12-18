import sys

import numpy as np
from helpers import xyz_to_smiles
from selfies import encoder


# worked in python 3
def selfies(dir="../data/xyz/DB/"):
    print(dir)
    ret = []
    names = []

    print("..........converting xyz to smiles.......")
    names, smil = xyz_to_smiles(dir)
    print("\n complete")

    print("files to describe: " + str(len(smil)))

    for tmp, i in enumerate(smil):

        try:
            selfies_temp = encoder(i)
            names.append(names[tmp])
            ret.append(selfies_temp)
        except:
            print("not encoded")

        sys.stdout.write("\r %s " % tmp + str(len(smil)))
        sys.stdout.flush()

    ret = np.array(ret)
    return names, ret

