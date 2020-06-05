import sys

import numpy as np
from helpers import xyz_to_smiles
from selfies import encoder


# worked in python 3
def selfies(dir="../data/xyz/DB/"):
    ret = []
    names = []

    print("..........converting xyz to smiles.......")
    smil = xyz_to_smiles(dir)

    print("complete")

    print("files to describe: " + str(len(ret)) )

    for tmp, i in enumerate(smil):

        try:
            selfies_temp = encoder(i)
            names.append(i)
            ret.append(selfies_temp)
        except:
            print("not encoded")

        sys.stdout.write("\r/ " % tmp + str(len(smil)))
        sys.stdout.flush()

    ret = np.array(ret)
    return names, ret

