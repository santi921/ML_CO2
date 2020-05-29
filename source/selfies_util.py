import numpy as np
from helpers import xyz_to_smiles
from selfies import encoder, decoder, selfies_alphabet

# worked in python 3
def selfies(dir="../data/xyz/"):

    temp = xyz_to_smiles(dir)
    ret = []
    names = []

    for i in temp:
        try:
            selfies_temp = encoder(i)
            names.append(i)
            ret.append(selfies_temp)
        except:
            print("not encoded")


    ret = np.array(ret)
    return names, ret

