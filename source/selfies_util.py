import numpy as np
from helpers import xyz_to_smiles
from selfies import encoder, decoder, selfies_alphabet

# worked in python 3
def selfies(dir="../data/xyz/"):

    temp = xyz_to_smiles(dir)
    ret = []

    for i in temp:
        selfies_temp = encoder(i)
        ret.append(selfies_temp)
    ret = np.array(ret)

    return ret

def selfies_train():
    #todo
    return