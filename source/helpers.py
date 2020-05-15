import os
import numpy as np
import pybel
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem import AllChem
from rdkit.Chem import SDMolSupplier


def morgan(bit_length=256, dir="../data/sdf/", bit=True):
    dir = "ls " + dir
    temp = os.popen(dir).read()
    temp = str(temp).split()
    morgan = []

    for item in temp:
        suppl = SDMolSupplier("../data/sdf/" + item)

        if (bit == True):
            fp_bit = AllChem.GetMorganFingerprintAsBitVect(suppl[0], 2, nBits=bit_length)
            morgan.append(fp_bit)
        else:
            fp = AllChem.GetMorganFingerprint(suppl[0], 2)
            morgan.append(fp)

    morgan = np.array(morgan)
    return morgan


def rdk(dir="../data/sdf/"):
    dir = "ls " + dir
    temp = os.popen(dir).read()
    temp = str(temp).split()
    rdk = []

    for item in temp:
        suppl = SDMolSupplier("../data/sdf/" + item)
        fp_rdk = AllChem.RDKFingerprint(suppl[0], maxPath=2)
        rdk.append(fp_rdk)
    rdk = np.array(rdk)
    return rdk


def aval(dir="../data/sdf/", bit_length=128):
    dir = "ls " + dir
    temp = os.popen(dir).read()
    temp = str(temp).split()
    avalon = []

    for item in temp:
        suppl = SDMolSupplier("../data/sdf/" + item)
        fp_aval = pyAvalonTools.GetAvalonFP(suppl[0], bit_length)
        avalon.append(fp_aval)

    avalon = np.array(avalon)
    return avalon


def layer(dir="../data/sdf/"):
    dir = "ls " + dir
    temp = os.popen(dir).read()
    temp = str(temp).split()
    layer = []

    for item in temp:
        suppl = SDMolSupplier("../data/sdf/" + item)
        fp_layer = AllChem.LayeredFingerprint(suppl[0])
        layer.append(fp_layer)
    layer = np.array(layer)
    return layer


# this script converts xyz files to rdkit/openbabel-readable sdf
# Input: not implemented here but a directory with xyz files

# Input: directory of xyz files
# Output: None, saves SDF type files to and sdf folder for later
def xyz_to_sdf(dir="../data/xyz/"):
    dir_str = "ls " + str(dir)
    temp = os.popen(dir_str).read()
    temp = str(temp).split()
    for i in temp:
        file_str = "python ./xyz2mol/xyz2mol.py " + dir + i + " -o sdf > ./sdf/" + i[0:-4] + ".sdf"
        os.system(file_str)



# Input: directory of xyz files
# Output: returns a list of smiles strings
def xyz_to_smiles(dir="../data/xyz/"):
    dir_str = "ls " + str(dir)
    temp = os.popen(dir_str).read()
    temp = str(temp).split()
    ret_list = []

    for i in temp:
        mol = next(pybel.readfile("xyz", dir + i))
        smi = mol.write(format="smi")
        ret_list.append(smi.split()[0].strip())

    return ret_list
