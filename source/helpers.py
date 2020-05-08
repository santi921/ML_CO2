import os
import pybel


# this script converts xyz files to rdkit/openbabel-readable sdf
# Input: not implemented here but a directory with xyz files


# currently these just print, waiting for how to send back to train nn

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
