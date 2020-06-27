import os
import sys
import time

import numpy as np
import pybel
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem import AllChem
from rdkit.Chem import SDMolSupplier


def morgan(bit_length=256, dir="../data/sdf/DB/", bit=True):
    ls_dir = "ls " + str(dir) + " | sort"
    temp = os.popen(ls_dir).read()
    temp = str(temp).split()
    morgan = []
    names = []

    for tmp, item in enumerate(temp):
        suppl = SDMolSupplier(dir + item)

        if (bit == True):
            try:

                fp_bit = AllChem.GetMorganFingerprintAsBitVect(suppl[0], int(2), nBits=int(bit_length))
                morgan.append(fp_bit)
                names.append(item)

                sys.stdout.write("\r %s / " % tmp + str(len(temp)))
                sys.stdout.flush()
            except:
                pass
        else:
            try:
                fp = AllChem.GetMorganFingerprint(suppl[0], int(2))
                morgan.append(fp)
                names.append(item)

                sys.stdout.write("\r %s / " % tmp + str(len(temp)))
                sys.stdout.flush()

            except:
                print("error")
                pass

    print(len(morgan))
    morgan = np.array(morgan)
    return names, morgan


def rdk(dir="../data/sdf/DB/"):
    ls_dir = "ls " + str(dir) + " | sort"
    temp = os.popen(ls_dir).read()
    temp = str(temp).split()
    rdk = []
    names = []

    for tmp,item in enumerate(temp):
        suppl = SDMolSupplier(dir + item)
        try:
            fp_rdk = AllChem.RDKFingerprint(suppl[0], maxPath=2)
            rdk.append(fp_rdk)
            names.append(item)

            sys.stdout.write("\r %s / " % tmp + str(len(temp)))
            sys.stdout.flush()

        except:
            print("error")
            pass
    rdk = np.array(rdk)

    print(len(rdk))
    return names, rdk


def aval(dir="../data/sdf/DB/", bit_length=128):
    ls_dir = "ls " + str(dir) + " | sort"
    temp = os.popen(ls_dir).read()
    temp = str(temp).split()
    avalon = []
    names = []

    for tmp,item in enumerate(temp):
        suppl = SDMolSupplier(dir + item)
        try:
            fp_aval = pyAvalonTools.GetAvalonFP(suppl[0], bit_length)
            avalon.append(fp_aval)
            names.append(item)

            sys.stdout.write("\r %s /" % tmp + str(len(temp)))
            sys.stdout.flush()

        except:
            print("error")
            pass

    print(len(avalon))
    avalon = np.array(avalon)
    return names, avalon


def layer(dir="../data/sdf/DB/"):
    ls_dir = "ls " + str(dir) + " | sort"
    temp = os.popen(ls_dir).read()
    temp = str(temp).split()
    layer = []
    names = []

    for tmp, item in enumerate(temp):
        try:
            suppl = SDMolSupplier( dir + item)
            fp_layer = AllChem.LayeredFingerprint(suppl[0])
            layer.append(fp_layer)
            names.append(item)

            sys.stdout.write("\r %s /" % tmp + str(len(temp)))
            sys.stdout.flush()
        except:
            print("error")
            pass
    layer = np.array(layer)
    return names, layer


# this script converts xyz files to rdkit/openbabel-readable sdf
# Input: not implemented here but a directory with xyz files

# Input: directory of xyz files
# Output: None, saves SDF type files to and sdf folder for later
def xyz_to_sdf(dir="../data/xyz/DB/"):
    dir_str = "ls " + str(dir) + " | sort "
    temp = os.popen(dir_str).read()
    temp = str(temp).split()
    for j,i in enumerate(temp):
        try:
            i = i.replace("(","\(").replace(")","\)").replace("[","\[").replace("]","\]")
            file_str = "python ./xyz2mol/xyz2mol.py " + dir + i + " -o sdf > ../data/sdf/" + i[0:-4] + ".sdf"
            os.system(file_str)

            sys.stdout.write("\r %s / " % j + str(len(temp)))
            sys.stdout.flush()

      
        except:
            print("not working")



# Input: directory of xyz files
# Output: returns a list of smiles strings
def xyz_to_smiles(dir="../data/xyz/DB2/"):
    dir_str = "ls " + str(dir) + " | sort -d "
    temp = os.popen(dir_str).read()
    temp = str(temp).split()
    ret_list = []
    names = []
    for j, i in enumerate(temp):
        try:
            mol = next(pybel.readfile("xyz", dir + i))
            smi = mol.write(format="smi")
            ret_list.append(smi.split()[0].strip())
            names.append(i)
            sys.stdout.write("\r %s / " % j + str(len(temp)))
            sys.stdout.flush()

        except:
            pass
    # print(ret_list[0:4])
    return names, ret_list


#splits a single large smi file into many smaller ones
def smi_split(file=""):
    for i, mol in enumerate(pybel.readfile("smi", "zz.smi")):
        temp = str(i)
        mol.write("smi", "%s.smi" % temp)


# converts a log files of smiles strings to a pandas db of xyz
def smiles_to_xyz( dir="../data/smiles/ZZ/"):
    dir_str = "ls " + str(dir) + " | sort"
    temp = os.popen(dir_str).read()
    temp = str(temp).split()
    t = []
    for i in temp:
        t1 = time.time()
        print("Current file: " + i)
        mol = next(pybel.readfile("smi", dir + i))
        mol.make3D(forcefield='mmff94', steps=10)
        mol.localopt()
        t2 = time.time()
        print("Smi Optimization Complete in " + str(t2-t1) + "s")
        mol.write("xyz", "%s.xyz" % i)
        t.append(t2-t1)
        #mol.write("xyz", "%s.xyz" % temp)
    time_array = np.array(t)
    print("time average for computation: " + np.mean(time_array))

# Convert Daniel's second db to sdf
#xyz_to_sdf("../data/xyz/DB_2/bis-25/")
#xyz_to_sdf("../data/xyz/DB_2/bis-26/")
#xyz_to_sdf("../data/xyz/DB_2/bis-23/")
#xyz_to_sdf("../data/xyz/DB_2/mono/")
# xyz_to_sdf("../data/xyz/DB_2/tris/")
# xyz_to_sdf("../data/xyz/DB2/")

# Convert ZZ's immense db to sdf
# xyz_to_sdf("../data/xyz/ZZ/3/")
# xyz_to_sdf("../data/xyz/ZZ/2_dir/")
# xyz_to_sdf("../data/xyz/ZZ/4/")
# xyz_to_sdf("../data/xyz/ZZ/56/")
# xyz_to_sdf("../data/xyz/ZZ/78/")
# xyz_to_sdf("../data/xyz/ZZ/9/")
# xyz_to_sdf("../data/xyz/DB2/")
# xyz_to_smiles("../data/xyz/DB2/")
