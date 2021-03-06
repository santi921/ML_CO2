import os
import sys
import time
import pandas as pd
import numpy as np
import pybel

from rdkit.Avalon import pyAvalonTools
from rdkit.Chem import AllChem
from rdkit.Chem import SDMolSupplier

def merge_dir_and_data(dir = "DB3"):
    # all files in the directory
    ls_dir = "ls " + str(dir) + " | sort"
    dir_fl_names = os.popen(ls_dir).read()
    dir_fl_names = str(dir_fl_names).split()
    dir_fl_names.sort()

    # all energies in the database
    list_to_sort = []
    remove_ind = []
    added_seg = "BQ"
    with open("../data/DATA_DB3") as fp:
        line = fp.readline()
        while line:
            if(line.split()[0] == "----"):
                added_seg = line.split()[1]
            else:
                list_to_sort.append(added_seg + "_" + line[0:-2])
            line = fp.readline()

    list_to_sort.sort()
    only_names=[i[0:-2].split(":")[0] for i in list_to_sort]

    # find the values which are in the dir and database
    files_relevant = []
    for i, file_name in enumerate(dir_fl_names):
        try:
            ind_find = only_names.index(file_name[0:-4])
            sys.stdout.write("\r %s /" % i + str(len(dir_fl_names)))
            sys.stdout.flush()
            files_relevant.append(file_name)
        except:
            ind_empty = dir_fl_names.index(file_name)
            remove_ind.append(ind_empty)
            pass

    remove_ind.reverse()
    [dir_fl_names.pop(i) for i in remove_ind]
    remove_ind_2 = []

    for ind, files in enumerate(only_names):
        try:
            try:
                ind_find = dir_fl_names.index(files+".sdf")
            except:
                ind_find = dir_fl_names.index(files + ".xyz")
            sys.stdout.write("\r %s /" % ind + str(len(only_names)))
            sys.stdout.flush()
        except:
            ind_empty = only_names.index(files)
            remove_ind_2.append(ind_empty)
            pass

    remove_ind_2.reverse()
    [only_names.pop(i) for i in remove_ind_2]
    [list_to_sort.pop(i) for i in remove_ind_2]
    # go back and remove files in directory that we don't have energies
    return dir_fl_names, list_to_sort

def morgan(bit_length=256, dir="../data/sdf/DB3/", bit=True):

    morgan = []
    names = []
    homo = []
    homo1 = []
    diff = []

    dir_fl_names, list_to_sort = merge_dir_and_data(dir = dir)
    #---------------------------------------------------------------------------
    for tmp, item in enumerate(dir_fl_names):
        try:
            suppl = SDMolSupplier(dir + item)

            if (bit == True):
                try:
                    fp = AllChem.GetMorganFingerprintAsBitVect(suppl[0], int(2), nBits=int(bit_length))
                except:
                    pass
            else:
                try:
                    fp = AllChem.GetMorganFingerprint(suppl[0], int(2))
                except:
                    print("error")
                    pass

            if (item[0:-4] == list_to_sort[tmp].split(":")[0] ):
                morgan.append(fp)
                names.append(item)
                homo_temp = float(list_to_sort[tmp].split(":")[1])
                homo1_temp = float(list_to_sort[tmp].split(":")[2])
                homo.append(homo_temp)
                homo1.append(homo1_temp)
                diff.append(homo_temp - homo1_temp)
            else:
                try:
                    if (item[0:-4] == list_to_sort[tmp+1].split(":")[0]):
                        morgan.append(fp)
                        names.append(item)
                        homo_temp = float(list_to_sort[tmp+1].split(":")[1])
                        homo1_temp = float(list_to_sort[tmp+1].split(":")[2])
                        homo.append(homo_temp)
                        homo1.append(homo1_temp)
                        diff.append(homo_temp - homo1_temp)
                except:
                    print(list_to_sort[tmp].split(":")[0], item[0:-4])
                    pass
            sys.stdout.write("\r %s /" % tmp + str(len(dir_fl_names)))
            sys.stdout.flush()
        except:
            pass
    morgan = np.array(morgan)
    return names, morgan, homo, homo1, diff

def rdk(dir="../data/sdf/DB/"):

    rdk = []
    names = []
    homo = []
    homo1 = []
    diff = []

    dir_fl_names, list_to_sort = merge_dir_and_data(dir = dir)
    #---------------------------------------------------------------------------
    for tmp, item in enumerate(dir_fl_names):
        try:
            suppl = SDMolSupplier(dir + item)
            fp_rdk = AllChem.RDKFingerprint(suppl[0], maxPath=2)

            if (item[0:-4] == list_to_sort[tmp].split(":")[0] ):
                rdk.append(fp_rdk)
                names.append(item)
                homo_temp = float(list_to_sort[tmp].split(":")[1])
                homo1_temp = float(list_to_sort[tmp].split(":")[2])
                homo.append(homo_temp)
                homo1.append(homo1_temp)
                diff.append(homo_temp - homo1_temp)
            else:
                try:
                    if (item[0:-4] == list_to_sort[tmp+1].split(":")[0]):
                        rdk.append(fp_rdk)
                        names.append(item)
                        homo_temp = float(list_to_sort[tmp+1].split(":")[1])
                        homo1_temp = float(list_to_sort[tmp+1].split(":")[2])
                        homo.append(homo_temp)
                        homo1.append(homo1_temp)
                        diff.append(homo_temp - homo1_temp)
                except:
                    print(list_to_sort[tmp].split(":")[0], item[0:-4])
                    pass
            sys.stdout.write("\r %s /" % tmp + str(len(dir_fl_names)))
            sys.stdout.flush()
        except:
            pass
    rdk = np.array(rdk)
    return names, rdk, homo, homo1, diff

def aval(dir="../data/sdf/DB/", bit_length=256):
    aval = []
    names = []
    homo = []
    homo1 = []
    diff = []
    dir_fl_names, list_to_sort = merge_dir_and_data(dir = dir)
    #---------------------------------------------------------------------------
    for tmp, item in enumerate(dir_fl_names):
        try:
            suppl = SDMolSupplier(dir + item)
            fp_aval = pyAvalonTools.GetAvalonFP(suppl[0], bit_length)

            if (item[0:-4] == list_to_sort[tmp].split(":")[0] ):
                aval.append(fp_aval)
                names.append(item)
                homo_temp = float(list_to_sort[tmp].split(":")[1])
                homo1_temp = float(list_to_sort[tmp].split(":")[2])
                homo.append(homo_temp)
                homo1.append(homo1_temp)
                diff.append(homo_temp - homo1_temp)
            else:
                try:
                    if (item[0:-4] == list_to_sort[tmp+1].split(":")[0]):
                        aval.append(fp_aval)
                        names.append(item)
                        homo_temp = float(list_to_sort[tmp+1].split(":")[1])
                        homo1_temp = float(list_to_sort[tmp+1].split(":")[2])
                        homo.append(homo_temp)
                        homo1.append(homo1_temp)
                        diff.append(homo_temp - homo1_temp)
                except:
                    print(list_to_sort[tmp].split(":")[0], item[0:-4])
                    pass
            sys.stdout.write("\r %s /" % tmp + str(len(dir_fl_names)))
            sys.stdout.flush()
        except:
            pass
    aval = np.array(layer)
    return names, aval, homo, homo1, diff

def layer(dir="../data/sdf/DB/"):

    layer = []
    names = []
    homo = []
    homo1 = []
    diff = []
    dir_fl_names, list_to_sort = merge_dir_and_data(dir = dir)
    #---------------------------------------------------------------------------
    for tmp, item in enumerate(dir_fl_names):
        try:
            suppl = SDMolSupplier(dir + item)
            fp_layer = AllChem.LayeredFingerprint(suppl[0])

            if (item[0:-4] == list_to_sort[tmp].split(":")[0] ):
                layer.append(fp_layer)
                names.append(item)
                homo_temp = float(list_to_sort[tmp].split(":")[1])
                homo1_temp = float(list_to_sort[tmp].split(":")[2])
                homo.append(homo_temp)
                homo1.append(homo1_temp)
                diff.append(homo_temp - homo1_temp)
            else:
                try:
                    if (item[0:-4] == list_to_sort[tmp+1].split(":")[0]):
                        layer.append(fp_layer)
                        names.append(item)
                        homo_temp = float(list_to_sort[tmp+1].split(":")[1])
                        homo1_temp = float(list_to_sort[tmp+1].split(":")[2])
                        homo.append(homo_temp)
                        homo1.append(homo1_temp)
                        diff.append(homo_temp - homo1_temp)
                except:
                    print(list_to_sort[tmp].split(":")[0], item[0:-4])
                    pass
            sys.stdout.write("\r %s /" % tmp + str(len(dir_fl_names)))
            sys.stdout.flush()
        except:
            pass
    layer = np.array(layer)
    return names, layer, homo, homo1, diff

# TODO: multiprocess all of this
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

def process_input_DB2(dir="DB2", desc="rdkit"):
    try:
        str = "../data/desc/" + dir + "/desc_calc_DB2_" + desc + ".pkl"
        print(str)
        df = pd.read_pickle(str)
        pkl = 1

    except:
        str = "../data/desc/" + dir + "/desc_calc_DB2_" + desc + ".h5"
        df = pd.read_hdf(str)
        pkl = 0

    df["HOMO"] = ""
    df["HOMO-1"] = ""
    df["diff"] = ""
    print(df.head())
    list_to_sort = []
    with open("../data/benzoquinone_DB/DATA_copy") as fp:
        line = fp.readline()
        while line:
            list_to_sort.append(line[0:-2])
            line = fp.readline()
    list_to_sort.sort()

    for i in range(df.copy().shape[0]):
        for j in list_to_sort:

            # print(df["name"].iloc[i][:-4])
            # print(j.split(":")[0])
            if (desc == "auto"):
                if (df["name"].iloc[i].split("/")[-1][:-4] in j.split(";")[0]):
                    temp1 = float(j.split(":")[1])
                    temp2 = float(j.split(":")[2])
                    df["HOMO"].loc[i] = float(j.split(":")[1])
                    df["HOMO-1"].loc[i] = float(j.split(":")[2])
                    df["diff"].loc[i] = temp2 - temp1
                    print(temp2 - temp1)

            if (df["name"].iloc[i][:-4] in j.split(";")[0]):
                temp1 = float(j.split(":")[1])
                temp2 = float(j.split(":")[2])

                print(j)

                df["HOMO"].loc[i] = float(j.split(":")[1])
                df["HOMO-1"].loc[i] = float(j.split(":")[2])
                df["diff"].loc[i] = temp2 - temp1
    if (pkl == 0):
        df.to_hdf(str, key="df", mode='a')
    else:
        df.to_pickle(str)

# TODO : this, once ZZ finishes getting his data
def process_input_ZZ(dir="ZZ", desc="vae"):
    # todo: process ZZ's
    return 0

def process_input_DB3(dir="DB3", desc="rdkit"):
    print(desc)
    try:
        str = "../data/desc/" + dir + "/desc_calc_"+ dir +"_" + desc + ".h5"
        df = pd.read_hdf(str)
        pkl = 0
    except:
        str = "../data/desc/" + dir + "/desc_calc_"+ dir +"_" + desc + ".pkl"
        df = pd.read_pickle(str)
        pkl = 1

    print(df.head())
    df["HOMO"] = 0
    df["HOMO-1"] = 0
    df["diff"] = 0
    list_to_sort = []
    with open("../data/DATA_DB3") as fp:
        line = fp.readline()
        while line:
            list_to_sort.append(line[0:-2])
            line = fp.readline()
    list_to_sort.sort()
    print("Dimensions of df {0}".format(np.shape(df)))
    iters = df.copy().shape[0]

    for i in range(iters):

        shift = 0
        if ("tris" in df["name"].iloc[i]):
            shift = 4
        elif ("tetra" in df["name"].iloc[i]):
            shift = 5
        elif ("bis-23" in df["name"].iloc[i]):
            shift = 6
        elif ("bis-25" in df["name"].iloc[i]):
            shift = 6
        elif ("bis-26" in df["name"].iloc[i]):
            shift = 6
        elif ("mono" in df["name"].iloc[i]):
            shift = 4
        elif ("BQ" in df["name"].iloc[i]):
            shift = -1
        else:
            pass
        name = df["name"].iloc[i][:-4]
        shifted_name = df["name"].iloc[i].split("/")[-1][shift + 1:-4]

        for j in list_to_sort:
            if (name in j.split(";")[0] and j[0:2] != "--"):
                temp1 = float(j.split(":")[1])
                temp2 = float(j.split(":")[2])
                df["HOMO"].loc[i] = float(j.split(":")[1])
                df["HOMO-1"].loc[i] = float(j.split(":")[2])
                df["diff"].loc[i] = temp2 - temp1
                sys.stdout.write("\r {0} / {1}".format(i, np.shape(df)[0]))
                sys.stdout.flush()
                print(j.split(":")[0])

                list_to_sort.remove(j)
                break

            else:
                if(shifted_name in j.split(":")[0]):
                    sys.stdout.write("\r {0} / {1}".format(i, np.shape(df)[0]))
                    sys.stdout.flush()

                    temp1 = float(j.split(":")[1])
                    temp2 = float(j.split(":")[2])
                    df["HOMO"].loc[i] = float(j.split(":")[1])
                    df["HOMO-1"].loc[i] = float(j.split(":")[2])
                    df["diff"].loc[i] = temp2 - temp1
                    print(j.split(":")[0])
                    list_to_sort.remove(j)
                    break

    print(df.head())
    db_integrity(dir=dir, desc=desc)
    if (pkl == 0):
        df.to_hdf(str, key="df", mode='a')
    else:
        df.to_pickle(str)

def db_full_integrity(dir="DB3", desc="rdkit"):
    db_integrity(dir="DB3", desc="morg")
    db_integrity(dir="DB3", desc="rdkit")
    db_integrity(dir="DB3", desc="layer")
    db_integrity(dir="DB3", desc="aval")
    db_integrity(dir="DB3", desc="persist")
    db_integrity(dir="DB3", desc="vae")
    db_integrity(dir="DB3", desc="self")
    db_integrity(dir="DB3", desc="auto")

def db_integrity(dir="DB3", desc="rdkit"):
    print(">>>>>>Now testing database integrity")
    print(desc)
    try:
        str = "../data/desc/" + dir + "/desc_calc_"+ dir +"_" + desc + ".h5"
        df = pd.read_hdf(str)
        pkl = 0
    except:
        str = "../data/desc/" + dir + "/desc_calc_"+ dir +"_" + desc + ".pkl"
        df = pd.read_pickle(str)
        pkl = 1

    try:
        HOMO = df["HOMO"]
    except:
        print("it appears you haven't transcribed HOMO correctly")
    try:
        HOMO1 = df["HOMO-1"]
    except:
        print("it appears you haven't transcribed HOMO-1 correctly")
    try:
        HOMO1 = df["diff"]
    except:
        print("it appears you haven't transcribed diff correctly")

    # all energies in the database
    list_to_sort = []
    final_lines = []

    added_seg = "BQ"
    with open("../data/DATA_DB3") as fp:
        line = fp.readline()
        while line:
            if(line.split()[0] == "----"):
                added_seg = line.split()[1]
            else:
                list_to_sort.append(added_seg + "_" + line[0:-2])
            line = fp.readline()

    list_to_sort.sort()
    only_names= [i[0:-2].split(":")[0] for i in list_to_sort]
    df.sort_values("name")
    list_len = len(list_to_sort)
    count_bad = 0
    for i in range(1000):
        rand_index = np.random.randint(0, df.shape[0])

        str_search = df.iloc[rand_index]["name"][0:-4]
        diff_search = df.iloc[rand_index]["diff"]
        HOMO_search = df.iloc[rand_index]["HOMO"]

        try:
            index_of_search = only_names.index(str_search)

            name_only_names = list_to_sort[index_of_search]
            name_of_search = list_to_sort[index_of_search].split(":")[0]

            homo_db = float(list_to_sort[index_of_search].split(":")[1])
            diff_db = float(list_to_sort[index_of_search].split(":")[1]) - float(list_to_sort[index_of_search].split(":")[2])

            if(HOMO_search != homo_db):
                print(str_search, name_of_search, name_only_names)
                print(HOMO_search, homo_db)
                count_bad +=1
            else:
                pass
        except:
            pass
    print(df.shape)
    print(count_bad/1000)
    if(count_bad == 0):
        print(desc +" dataset looks good!")

