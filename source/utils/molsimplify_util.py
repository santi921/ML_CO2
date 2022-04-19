import sys
import numpy as np
from source.utils.helpers import merge_dir_and_data
from molSimplify.Classes.mol3D import *
from molSimplify.Informatics.autocorrelation import *
from molSimplify.Informatics.graph_analyze import *
from molSimplify.Informatics.misc_descriptors import *


def full_autocorr(dir="../data/xyz/"):
    res = []
    names = []
    homo = []
    homo1 = []
    diff = []

    dir_fl_names, list_to_sort = merge_dir_and_data(dir=dir)
    # ---------------------------------------------------------------------------
    for tmp, item in enumerate(dir_fl_names):
        try:
            this_mol = mol3D()  # mol3D instance
            this_mol.readfromxyz(dir + item)  # read geo
            results_auto = generate_full_complex_autocorrelations(
                this_mol, depth=3, loud=True
            )["results"]

            if item[0:-4] == list_to_sort[tmp].split(":")[0]:
                res.append(results_auto)
                names.append(item)
                homo_temp = float(list_to_sort[tmp].split(":")[1])
                homo1_temp = float(list_to_sort[tmp].split(":")[2])
                homo.append(homo_temp)
                homo1.append(homo1_temp)
                diff.append(homo_temp - homo1_temp)
            else:
                try:
                    if item[0:-4] == list_to_sort[tmp + 1].split(":")[0]:
                        res.append(results_auto)
                        names.append(item)
                        homo_temp = float(list_to_sort[tmp + 1].split(":")[1])
                        homo1_temp = float(list_to_sort[tmp + 1].split(":")[2])
                        homo.append(homo_temp)
                        homo1.append(homo1_temp)
                        diff.append(homo_temp - homo1_temp)
                except:
                    pass
            sys.stdout.write("\r %s /" % tmp + str(len(dir_fl_names)))
            sys.stdout.flush()
        except:
            pass
    res = np.array(res)
    return names, res, homo, homo1, diff


def deltametrics_gen(dir="../data/xyz/"):
    results_delta = []
    names = []
    homo = []
    homo1 = []
    diff = []

    dir_fl_names, list_to_sort = merge_dir_and_data(dir=dir)
    # ---------------------------------------------------------------------------
    for tmp, item in enumerate(dir_fl_names):
        try:
            this_mol = mol3D()  # mol3D instance
            this_mol.readfromxyz(item)  # read geo
            results_delta_temp = deltametric(this_mol)["results"]

            if item[0:-4] == list_to_sort[tmp].split(":")[0]:
                results_delta.append(results_delta_temp)
                names.append(item)
                homo_temp = float(list_to_sort[tmp].split(":")[1])
                homo1_temp = float(list_to_sort[tmp].split(":")[2])
                homo.append(homo_temp)
                homo1.append(homo1_temp)
                diff.append(homo_temp - homo1_temp)
            else:
                try:
                    if item[0:-4] == list_to_sort[tmp + 1].split(":")[0]:
                        results_delta.append(results_delta_temp)
                        names.append(item)
                        homo_temp = float(list_to_sort[tmp + 1].split(":")[1])
                        homo1_temp = float(list_to_sort[tmp + 1].split(":")[2])
                        homo.append(homo_temp)
                        homo1.append(homo1_temp)
                        diff.append(homo_temp - homo1_temp)
                except:
                    pass
            sys.stdout.write("\r %s /" % tmp + str(len(dir_fl_names)))
            sys.stdout.flush()
        except:
            pass
    ret = np.array(results_delta)
    return names, ret, homo, homo1, diff


def metal_deltametrics(dir="../data/xyz/"):
    res = []
    names = []
    homo = []
    homo1 = []
    diff = []

    dir_fl_names, list_to_sort = merge_dir_and_data(dir=dir)
    # ---------------------------------------------------------------------------
    for tmp, item in enumerate(dir_fl_names):
        try:
            this_mol = mol3D()  # mol3D instance
            this_mol.readfromxyz(dir + item)  # read geo
            results_delta = generate_metal_deltametrics(this_mol, loud="something")[
                "results"
            ]

            if item[0:-4] == list_to_sort[tmp].split(":")[0]:
                res.append(results_delta)
                names.append(item)
                homo_temp = float(list_to_sort[tmp].split(":")[1])
                homo1_temp = float(list_to_sort[tmp].split(":")[2])
                homo.append(homo_temp)
                homo1.append(homo1_temp)
                diff.append(homo_temp - homo1_temp)
            else:
                try:
                    if item[0:-4] == list_to_sort[tmp + 1].split(":")[0]:
                        res.append(results_delta)
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
    res = np.array(res)
    return names, res, homo, homo1, diff
