import glob
import numpy as np
from molSimplify.Classes.mol3D import *
from molSimplify.Informatics.autocorrelation import *
from molSimplify.Informatics.graph_analyze import *
from molSimplify.Informatics.misc_descriptors import *


def full_autocorr(dir="../data/xyz/"):
    # begin parsing
    str_temp = dir + "/*.xyz"
    target_paths = sorted(glob.glob(str_temp))
    auto_corr_mat = []
    names = []

    for geos in target_paths:
        this_mol = mol3D()  # mol3D instance
        try:
            this_mol.readfromxyz(geos)  # read geo
            results_auto = generate_full_complex_autocorrelations(this_mol, depth=3, loud=True)
            auto_corr_mat.append(results_auto["results"])
            names.append(geos)
        except:
            pass
    auto_corr_mat = np.array(auto_corr_mat)
    return names, auto_corr_mat


def ligand_autocorr(dir="../data/xyz/"):
    # begin parsing
    str_temp = dir + "/*.xyz"
    target_paths = sorted(glob.glob(str_temp))
    auto_corr_mat = []
    names = []

    for geos in target_paths:
        this_mol = mol3D()  # mol3D instance
        try:
            this_mol.readfromxyz(geos)  # read geo
            # requires metal in the system
            results_auto = generate_all_ligand_autocorrelations(this_mol, depth=3, loud=True)
            auto_corr_mat.append(results_auto["results"])
            names.append(geos)
        except:
            pass

    auto_corr_mat = np.array(auto_corr_mat)
    return names, auto_corr_mat


def deltametrics_gen(dir="../data/xyz/"):
    # begin parsing
    str_temp = dir + "/*.xyz"
    target_paths = sorted(glob.glob(str_temp))

    results_delta = []
    names = []

    for geos in target_paths:
        this_mol = mol3D()  # mol3D instance
        #try:
        this_mol.readfromxyz(geos)  # read geo
        # if metals present
        results_delta_temp = deltametric(this_mol)

        results_delta.append(results_delta_temp["results"])
        names.append(geos)
        #except:
        #    pass

    results_delta = np.array(results_delta)
    return names, results_delta


def metal_deltametrics(dir="../data/xyz/"):
    # begin parsing
    str_temp = dir + "/*.xyz"
    target_paths = sorted(glob.glob(str_temp))
    results_delta_metal = []
    names = []

    for geos in target_paths:
        this_mol = mol3D()  # mol3D instance
        try:
            this_mol.readfromxyz(geos)  # read geo
            # if metals present
            results_delta_temp = generate_metal_deltametrics(this_mol, loud="something")
            results_delta_metal.append(results_delta_temp["results"])
            names.append(geos)
        except:
            pass

    results_delta_metal = np.array(results_delta_metal)
    return names, results_delta_metal
