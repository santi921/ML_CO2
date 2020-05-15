import os
import numpy as np
import matplotlib.pyplot as plt

from rdkit.Chem import AllChem, Descriptors, DataStructs
from rdkit.Chem import Fingerprints, SDMolSupplier, RDKFingerprint
from rdkit.Avalon import pyAvalonTools
import rdkit.Chem as rdkit_util

from helpers import xyz_to_smiles


def rd_kit(dir_sdf = "../data/sdf/"):

    temp_str = "ls " + dir_sdf
    temp = os.popen(temp_str).read()
    temp = str(temp).split()

    bit_length = 256

    sim_matrix_morgan = []
    sim_matrix_rdk = []
    sim_matrix_aval = []
    sim_matrix_layer = []

    baseline = SDMolSupplier("../data/sdf/" + temp[0])

    baseline_morgan = AllChem.GetMorganFingerprintAsBitVect(baseline[0], 2, nBits=bit_length)
    baseline_rdk = AllChem.RDKFingerprint(baseline[0], maxPath=2)
    baseline_aval = pyAvalonTools.GetAvalonFP(baseline[0], 128)
    baseline_layer = AllChem.LayeredFingerprint(baseline[0])

    for item in temp:
        suppl = SDMolSupplier("../data/sdf/" + item)

        fp = AllChem.GetMorganFingerprint(suppl[0], 2)

        fp_bit = AllChem.GetMorganFingerprintAsBitVect(suppl[0], 2, nBits=bit_length)
        fp_rdk = AllChem.RDKFingerprint(suppl[0], maxPath=2)
        fp_aval = pyAvalonTools.GetAvalonFP(suppl[0], 128)
        fp_layer = AllChem.LayeredFingerprint(suppl[0])

        sim_matrix_morgan.append(
            DataStructs.FingerprintSimilarity(baseline_morgan, fp_bit, metric=DataStructs.TanimotoSimilarity))
        sim_matrix_rdk.append(
            DataStructs.FingerprintSimilarity(baseline_rdk, fp_rdk, metric=DataStructs.TanimotoSimilarity))
        sim_matrix_aval.append(
            DataStructs.FingerprintSimilarity(baseline_aval, fp_aval, metric=DataStructs.TanimotoSimilarity))
        sim_matrix_layer.append(
            DataStructs.FingerprintSimilarity(baseline_layer, fp_layer, metric=DataStructs.TanimotoSimilarity))

    sim_matrix_morgan = np.array(sim_matrix_morgan)
    sim_matrix_rdk = np.array(sim_matrix_rdk)
    sim_matrix_aval = np.array(sim_matrix_aval)
    sim_matrix_layer = np.array(sim_matrix_layer)

    label_morgan = "morgan" + str(bit_length)
    plt.hist(sim_matrix_morgan, label = label_morgan)
    plt.hist(sim_matrix_rdk, label = "rdk2")
    plt.hist(sim_matrix_aval, label = "avalon128")
    plt.hist(sim_matrix_layer, label = "layer")
    print(np.mean(sim_matrix_rdk))
    plt.xlabel("Similarity to Baseline")
    plt.ylabel("Counts")
    plt.title("Different Fingerprinting Methods, Similarity to Baseline")
    plt.legend()
    plt.show()

