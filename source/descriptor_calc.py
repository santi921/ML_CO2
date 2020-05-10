import os
import numpy as np
from helpers import xyz_to_smiles

from rdkit.Chem import AllChem, Descriptors, DataStructs
from rdkit.Chem import Fingerprints, SDMolSupplier, RDKFingerprint
from rdkit.Avalon import pyAvalonTools
import rdkit.Chem as rdkit_util



def rd_kit_morgan(dir_sdf = "../data/sdf/"):

	temp_str = "ls " + dir_sdf
	temp = os.popen(temp_str).read()
	temp = str(temp).split()
	bit_length = 256
	sim_matrix_morgan = []
	baseline = SDMolSupplier("../data/sdf/" + temp[0])
	for item in temp:
		suppl = SDMolSupplier("../data/sdf/" + item)
		#Note: morgan can output vectors as two types
		fp = AllChem.GetMorganFingerprint(suppl[0], 2)
		fp_bit = AllChem.GetMorganFingerprintAsBitVect(suppl[0], 2, nBits=bit_length)
		sim_matrix_morgan.append(fp_bit)

	sim_matrix_morgan = np.array(sim_matrix_morgan)
	return sim_matrix_morgan


def rd_kit_rd(dir_sdf = "../data/sdf/"):

	temp_str = "ls " + dir_sdf
	temp = os.popen(temp_str).read()
	temp = str(temp).split()

	sim_matrix_rdk = []
	baseline = SDMolSupplier("../data/sdf/" + temp[0])
	baseline_rdk = AllChem.RDKFingerprint(baseline[0], maxPath=2)

	for item in temp:
		suppl = SDMolSupplier("../data/sdf/" + item)
		fp_rdk = AllChem.RDKFingerprint(suppl[0], maxPath=2)
		sim_matrix_rdk.append(DataStructs.FingerprintSimilarity(baseline_rdk, fp_rdk, metric=DataStructs.TanimotoSimilarity))

	sim_matrix_rdk = np.array(sim_matrix_rdk)
	return sim_matrix_rdk


def rd_kit_aval(dir_sdf = "../data/sdf/"):

	temp_str = "ls " + dir_sdf
	temp = os.popen(temp_str).read()
	temp = str(temp).split()

	bit_length = 256
	sim_matrix_aval = []
	baseline = SDMolSupplier("../data/sdf/" + temp[0])

	baseline_aval = pyAvalonTools.GetAvalonFP(baseline[0], 128)

	for item in temp:
		suppl = SDMolSupplier("../data/sdf/" + item)
		fp_aval = pyAvalonTools.GetAvalonFP(suppl[0], 128)
		sim_matrix_aval.append(fp_aval)
	sim_matrix_aval = np.array(sim_matrix_aval)
	return sim_matrix_aval


def rd_kit_morgan(dir_sdf = "../data/sdf/"):
	temp_str = "ls " + dir_sdf
	temp = os.popen(temp_str).read()
	temp = str(temp).split()
	sim_matrix_layer = []

	for item in temp:
		suppl = SDMolSupplier("../data/sdf/" + item)
		fp_layer = AllChem.LayeredFingerprint(suppl[0])
		sim_matrix_layer.append(fp_layer)
		sim_matrix_layer = np.array(sim_matrix_layer)

	return sim_matrix_layer


# tonight
# todo: train our own VAEs? use ZZ and daniel's database
# todo: lasso\pca to select features