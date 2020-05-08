

import glob
import numpy as np
from molSimplify.Classes.mol3D import *
from molSimplify.Informatics.autocorrelation import *
from molSimplify.Informatics.graph_analyze import *
from molSimplify.Informatics.misc_descriptors import *



def full_autocorr(dir = "../data/xyz/"):

	# begin parsing
	str_temp = dir + "/*.xyz"
	target_paths=sorted(glob.glob(str_temp))
	count = 0
	max_size = 0
	list_of_runs = list()
	smi_dict = dict()
	auto_corr_mat = []

	for geos in target_paths:

		this_mol = mol3D() # mol3D instance
		this_mol.readfromxyz(geos) # read geo
		results_auto = generate_full_complex_autocorrelations(this_mol, depth=3, loud=True)
		auto_corr_mat.append(results_auto["results"])

	auto_corr_mat = np.array(auto_corr_mat)
	print(auto_corr_mat)
	return auto_corr_mat

def ligand_autocorr(dir = "../data/xyz/"):
	# begin parsing
	str_temp = dir + "/*.xyz"
	target_paths=sorted(glob.glob(str_temp))
	count = 0
	max_size = 0
	list_of_runs = list()
	smi_dict = dict()
	auto_corr_mat = []

	for geos in target_paths:

		this_mol = mol3D() # mol3D instance
		this_mol.readfromxyz(geos) # read geo
		#requires metal in the system
		results_auto = generate_all_ligand_autocorrelations(this_mol, depth=3, loud=True)
		auto_corr_mat.append(results_auto["results"])

	auto_corr_mat = np.array(auto_corr_mat)
	return auto_corr_mat

def deltametrics(dir = "../data/xyz/"):
	# begin parsing
	str_temp = dir + "/*.xyz"
	target_paths=sorted(glob.glob(str_temp))
	count = 0
	max_size = 0
	list_of_runs = list()
	smi_dict = dict()
	results_delta = []
	results_delta_all = []

	for geos in target_paths:

		this_mol = mol3D() # mol3D instance
		this_mol.readfromxyz(geos) # read geo

		#if metals present
		#results_delta = generate_metal_deltametrics(this_mol, loud = "something")
		results_delta_all = deltametrics(this_mol)
		print(results_delta["results"])

	results_delta = np.array(results_delta)
	return results_delta

def metal_deltametrics(dir = "../data/xyz/"):
	# begin parsing
	str_temp = dir + "/*.xyz"
	target_paths=sorted(glob.glob(str_temp))
	count = 0
	max_size = 0
	list_of_runs = list()
	smi_dict = dict()
	results_delta = []
	results_delta_all = []

	for geos in target_paths:

		this_mol = mol3D() # mol3D instance
		this_mol.readfromxyz(geos) # read geo

		#if metals present
		#results_delta = generate_metal_deltametrics(this_mol, loud = "something")
		results_delta_all = deltametrics(this_mol)
		print(results_delta["results"])

	results_delta = np.array(results_delta)
	return results_delta



