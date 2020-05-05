

import glob
import os
import sys

from molSimplify.Classes.mol3D import *
from molSimplify.Informatics.autocorrelation import *
from molSimplify.Informatics.graph_analyze import *
from molSimplify.Informatics.misc_descriptors import *




# check and create new folder
if not os.path.isdir('qm9_geos/'):
		os.makedirs('qm9_geos')

# begin parsing
print('starting loop over data, please be patient...')
target_paths=sorted(glob.glob('../data/xyz/*.xyz'))
print('found ' + str(len(target_paths)) + ' molecules to read')

count = 0
max_size = 0
list_of_runs = list()
smi_dict = dict()
for geos in target_paths:

		this_mol = mol3D() # mol3D instance
		this_mol.readfromxyz(geos) # read geo

		results_dictionary = generate_full_complex_autocorrelations(this_mol, depth=3, loud=True)
		print(results_dictionary["results"])
		count += 1
		sys.stdout.write('\r number of molecules read = '+str(count) + "/"+str(len(target_paths)))
		sys.stdout.flush()
print('complete!')

#functions of note
#generate_full_complex_autocorrelations

#def generate_full_complex_autocorrelations(mol, loud,  depth=4, oct=True,flag_name=False, modifier=False,use_dist=False, NumB=False, Zeff=False):
#generate_multiatom_deltametrics(mol, loud, depth=4, oct=True, flag_name=False, additional_elements=False):

# guzik uses autocorrelation and deltametric functions
# MAD3 features





