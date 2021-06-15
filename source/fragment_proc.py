import joblib, argparse, uuid, sigopt
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import BRICS
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Draw
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import Recap
from rdkit.Chem import rdRGroupDecomposition as rdRGD

from sklearn import preprocessing
from utils.sklearn_utils import *
import matplotlib.pyplot as plt
import seaborn as sns

import selfies as sf
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,regularizers

from utils.selfies_util import selfies,smile_to_hot, \
multiple_smile_to_hot, selfies_to_hot, multiple_selfies_to_hot,\
get_selfie_and_smiles_encodings_for_dataset, compare_equality, tanimoto_dist


names, ret, homo, homo1, diff = selfies()
print(len(names))

selfies_list, selfies_alphabet, largest_selfies_len,\
smiles_list, smiles_alphabet, largest_smiles_len\
= get_selfie_and_smiles_encodings_for_dataset(names)


data = multiple_selfies_to_hot(selfies_list, largest_selfies_len,\
                                       selfies_alphabet)


max_mol_len = data.shape[1]
alpha_len = data.shape[2]
len_alphabet_mol = alpha_len * max_mol_len
print(len(names))
print(len(data))

# removes functional groups from quinone backbone, this works for IDing what functional 
# groups are in a given file, also reconstructs
# TODO: generate structure from molecules
arom = Chem.MolFromSmiles("c1(ccc(cc1))")
quinone = Chem.MolFromSmiles("c1(ccc(cc1)[O])[O]")
quinone2 = Chem.MolFromSmiles("C1=CC(=O)C=CC1=O")



count1 = 0
count2 = 0
count3 = 0

count_fail_no_match = 0
count_fail = 0 
pattern = "(*)"
pattern_double = "[*]"
functional_list = []

for i, can_smi in enumerate(names):
    
    try:
        temp = Chem.MolFromSmiles(can_smi)
        rm = Chem.DeleteSubstructs(temp, quinone)
        rm2 = Chem.DeleteSubstructs(temp, arom)
        rm3 = Chem.DeleteSubstructs(temp, quinone2)

        #print(can_smi)
        if (len(Chem.MolToSmiles(rm3).split(".")) > 1):
            count1 = count1 + 1
            [functional_list.append(i) for i in Chem.MolToSmiles(rm3).split(".")]
        else: 
            if(len(Chem.MolToSmiles(rm2).split(".")) > 1):
                count2 = count2 + 1
                [functional_list.append(i) for i in Chem.MolToSmiles(rm2).split(".")]

            else:
                if(len(Chem.MolToSmiles(rm).split(".")) > 1):
                    count3 = count3 + 1
                    [functional_list.append(i) for i in Chem.MolToSmiles(rm).split(".")]
                    
                else:
                    pieces_smi = Chem.BRICS.BRICSDecompose(temp)
                    pieces = [Chem.MolFromSmiles(x) for x in BRICS.BRICSDecompose(temp)]
                    count_fail_no_match += 1
                    print(can_smi)

                    
    except:
        count_fail += 1
print(list(set(functional_list))) #retrieve only the found functional groups
print(len(list(set(functional_list))))
print(count1, count2, count3)
print("total processed: "+ str(count1+count2+count3))
print("no substructured: "+ str(count_fail_no_match))
print("fail processed: "+ str(count_fail))



can_smi = names[100]
mol_set = [Chem.MolFromSmiles(can_smi) for can_smi in names]
quinone = Chem.MolFromSmiles("c1(ccc(cc1)[O])[O]")
quinone_2 = Chem.MolFromSmiles("C1=CC(=O)C=CC1=O")
quinone_3 = Chem.MolFromSmiles("C1=CC(=O)CCC1=O")
quinone_4 = Chem.MolFromSmiles("C1CC(=O)CCC1=O")
quinone_5 = Chem.MolFromSmiles("C1=CC(=O)C=CC1[O]")
quinone_6 = Chem.MolFromSmiles("C1CC(=O)CC=C1[O]")
quinone_7 = Chem.MolFromSmiles("C1CC(=O)CCC1[O]")
quinone_8 = Chem.MolFromSmiles("[CH]=1[CH]C(=O)C=CC1[O]")


test = Chem.MolFromSmiles("[CH]=1[CH]C(=O)C=CC1[O]")
test2 = Chem.MolFromSmiles("C1(=C[CH]C(=O)C(=C1OC)[O])")
test3 = Chem.MolFromSmiles("C1(=O)[C]=[C]C(=O)[C]=[C]1")

temp = Chem.MolFromSmiles("c1(c(c(c(cc1)[O])C(C)(C)C)N)[O]")
temp_fail = Chem.MolFromSmiles("C1(=C[CH]C(=O)C(=C1OC)C(F)(F)F)[O]")


mol_set = [Chem.MolFromSmiles(can_smi) for can_smi in names[0:1000]]

res_1, res_2, res_3, res_4, res_5, res_6, res_7, res_8 = [], [], [], [], [], [], [], []
fail_list = []
frag_list = []
homo_frag = []
homo1_frag = []
diff_frag = []

for ind,i in enumerate(mol_set):
        fail = 0 
        try:
            res1, unmatched = rdRGD.RGroupDecompose([quinone], [i], asSmiles=True)
        except:pass
        try:
            res2, unmatched = rdRGD.RGroupDecompose([quinone_2], [i], asSmiles=True)
        except:pass
        try:
            res3, unmatched = rdRGD.RGroupDecompose([quinone_3], [i], asSmiles=True)
        except:pass
        try:
            res4, unmatched = rdRGD.RGroupDecompose([quinone_4], [i], asSmiles=True)
        except:pass
        try:
            res5, unmatched = rdRGD.RGroupDecompose([quinone_5], [i], asSmiles=True)
        except:pass
        try:
            res6, unmatched = rdRGD.RGroupDecompose([quinone_6], [i], asSmiles=True)
        except:pass
        try:
            res7, unmatched = rdRGD.RGroupDecompose([quinone_7], [i], asSmiles=True)
        except:pass
        try:
            res8, unmatched = rdRGD.RGroupDecompose([quinone_8], [i], asSmiles=True)
        except:pass
     
        if(len(res8) > 1 or len(res7) > 1 or len(res6) > 1 or len(res5) > 1 or len(res4) > 1 or len(res3) > 1 or len(res2) > 1 or len(res1) > 1):
            print("longer than 1, shit's wrong")
            
        if(len(res1) != 0):
            frag_list.append(res1)
        else: 
            if(len(res2) != 0):
                frag_list.append(res2)
            else:
                if(len(res3) != 0):
                    frag_list.append(res3)
                else:
                    if(len(res4) != 0):
                        frag_list.append(res4)
                    else:
                        if(len(res5) != 0):
                            frag_list.append(res5)
                        else:
                            if(len(res6) != 0):
                                frag_list.append(res6)
                            else:
                                if(len(res7) != 0):
                                    frag_list.append(res7)
                                else:
                                    if(len(res8) != 0):
                                        frag_list.append(res8)
                                    else:
                                        fail_list.append(i)
                                        fail = 1
        if(fail == 0):
            homo_frag.append(homo[ind])
            homo1_frag.append(homo1[ind])
            diff_frag.append(diff[ind])
            fail = 0
                    
print(len(frag_list))
print(len(homo_frag))
print(len(homo1_frag))
print(len(diff_frag))

r1_list = []
r2_list = []
r3_list = []
r4_list = []

for i in frag_list:
    temp_r1, temp_r2, temp_r3, temp_r4 = '', '', '', ''

    try:
        temp_r1 = i[0].get("R1")
        r1_list.append(temp_r1)
        if(len(split_list) == 2):
            trial_string = split_list[0] + "[H]"
            Chem.MolFromSmiles(trial_string)
            r1_list.append(trial_string)
        else:
            pass
            #r1_list.append(split_list[0] + split_list[-1])

    except:pass
    
    try:
        temp_r2 = i[0].get("R2") 
        r2_list.append(temp_r2)
        if(len(split_list) == 2):
            trial_string = split_list[0] + "[H]"
            Chem.MolFromSmiles(trial_string)
            r2_list.append(trial_string)
        else:
            pass
            #r2_list.append(split_list[0] + split_list[-1])

    except:pass

    try:
        temp_r3 = i[0].get("R3")
        r3_list.append(temp_r3)
        if(len(split_list) == 2):
            trial_string = split_list[0] + "[H]"
            Chem.MolFromSmiles(trial_string)
            r3_list.append(trial_string)
        else:
            pass
            #r3_list.append(split_list[0] + split_list[-1])

    except:pass

    try:
        temp_r4 = i[0].get("R4")
        split_list = temp_r4.split("[*:4]")
        if(len(split_list) == 2):
            trial_string = split_list[0] + "[H]"
            Chem.MolFromSmiles(trial_string)
            r4_list.append(trial_string)
        else:
            pass
            #r4_list.append(split_list[0] + split_list[-1])
    except:pass

selfies_list, selfies_alphabet, largest_selfies_len,\
smiles_list, smiles_alphabet, largest_smiles_len\
= get_selfie_and_smiles_encodings_for_dataset(r4_list)
