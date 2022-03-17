from re import I
import argparse, random
import numpy as np
import pandas as pd
import scipy as sp
from tqdm import tqdm
import pandas as pd
from scipy.stats import spearmanr
from rdkit import RDLogger

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)
from rdkit import Chem
from rdkit.Chem import rdRGroupDecomposition as rdRGD


from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from utils.tensorflow_util import *

import os
import random
import argparse
import pandas as pd
import selfies as sf
from tqdm import tqdm
from rdkit import Chem

from utils.sklearn_util import *
from utils.genetic_util import *
from boruta import BorutaPy

def quinone_check(mol_smiles):

    mol = Chem.MolFromSmiles(mol_smiles)
    res_1, res_2, res_3, res_4, res_5, res_6, res_7, res_8 = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )

    quinone = Chem.MolFromSmiles("c1(ccc(cc1)[O])[O]")
    quinone_2 = Chem.MolFromSmiles("C1=CC(=O)C=CC1=O")
    quinone_3 = Chem.MolFromSmiles("C1=CC(=O)CCC1=O")
    quinone_4 = Chem.MolFromSmiles("C1CC(=O)CCC1=O")
    quinone_5 = Chem.MolFromSmiles("C1=CC(=O)C=CC1[O]")
    quinone_6 = Chem.MolFromSmiles("C1CC(=O)CC=C1[O]")
    quinone_7 = Chem.MolFromSmiles("C1CC(=O)CCC1[O]")
    quinone_8 = Chem.MolFromSmiles("[CH]=1[CH]C(=O)C=CC1[O]")

    frag_list = []

    success = 0
    try:
        res1, unmatched = rdRGD.RGroupDecompose([quinone], [mol], asSmiles=True)
    except:
        pass
    try:
        res2, unmatched = rdRGD.RGroupDecompose([quinone_2], [mol], asSmiles=True)
    except:
        pass
    try:
        res3, unmatched = rdRGD.RGroupDecompose([quinone_3], [mol], asSmiles=True)
    except:
        pass
    try:
        res4, unmatched = rdRGD.RGroupDecompose([quinone_4], [mol], asSmiles=True)
    except:
        pass
    try:
        res5, unmatched = rdRGD.RGroupDecompose([quinone_5], [mol], asSmiles=True)
    except:
        pass
    try:
        res6, unmatched = rdRGD.RGroupDecompose([quinone_6], [mol], asSmiles=True)
    except:
        pass
    try:
        res7, unmatched = rdRGD.RGroupDecompose([quinone_7], [mol], asSmiles=True)
    except:
        pass
    try:
        res8, unmatched = rdRGD.RGroupDecompose([quinone_8], [mol], asSmiles=True)
    except:
        pass

    if (
        len(res8) > 1
        or len(res7) > 1
        or len(res6) > 1
        or len(res5) > 1
        or len(res4) > 1
        or len(res3) > 1
        or len(res2) > 1
        or len(res1) > 1
    ):
        print("longer than 1, shit's wrong")

    if len(res1) != 0:
        frag_list.append(res1)
        success = 1
    else:
        if len(res2) != 0:
            frag_list.append(res2)
            success = 1
        else:
            if len(res3) != 0:
                frag_list.append(res3)
                success = 1
            else:
                if len(res4) != 0:
                    frag_list.append(res4)
                    success = 1
                else:
                    if len(res5) != 0:
                        frag_list.append(res5)
                        success = 1
                    else:
                        if len(res6) != 0:
                            frag_list.append(res6)
                            success = 1
                        else:
                            if len(res7) != 0:
                                frag_list.append(res7)
                                success = 1
                            else:
                                if len(res8) != 0:
                                    frag_list.append(res8)
                                    success = 1
                                else:
                                    success = 0

    return frag_list

    # from utils.selfies_util import smiles
    # mols_smiles = pd.read_hdf('../data/desc/DB3/desc_calc_DB3_self.h5')['name'].tolist()
    homo = []

def get_one_hot_data():

    homo = []
    homo1 = []
    str_preparse = []

    with open("../data/DATA_DB3") as f:
        lines = f.readlines()

    for i in lines:
        if len(i.split(":")) > 1:
            homo.append(float(i.split(":")[2]))
            homo1.append(float(i.split(":")[1]))
            str_preparse.append(i.split(":")[0])
            


    list_of_lists = []
    list_one_hot = []


    for ind2, str_split in enumerate(str_preparse):
        if len(str_split.split("_")) == 0:
            partition = ["H", "H", "H", "H"]
        elif len(str_split.split("_")) == 1:
            partition = str_split.split("_")
            partition.append("H")
            partition.append("H")
            partition.append("H")
        elif len(str_split.split("_")) == 2:
            partition = str_split.split("_")
            partition.append("H")
            partition.append("H")
        elif len(str_split.split("_")) == 3:
            partition = str_split.split("_")
            partition.append("H")
        else:
            partition = str_split.split("_")

        list_groups = []
        list_temp = [0 for i in range(25)]

        for ind, split in enumerate(partition):
            sulf = 0
            if "[N](C)(C)C" == split:
                list_temp[0] += 1
                list_groups.append(0)

            elif split == "N(=O)=O":
                list_temp[1] += 1
                list_groups.append(1)
            elif split == "[C][N]":
                list_temp[2] += 1
                list_groups.append(2)

            elif split == "C[N](C)(C)C":
                list_temp[3] += 1
                list_groups.append(3)

            elif split == "OC(=O)C":
                list_temp[4] += 1
                list_groups.append(4)

            elif split == "Br":
                list_temp[5] += 1
                list_groups.append(5)

            elif split == "Cl":
                list_temp[6] += 1
                list_groups.append(6)

            elif split == "C(=O)OC":
                list_temp[7] += 1
                list_groups.append(7)

            elif split == "C(F)(F)F":
                list_temp[8] += 1
                list_groups.append(8)

            elif split == "NC(=O)C":
                list_temp[9] += 1
                list_groups.append(9)

            elif split == "C2=CC=CC=C2":
                list_temp[10] += 1
                list_groups.append(10)

            elif split == "F":
                list_temp[11] += 1
                list_groups.append(11)

            elif split == "OC2=CC=CC=C2":
                list_temp[12] += 1
                list_groups.append(12)

            elif split == "NC2=CC=CC=C2":
                list_temp[13] += 1
                list_groups.append(13)

            elif split == "O":
                list_temp[14] += 1
                list_groups.append(14)

            elif split == "CO":
                list_temp[15] += 1
                list_groups.append(15)

            elif split == "CC2=CC=CC=C2":
                list_temp[16] += 1
                list_groups.append(16)

            elif split == "OC":
                list_temp[17] += 1
                list_groups.append(17)

            elif split == "N":
                list_temp[18] += 1
                list_groups.append(18)

            elif split == "NC":
                list_temp[19] += 1
                list_groups.append(19)

            elif split == "C":
                list_temp[20] += 1
                list_groups.append(20)

            elif split == "CC":
                list_temp[21] += 1
                list_groups.append(21)

            elif split == "C(C)(C)C":
                list_temp[22] += 1
                list_groups.append(22)

            elif split == "N(C)C":
                list_temp[23] += 1
                list_groups.append(23)

            elif split == "H":
                list_temp[24] += 1
                list_groups.append(24)
            else:
                pass
                
        list_one_hot.append(list_temp)
        list_of_lists.append(list_groups)

   
    homo = np.array(homo)
    homo1 = np.array(homo1)
    diff = homo - homo1

    print("---------- HOMO ----------")
    for ind in range(25):
        corr, _ = spearmanr(np.array([i[ind] for i in list_one_hot]), np.array(homo))
        print("Spearmans correlation: %.3f" % corr)

    print("---------- HOMO-1 ----------")
    for ind in range(25):
        corr, _ = spearmanr(np.array([i[ind] for i in list_one_hot]), np.array(homo1))
        print("Spearmans correlation: %.3f" % corr)


    rf = RandomForestRegressor(max_depth = 4, n_estimators=50)
    feat_selector = BorutaPy(rf,verbose = 2,  max_iter = 50)
    feat_selector.fit(np.array(list_one_hot), diff)

    print(feat_selector.support_)

    return list_one_hot, homo, homo1

get_one_hot_data()
