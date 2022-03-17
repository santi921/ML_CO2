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
from utils.sklearn_util import *
from rdkit import Chem
from rdkit.Chem import rdRGroupDecomposition as rdRGD


from xgboost import XGBRegressor

from utils.tensorflow_util import *

import os
import random
import argparse
import pandas as pd
import selfies as sf
from tqdm import tqdm
from rdkit import Chem

from utils.genetic_util import *



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
            homo.append(float(i.split(":")[2][:2]))
            homo1.append(float(i.split(":")[1]))
            str_preparse.append(i.split(":")[0])

    df_sulph = pd.read_hdf("../data/benzo/compiled.h5")
    sulp_smiles = df_sulph["smiles"]
    sulp_homo = df_sulph["homo"]
    sulp_homo1 = df_sulph["homo1"]
    
    str_sulf = []

    for ind, i in enumerate(sulp_smiles):
        ret = quinone_check(i)
        frag_list_temp = []

        if len(ret) > 0:
            try:
                frag_list_temp.append(ret[0][0]["R1"].split("*")[0][:-1])
            except:
                frag_list_temp.append("H")
            try:
                frag_list_temp.append(ret[0][0]["R2"].split("*")[0][:-1])
            except:
                frag_list_temp.append("H")
            try:
                frag_list_temp.append(ret[0][0]["R3"].split("*")[0][:-1])
            except:
                frag_list_temp.append("H")
            try:
                frag_list_temp.append(ret[0][0]["R4"].split("*")[0][:-1])
                # if(ret[0][0]['R4'].split("*")[0][:-1] == 'c1ccc(' or ret[0][0]['R4'].split("*")[0][:-1] == 'OC(O)(O)C'):
                #    print(i)
            except:
                frag_list_temp.append("H")

        str_sulf.append(frag_list_temp)


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
        list_temp = [0 for i in range(32)]

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
            elif split == "S":
                list_temp[24] += 1
                list_groups.append(24)
            elif split == "SC":
                list_temp[25] += 1
                list_groups.append(25)
            elif split == "S(=O)C":
                list_temp[26] += 1
                list_groups.append(26)
            elif split == "S(=O)(=O)O":
                list_temp[27] += 1
                list_groups.append(27)
            elif split == "S(=O)(=O)C":
                list_temp[28] += 1
                list_groups.append(28)
            elif split == "S(=O)(=O)OC":
                list_temp[29] += 1
                list_groups.append(29)
            elif split == "S(=O)(=O)c1ccccc1":

                list_temp[30] += 1
                list_groups.append(30)
            elif split == "H":
                list_temp[31] += 1
                list_groups.append(31)
            else:
                pass
                
        list_one_hot.append(list_temp)
        list_of_lists.append(list_groups)

    fail = 0
    for ind, partition in enumerate(str_sulf):

        list_groups = []
        list_temp = [0 for i in range(32)]

        for ind, split in enumerate(partition):
            sulf = 0
            add = True
            if "[N](C)(C)C" == split:
                list_temp[0] += 1
                list_groups.append(0)

            elif split == "N(=O)=O" or split == "O=[N+]([O-])":
                list_temp[1] += 1
                list_groups.append(1)
            elif split == "[C][N]":
                list_temp[2] += 1
                list_groups.append(2)

            elif split == "C[N](C)(C)C":
                list_temp[3] += 1
                list_groups.append(3)

            elif split == "OC(=O)C" or split == "O=COC":
                list_temp[4] += 1
                list_groups.append(4)

            elif split == "Br":
                list_temp[5] += 1
                list_groups.append(5)

            elif split == "Cl":
                list_temp[6] += 1
                list_groups.append(6)

            elif split == "C(=O)OC" or split == "O=C(O)C":
                list_temp[7] += 1
                list_groups.append(7)

            elif split == "C(F)(F)F" or split == "FC(F)(F)":
                list_temp[8] += 1
                list_groups.append(8)

            elif split == "NC(=O)C" or split == "CC(N)=O":
                list_temp[9] += 1
                list_groups.append(9)

            elif split == "C2=CC=CC=C2" or split == "c1ccc(C" or split == "c1ccc(":
                list_temp[10] += 1
                list_groups.append(10)

            elif split == "F":
                list_temp[11] += 1
                list_groups.append(11)

            elif split == "OC2=CC=CC=C2" or split == "c1ccc(O":
                list_temp[12] += 1
                list_groups.append(12)

            elif split == "NC2=CC=CC=C2" or split == "c1ccc(N":
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

            elif split == "NC" or split == "N#C":
                list_temp[19] += 1
                list_groups.append(19)

            elif split == "C":
                list_temp[20] += 1
                list_groups.append(20)

            elif split == "CC":
                list_temp[21] += 1
                list_groups.append(21)

            elif split == "C(C)(C)C" or split == "CC(C)C":
                list_temp[22] += 1
                list_groups.append(22)

            elif split == "N(C)C" or split == "CNC":
                list_temp[23] += 1
                list_groups.append(23)
            elif split == "S":
                list_temp[24] += 1
                list_groups.append(24)
            elif split == "SC" or split == "CS":
                list_temp[25] += 1
                list_groups.append(25)
            elif split == "S(=O)C" or split == "CS(=O)":
                list_temp[26] += 1
                list_groups.append(26)
            elif split == "S(=O)(=O)O" or split == "O=S(=O)(O)":
                list_temp[27] += 1
                list_groups.append(27)
            elif split == "S(=O)(=O)C" or split == "CS(=O)(=O)":
                list_temp[28] += 1
                list_groups.append(28)
            elif split == "S(=O)(=O)OC" or split == "COS(=O)(=O)":
                list_temp[29] += 1
                list_groups.append(29)
            elif split == "S(=O)(=O)c1ccccc1" or split == "O=S(=O)(c1ccccc1)":
                list_temp[30] += 1
                list_groups.append(30)
            elif split == "H":
                list_temp[31] += 1
                list_groups.append(31)
            else:
                print(split)
                fail += 1
                add = False
            if add:
                homo.append(sulp_homo[ind])
                homo1.append(sulp_homo1[ind])
                list_one_hot.append(list_temp)
    print(str(fail) + "/" + str(len(str_sulf)))
    homo = np.array(homo)
    homo1 = np.array(homo1)
    diff = homo - homo1

    print("---------- HOMO ----------")
    for ind in range(32):
        corr, _ = spearmanr(np.array([i[ind] for i in list_one_hot]), np.array(homo))
        print("Spearmans correlation: %.3f" % corr)

    print("---------- HOMO-1 ----------")
    for ind in range(32):
        corr, _ = spearmanr(np.array([i[ind] for i in list_one_hot]), np.array(homo1))
        print("Spearmans correlation: %.3f" % corr)

    return list_one_hot, homo, homo1


class optimizer_genetic(object):
    def __init__(
        self,
        population,
        homo,
        homo1,
        mut_prob=0.1,
        steps=10,
        start_pop_size=100,
        ckpt=True,
    ):
        self.ckpt = ckpt
        self.total_population = population
        self.start_pop_size = start_pop_size
        self.homo = homo
        self.homo1 = homo1
        self.mut_prob = mut_prob,
        self.step  = 10,


        self.population_sample = [
            population[int(i)]
            for i in np.random.randint(0, len(population), int(start_pop_size))
        ]

    def loss(self, xmat):
        homo_pred = self.homo_model.predict(np.array(xmat).reshape(1, -1))
        homo1_pred = self.homo1_model.predict(np.array(xmat).reshape(1, -1))

        return np.abs(homo_pred) - np.abs(homo1_pred)

    def train_models(self):
        X_train, X_test, y_train, y_test = train_test_split(
            np.array(self.total_population),
            np.array(self.homo1),
            test_size=0.2,
            random_state=0,
        )
        self.homo1_model = XGBRegressor()
        self.homo1_model.fit(X_train, y_train)
        y_pred = self.homo1_model.predict(X_test)
        print(r2_score(y_test, y_pred))

        X_train, X_test, y_train, y_test = train_test_split(
            np.array(self.total_population),
            np.array(self.homo),
            test_size=0.2,
            random_state=0,
        )
        self.homo_model = XGBRegressor()
        self.homo_model.fit(X_train, y_train)
        y_pred = self.homo_model.predict(X_test)
        print(r2_score(y_test, y_pred))

    def selection(self):
        pop_loss = []
        pop_new = []
        quinone_tf_arr = []
        total_loss = 0.0
        ratio_children = 1.0
        population = self.population_sample
        successful_mol_count = 0

        for ind in tqdm(range(len(population))):
            pop_temp = []
            i = population[ind]

            temp_loss = self.loss(i)
            temp_loss = np.abs(temp_loss)
            pop_loss.append(temp_loss)
            pop_temp.append(temp_loss)
            total_loss += temp_loss
            successful_mol_count += 1

        parent_ind, parent_gen_loss, parent_prob_dist = [], [], []
        #pop_loss_temp = pop_loss
        #pop_loss_temp = np.array(pop_loss_temp)
        pop_loss_temp = np.array([i[0] for i in pop_loss])

        while len(parent_ind) <= int(successful_mol_count * ratio_children - 1):
            boltz = True
            # print('draw')
            draw = draw_from_pop_dist_no_tf(pop_loss_temp, boltz=boltz)
            # if (parent_ ind.count(draw) == 0):
            parent_ind.append(draw)
            parent_gen_loss.append(pop_loss[draw])

        if len(parent_ind) % 2 == 1:  # chop off member from parent gen if odd numbered
            parent_ind = parent_ind[0:-1]
            parent_gen_loss = parent_gen_loss[0:-1]

        parent_gen = [population[i] for i in parent_ind]
        parent_gen_index_tracker = [i for i in range(len(parent_gen))]

        while len(pop_new) < len(parent_gen):

            draw1 = random.choice(parent_gen_index_tracker)
            draw2 = random.choice(parent_gen_index_tracker)

            cross_res1, cross_res2 = cross_one_hot(parent_gen[draw1], parent_gen[draw2])
            if(np.sum(np.array(cross_res1)) < 5):
                pop_new.append(cross_res1)
            if(np.sum(np.array(cross_res2)) < 5):
                pop_new.append(cross_res2)

        return pop_new

    def enrich_pop(self, gens=2, longitudnal_save=False):
        mean_loss_arr = []
        total_loss = 0
        successful_mol_count = 0
        write_to_file = str(self.start_pop_size) + "_start_" + str(gens) + "_gens.h5"
        prev_file = os.path.exists(write_to_file)

        if prev_file == True and self.ckpt == True:  # handles checkpoint resuming
            df = pd.read_hdf(write_to_file)
            gen_mols = df["str"][df["gen"] == df["gen"].max()].tolist()
            gen_start = df["gen"].max()        
            self.population_sample = gen_mols
            print(
                "resuming from generation: "
                + str(gen_start)
                + " with ["
                + str(len(self.population_sample))
                + "] molecules"
            )

        else:
            gen_start = 1

        for gen in range(gen_start, gens + 1):

            if int(self.start_pop_size * 0.1) > len(self.population_sample):
                print("terminating population study early at gen" + str(gen))
                break
            print("Gen " + str(gen) + "/" + str(gens))
            print("selection + cross...")
            pop_new = self.selection()
            pop_mut_new = []
            pop_loss = []
            quinone_tf_arr = []

            print("mutation...")
            for i in pop_new:
                pop_mut_new.append(random_mutation_one_hot(i, mut_chance=0.15))

            self.population_sample = pop_mut_new

            smiles_scored = []
            print("computing performance of new generation...")
            for ind in tqdm(range(len(pop_mut_new))):
                i = pop_mut_new[ind]
                temp_loss = self.loss(i)[0]
                temp_loss = np.abs(temp_loss)
                smiles_scored.append(i)
                pop_loss.append(temp_loss)
                total_loss += temp_loss
                successful_mol_count += 1


            total_loss = 0
            mean_loss_arr.append(np.array(pop_loss).mean())
            print("mean loss: " + str(np.array(pop_loss).mean()))
            print("GENERATION " + str(gen) + " COMPLETED")

            df_temp_list = []

            if longitudnal_save == True:
                print("saving generation for longitudinal study")
                for ind, element in enumerate(smiles_scored):
                    df_temp_row = [element, pop_loss[ind], gen]
                    df_temp_list.append(df_temp_row)

                if os.path.exists(write_to_file):
                    df_old = pd.read_hdf(write_to_file)
                    df = pd.DataFrame(df_temp_list, columns=["str", "loss", "gen"])
                    df = df_old.append(df, ignore_index=True)
                    df.to_hdf(write_to_file, key="df")

                else:
                    df = pd.DataFrame(
                        df_temp_list, columns=["str", "loss", "gen"]
                    )
                    df.to_hdf(write_to_file, key="df")

        return mean_loss_arr


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="parameters of genetic study")

    parser.add_argument(
        "--longi",
        action="store_true",
        dest="longitudnal",
        default=False,
        help="longitudnal saving of molecules over gens",
    )
    parser.add_argument(
        "--ckpt",
        action="store_true",
        dest="ckpt",
        help="resume from previous job",
    )
    parser.add_argument(
        "-mols_data",
        action="store",
        dest="mols_data",
        default=1000,
        help="mols in population",
    )
    parser.add_argument(
        "-start_pop",
        action="store",
        dest="start_pop_size",
        default=20,
        help="start pop size",
    )

    parser.add_argument(
        "-gens", action="store", dest="gens", default=2, help="generations"
    )

    results = parser.parse_args()
    long = bool(results.longitudnal)
    ckpt = bool(results.ckpt)
    gens = int(results.gens)
    list_one_hot, homo, homo1 = get_one_hot_data()

    start_pop_size = int(results.start_pop_size)

    print("Longitudinal study:" + str(long))
    print("number of molecules in sample: " + str(len(list_one_hot)))

    #######################
    opt = optimizer_genetic(
        list_one_hot, homo, homo1,
        ckpt=ckpt,
        start_pop_size=start_pop_size,
    )
    opt.train_models()
    mean_loss = opt.enrich_pop(gens=gens, longitudnal_save=long)
    print(":" * 50)
    print("Mean Loss array")
    print(":" * 50)
    print(mean_loss)
    final_molecules = opt.population_sample
    print("number of molecules at the end: " + str(len(final_molecules)))
    df = pd.read_hdf(str(start_pop_size) + "_start_" + str(gens) + "_gens.h5")
    print("Performance averages")
    len_gens = df["gen"].max()

    for i in range(len_gens):
        print("Stats for gen: " + str(i + 1))
        print("mean: " + str(df["loss"][df["gen"] == int(i + 1)].mean()))
        print("max: " + str(df["loss"][df["gen"] == int(i + 1)].max()))
