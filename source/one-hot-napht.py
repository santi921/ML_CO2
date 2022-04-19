import argparse, random
import numpy as np
import pandas as pd
import scipy as sp
from tqdm import tqdm
import pandas as pd
from scipy.stats import spearmanr
from rdkit import RDLogger
from xgboost import XGBRegressor

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)
from rdkit import Chem
from rdkit.Chem import rdRGroupDecomposition as rdRGD

from utils.tensorflow_util import *

import os
import random
import argparse
import pandas as pd
from tqdm import tqdm
from rdkit import Chem

#from utils.sklearn_util import *
from utils.genetic_util import *
from utils.helpers import nap, napht_check
from sklearn.kernel_ridge import KernelRidge


def get_one_hot_data():
    df_napht= pd.read_hdf("../data/napth/compiled.h5")
    napht_smiles = df_napht["smiles"]
    homo_napht = df_napht["homo"]
    homo1_napht = df_napht['homo1']
    str_napht = []
    homo = []
    homo1 = []
    succ = 0
                 
    for ind, i in enumerate(napht_smiles):

        ret = napht_check(i)
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
            except:
                frag_list_temp.append("H")
        str_napht.append(frag_list_temp)

    
    list_one_hot = []
    fail = 0

    for ind, partition in enumerate(str_napht):

        list_groups = []
        list_temp = [0 for i in range(27)]
        list_temp2 = [0 for i in range(16)]
        

        for ind2, split in enumerate(partition[:2]):
            add = True

            if split == "O=[N+]([O-])":
                list_temp[0] += 1
                list_groups.append(0)

            elif split == "O=COC":
                list_temp[1] += 1
                list_groups.append(1)

            elif split == "OC(O)(O)C":
                list_temp[2] += 1
                list_groups.append(2)

            elif split == "Br":
                list_temp[3] += 1
                list_groups.append(3)

            elif split == "Cl":
                list_temp[4] += 1
                list_groups.append(4)

            elif split == "C(=O)OC" or split == "O=C(O)C":
                list_temp[5] += 1
                list_groups.append(5)

            elif split == "C(F)(F)F" or split == "FC(F)(F)":
                list_temp[6] += 1
                list_groups.append(6)

            elif split == "NC(=O)C" or split == "CC(N)=O":
                list_temp[7] += 1
                list_groups.append(7)

            elif split == "C2=CC=CC=C2" or split == "c1ccc(C" or split == "c1ccc(":
                list_temp[8] += 1
                list_groups.append(8)

            elif split == "F":
                list_temp[9] += 1
                list_groups.append(9)

            elif split == "OC2=CC=CC=C2" or split == "c1ccc(O":
                list_temp[10] += 1
                list_groups.append(10)

            elif split == "NC2=CC=CC=C2" or split == "c1ccc(N":
                list_temp[11] += 1
                list_groups.append(11)

            elif split == "O":
                list_temp[12] += 1
                list_groups.append(12)

            elif split == "S(=O)(=O)c1ccccc1" or split == "O=S(=O)(c1ccccc1)":
                list_temp[13] += 1
                list_groups.append(13)
                
            elif split == "S(=O)(=O)OC" or split == "COS(=O)(=O)":
                list_temp[14] += 1
                list_groups.append(14)

            elif split == "OC":
                list_temp[15] += 1
                list_groups.append(15)

            elif split == "N":
                list_temp[16] += 1
                list_groups.append(16)

            elif split == "NC" or split == "N#C":
                list_temp[17] += 1
                list_groups.append(17)

            elif split == "C":
                list_temp[18] += 1
                list_groups.append(18)

            elif split == "CC":
                list_temp[19] += 1
                list_groups.append(19)

            elif split == "C(C)(C)C" or split == "CC(C)C":
                list_temp[20] += 1
                list_groups.append(20)

            elif split == "N(C)C" or split == "CNC":
                list_temp[21] += 1
                list_groups.append(21)
            elif split == "S":
                list_temp[22] += 1
                list_groups.append(22)
            elif split == "SC" or split == "CS":
                list_temp[23] += 1
                list_groups.append(23)
            elif split == "S(=O)C" or split == "CS(=O)":
                list_temp[24] += 1
                list_groups.append(24)
            elif split == "S(=O)(=O)O" or split == "O=S(=O)(O)":
                list_temp[25] += 1
                list_groups.append(25)
            elif split == "S(=O)(=O)C" or split == "CS(=O)(=O)":
                list_temp[26] += 1
                list_groups.append(26)
            else:
                fail += 1
                add = False
                print(split)
        
        for ind2, split in enumerate(partition[2:]):
        
            
            if split == "O=COC":
                list_temp2[1] += 1
                list_groups.append(1)

            elif split == "NC" or split == "N#C":
                list_temp2[0] += 1
                list_groups.append(0)

            elif split == "OC(O)(O)C":
                list_temp2[2] += 1
                list_groups.append(2)
            
            elif split == "H":
                list_temp2[3] += 1
                list_groups.append(3)

            elif split == "OC":
                list_temp2[4] += 1
                list_groups.append(4)

            elif split == "C(=O)OC" or split == "O=C(O)C":
                list_temp2[5] += 1
                list_groups.append(5)

            elif split == "N":
                list_temp2[6] += 1
                list_groups.append(6)

            elif split == "NC(=O)C" or split == "CC(N)=O":
                list_temp2[7] += 1
                list_groups.append(7)

            elif split == "C2=CC=CC=C2" or split == "c1ccc(C" or split == "c1ccc(":
                list_temp2[8] += 1
                list_groups.append(8)
            
            elif split == "N(C)C" or split == "CNC":
                list_temp2[9] += 1
                list_groups.append(9)

            elif split == "OC2=CC=CC=C2" or split == "c1ccc(O":
                list_temp2[10] += 1
                list_groups.append(10)

            elif split == "NC2=CC=CC=C2" or split == "c1ccc(N":
                list_temp2[11] += 1
                list_groups.append(11)

            elif split == "O":
                list_temp2[12] += 1
                list_groups.append(12)
            elif split == "CC":
                list_temp2[13] += 1
                list_groups.append(13)

            elif split == "C(C)(C)C" or split == "CC(C)C":
                list_temp2[14] += 1
                list_groups.append(14)
            elif split == "C":
                list_temp2[15] += 1
                list_groups.append(15)
            else:
                fail += 1
                add = False
                print(split)
        if add:
            homo.append(homo_napht[ind])
            homo1.append(homo1_napht[ind])
            list_one_hot.append(list_temp+list_temp2)

    print(str(fail) + "/" + str(len(str_napht)))
    homo = np.array(homo)
    homo1 = np.array(homo1)
    diff = homo - homo1
    
        
    print("---------- HOMO ----------")
    for ind in range(27):
        corr, _ = spearmanr(np.array([i[ind] for i in list_one_hot]), np.array(homo))
        print("Spearmans correlation: %.3f" % corr)

    print("---------- HOMO-1 ----------")
    for ind in range(27+16):
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
        y_pred_train = self.homo1_model.predict(X_train)
        print(r2_score(y_pred_train, y_train))
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
        y_pred_train = self.homo_model.predict(X_train)
        print(r2_score(y_pred_train, y_train))
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
        write_to_file = str(self.start_pop_size) + "_start_" + str(gens) + "_gens_napht.h5"
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
    df = pd.read_hdf(str(start_pop_size) + "_start_" + str(gens) + "_gens_napht.h5")
    print("Performance averages")
    len_gens = df["gen"].max()

    for i in range(len_gens):
        print("Stats for gen: " + str(i + 1))
        print("mean: " + str(df["loss"][df["gen"] == int(i + 1)].mean()))
        print("max: " + str(df["loss"][df["gen"] == int(i + 1)].max()))
