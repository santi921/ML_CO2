import os
import random
import argparse
import warnings
import pandas as pd
import selfies as sf
from tqdm import tqdm
from rdkit import Chem
warnings.filterwarnings("ignore")

from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.linear_model import HuberRegressor

from sklearn.preprocessing import StandardScaler
from terminaltables import AsciiTable

from utils.desc_helpers import (
    calc_merck,
    desc_calc,
    top_keys,
    no_outlier_list,
    desc_calc,
)

from utils.pk_util import PBPKsim
from utils.sklearn_util import *
from utils.genetic_util import *
from utils.selfies_util import (
    selfies_to_hot,
    multiple_selfies_to_hot,
    get_selfie_and_smiles_encodings_for_dataset,
)


def calc(
    x,
    y,
    des,
    scale,
    rand_tf=False,
    grid_tf=False,
    bayes_tf=False,
    sigopt_tf=False,
    algo="sgd",
):

    if rand_tf == True:
        # todo incorp all sklearn algos here
        rand(x, y, algo, des)

    if grid_tf == True:
        print("........starting grid search........")
        grid_obj = grid(x, y, method=algo, des=des)
        uuid_temp = uuid.uuid4()
        str = (
            "../data/train/grid/complete_grid_"
            + algo
            + "_"
            + des
            + "_"
            + uuid_temp.urn[9:]
            + ".pkl"
        )
        joblib.dump(grid_obj, str)
        return grid_obj

    elif bayes_tf == True:
        print("........starting bayes search........")
        bayes_obj = bayes(x, y, method=algo, des=des)
        uuid_temp = uuid.uuid4()
        str = (
            "../data/train/bayes/complete_bayes_"
            + algo
            + "_"
            + des
            + "_"
            + uuid_temp.urn[9:]
            + ".pkl"
        )
        joblib.dump(bayes_obj, str)
        return bayes_obj

    elif sigopt_tf == True:

        print("........starting sigopt bayes search........")
        bayes_obj = bayes_sigopt(x, y, method=algo)
        uuid_temp = uuid.uuid4()
        str = (
            "../data/train/bayes/complete_bayes_"
            + algo
            + "_"
            + des
            + "_"
            + uuid_temp.urn[9:]
            + ".pkl"
        )
        joblib.dump(bayes_obj, str)
        return bayes_obj

    else:
        print("........starting single algo evaluation........")
        if algo == "nn":
            print("nn reg selected")
            reg = sk_nn(x, y, scale)
        elif algo == "rf":
            print("random forest selected")
            reg = random_forest(x, y, scale)
        elif algo == "extra":
            print("extra trees selected")
            reg = extra_trees(x, y, scale)
        elif algo == "grad":
            print("grad algo selected")
            reg = gradient_boost_reg(x, y, scale)
        elif algo == "svr":
            print("svr algo selected")
            reg = svr(x, y, scale)
        elif algo == "bayes":
            print("bayes regression selected")
            reg = bayesian(x, y, scale)
        elif algo == "kernel":
            print("kernel regression selected")
            reg = kernel(x, y, scale)
        elif algo == "gaussian":
            print("gaussian algo selected")
            reg = gaussian(x, y, scale)
        elif algo == "xgboost":
            from source.utils.xgboost_util import xgboost

            print("xgboost algo selected")
            reg = xgboost(x, y, scale)
        elif algo == "tf_nn":
            from source.utils.tensorflow_util import nn_basic

            x = x.astype("float32")
            y = y.astype("float32")

            reg = nn_basic(x, y, scale)
        elif algo == "tf_cnn":
            from source.utils.tensorflow_util import cnn_basic

            x = x.astype("float32")
            y = y.astype("float32")

            reg = cnn_basic(x, y, scale)
        elif algo == "tf_cnn_norm":
            from source.utils.tensorflow_util import cnn_norm_basic

            x = x.astype("float32")
            y = y.astype("float32")

            reg = cnn_norm_basic(x, y, scale)
        elif algo == "resnet":
            from source.utils.tensorflow_util import resnet34

            x = x.astype("float32")
            y = y.astype("float32")

            reg = resnet34(x, y, scale)
        else:
            print("stochastic gradient descent selected")
            reg = sgd(x, y, scale)
        return reg



class optimizer_genetic(object):
    def __init__(
            self, population, alphabet, largest_selfies_len, mut_prob=0.1, steps=10, start_pop_size=100, ckpt = True
    ):
        self.ckpt = ckpt
        self.mut_prob = mut_prob
        self.steps = steps
        self.total_population = population
        self.start_pop_size = start_pop_size
        self.selfies_alphabet = alphabet
        self.largest_selfies_len = largest_selfies_len
        self.population_sample = [
            population[int(i)]
            for i in np.random.randint(
                0, len(population), int(start_pop_size)
            )
        ]


    def vect_to_struct(self, coordinates, selfies = True):
        self_out = sf.encoding_to_selfies(
            coordinates.tolist(), self.selfies_alphabet, enc_type="one_hot"
        )
        if (selfies == True):
            decoded_smiles = sf.decoder(self_out)
        else:
            decoded_smiles = self_out
        return decoded_smiles

    def loss(self, smiles):

        #df, df_var_thresh, x_scale = desc_calc([smiles])
        #psb_desc = self.psb_scaler_x.transform(df[self.psb_keys].to_numpy())

        x_mat = self.scaler_x.transform()
        homo_pred = self.homo_model.predict(x_mat)
        homo1_pred = self.homo1_model.predict(x_mat)

        return integration_val

    def struct_to_latent(self, smiles):
        data = multiple_selfies_to_hot(
            [sf.encoder(smiles)], self.largest_selfies_len, self.selfies_alphabet
        )
        data_reshape = data.reshape(
            1,
            self.data_dim_1 * self.data_dim_2,
        )
        encoded = self.encoder.predict(data_reshape)[0]
        return encoded

    def train_models(
            self, test=False, transfer=False, ret_list=[]
    ):
        des = 'persist'
        print("done processing dataframe")
        str = "../data/desc/DB3/desc_calc_DB3_" + des + ".h5"
        print(str)
        # all of the models
        df = pd.read_pickle(str)

        print(len(df))
        print(df.head())
        HOMO = df["HOMO"].to_numpy()
        HOMO_1 = df["HOMO-1"].to_numpy()
        diff = df["diff"].to_numpy()


        if des == "vae":
            temp = df["mat"].tolist()
            mat = list([i.flatten() for i in temp])

        elif des == "auto":
            temp = df["mat"].tolist()
            mat = list([i.flatten() for i in temp])
        else:
            mat = df["mat"].to_numpy()

        try:
            mat = preprocessing.scale(np.array(mat))
        except:
            mat = list(mat)
            mat = preprocessing.scale(np.array(mat))

        print("Using " + des + " as the descriptor")
        print("Matrix Dimensions: {0}".format(np.shape(mat)))

        # finish optimization

        des = des + "_homo"
        print(".........................HOMO..................")

        self.scale_homo = np.max(HOMO) - np.min(HOMO)
        self.min_homo = np.min(HOMO)
        self.scale_homo1 = np.max(HOMO) - np.min(HOMO)
        self.min_homo1 = np.min(HOMO)
        HOMO = (HOMO - self.min_homo ) / self.scale_HOMO
        self.homo_model = calc(
            mat, HOMO, des, self.scale_homo, grid_tf, bayes_tf, sigopt_tf, algo
        )
        self.homo1_model = calc(
            mat, HOMO_1, des, self.scale_homo1, rand_tf, grid_tf, bayes_tf, sigopt_tf, algo
        )
        x_train_cl2 = self.cl2_scaler_x.transform(np.array(x_subset_cl2))
        y_train_cl2 = self.cl2_scaler_y.transform(
            np.array(y_subset_cl2["CL2"]).reshape(-1, 1)
        )

        if test == True:
            x_train_psb, x_test_psb, y_train_psb, y_test_psb = train_test_split(
                x_train_psb, y_train_psb, random_state=0
            )

        else:
            self.model_cl2.fit(x_train_cl2, y_train_cl2)
            self.model_pst.fit(x_train_pst, y_train_pst)
            self.model_kpt.fit(x_train_kpt, y_train_kpt)
            self.model_psb.fit(x_train_psb, y_train_psb)
            self.model_kpcsf.fit(x_train_kpcsf, y_train_kpcsf)

        ########################################################### cl (20)
        # this gets index of the values for pst, to get newer transfer learning
        if transfer == True:
            x_subset_kpt_to_cl = np.nan_to_num(
                df[self.kpt_keys].iloc[no_outlier_ind_cl]
            )
            x_subset_cl2_to_cl = np.nan_to_num(
                df[self.cl2_keys].iloc[no_outlier_ind_cl]
            )
            x_transfer_desc_2 = self.kpt_scaler_x.transform(x_subset_kpt_to_cl)
            x_transfer_desc_1 = self.cl2_scaler_x.transform(x_subset_cl2_to_cl)
            transfer_out_2 = np.array(
                [
                    self.model_kpt.predict(np.array(i).reshape(1, -1)).ravel()
                    for i in x_transfer_desc_2
                ]
            ).reshape(-1, 1)
            transfer_out_1 = np.array(
                [
                    self.model_cl2.predict(np.array(i).reshape(1, -1)).ravel()
                    for i in x_transfer_desc_1
                ]
            ).reshape(-1, 1)
            transfer = np.array([transfer_out_1, transfer_out_2]).reshape(-1, 2)
            x_train_cl = np.append(x_train_cl, transfer, axis=1)  # add cl2, kpt

        ########################################################### kpisf (10/21)
        if test == True:
            x_train_cl, x_test_cl, y_train_cl, y_test_cl = train_test_split(
                x_train_cl, y_train_cl, random_state=0
            )
            x_train_kpisf, x_test_kpisf, y_train_kpisf, y_test_kpisf = train_test_split(
                x_train_kpisf, y_train_kpisf, random_state=0
            )

        ############################################ Testing performance
        if test == True:
            pred_train = self.model_cl2.predict(x_train_cl2)
            pred_test = self.model_cl2.predict(x_test_cl2)
            score_cl2_train = str(r2_score(pred_train, y_train_cl2))
            score_cl2_test = str(r2_score(pred_test, y_test_cl2))


            table_data = [
                ["Set", "HOMO", "HOMO_1", "gap"],
                [
                    "Train",
                    score_homo_train,
                    score_homo1_train,
                    score_gap_train,
                ],
                [
                    "Test",
                    score_homo_test,
                    score_homo1_test,
                    score_gap_test,
                ],
            ]
            table = AsciiTable(table_data)
            print(table.table)
            ret_list = [
                score_cl2_test,
                score_pst_test,
                score_kpt_test,
                score_psb_test,
                score_cl_test,
                score_kpcsf_test,
                score_kpisf_test,
            ]
        else:
            pred_train = self.model_cl2.predict(x_train_cl2)
            score_cl2_train = str(r2_score(pred_train, y_train_cl2))

            pred_train = self.model_pst.predict(x_train_pst)
            score_pst_train = str(r2_score(pred_train, y_train_pst))

            pred_train = self.model_kpt.predict(x_train_kpt)
            score_kpt_train = str(r2_score(pred_train, y_train_kpt))

            pred_train = self.model_psb.predict(x_train_psb)
            score_psb_train = str(r2_score(pred_train, y_train_psb))

            pred_train = self.model_cl.predict(x_train_cl)
            score_cl_train = str(r2_score(pred_train, y_train_cl))

            pred_train = self.model_kpcsf.predict(x_train_kpcsf)
            score_kpcsf_train = str(r2_score(pred_train, y_train_kpcsf))

            pred_train = self.model_kpisf.predict(x_train_kpisf)
            score_kpisf_train = str(r2_score(pred_train, y_train_kpisf))

            table_data = [
                ["Set", "CL2", "PST", "KpT", "PSB", "CL", "KpCSF", "KpISF"],
                [
                    "Train",
                    score_cl2_train,
                    score_pst_train,
                    score_kpt_train,
                    score_psb_train,
                    score_cl_train,
                    score_kpcsf_train,
                    score_kpisf_train,
                ],
            ]
            table = AsciiTable(table_data)
            print(table.table)

            ret_list = [
                score_cl2_train,
                score_pst_train,
                score_kpt_train,
                score_psb_train,
                score_cl_train,
                score_kpcsf_train,
                score_kpisf_train,
            ]
        return ret_list

    def selection(self):
        pop_loss = []
        pop_new = []
        total_loss = 0.0
        ratio_children = 0.5
        population = self.population_sample
        successful_mol_count = 0

        for ind in tqdm(range(len(population))):
            pop_temp = []
            i = population[ind]
            self_i = sf.encoding_to_selfies(
                i.tolist(), self.selfies_alphabet, "one_hot"
            )
            try:
                smiles_i = sf.decoder(self_i)
                temp_loss = self.loss(smiles_i)
                pop_loss.append(temp_loss)
                pop_temp.append(temp_loss)
                total_loss += temp_loss
                successful_mol_count += 1
            except:
                print("---------invalid molecule @ selection for keys---------")
                np.delete(population, ind)

        parent_ind, parent_gen_loss, parent_prob_dist = [], [], []
        pop_loss_temp = pop_loss

        while len(parent_ind) <= int(successful_mol_count * ratio_children + 1):
            log = True
            draw = draw_from_pop_dist(pop_loss_temp, log=True)
            if (parent_ind.count(draw) == 0):
                parent_ind.append(draw)
                parent_gen_loss.append(pop_loss[draw])
                if (log == False):
                    pop_loss_temp[draw] = 0
                else:
                    pop_loss_temp[draw] = 1.1

        if len(parent_ind) % 2 == 1:
            parent_ind = parent_ind[0:-1]
            parent_gen_loss = parent_gen_loss[0:-1]

        parent_gen = [population[i] for i in parent_ind]
        parent_gen_order = np.array(parent_gen_loss).argsort().tolist()[::-1]
        parent_gen_index_tracker = [i for i in range(len(parent_gen))]

        # here we want to shuffle indexes, not take just the best in order

        for i in range(int(len(parent_gen) / 2)):
            draw1 = random.choice(parent_gen_index_tracker)
            parent_gen_index_tracker.remove(draw1)
            draw2 = random.choice(parent_gen_index_tracker)
            parent_gen_index_tracker.remove(draw2)
            cross_res1, cross_res2 = cross(parent_gen[draw1], parent_gen[draw2])

            pop_new.append(cross_res1)
            pop_new.append(cross_res2)

        [pop_new.append(parent_gen[i])
         for i in parent_gen_order[0:int(len(parent_gen_order) / 2 + 1)]]

        return pop_new

    def enrich_pop(self,  gens=2, longitudnal_save=False):
        mean_loss_arr = []
        pop_loss = []
        df_temp_list = []
        total_loss = 0
        successful_mol_count = 0
        write_to_file = str(self.start_pop_size) + "_start_" + str(gens) + "_gens.h5"
        prev_file = os.path.exists(write_to_file)

        if (prev_file == True and self.ckpt == True):
            df = pd.read_hdf(write_to_file)
            gen_mols = df["str"][df["gen"] == df['gen'].max()].tolist()
            selfies = [sf.encoder(smiles) for smiles in gen_mols]
            gen_start = df['gen'].max()
            data = multiple_selfies_to_hot(selfies, self.largest_selfies_len, self.selfies_alphabet)
            print(len(data))
            self.population_sample = [np.array(i) for i in data]

            print("resuming from generation: " + str(gen_start) + " with [" + str(len(self.population_sample)) + "] molecules")

        else:
           gen_start = 1


        for gen in range(gen_start, gens + 1):

            if (int(self.start_pop_size * 0.1) > len(self.population_sample)):
                print("terminating population study early at gen" + str(gen))
                break
            print("Gen " + str(gen) + "/" + str(gens))
            print("selection + cross...")
            pop_new = self.selection()
            pop_mut_new = []
            pop_loss = []

            print("mutation...")
            for i in pop_new:
                pop_mut_new.append(random_mutation(i, mut_chance=0.2))

            self.population_sample = pop_mut_new

            smiles_scored = []
            print("computing performance of new generation...")
            for ind in tqdm(range(len(pop_mut_new))):
                i = pop_mut_new[ind].tolist()
                try:
                    self_i = sf.encoding_to_selfies(
                        i, self.selfies_alphabet, "one_hot")
                    smiles_i = sf.decoder(self_i)
                    temp_loss = self.pk_model(smiles_i)
                    smiles_scored.append(smiles_i)
                    pop_loss.append(temp_loss)
                    total_loss += temp_loss
                    successful_mol_count += 1
                except:
                    print("---------invalid molecule in new gen---------")
                    pop_mut_new = np.delete(pop_mut_new, ind)
            print(len(pop_mut_new))

            total_loss = 0
            mean_loss_arr.append(np.array(pop_loss).mean())
            print('mean loss: ' + str(np.array(pop_loss).mean()))
            print("GENERATION " + str(gen) + " COMPLETED")

            df_temp_list = []

            if (longitudnal_save == True):
                print("saving generation for longitudinal study")
                for ind, element in enumerate(smiles_scored):
                    df_temp_row = [element, pop_loss[ind], gen]
                    df_temp_list.append(df_temp_row)

                if (os.path.exists(write_to_file)):
                    df_old = pd.read_hdf(write_to_file)
                    df = pd.DataFrame(df_temp_list, columns=['str', 'loss', 'gen'])
                    df = df_old.append(df, ignore_index = True)
                    df.to_hdf(write_to_file, key='df')

                else:
                    df = pd.DataFrame(df_temp_list, columns=['str', 'loss', 'gen'])
                    df.to_hdf(write_to_file, key='df')



        return mean_loss_arr


if __name__ == "__main__":

    mols = Chem.SDMolSupplier("./data/mols_filter.sdf")
    mols_smiles = []
    for mol in mols:
        if mol is None:
            continue
        mols_smiles.append(Chem.MolToSmiles(mol))

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
        "-mols_data", action="store", dest="mols_data", default=1000, help="mols in population"
    )
    parser.add_argument(
        "-start_pop", action="store", dest="start_pop_size", default=20, help="start pop size"
    )

    parser.add_argument(
        "-gens", action="store", dest="gens", default=2, help="generations"
    )

    results = parser.parse_args()
    long = bool(results.longitudnal)
    print("Longitudinal study:" + str(long))
    mols_smiles = mols_smiles[0:int(results.mols_data)]
    gens = int(results.gens)
    start_pop_size = int(results.start_pop_size)
    ckpt = bool(results.ckpt)
    print("number of molecules in sample: " + str(len(mols_smiles)))
    (
        selfies_list,
        selfies_alphabet,
        largest_selfies_len,
        smiles_list,
        smiles_alphabet,
        largest_smiles_len,
    ) = get_selfie_and_smiles_encodings_for_dataset(mols_smiles)

    data = multiple_selfies_to_hot(selfies_list, largest_selfies_len, selfies_alphabet)

    opt = optimizer_genetic(data, selfies_alphabet, largest_selfies_len, ckpt = ckpt, start_pop_size=start_pop_size)

    model_performance = []
    retrain = False

    while True:
        model_performance = opt.train_models(test=True, retrain=retrain, ret_list=model_performance)
        if all(float(i) >= 0.05 for i in model_performance):
            break
        retrain = True

    mean_loss = opt.enrich_pop(gens=gens, longitudnal_save=long)
    print(":" * 50)
    print("Mean Loss array")
    print(":" * 50)
    print(mean_loss)
    final_molecules = opt.population_sample
    final_molecules_selfies = [opt.vect_to_struct(i) for i in final_molecules]
    print("number of molecules at the end: " + str(len(final_molecules_selfies)))


    df = pd.read_hdf(str(start_pop_size) + "_start_" + str(gens) + "_gens.h5")
    print("Performance averages")
    len_gens = df['gen'].max()

    for i in range(len_gens):
        print("Stats for gen: " + str(i+1))
        print("mean: " + str(df["loss"][df['gen'] == int(i + 1)].mean()))
        print("max: " + str(df["loss"][df['gen'] == int(i+1)].max()))


