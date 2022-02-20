import tensorflow as tf
import tensorflow.keras as keras

import os
import uuid 
import joblib
import argparse
import numpy as np
import pandas as pd
import selfies as sf
sf.set_semantic_constraints("hypervalent")

from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
from terminaltables import AsciiTable

from sklearn.svm import SVR, LinearSVR
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

from Element_PI import VariancePersistv1

from utils.sklearn_util import *
from utils.genetic_util import *
from utils.selfies_util import (
    multiple_selfies_to_hot,
    get_selfie_and_smiles_encodings_for_dataset,
    selfies_to_hot,
)
from utils.helpers import quinone_check



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
            from utils.xgboost_util import xgboost

            print("xgboost algo selected")
            reg = xgboost(x, y, scale)
        elif algo == "tf_nn":

            x = x.astype("float32")
            y = y.astype("float32")

            reg = nn_basic(x, y, scale)
        elif algo == "tf_cnn":


            x = x.astype("float32")
            y = y.astype("float32")

            reg = cnn_basic(x, y, scale)
        elif algo == "tf_cnn_norm":

            x = x.astype("float32")
            y = y.astype("float32")

            reg = cnn_norm_basic(x, y, scale)
        elif algo == "resnet":

            x = x.astype("float32")
            y = y.astype("float32")

            reg = resnet34(x, y, scale)
        else:
            print("stochastic gradient descent selected")
            reg = sgd(x, y, scale)
        return reg



class optimizer_vae(object):

    def __init__(self, data, selfies_alphabet, longi=True, ckpt=True, test_vae=False, gens=2, start_pop_size=100, population_sample = []):
        self.encoder = keras.models.load_model("./encoder")  # need to get newer vae
        self.decoder = keras.models.load_model("./decoder")
        self.ckpt = ckpt
        self.longi = longi
        self.gens = gens
        self.start_pop_size = start_pop_size
        self.selfies_alphabet = selfies_alphabet
        self.start_pop_size = start_pop_size
        self.population = data
        self.max_mol_len = data.shape[1]
        self.alpha_len = data.shape[2]
        if(population_sample == []):
            self.population_sample = [
            self.population[int(i)]
            for i in np.random.randint(0, len(self.population), int(self.start_pop_size))
            ]
        else:
             self.population_sample = population_sample

        if test_vae == True:
            self.test()
        else:
            self.stdev = [2 for _ in range(75)]


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
        mol_obj = Chem.MolFromSmiles(smiles)
        if (self.desc == 'persist'):
            x_mat = VariancePersistv1(
                mol_obj,
                pixelx=50, pixely=50,
                myspread=0.28, myspecs={"maxBD": 2.5, "minBD": -.10}, showplot=False)
            x_mat = self.scaler.transform(x_mat.reshape(1, -1))
            if (self.algo == 'tf_cnn' or self.algo == 'tf_nn'):
                x_mat = x_mat.reshape((np.shape(x_mat)[0], 50, 50))
                x_mat = np.expand_dims(x_mat, -1)

        else:
            #x_mat = AllChem.GetMorganFingerprint(mol_obj, int(2))
            bit_obj = AllChem.GetMorganFingerprintAsBitVect(mol_obj, 2, nBits=int(1024))
            x_mat = np.array([int(i) for i in bit_obj]).reshape(1, -1)

        homo_pred = self.homo_model.predict(x_mat)[0]
        homo1_pred = self.homo1_model.predict(x_mat)[0]
        quinone_tf = quinone_check(smiles)

        return np.abs(homo_pred) + np.abs(homo_pred - homo1_pred) , quinone_tf


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
        #des = 'persist'
        des = self.desc
        print("done processing dataframe")
        str = "../data/desc/DB3/desc_calc_DB3_" + des + ".h5"
        print(str)
        # all of the models
        df = pd.read_hdf(str)

        print(len(df))
        print(df.head())
        HOMO = df["HOMO"].to_numpy()
        HOMO_1 = df["HOMO-1"].to_numpy()
        #diff = df["diff"].to_numpy()

        if des == "vae":
            temp = df["mat"].tolist()
            mat = list([i.flatten() for i in temp])

        elif des == "auto":
            temp = df["mat"].tolist()
            mat = list([i.flatten() for i in temp])
        else:
            mat = df["mat"].to_numpy()

        try:
            self.scaler = StandardScaler()
            self.scaler.fit(np.array(mat))
            mat = self.scaler.transform(X = np.array(mat))

        except:
            mat = list(mat)
            self.scaler = StandardScaler()
            self.scaler.fit(np.array(mat))
            mat = self.scaler.transform(X = np.array(mat))

        print("Using " + des + " as the descriptor")
        print("Matrix Dimensions: {0}".format(np.shape(mat)))

        # finish optimization

        des = des + "_homo"
        print(".........................HOMO..................")

        self.scale_homo = np.max(HOMO) - np.min(HOMO)
        self.scale_homo1 = np.max(HOMO_1) - np.min(HOMO_1)

        self.min_homo = np.min(HOMO)
        self.min_homo1 = np.min(HOMO)

        HOMO = (HOMO - self.min_homo ) / self.scale_homo
        HOMO_1 = (HOMO_1 - self.min_homo1 ) / self.scale_homo1
        #from utils.tensorflow_util import cnn_basic
        algo = self.algo

        self.homo_model = calc(
            mat, HOMO, des, self.scale_homo, algo = algo
        )
        self.homo1_model = calc(
            mat, HOMO_1, des, self.scale_homo1, algo = algo
        )

    def smiles_to_encoded(self, smiles):
        onehot_encoded = self.smiles_to_one_hot(smiles)

        onehot_encoded = onehot_encoded.reshape(
            1, self.max_mol_len * self.alpha_len,
        )
        encoded = self.encoder.predict(onehot_encoded)[0][0]
        return encoded

    def encoding_to_smiles(self, encoded):
        encoded = encoded.reshape(-1, 75)
        #print(encoded)
        decode = self.decoder.predict(encoded)

        try:
            decode = self.decoder.predict(encoded)

        except:
            decode = self.decoder.predict(encoded[0])

            # single point - through vae
        one_hot = np.zeros((self.max_mol_len, self.alpha_len))

        for ind, row in enumerate(decode.reshape(self.max_mol_len, self.alpha_len)):
            lab_temp = np.argmax(row)
            one_hot[ind][lab_temp] = 1

        self_test = sf.encoding_to_selfies(one_hot.tolist(), self.selfies_alphabet, "one_hot")
        smiles_decoded = sf.decoder(self_test)
        canonical_smiles = Chem.CanonSmiles(smiles_decoded)
        return canonical_smiles

    def one_hot_to_smiles(self, one_hot):
        try:
            self_i = sf.encoding_to_selfies(one_hot.tolist(), self.selfies_alphabet, "one_hot")
        except:
            self_i = sf.encoding_to_selfies(one_hot, self.selfies_alphabet, "one_hot")

        smiles_i = sf.decoder(self_i)
        return smiles_i

    def smiles_to_one_hot(self, smiles):
        selfies = sf.encoder(smiles)
        _, onehot_encoded = selfies_to_hot(selfies, self.max_mol_len, self.selfies_alphabet)
        onehot_encoded  = onehot_encoded.reshape(1, self.max_mol_len, self.alpha_len)
        return onehot_encoded

    def encoded_to_loss(self, encoded, save_calc_smiles):
        canonical_smiles = self.encoding_to_smiles(encoded)
        if(canonical_smiles == save_calc_smiles):
            return -1
        else:
            return self.loss(canonical_smiles)

    def pull_gradient(self, smiles):
        y_0 = self.loss(smiles) # for refrence molecules 
        x_0 = self.smiles_to_encoded(smiles)
        dy_arr = []
        dx_arr = []
        for ind, i in enumerate(self.stdev):
            encoded_x_0 = x_0
            encoded_x_0[ind] += i
            dx_arr.append(encoded_x_0)
            try:
                encoded_loss = self.encoded_to_loss(encoded_x_0, smiles)
            except:
                print("failed subgradient")
                encoded_loss = -1    
            if (encoded_loss == -1):
                dy_arr.append(0)
            else:
                dy_arr.append(encoded_loss)
        dy_arr = np.array(dy_arr) - y_0 
        dy_arr /= (self.stdev)
        return dy_arr, dx_arr

    def grad_iter(self):
        pop_new = []
        smiles_ret = []
        succ_ind = []
        for mol_ind, molecule in enumerate(self.population_sample):
            ind = 0
            filter_val = False
            print("--------------------------------------------")
            smiles = self.one_hot_to_smiles(molecule)
            dy_dx_arr, encoded_grad_arr = self.pull_gradient(smiles)
            order_arr = np.argsort(dy_dx_arr)

            while(not(filter_val)):
                try:
                    smiles_grad = self.encoding_to_smiles(encoded_grad_arr[order_arr[ind]]) # works 
                    filter_val = quinone_check(smiles_grad) # works

                except:
                    print("failed molecular conversion")
                    filter_val = False

                if(not(filter_val)):
                    print("non lipinski, redrawing")
                else:
                    try:
                        smiles_ret.append(smiles_grad) 
                        succ_ind.append(mol_ind)
                        print(smiles, smiles_grad)
                    except:
                        print("alphabet missing element likely")
                ind += 1
                if (ind == len(dy_dx_arr)):
                    print("no valid molecules around this seed")
                    filter_val = True

        selfies_ret = [sf.encoder(i) for i in smiles_ret]
        data = multiple_selfies_to_hot(selfies_ret, self.max_mol_len, self.selfies_alphabet)
        self.population_sample = [i for i in data]
        return [i for i in data], smiles_ret, succ_ind

    def enrich_pop(self):
        gens = self.gens
        mean_loss_arr = []
        total_loss = 0
        successful_mol_count = 0
        write_to_file = "vae_" + str(self.start_pop_size) + "_start_" + str(gens) + "_gens.h5"
        prev_file = os.path.exists(write_to_file)

        if prev_file == True and self.ckpt == True:
            df = pd.read_hdf(write_to_file)
            gen_mols = df["str"][df["gen"] == df["gen"].max()].tolist()
            selfies = [sf.encoder(smiles) for smiles in gen_mols]
            gen_start = df["gen"].max()
            data = multiple_selfies_to_hot(
                selfies, self.max_mol_len, self.selfies_alphabet
            )
            self.population_sample = [np.array(i) for i in data]
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
            if 3 > len(self.population_sample):
                print("terminating population study early at gen" + str(gen))
                break

            print("Gen " + str(gen) + "/" + str(gens))
            print("gradient computation + ...")
            pop_new, smiles_new, succ_ind = self.grad_iter()

            pop_loss = []
            lip_score = []
            smiles_scored = []
            success_mol_track = []
            print("computing performance of new generation...")
            for ind in tqdm(range(len(pop_new))):
                smiles = smiles_new[ind] 
                try:
                    temp_loss = self.loss(smiles)
                    smiles_scored.append(smiles)
                    pop_loss.append(temp_loss)
                    total_loss += temp_loss
                    successful_mol_count += 1
                    success_mol_track.append(succ_ind[ind])
                except:
                    print("---------invalid molecule in new gen---------")
                    pop_new = np.delete(pop_new, ind)
            print(len(pop_new))

            total_loss = 0
            mean_loss_arr.append(np.array(pop_loss).mean())
            print("mean loss: " + str(np.array(pop_loss).mean()))
            print("GENERATION " + str(gen) + " COMPLETED")

            df_temp_list = []

            if self.longi == True:
                print("saving generation for longitudinal study")
                for ind, element in enumerate(smiles_scored):
                    df_temp_row = [element, pop_loss[ind], lip_score[ind], success_mol_track[ind], gen]
                    df_temp_list.append(df_temp_row)

                if os.path.exists(write_to_file):
                    df_old = pd.read_hdf(write_to_file)
                    df = pd.DataFrame(
                        df_temp_list, columns=["str", "loss", "lip_score", "track", "gen"]
                    )
                    df = df_old.append(df, ignore_index=True)
                    df.to_hdf(write_to_file, key="df")

                else:
                    df = pd.DataFrame(
                        df_temp_list, columns=["str", "loss", "lip_score", "track", "gen"]
                    )
                    df.to_hdf(write_to_file, key="df")

        return mean_loss_arr

    def test(self):
        data_reshape = self.population.reshape(
            self.population.shape[0],
            self.max_mol_len * self.alpha_len,
        )
        train_ind, test_ind = train_test_split(range(self.population.shape[0]), test_size=0.15)
        x_train = data_reshape[train_ind]
        x_test = data_reshape[test_ind]

        train_data = x_train.reshape(x_train.shape[0], self.max_mol_len * self.alpha_len)
        test_data = x_test.reshape(x_test.shape[0], self.max_mol_len * self.alpha_len)

        encoder_train = self.encoder.predict(train_data)
        encoder_test = self.encoder.predict(test_data)
        try:
            code_decode_train = self.decoder.predict(encoder_train)
            code_decode_test = self.decoder.predict(encoder_test)
        except:
            code_decode_train = self.decoder.predict(encoder_train[0])
            code_decode_test = self.decoder.predict(encoder_test[0])
        print("..............statistics for training dataset..............")
        compare_equality(
            train_data,
            code_decode_train,
            (self.max_mol_len, self.alpha_len),
            self.selfies_alphabet,
        )
        print("..............statistics for test dataset..............")
        compare_equality(
            test_data,
            code_decode_test,
            (self.max_mol_len, self.alpha_len),
            self.selfies_alphabet,
        )
        print(np.shape(np.var(code_decode_train, axis = 1)))
        self.stdev = np.var(code_decode_train, axis = 1)

    def hall_of_fame(self, pop, loss):
        max_temp = -1
        max_temp_ind = -1
        print(len(pop))
        print(len(loss))
        for ind, i in enumerate(pop):
            if loss[ind] > self.max_loss and loss[ind] > max_temp:
                max_temp = loss[ind]
                max_temp_ind = ind

        if max_temp_ind != -1:
            self.max_indiv = pop[max_temp_ind]
            self.max = max_temp

        return self.max_indiv



if __name__ == "__main__":
    from utils.selfies_util import smiles
    mols_smiles = pd.read_hdf('../data/desc/DB3/desc_calc_DB3_self.h5')['name'].tolist()
    parser = argparse.ArgumentParser(description="parameters of VAE study")

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
    parser.add_argument(
        "--test",
        action="store_true",
        dest="test",
        help="enforce lower bound on regre algos",
    )
    parser.add_argument(
        "--random", action="store_true", dest="random_bool", help="random crossing"
    )

    results = parser.parse_args()
    long = bool(results.longitudnal)
    test = bool(results.test)
    print("Longitudinal study:" + str(long))
    gens = int(results.gens)
    start_pop_size = int(results.start_pop_size)
    ckpt = bool(results.ckpt)
    random_bool = bool(results.random_bool)
    algo = str(results.algo)
    desc = str(results.desc)
    
    
    mols_smiles = mols_smiles[0:int(results.mols_data)]
    SMILES_clean = list(filter(None, mols_smiles))

    
    (
        selfies_list,
        selfies_alphabet,
        largest_selfies_len,
        smiles_list,
        smiles_alphabet,
        largest_smiles_len,
    ) = get_selfie_and_smiles_encodings_for_dataset(SMILES_clean)

    text_file = open("vae_alpha.txt", "r")
    selfies_alphabet = [l.strip("\n") for l in text_file.readlines()]
    data = multiple_selfies_to_hot(selfies_list, largest_selfies_len, selfies_alphabet)
    data_merck = multiple_selfies_to_hot([sf.encoder(i) for i in SMILES_clean], largest_selfies_len, selfies_alphabet)

    opt = optimizer_vae(
        data,
        selfies_alphabet,
        ckpt=True,
        longi=True,
        gens= gens,
        start_pop_size=start_pop_size,
        population_sample = [i for i in data_merck]
    )

    model_performance = []
    retrain = False
    test = False
    if test == False:
        while True:
            model_performance = opt.train_models(
                test=True, retrain=retrain, ret_list=model_performance
            )
            if all(float(i) >= 0.1 for i in model_performance):
                break
            retrain = True
    else:
        model_performance = opt.train_models(
            test=True, retrain=retrain, ret_list=model_performance
        )

    opt.enrich_pop()