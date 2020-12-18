import joblib, argparse, uuid, sigopt
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn_utils import *

# todo: work on interpretability algo/aspects
# todo: plots of parameter space
# todo: process zz's stuff


def process_input_DB2(dir="DB2", desc="rdkit"):
    try:
        str = "../data/desc/" + dir + "/desc_calc_DB2_" + desc + ".pkl"
        print(str)
        df = pd.read_pickle(str)
        pkl = 1

    except:
        str = "../data/desc/" + dir + "/desc_calc_DB2_" + desc + ".h5"
        df = pd.read_hdf(str)
        pkl = 0

    df["HOMO"] = ""
    df["HOMO-1"] = ""
    df["diff"] = ""
    print(df.head())
    list_to_sort = []
    with open("../data/benzoquinone_DB/DATA_copy") as fp:
        line = fp.readline()
        while line:
            list_to_sort.append(line[0:-2])
            line = fp.readline()
    list_to_sort.sort()

    for i in range(df.copy().shape[0]):
        for j in list_to_sort:

            # print(df["name"].iloc[i][:-4])
            # print(j.split(":")[0])
            if (desc == "auto"):
                if (df["name"].iloc[i].split("/")[-1][:-4] in j.split(";")[0]):
                    temp1 = float(j.split(":")[1])
                    temp2 = float(j.split(":")[2])
                    df["HOMO"].loc[i] = float(j.split(":")[1])
                    df["HOMO-1"].loc[i] = float(j.split(":")[2])
                    df["diff"].loc[i] = temp2 - temp1
                    print(temp2 - temp1)

            if (df["name"].iloc[i][:-4] in j.split(";")[0]):
                temp1 = float(j.split(":")[1])
                temp2 = float(j.split(":")[2])

                print(j)

                df["HOMO"].loc[i] = float(j.split(":")[1])
                df["HOMO-1"].loc[i] = float(j.split(":")[2])
                df["diff"].loc[i] = temp2 - temp1
    if (pkl == 0):
        df.to_hdf(str, key="df", mode='a')
    else:
        df.to_pickle(str)

# TODO : this, once ZZ finishes getting his data
def process_input_ZZ(dir="ZZ", desc="vae"):
    # todo: process ZZ's
    return 0

def process_input_DB3(dir="DB3", desc="rdkit"):
    try:
        str = "../data/desc/" + dir + "/desc_calc_"+ dir +"_" + desc + ".h5"
        df = pd.read_hdf(str)
        pkl = 0

    except:
        str = "../data/desc/" + dir + "/desc_calc_"+ dir +"_" + desc + ".pkl"
        df = pd.read_pickle(str)
        pkl = 1

    print(df.head())
    df["HOMO"] = 0
    df["HOMO-1"] = 0
    df["diff"] = 0
    list_to_sort = []
    with open("../data/DATA_DB3") as fp:
        line = fp.readline()
        while line:
            list_to_sort.append(line[0:-2])
            line = fp.readline()
    list_to_sort.sort()
    print("Dimensions of df {0}".format(np.shape(df)))

    for i in range(df.copy().shape[0]):
        for j in list_to_sort:
            if (desc == "auto"):
                print()
                if (df["name"].iloc[i].split("/")[-1][:-4] in j.split(";")[0]):
                    temp1 = float(j.split(":")[1])
                    temp2 = float(j.split(":")[2])
                    df["HOMO"].loc[i] = float(j.split(":")[1])
                    df["HOMO-1"].loc[i] = float(j.split(":")[2])
                    df["diff"].loc[i] = temp2 - temp1
                    print(temp2 - temp1)


            if (df["name"].iloc[i][:-4] in j.split(";")[0] and j[0:2] != "--"):
                temp1 = float(j.split(":")[1])
                temp2 = float(j.split(":")[2])
                df["HOMO"].loc[i] = float(j.split(":")[1])
                df["HOMO-1"].loc[i] = float(j.split(":")[2])
                df["diff"].loc[i] = temp2 - temp1

            shift = 1
            if (df["name"].iloc[i][0:4] == "tris"):
                shift = 5
            elif(df["name"].iloc[i][0:5] == "tetra"):
                shift = 6
            elif (df["name"].iloc[i][0:6] == "bis-23"):
                shift = 7
            elif (df["name"].iloc[i][0:6] == "bis-25"):
                shift = 7
            elif (df["name"].iloc[i][0:6] == "bis-26"):
                shift = 7
            else:
                pass

            if(df["name"].iloc[i][shift:-4] in j.split(";")[0] and j[0:2] != "--"):
                print(j)
                temp1 = float(j.split(":")[1])
                temp2 = float(j.split(":")[2])
                df["HOMO"].loc[i] = float(j.split(":")[1])
                df["HOMO-1"].loc[i] = float(j.split(":")[2])
                df["diff"].loc[i] = temp2 - temp1

    print(df.head())

    if (pkl == 0):
        df.to_hdf(str, key="df", mode='a')
    else:
        df.to_pickle(str)

def calc(x, y, des, scale, rand_tf = False, grid_tf=False, bayes_tf=False, sigopt_tf = False, algo="sgd"):

    if (rand_tf == True):
        #todo incorp all sklearn algos here
        rand(x,y, algo,  des)

    if (grid_tf == True):
        print("........starting grid search........")
        grid_obj = grid(x, y, method=algo, des = des)
        uuid_temp = uuid.uuid4()
        str = "../data/train/grid/complete_grid_" + algo + "_" + des + "_" + uuid_temp.urn[9:] + ".pkl"
        joblib.dump(grid_obj, str)
        return grid_obj

    elif (bayes_tf == True):
        print("........starting bayes search........")

        bayes_obj = bayes(x, y, method=algo, des = des)
        uuid_temp = uuid.uuid4()
        str = "../data/train/bayes/complete_bayes_" + algo + "_" + des + "_" + uuid_temp.urn[9:] + ".pkl"
        joblib.dump(bayes_obj, str)
        return bayes_obj

    elif (sigopt_tf == True):

        print("........starting sigopt bayes search........")
        bayes_obj = bayes_sigopt(x, y, method=algo)
        uuid_temp = uuid.uuid4()
        str = "../data/train/bayes/complete_bayes_" + algo + "_" + des + "_" + uuid_temp.urn[9:] + ".pkl"
        joblib.dump(bayes_obj, str)
        return bayes_obj

    else:
        print("........starting single algo evaluation........")
        if (algo == "nn"):
            print("nn reg selected")
            reg = sk_nn(x, y, scale)
        elif (algo == "rf"):
            print("random forest selected")
            reg = random_forest(x, y, scale)
        elif (algo == "grad"):
            print("grid algo selected")
            reg = gradient_boost_reg(x, y, scale)
        elif (algo == "svr"):
            print("svr algo selected")
            reg = svr(x, y, scale)
        elif (algo == "bayes"):
            print("bayes regression selected")
            reg = bayesian(x, y, scale)
        elif (algo == "kernel"):
            print("kernel regression selected")
            reg = kernel(x, y, scale)
        elif (algo == "gaussian"):
            print("gaussian algo selected")
            reg = gaussian(x, y, scale)
        elif (algo == "xgboost"):
            from xgboost_util import xgboost
            print("xgboost algo selected")
            reg = xgboost(x, y, scale)
        elif (algo == "tf_nn"):
            from tensorflow_util import nn_basic
            reg = nn_basic(x, y, scale)
        elif (algo == "tf_cnn"):
            from tensorflow_util import cnn_basic
            reg = cnn_basic(x, y, scale)
        elif (algo == "tf_cnn_norm"):
            from tensorflow_util import cnn_norm_basic
            reg = cnn_norm_basic(x, y, scale)
        elif (algo == "resnet"):
            from tensorflow_util import resnet34
            reg = resnet34(x, y, scale)
        else:
            print("stochastic gradient descent selected")
            reg = sgd(x, y, scale)
        return reg

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='select descriptor, and directory of files')
    parser.add_argument("--des", action='store', dest="desc", default="rdkit",
                        help="select descriptor to convert to")
    parser.add_argument("--dir", action="store", dest="dir", default="DB", help="select directory")
    parser.add_argument("--algo", action="store", dest="algo", default="DB",
                        help="options: [svr_rbf, svr_poly, svr_lin, grad, rf, sgd, bayes, kernel, gaussian, nn]")
    parser.add_argument('--grid', dest="grid_tf", action='store_true')
    parser.add_argument('--bayes', dest="bayes_tf", action='store_true')
    parser.add_argument('--sigopt', dest="sigopt", action='store_true')
    parser.add_argument('--rand', dest="rand_tf", action='store_true')

    parser.add_argument('--diff', dest="diff", action='store_true')
    parser.add_argument('--homo', dest="homo", action='store_true')
    parser.add_argument('--homo1', dest="homo1", action='store_true')

    results = parser.parse_args()
    des = results.desc
    dir_temp = results.dir
    algo = results.algo

    rand_tf = results.rand_tf
    grid_tf = results.grid_tf
    bayes_tf = results.bayes_tf
    sigopt_tf = results.sigopt

    diff_tf = results.diff
    homo_tf = results.homo
    homo1_tf = results.homo1

    print("parser parsed")
    print("pulling directory: " + dir_temp + " with descriptor: " + des)

    if (homo1_tf == False and homo_tf == False):
        diff_tf = True

    if (dir_temp == "DB3" or dir_temp == "DB2"):
        try:
            print("done processing dataframe")
            str = "../data/desc/" + dir_temp + "/desc_calc_" + dir_temp + "_" + des + ".pkl"
            print(str)
            df = pd.read_pickle(str)
            pkl = 1
        except:
            print("done processing dataframe")
            str = "../data/desc/" + dir_temp + "/desc_calc_" + dir_temp + "_" + des + ".h5"
            print(str)
            df = pd.read_hdf(str)
            pkl = 0
    print(len(df))

    HOMO = df["HOMO"].to_numpy()
    HOMO_1 = df["HOMO-1"].to_numpy()
    diff = df["diff"].to_numpy()

    if (des == "vae"):
        temp = df["mat"].tolist()
        mat = list([i.flatten() for i in temp])

    elif (des == "auto"):
        temp = df["mat"].tolist()
        mat = list([i.flatten() for i in temp])
    else:
        mat = df["mat"].to_numpy()

    if (sigopt_tf == True):
        sigopt.log_dataset(name = dir_temp + " " +des)
        sigopt.log_model(type=algo)
        sigopt.log_metadata('input_features', np.shape(mat[0]))
    try:
        mat = preprocessing.scale(np.array(mat))
    except:
        mat = list(mat)
        mat = preprocessing.scale(np.array(mat))

    print("Using " + des + " as the descriptor")
    print("Matrix Dimensions: {0}".format(np.shape(mat)))

    # finish optimization
    if (homo_tf == True):
        des = des + "_homo"
        print(".........................HOMO..................")
        scale_HOMO = (np.max(HOMO) - np.min(HOMO))
        HOMO = (HOMO - np.min(HOMO)) / scale_HOMO
        reg_HOMO = calc(mat, HOMO, des, scale_HOMO, rand_tf,
                        grid_tf, bayes_tf, sigopt_tf, algo)

    if(homo1_tf == True):
        des = des + "_homo_1"

        print(".........................HOMO1..................")
        scale_HOMO_1 = (np.max(HOMO_1) - np.min(HOMO_1))
        HOMO_1 = (HOMO_1 - np.min(HOMO_1)) / scale_HOMO_1
        reg_HOMO = calc(mat, HOMO_1, des, scale_HOMO_1,rand_tf,
                        grid_tf, bayes_tf, sigopt_tf, algo)

    if(diff_tf == True):
        des = des + "_diff"

        print(".........................diff..................")

        scale_diff = (np.max(diff) - np.min(diff))
        diff = (diff - np.min(diff)) / scale_diff
        reg_diff = calc(mat, diff, des, scale_diff, rand_tf,
                        grid_tf, bayes_tf, sigopt_tf, algo)


