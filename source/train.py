import argparse
import joblib
import numpy as np
import pandas as pd
import uuid
from sklearn import preprocessing
from sklearn_utils import gradient_boost_reg, \
    random_forest, sk_nn, grid, sgd, gaussian, kernel, \
    bayesian, svr, bayes


# todo: compile and xgboost on xsede/test gpu
# todo: upload descs to xsede
# todo: work on interpretability algo/aspects
# todo: plots of parameter space
# todo: process zz's stuff
# todo: make a standard method of storing results


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
        str = "../data/desc/" + dir + "/desc_calc_DB3_" + desc + ".pkl"
        print(str)
        df = pd.read_pickle(str)
        pkl = 1

    except:
        str = "../data/desc/" + dir + "/desc_calc_DB3_" + desc + ".h5"
        df = pd.read_hdf(str)
        pkl = 0

    print(df.head())

    list_to_sort = []
    with open("../data/DATA_DB3") as fp:
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


def calc(x, y, des, scale, grid_tf=True, bayes_tf=False, algo="sgd"):
    if (grid_tf == True):
        print("........starting grid search........")
        grid_obj = grid(x, y, method=algo)
        uuid_temp = uuid.uuid4()
        str = "../data/train/grid/complete_grid_" + algo + "_" + des + "_" + uuid_temp.urn[9:] + ".pkl"
        joblib.dump(grid_obj, str)

    elif (bayes_tf == True):
        print("........starting bayes search........")
        bayes_obj = bayes(x, y, method=algo)
        uuid_temp = uuid.uuid4()
        str = "../data/train/bayes/complete_bayes_" + algo + "_" + des + "_" + uuid_temp.urn[9:] + ".pkl"
        joblib.dump(bayes_obj, str)

    else:
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
    parser.add_argument('--scale', dest="scale_x_tf", action='store_true')

    results = parser.parse_args()
    des = results.desc
    print("parser parsed")
    dir_temp = results.dir
    print("pulling directory: " + dir_temp + " with descriptor: " + des)
    algo = results.algo
    grid_tf = results.grid_tf
    bayes_tf = results.bayes_tf
    scale_x_tf = results.scale_x_tf

    if (dir_temp == "DB3" or dir_temp == "DB2"):
        try:
            print("done processing dataframe")
            str = "../data/desc/" + dir_temp + "/desc_calc_" + dir_temp + "_" + des + ".pkl"
            df = pd.read_pickle(str)
            pkl = 1
        except:
            print("done processing dataframe")
            str = "../data/desc/" + dir_temp + "/desc_calc_" + dir_temp + "_" + des + ".h5"
            df = pd.read_hdf(str)
            pkl = 0

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

    scale_x_tf = True
    if (scale_x_tf == True):
        try:
            mat = preprocessing.scale(np.array(mat))

        except:
            mat = list(mat)
            mat = preprocessing.scale(np.array(mat))
    '''
    scale_HOMO = (np.max(HOMO) - np.min(HOMO))
    HOMO = (HOMO - np.min(HOMO)) / scale_HOMO
    print("Using " + des + " as the descriptor")
    print(".........................HOMO..................")
    reg_HOMO = calc(mat, HOMO, des, scale_HOMO, grid_tf, bayes_tf, algo)


    print(".........................HOMO1..................")
    scale_HOMO_1 = (np.max(HOMO_1) - np.min(HOMO_1))
    HOMO_1 = (HOMO_1 - np.min(HOMO_1)) / scale_HOMO_1
    reg_HOMO_1 = calc(mat, HOMO_1, des, scale_HOMO_1, grid_tf, bayes_tf, algo)
    '''

    scale_diff = (np.max(diff) - np.min(diff))
    # diff = diff - np.min(diff)
    diff = (diff - np.min(diff)) / scale_diff
    print(".........................diff..................")
    reg_diff = calc(mat, diff, des, scale_diff, grid_tf, bayes_tf, algo)
