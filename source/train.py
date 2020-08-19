import argparse
import uuid

import pandas as pd
from sklearn_utils import gradient_boost_reg, \
    random_forest, sk_nn, grid, sgd, gaussian, kernel, \
    bayesian, svr, bayes


# TODO: implement with and without standardization

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

# TODO : this
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


def calc(x, y, des, grid_tf=True, bayes_tf=False, algo="sgd"):
    if (grid_tf == True):
        print("........starting grid search........")
        grid_obj = grid(x, y, method=algo)
        uuid_temp = uuid.uuid4()
        str = "../data/train/grid/" + algo + "_" + des + "_" + uuid_temp.urn[9:] + ".pkl"
        # joblib.dump(grid_obj, str)

    elif (bayes_tf == True):
        print("........starting bayes search........")
        bayes_obj = bayes(x, y, method=algo)
        uuid_temp = uuid.uuid4()
        str = "../data/train/bayes/" + algo + "_" + des + "_" + uuid_temp.urn[9:] + ".pkl"
        # joblib.dump(bayes_obj, str)

    else:
        if (algo == "nn"):
            print("nn reg selected")
            sk_nn(x, y)
        elif (algo == "rf"):
            print("random forest selected")
            random_forest(x, y)
        elif (algo == "grad"):
            print("grid algo selected")
            gradient_boost_reg(x, y)
        elif (algo == "svr"):
            print("svr algo selected")
            svr(x, y)
        elif (algo == "bayes"):
            print("bayes regression selected")
            bayesian(x, y)
        elif (algo == "kernel"):
            print("kernel regression selected")
            kernel(x, y)
        elif (algo == "gaussian"):
            print("gaussian algo selected")
            gaussian(x, y)
        elif (algo == "xgboost"):
            from xgboost_util import xgboost
            print("xgboost algo selected")
            xgboost(x, y)
        else:
            print("stochastic gradient descent selected")
            sgd(x, y)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='select descriptor, and directory of files')
    parser.add_argument("--des", action='store', dest="desc", default="rdkit", help="select descriptor to convert to")
    parser.add_argument("--dir", action="store", dest="dir", default="DB", help="select directory")
    parser.add_argument("--algo", action="store", dest="algo", default="DB",
                        help="options: [svr_rbf, svr_poly, svr_lin, grad, rf, sgd, bayes, kernel, gaussian, nn]")
    parser.add_argument('--grid', dest="grid_tf", action='store_true')
    parser.add_argument('--bayes', dest="bayes_tf", action='store_true')

    results = parser.parse_args()
    des = results.desc
    print("parser parsed")
    dir_temp = results.dir
    print("pulling directory: " + dir_temp + " with descriptor: " + des)
    algo = results.algo
    grid_tf = results.grid_tf
    bayes_tf = results.bayes_tf

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

    print("Using " + des + " as the descriptor")
    print(".........................HOMO..................")
    calc(mat, HOMO, des, grid_tf, bayes_tf, algo)
    print(".........................HOMO1..................")
    calc(mat, HOMO_1, des, grid_tf, bayes_tf, algo)
    print(".........................diff..................")
    calc(mat, diff, des, grid_tf, bayes_tf, algo)
