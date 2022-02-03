import joblib, argparse, uuid, sigopt
import pandas as pd
from sklearn import preprocessing
from utils.sklearn_util import *
import random

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


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="select descriptor, and directory of files"
    )
    parser.add_argument(
        "--des",
        action="store",
        dest="desc",
        default="rdkit",
        help="select descriptor to convert to",
    )
    parser.add_argument(
        "--dir", action="store", dest="dir", default="DB", help="select directory"
    )
    parser.add_argument(
        "--algo",
        action="store",
        dest="algo",
        default="DB",
        help="options: [svr_rbf, svr_poly, svr_lin, grad, rf, sgd, bayes, kernel, gaussian, nn]",
    )

    parser.add_argument('--benzo', dest = 'benzo', action='store_true')
    parser.add_argument("--grid", dest="grid_tf", action="store_true")
    parser.add_argument("--bayes", dest="bayes_tf", action="store_true")
    parser.add_argument("--sigopt", dest="sigopt", action="store_true")
    parser.add_argument("--rand", dest="rand_tf", action="store_true")

    parser.add_argument("--diff", dest="diff", action="store_true")
    parser.add_argument("--homo", dest="homo", action="store_true")
    parser.add_argument("--homo1", dest="homo1", action="store_true")

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
    benzo_tf = results.benzo

    print("parser parsed")
    print("pulling directory: " + dir_temp + " with descriptor: " + des)
    print(des)
    print(dir_temp)
    if homo1_tf == False and homo_tf == False:
        diff_tf = True

    if dir_temp == "DB3" or dir_temp == "DB2":
        try:
            print("done processing dataframe")
            str_temp = "../data/desc/" + dir_temp + "/desc_calc_" + dir_temp + "_"  + str(des)  + ".h5"
            print(str_temp)
            df = pd.read_hdf(str_temp)

        except:
            print("done processing dataframe")
            str_temp = "../data/desc/" + dir_temp + "/desc_calc_" + dir_temp + "_"  + str(des)  + ".pkl"
            print(str_temp)
            df = pd.read_pickle(str_temp)

    print("done processing dataframe")
    str_temp = "../data/desc/" + dir_temp + "/desc_calc_" + dir_temp + "_"  + str(des)  + ".h5"
    print(str_temp)
    df = pd.read_hdf(str_temp)

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

    if sigopt_tf == True:
        sigopt.log_dataset(name=dir_temp + " " + des)
        sigopt.log_model(type=algo)
        sigopt.log_metadata("input_features", np.shape(mat[0]))
    try:
        mat = preprocessing.scale(np.array(mat))
    except:
        mat = list(mat)
        mat = preprocessing.scale(np.array(mat))

    print("Using " + des + " as the descriptor")
    print("Matrix Dimensions: {0}".format(np.shape(mat)))
    df_benzo = pd.read_hdf('../data/benzo/compiled.h5')
    mat_benzo = df_benzo['mat']

    # finish optimization
    if homo_tf:
        des = des + "_homo"
        print(".........................HOMO..................")
        scale_HOMO = np.max(HOMO) - np.min(HOMO)
        HOMO = (HOMO - np.min(HOMO)) / scale_HOMO
        reg_HOMO = calc(
            mat, HOMO, des, scale_HOMO, rand_tf, grid_tf, bayes_tf, sigopt_tf, algo
        )

        if (benzo_tf):
            homo_benzo = df_benzo['homo']
            #X_train, X_test, y_train, y_test = train_test_split(mat, homo_benzo, test_size=0.2)
            y_test_pred = reg_HOMO.predict(mat)
            print("extrapolate benzo r^2: " + str(r2_score(homo_benzo, y_test_pred)))

    if homo1_tf:
        des = des + "_homo_1"

        print(".........................HOMO1..................")
        scale_HOMO_1 = np.max(HOMO_1) - np.min(HOMO_1)
        HOMO_1 = (HOMO_1 - np.min(HOMO_1)) / scale_HOMO_1
        reg_HOMO = calc(
            mat, HOMO_1, des, scale_HOMO_1, rand_tf, grid_tf, bayes_tf, sigopt_tf, algo
        )
        if (benzo_tf):
            homo1_benzo = df_benzo['homo1']
            #X_train, X_test, y_train, y_test = train_test_split(mat, homo1_benzo, test_size=0.2)
            y_test_pred = reg_HOMO.predict(mat)
            print("extrapolate benzo r^2: " + str(r2_score(homo1_benzo, y_test_pred)))


    if diff_tf:

        des = des + "_diff"

        print(".........................diff..................")

        scale_diff = np.max(diff) - np.min(diff)
        diff = (diff - np.min(diff)) / scale_diff
        reg_diff = calc(
            mat, diff, des, scale_diff, rand_tf, grid_tf, bayes_tf, sigopt_tf, algo
        )
        if (benzo_tf):
            diff_benzo = df_benzo['diff']
            #X_train, X_test, y_train, y_test = train_test_split(mat, diff_benzo, test_size=0.2)
            y_test_pred = reg_diff.predict(mat)
            print("extrapolate benzo r^2: " + str(r2_score(diff_benzo, y_test_pred)))


