import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import SGDRegressor, BayesianRidge
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


def svr(x, y):
    # change C
    # scale data
    # L1/L2 normalization

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    svr_rbf = SVR(kernel='rbf', C=1, gamma=0.1, epsilon=.1, cache_size=4000)

    est_rbf = make_pipeline(StandardScaler(), svr_rbf)
    t1 = time.time()
    est_rbf.fit(list(x_train), y_train)
    t2 = time.time()
    time_el = t2 - t1
    s1 = svr_rbf.score(list(x_test), y_test)
    print("radial basis svr score:              " + str(s1) + " time: " + str(time_el))

    svr_lin = SVR(kernel='linear', C=1, gamma='auto', cache_size=4000)

    est_lin = make_pipeline(StandardScaler(), svr_lin)
    t1 = time.time()
    est_lin.fit(list(x_train), y_train)
    t2 = time.time()
    time_el = t2 - t1
    s2 = svr_lin.score(list(x_test), y_test)
    print("linear svr score:                    " + str(s2) + " time: " + str(time_el))

    svr_poly = SVR(kernel='poly', C=1, gamma='auto', degree=6, epsilon=.1,
                   coef0=0.5, cache_size=4000)
    est_poly = make_pipeline(StandardScaler(), svr_poly)
    t1 = time.time()
    est_poly.fit(list(x_train), y_train)
    t2 = time.time()
    time_el = t2 - t1
    score = svr_poly.score(list(x_test), y_test)
    print("polynomial svr score:                " + str(score) + " time: " + str(time_el))

def sgd (x,y):
    # requires scaling
    # loss = squared_loss, huber, epsilon_insensitive

    max = np.ceil(10 ** 7)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    reg = SGDRegressor(loss='huber', max_iter=max, tol=0.00001, penalty='l2', l1_ratio=0.15, epsilon=0.01,
                       validation_fraction=0.2, learning_rate='invscaling')
    est = make_pipeline(StandardScaler(), reg)
    t1 = time.time()
    est.fit(list(x_train), y_train)
    t2 = time.time()
    time_el = t2 - t1
    score = str(est.score(list(x_test), y_test))
    print("stochastic gradient descent score:   " + str(score) + " time: " + str(time_el))


def gradient_boost_reg(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    reg = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.3,
                                    max_depth=15)
    est = make_pipeline(StandardScaler(), reg)
    t1 = time.time()
    est.fit(list(x_train), y_train)
    t2 = time.time()
    time_el = t2 - t1
    score = est.score(list(x_test), y_test)
    print("gradient boost score:                " + str(score) + " time: " + str(time_el))


def random_forest(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    reg = RandomForestRegressor(max_depth=40, min_samples_split=3, n_estimators=4000, n_jobs=8)
    est = make_pipeline(StandardScaler(), reg)
    t1 = time.time()
    est.fit(list(x_train), y_train)
    t2 = time.time()
    time_el = t2 - t1
    score = est.score(list(x_test), y_test)
    print("random forest score:                 " + str(score) + " time: " + str(time_el))


def gaussian(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    kernel = DotProduct() + WhiteKernel()
    reg = GaussianProcessRegressor(kernel=kernel, alpha=1e-10, random_state=0)
    est = make_pipeline(StandardScaler(), reg)
    t1 = time.time()
    est.fit(list(x_train), y_train)
    t2 = time.time()
    time_el = t2 - t1
    score = est.score(list(x_test), y_test)
    print("gaussian process score:              " + str(score) + " time: " + str(time_el))


def kernel(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    # reg = KernelRidge(alpha=0.0001, degree = 10,kernel = "polynomial")
    reg = KernelRidge(kernel='rbf', alpha=0.00005, gamma=0.0001)

    est = make_pipeline(StandardScaler(), reg)
    t1 = time.time()
    est.fit(list(x_train), y_train)
    t2 = time.time()
    time_el = t2 - t1
    score = est.score(list(x_test), y_test)
    print("kernel regression score:             " + str(score) + " time: " + str(time_el))


def bayesian(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    reg = BayesianRidge(n_iter=10000, tol=1e-7, copy_X=True, alpha_1=1e-03, alpha_2=1e-03,
                        lambda_1=1e-03, lambda_2=1e-03)
    est = make_pipeline(StandardScaler(), reg)
    t1 = time.time()
    est.fit(list(x_train), y_train)
    t2 = time.time()
    time_el = t2 - t1
    score = est.score(list(x_test), y_test)
    print("bayesian score:                      " + str(score) + " time: " + str(time_el))


def sk_nn(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    reg = MLPRegressor(random_state=1, max_iter=100000, learning_rate_init=0.0001, learning_rate="adaptive",
                       early_stopping=True, tol=1e-9, shuffle=True, solver="adam", activation="logistic",
                       hidden_layer_sizes=(1000, 1000, 1000, 1000, 1000), verbose=True, alpha=0.00001, )

    est = make_pipeline(StandardScaler(), reg)
    t1 = time.time()
    est.fit(list(x_train), y_train)
    t2 = time.time()
    time_el = t2 - t1
    score = est.score(list(x_test), y_test)
    print("Neural Network score:                " + str(score) + " time: " + str(time_el))


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

    list_to_sort = []
    with open("../data/benzoquinone_DB/DATA_copy") as fp:
        line = fp.readline()
        while line:
            list_to_sort.append(line[0:-2])
            line = fp.readline()
    list_to_sort.sort()

    for i in range(df.copy().shape[0]):
        for j in list_to_sort:
            if (df["name"].iloc[i][:-4] in j.split(";")[0]):
                temp1 = float(j.split(":")[1])
                temp2 = float(j.split(":")[2])

                df["HOMO"].loc[i] = float(j.split(":")[1])
                df["HOMO-1"].loc[i] = float(j.split(":")[2])
                df["diff"].loc[i] = temp2 - temp1
                print(temp2 - temp1)
        # print(df)
    if (pkl == 0):
        df.to_hdf(str, key="df", mode='a')
    else:
        df.to_pickle(str)


def process_input_ZZ(dir="ZZ", desc="rdkit"):
    try:
        str = "../data/desc/" + dir + "/desc_calc_DB2_" + desc + ".pkl"
        df = pd.read_pickle(str)
        pkl = 1
    except:
        str = "../data/desc/" + dir + "/desc_calc_DB2_" + desc + ".h5"
        df = pd.read_hdf(str)
        pkl = 0

    list_to_sort = []
    with open("../data/benzoquinone_DB/DATA_copy") as fp:
        line = fp.readline()
        while line:
            list_to_sort.append(line[0:-2])
            line = fp.readline()
    list_to_sort.sort()
    # print(list_to_sort)

    index_search = 0
    # df.insert(2,"HOMO","")
    # df.insert(3,"HOMO-1","")

    for i in range(df.shape[0]):
        # print(type(df["name"].iloc[i][:-4]))
        for j in list_to_sort:
            # print(j.split(":")[0])
            # print(df["name"].iloc[i][:-4])
            # if ( df["name"].iloc[i][:-4] == j.split(";")[0] ):
            if (df["name"].iloc[i][:-4] in j.split(";")[0]):
                df["HOMO"][i] = j.split(":")[1]
                df["HOMO-1"][i] = j.split(":")[2]
                df["diff"] = df["HOMO"][i] - df["HOMO-1"][i]
        # print(df)
    if (pkl == 0):
        df.to_hdf(str, key="df", mode='a')

    else:
        df.to_pickle(str)


def calc(dir="DB2", desc="rdkit"):
    try:
        # process_input_DB2()
        print("done processing dataframe")
        str = "../data/desc/" + dir + "/desc_calc_" + dir + "_" + desc + ".pkl"
        df = pd.read_pickle(str)
        pkl = 1
    except:
        # process_input_DB2()
        print("done processing dataframe")
        str = "../data/desc/" + dir + "/desc_calc_" + dir + "_" + desc + ".h5"
        df = pd.read_hdf(str)
        pkl = 0
    print(df.head())

    mat = df["mat"].to_numpy()

    HOMO = df["HOMO"].to_numpy()
    HOMO_1 = df["HOMO-1"].to_numpy()
    diff = df["diff"].to_numpy()

    # sgd(mat, diff)
    # gradient_boost_reg(mat, diff)
    # svr(mat, diff)
    # random_forest(mat,diff)
    # sk_nn(mat,HOMO)
    # bayesian(mat,diff)
    # kernel(mat,diff)
    # gaussian(mat,diff)


'''

with open("../data/benzoquinone_DB/DATA_copy") as fp:
    line = fp.readline()
    while line:
        list_to_sort.append(line)
        line = fp.readline()

list_to_sort.sort()
print(list_to_sort[0:4])
#print(sorted[0:4])
'''
'''
        if(line.split(":")[0] == df["name"].iloc[index_search][:-4]):
            #print(line.split(":")[0])
            print("yeet")
            #index_search += 1
        if (index_search < 4):
            #print(df["mat"].iloc[index_search])
            print(df["name"].iloc[index_search][:-4])
            print(line.split(":")[0])
            index_search+=1
            #if (line.split(":")[0][0] != " " and line.split(":")[0][0] != "-"):

        line = fp.readline()

'''

# process_input_DB2(dir = "DB2", desc="aval")
# calc()

'''
df_reload = pd.read_pickle("../data/desc/DB/desc_calc_DB_aval.pkl")
df_y = pd.read_excel("../data/quinones_for_Santiago.xlsx")
x_aval = df_reload["mat"].values
y_lnk_biradical = df_y["lnK_biradical"].values
y_rad_HOMO = df_y["radical HOMO"].values
y_birad_HOMO = df_y["biradical HOMO"].values
y_CO2_HOMO = df_y["biradical_CO2 HOMO"].values
sgd(x_aval, y_lnk_biradical)
sgd(x_aval, y_rad_HOMO)
sgd(x_aval, y_birad_HOMO)
sgd(x_aval, y_CO2_HOMO)
'''
