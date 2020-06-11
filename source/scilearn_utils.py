import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


def svr(x, y):
    # change C
    # scale data
    # L1/L2 normalization

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    svr_rbf = SVR(kernel='rbf', C=10, gamma=0.1, epsilon=.1, cache_size=4000)
    est_rbf = make_pipeline(StandardScaler(), svr_rbf)
    est_rbf.fit(list(x_train), y_train)
    s1 = svr_rbf.score(list(x_test), y_test)
    print("rbf fit: " + str(s1))

    svr_lin = SVR(kernel='linear', C=10, gamma='auto', cache_size=4000)
    est_lin = make_pipeline(StandardScaler(), svr_lin)
    est_lin.fit(list(x_train), y_train)
    s2 = svr_lin.score(list(x_test), y_test)
    print("lin fit: " + str(s2))

    svr_poly = SVR(kernel='poly', C=10, gamma='auto', degree=6, epsilon=.1,
                   coef0=0.5, cache_size=4000)
    est_poly = make_pipeline(StandardScaler(), svr_poly)
    est_poly.fit(list(x_train), y_train)
    s3 = svr_poly.score(list(x_test), y_test)
    print("poly  fit: " + str(s3))


def logistic(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # reg =

    est = make_pipeline(StandardScaler(), reg)
    est.fit(list(x_train), y_train)
    # hyperparameter tuning 10.0 ** -np.arange(1, 7)
    print(est.score(list(x_test), y_test))

def sgd (x,y):
    # requires scaling
    # loss = squared_loss, huber, epsilon_insensitive

    max = np.ceil(10 ** 6)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    reg = SGDRegressor(loss='squared_loss', max_iter=max, tol=0.001, penalty='l1', l1_ratio=0.1, epsilon=0.05,
                       validation_fraction=0.25, learning_rate='invscaling')
    est = make_pipeline(StandardScaler(), reg)
    est.fit(list(x_train), y_train)
    # hyperparameter tuning 10.0 ** -np.arange(1, 7)
    print("sgd : " + str(est.score(list(x_test), y_test)))


def gradient_boost_reg(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    reg = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.7,
                                    max_depth=2, random_state=0)
    est = make_pipeline(StandardScaler(), reg)
    est.fit(list(x_train), y_train)
    s1 = est.score(list(x_test), y_test)
    s1 = r2_score(est.predict(list(x_test)), y_test)
    print("gradient score: " + str(s1))


def process_input_DB2(dir="DB2", desc="rdkit"):
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
        # print(df)
    if (pkl == 0):
        df.to_hdf(str, key="df", mode='a')

    else:
        df.to_pickle(str)


def calc(dir="DB2", desc="aval"):
    try:
        str = "../data/desc/" + dir + "/desc_calc_DB2_" + desc + ".pkl"
        df = pd.read_pickle(str)
        pkl = 1
    except:
        str = "../data/desc/" + dir + "/desc_calc_DB2_" + desc + ".h5"
        df = pd.read_hdf(str)
        pkl = 0
    mat = df["mat"].to_numpy()
    HOMO = df["HOMO"].to_numpy()
    # print(mat)
    # print(HOMO)
    sgd(mat, HOMO)
    gradient_boost_reg(mat, HOMO)
    svr(mat, HOMO)

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
calc()
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
