import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


def svr(x, y):
    # change C
    # scale data
    # L1/L2 normalization
    svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1, cache_size=2000)
    svr_lin = SVR(kernel='linear', C=100, gamma='auto', cache_size=2000)
    svr_poly = SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=.1,
                   coef0=1, cache_size=2000)

    svr_rbf.fit(x, y)
    svr_lin.fit(x, y)
    svr_poly.fit(x, y)


def sgd (x,y):
    # requires scaling
    # loss = squared_loss, huber, epsilon_insensitive
    n = len(x)

    max = np.ceil(10 ** 6)

    reg = SGDRegressor(loss='squared_loss', max_iter=max, tol=0.001, penalty='l1', l1_ratio=0.15, epsilon=0.1,
                       validation_fraction=0.25, learning_rate='invscaling')
    est = make_pipeline(StandardScaler(), reg)

    est.fit(list(x), y)
    print(est.score(list(x), y))
    # hyperparameter tuning 10.0 ** -np.arange(1, 7)


# def log_reg(x,y):
#


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
