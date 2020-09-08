import sklearn.utils.fixes
from numpy.ma import MaskedArray

sklearn.utils.fixes.MaskedArray = MaskedArray

import time
import numpy as np
from skopt.searchcv import BayesSearchCV
from skopt.space import Real, Integer

from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import SGDRegressor, BayesianRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.model_selection import train_test_split, GridSearchCV


def bayes(x, y, method="sgd"):
    if (method == "nn"):
        print(".........neural network optimization selected.........")
        params = {"alpha": Real(1e-10, 1e-1, prior='log-uniform'),
                  "max_iter": Integer(100, 10000),
                  "tol": Real(1e-10, 1e-1, prior='log-uniform'),
                  "learning_rate_init": Real(1e-3, 1e-1, prior='log-uniform')}

        reg = MLPRegressor(hidden_layer_sizes=(100, 1000, 100,), activation="relu",
                           solver="adam", learning_rate="adaptive")

    elif (method == "rf"):
        print(".........random forest optimization selected.........")
        params = {"max_depth": Integer(10, 40),
                  "min_samples_split": Integer(2, 6),
                  "n_estimators": Integer(500, 5000)}
        reg = RandomForestRegressor(n_jobs=1)

    elif (method == "grad"):
        print(".........gradient boost optimization selected.........")

        params = {"loss": ["ls"],
                  "n_estimators": Integer(500, 5000),
                  "learning_rate": Real(0.001, 0.3),
                  "subsample": Real(0.2, 0.8),
                  "max_depth": Integer(10, 30),
                  "tol": Real(1e-6, 1e-3, prior='log-uniform')}
        reg = GradientBoostingRegressor(criterion="mse", loss="ls")

    elif (method == "svr_rbf"):
        print(".........svr optimization selected.........")
        params = {"C": Real(1e-5, 1e+1, prior='log-uniform'),
                  "gamma": Real(1e-5, 1e-1, prior='log-uniform'),
                  "epsilon": Real(1e-2, 1e+1, prior='log-uniform'),
                  "cache_size": Integer(500, 8000)}
        reg = SVR(kernel="rbf")

    elif (method == "svr_poly"):
        print(".........svr optimization selected.........")

        params = {"C": Real(1e-5, 1e+1, prior='log-uniform'),
                  "gamma": Real(1e-5, 1e-1, prior='log-uniform'),
                  "epsilon": Real(1e-2, 1e+1, prior='log-uniform'),
                  "degree": Integer(5, 20),
                  "coef0": Real(0.2, 0.8),
                  "cache_size": Integer(500, 8000)}
        reg = SVR(kernel="poly")

    elif (method == "svr_lin"):
        print(".........svr optimization selected.........")

        params = {"C": Real(1e-6, 1e+1, prior='log-uniform'),
                  "gamma": Real(1e-5, 1e-1, prior='log-uniform'),
                  "cache_size": Integer(500, 8000)}
        reg = SVR(kernel="linear")

    elif (method == "bayes"):
        print(".........bayes optimization selected.........")

        params = {
            "n_iter": Integer(1000, 10000),
            "tol": Real(1e-9, 1e-3, prior='log-uniform'),
            "alpha_1": Real(1e-6, 1e+1, prior='log-uniform'),
            "alpha_2": Real(1e-6, 1e+1, prior='log-uniform'),
            "lambda_1": Real(1e-6, 1e+1, prior='log-uniform'),
            "lambda_2": Real(1e-6, 1e+1, prior='log-uniform')}
        reg = BayesianRidge()


    elif (method == "kernel"):
        print(".........kernel optimization selected.........")

        params = {"alpha": Real(1e-6, 1e0, prior='log-uniform'),
                  "gamma": Real(1e-8, 1e0, prior='log-uniform')}
        reg = KernelRidge(kernel="rbf")

    elif (method == "gaussian"):
        print(".........gaussian optimization selected.........")
        params = {"alpha": Real(1e-7, 1e+1, prior='log-uniform')}
        kernel = DotProduct() + WhiteKernel()
        reg = GaussianProcessRegressor(kernel=kernel)

    else:
        params = {'l1_ratio': Real(0.1, 0.3),
                  'tol': Real(1e-3, 1e-1, prior="log-uniform"),
                  "epsilon": Real(1e-3, 1e0, prior="log-uniform"),
                  "eta0": Real(0, 0.2)}
        reg = SGDRegressor(penalty="l1", loss='squared_loss')

    if (method == "xgboost"):
        from xgboost_util import xgboost_bayes_basic
        print(".........xgboost optimization selected.........")

        reg = xgboost_bayes_basic(x, y)
    else:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        reg = BayesSearchCV(reg, params, n_iter=1000, verbose=3, cv=3)

        reg.fit(list(x_train), y_train)
        print(reg.best_params_)
        print(reg.best_score_)
        print("Score on test data: " + str(reg.score(list(x_test), y_test)))

        score = str(mean_squared_error(reg.predict(x_test), y_test))
        print("MSE score:   " + str(score))

        score = str(mean_absolute_error(reg.predict(x_test), y_test))
        print("MAE score:   " + str(score))

        score = str(r2_score(reg.predict(x_test), y_test))
        print("r2 score:   " + str(score))

    return reg


def grid(x, y, method="sgd"):
    if (method == "nn"):
        print(".........neural network grid optimization selected.........")
        params = {"alpha": [1e-10, 1e-7, 1e-4, 1e-1],
                  "activation": ["relu"],
                  "solver": ["adam"],
                  "learning_rate": ["adaptive"],
                  "max_iter": [100, 10000, 1000000],
                  "tol": [1e-11, 1e-7, 1e-5, 1e-3, 1e-1, 1],
                  "learning_rate_init": [0.00001, 0.0001, 0.001, 0.01],
                  "shuffle": [True]
                  }
        reg = MLPRegressor(hidden_layer_sizes=(100, 1000, 100,))
    elif (method == "rf"):
        print(".........random forest grid optimization selected.........")

        params = {"max_depth": [10, 20, 30, 40],
                  "min_samples_split": [2, 4, 6],
                  "n_jobs": [1],
                  "n_estimators": [500, 1000, 2000, 5000]
                  }
        reg = RandomForestRegressor()

    elif (method == "grad"):
        print(".........gradient boost grid optimization selected.........")

        params = {"loss": ["ls"],
                  "n_estimators": [500, 1000, 2000, 4000],
                  "learning_rate": [i * 0.03 for i in range(1, 10)],
                  "subsample": [(i * 0.06) for i in range(1, 10)],
                  "criterion": ["mse"],
                  "max_depth": [i * 10 for i in range(1, 3)],
                  "tol": [0.0001, 0.000001]
                  }
        reg = GradientBoostingRegressor()

    elif (method == "svr_rbf"):
        print(".........svr grid optimization selected.........")

        params = {"kernel": ["rbf"],
                  "C": [10, 1, 0.1, 0.01, 0.001, 0.0001],
                  "gamma": [0.1, 0.0001, 0.00001],
                  "epsilon": [0.01, 0.1, 1, 5, 10, 20],
                  "cache_size": [500, 1000, 2000, 4000, 8000]
                  }

        reg = SVR()

    elif (method == "svr_poly"):
        print(".........svr grid optimization selected.........")

        params = {"kernel": ["poly"],
                  "C": [1, 0.1, 0.01, 0.001, 0.0001],
                  "gamma": [0.1, 0.0001, 0.00001],
                  "epsilon": [0.01, 0.1, 1, 5, 10, 20],
                  "degree": [5, 7, 9, 20],
                  "coef0": [0.2, 0.4, 0.5, 0.6, 0.8],
                  "cache_size": [500, 1000, 2000, 4000, 8000]
                  }

        reg = SVR()
    elif (method == "svr_lin"):
        print(".........svr grid optimization selected.........")

        params = {"kernel": ["linear"],
                  "C": [10, 1, 0.1, 0.01, 0.001, 0.0001],
                  "gamma": [0.1, 0.0001, 0.00001],
                  "cache_size": [500, 1000, 2000, 4000, 8000]
                  }

        reg = SVR()

    elif (method == "bayes"):
        print(".........bayes grid optimization selected.........")

        params = {
            "n_iter": [1000, 2000, 5000, 10000],
            "tol": [1e-3, 1e-5, 1e-7, 1e-9],
            "alpha_1": [1e-01, 1e-03, 1e-05, 1e-07],
            "alpha_2": [1e-01, 1e-03, 1e-05, 1e-07],
            "lambda_1": [1e-01, 1e-03, 1e-05, 1e-07],
            "lambda_2": [1e-01, 1e-03, 1e-05, 1e-07]
        }

        reg = BayesianRidge()


    elif (method == "kernel"):
        print(".........kernel grid optimization selected.........")

        params = {"kernel": ["rbf"],
                  "alpha": [1e-6, 1e-4, 1e-2, 1, 2],
                  "gamma": [1e-8, 1e-6, 1e-4, 1e-2, 1]
                  }

        reg = KernelRidge()

    elif (method == "gaussian"):
        print(".........gaussian grid optimization selected.........")

        params = {"alpha": [1e-10, 1e-7, 1e-4, 1e-1]
                  }

        kernel = DotProduct() + WhiteKernel()
        reg = GaussianProcessRegressor(kernel=kernel)


    else:

        params = {"loss": ['squared_loss', "huber"],
                  "tol": [0.01, 0.001, 0.0001],
                  "shuffle": [True],
                  "penalty": ["l1"],
                  "l1_ratio": [0.15, 0.20, 0.25],
                  "epsilon": [0.01, 0.1, 1],
                  "eta0": [10 ** (-2 * i) for i in range(1, 5)],
                  }
        reg = SGDRegressor()

    if (method == "xgboost"):
        from xgboost_util import xgboost_grid

        print(".........xgboost grid optimization selected.........")
        reg = xgboost_grid(x, y)
        return reg
    else:


        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        reg = GridSearchCV(reg, params, verbose=6, cv=3)
        reg.fit(list(x_train), y_train)

        print(reg.best_params_)
        print(reg.best_score_)
        print("Score on test data: " + str(reg.score(x_test, y_test)))

        return reg


def sgd(x, y, scale):
    x = np.array(x)
    y = np.array(y)

    params = {"loss": 'squared_loss',
              "max_iter": 10 ** 7,
              "tol": 0.0000001,
              "penalty": "l2", "l1_ratio": 0.15,
              "epsilon": 0.01,
              "learning_rate": 'invscaling'
              }

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    reg = SGDRegressor(**params)

    t1 = time.time()
    reg.fit(list(x_train), y_train)
    t2 = time.time()
    time_el = t2 - t1

    score = str(reg.score(list(x_test), y_test))
    print("stochastic gradient descent score:   " + str(score) + " time: " + str(time_el))

    score = str(mean_squared_error(reg.predict(x_test), y_test))
    print("MSE score:   " + str(score) + " time: " + str(time_el))

    score = str(mean_absolute_error(reg.predict(x_test), y_test))
    print("MAE score:   " + str(score) + " time: " + str(time_el))

    score = str(r2_score(reg.predict(x_test), y_test))
    print("r2 score:   " + str(score) + " time: " + str(time_el))

    score_mae = mean_absolute_error(reg.predict(x_test), y_test)
    print("scaled MAE")
    print(scale * score_mae)

    return reg


def gradient_boost_reg(x, y, scale):
    params = {"loss": "ls",
              "n_estimators": 2000,
              "learning_rate": 0.1,
              "subsample": 0.8,
              "criterion": "mse",
              "max_depth": 10,
              "tol": 0.0001
              }

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    reg = GradientBoostingRegressor(**params)

    t1 = time.time()
    reg.fit(list(x_train), y_train)
    t2 = time.time()
    time_el = t2 - t1
    score = reg.score(list(x_test), y_test)
    print("gradient boost score:                " + str(score) + " time: " + str(time_el))

    score = str(mean_squared_error(reg.predict(x_test), y_test))
    print("MSE score:   " + str(score) + " time: " + str(time_el))

    score = str(mean_absolute_error(reg.predict(x_test), y_test))
    print("MAE score:   " + str(score) + " time: " + str(time_el))

    score = str(r2_score(reg.predict(x_test), y_test))
    print("r2 score:   " + str(score) + " time: " + str(time_el))

    score_mae = mean_absolute_error(reg.predict(x_test), y_test)
    print("scaled MAE")
    print(scale * score_mae)

    return reg


def random_forest(x, y, scale):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    params = {"max_depth": 30,
              "min_samples_split": 3,
              "n_estimators": 5000,
              "n_jobs": 16, "verbose": False
              }

    reg = RandomForestRegressor(**params)

    t1 = time.time()
    reg.fit(list(x_train), y_train)
    t2 = time.time()
    time_el = t2 - t1
    score = reg.score(list(x_test), y_test)
    print("random forest score:                 " + str(score) + " time: " + str(time_el))

    score = str(mean_squared_error(reg.predict(x_test), y_test))
    print("MSE score:   " + str(score) + " time: " + str(time_el))

    score = str(mean_absolute_error(reg.predict(x_test), y_test))
    print("MAE score:   " + str(score) + " time: " + str(time_el))

    score = str(r2_score(reg.predict(x_test), y_test))
    print("r2 score:   " + str(score) + " time: " + str(time_el))

    score_mae = mean_absolute_error(reg.predict(x_test), y_test)
    print("scaled MAE")
    print(scale * score_mae)

    return reg


def gaussian(x, y, scale):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    kernel = DotProduct() + WhiteKernel()
    reg = GaussianProcessRegressor(kernel=kernel, alpha=1e-10, random_state=0)

    t1 = time.time()
    reg.fit(list(x_train), y_train)
    t2 = time.time()
    time_el = t2 - t1
    score = reg.score(list(x_test), y_test)
    print("gaussian process score:              " + str(score) + " time: " + str(time_el))

    score = str(mean_squared_error(reg.predict(x_test), y_test))
    print("MSE score:   " + str(score) + " time: " + str(time_el))

    score = str(mean_absolute_error(reg.predict(x_test), y_test))
    print("MAE score:   " + str(score) + " time: " + str(time_el))

    score = str(r2_score(reg.predict(x_test), y_test))
    print("r2 score:   " + str(score) + " time: " + str(time_el))

    score_mae = mean_absolute_error(reg.predict(x_test), y_test)
    print("scaled MAE")
    print(scale * score_mae)
    return reg


def kernel(x, y, scale):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    # reg = KernelRidge(alpha=0.0001, degree = 10,kernel = "polynomial")
    reg = KernelRidge(kernel='rbf', alpha=0.00005, gamma=0.0001)

    t1 = time.time()
    reg.fit(list(x_train), y_train)
    t2 = time.time()
    time_el = t2 - t1
    score = reg.score(list(x_test), y_test)
    print("kernel regression score:             " + str(score) + " time: " + str(time_el))

    score = str(mean_squared_error(reg.predict(x_test), y_test))
    print("MSE score:   " + str(score) + " time: " + str(time_el))

    score = str(mean_absolute_error(reg.predict(x_test), y_test))
    print("MAE score:   " + str(score) + " time: " + str(time_el))

    score = str(r2_score(reg.predict(x_test), y_test))
    print("r2 score:   " + str(score) + " time: " + str(time_el))

    score_mae = mean_absolute_error(reg.predict(x_test), y_test)
    print("scaled MAE")
    print(scale * score_mae)
    return reg


def bayesian(x, y, scale):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    reg = BayesianRidge(n_iter=10000, tol=1e-7, copy_X=True, alpha_1=1e-03, alpha_2=1e-03,
                        lambda_1=1e-03, lambda_2=1e-03)

    t1 = time.time()
    reg.fit(list(x_train), y_train)
    t2 = time.time()
    time_el = t2 - t1
    score = reg.score(list(x_test), y_test)
    print("bayesian score:                      " + str(score) + " time: " + str(time_el))

    score = str(mean_squared_error(reg.predict(x_test), y_test))
    print("MSE score:   " + str(score) + " time: " + str(time_el))

    score = str(mean_absolute_error(reg.predict(x_test), y_test))
    print("MAE score:   " + str(score) + " time: " + str(time_el))

    score = str(r2_score(reg.predict(x_test), y_test))
    print("r2 score:   " + str(score) + " time: " + str(time_el))

    score_mae = mean_absolute_error(reg.predict(x_test), y_test)
    print("scaled MAE")
    print(scale * score_mae)

    return reg


def svr(x, y, scale):
    # change C
    # scale data
    # L1/L2 normalization

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    svr_rbf = SVR(kernel='rbf', C=0.001, gamma=0.1, epsilon=.1, cache_size=4000)
    est_rbf = svr_rbf
    svr_lin = SVR(kernel='linear', C=0.1, gamma='auto', cache_size=4000)
    est_lin = svr_lin
    svr_poly = SVR(kernel='poly', C=100, gamma='auto', degree=6, epsilon=.1, coef0=0.5, cache_size=4000)
    est_poly = svr_poly

    t1 = time.time()
    est_rbf.fit(list(x_train), y_train)
    t2 = time.time()
    time_rbf = t2 - t1
    s1 = svr_rbf.score(list(x_test), y_test)

    t1 = time.time()
    est_lin.fit(list(x_train), y_train)
    t2 = time.time()
    time_svr= t2 - t1
    s2 = svr_lin.score(list(x_test), y_test)

    t1 = time.time()
    est_poly.fit(list(x_train), y_train)
    t2 = time.time()
    time_poly = t2 - t1
    score = svr_poly.score(list(x_test), y_test)

    print("linear svr score:                    " + str(s2) + " time: " + str(time_rbf))
    print("radial basis svr score:              " + str(s1) + " time: " + str(time_svr))
    print("polynomial svr score:                " + str(score) + " time: " + str(time_poly))

    score = str(mean_squared_error(reg.predict(x_test), y_test))
    print("MSE score:   " + str(score) + " time: " + str(time_el))

    score = str(mean_absolute_error(reg.predict(x_test), y_test))
    print("MAE score:   " + str(score) + " time: " + str(time_el))

    score = str(r2_score(reg.predict(x_test), y_test))
    print("r2 score:   " + str(score) + " time: " + str(time_el))

    score_mae = mean_absolute_error(reg.predict(x_test), y_test)
    print("scaled MAE")
    print(scale * score_mae)

    return svr_poly


def sk_nn(x, y, scale):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    reg = MLPRegressor(random_state=1, max_iter=100000, learning_rate_init=0.00001, learning_rate="adaptive",
                       early_stopping=True, tol=1e-7, shuffle=True, solver="adam", activation="relu",
                       hidden_layer_sizes=(1000,), verbose=False, alpha=0.00001)

    t1 = time.time()
    reg.fit(list(x_train), y_train)
    t2 = time.time()
    time_el = t2 - t1

    score = reg.score(list(x_test), y_test)
    print("Neural Network score:                " + str(score) + " time: " + str(time_el))

    score = str(mean_squared_error(reg.predict(x_test), y_test))
    print("MSE score:   " + str(score) + " time: " + str(time_el))

    score = str(mean_absolute_error(reg.predict(x_test), y_test))
    print("MAE score:   " + str(score) + " time: " + str(time_el))

    score = str(r2_score(reg.predict(x_test), y_test))
    print("r2 score:   " + str(score) + " time: " + str(time_el))

    score_mae = mean_absolute_error(reg.predict(x_test), y_test)
    print("scaled MAE")
    print(scale * score_mae)

    return reg
