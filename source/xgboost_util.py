import sklearn.utils.fixes
from numpy.ma import MaskedArray

sklearn.utils.fixes.MaskedArray = MaskedArray

import time
import numpy as np
import xgboost as xgb
import scipy.stats as stats
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV


def xgboost(x, y, scale):
    x = np.array(x)
    y = np.array(y)
    try:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    except:
        x = list(x)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    params = {
        "colsample_bytree": 0.4,
        "learning_rate": 0.05,
        "max_depth": 10, "gamma": 0.0,
        "lambda": 0.0,
        "alpha": 0.1,
        "eta": 0.0,
        "n_estimators": 10000}

    reg = xgb.XGBRegressor(**params, objective="reg:squarederror", tree_method="gpu_hist")

    t1 = time.time()
    # non grid
    print(y_train)
    reg.fit(x_train, y_train)
    t2 = time.time()

    time_el = t2 - t1
    score = reg.score(x_test, y_test)
    print("xgboost score:               " + str(score) + " time: " + str(time_el))

    score = str(mean_squared_error(reg.predict(x_test), y_test))
    print("MSE score:   " + str(score))

    score = str(mean_absolute_error(reg.predict(x_test), y_test))
    print("MAE score:   " + str(score))

    score = str(r2_score(reg.predict(x_test), y_test))
    print("r2 score:   " + str(score))

    score_mae = mean_absolute_error(reg.predict(x_test), y_test)
    print("scaled MAE")
    print(scale * score_mae)

    return reg


def xgboost_grid(x, y):
    try:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    except:
        x = list(x)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    params = {"objective": ['reg:squarederror'],
              "colsample_bytree": [0.25, 0.5, 0.75],
              "learning_rate": [0.01, 0.1, 0.2, 0.3],
              "max_depth": [10, 20, 50], "gamma": [i * 0.05 for i in range(0, 5)],
              "lambda": [i * 0.05 for i in range(0, 4)],
              "alpha": [i * 0.05 for i in range(0, 4)],
              "eta": [i * 0.05 for i in range(0, 4)],
              "n_estimators": [400, 4000],
              "tree_method": ["gpu_hist"]}

    xgb_temp = xgb.XGBRegressor()
    reg = GridSearchCV(xgb_temp, params, verbose=5, cv=3)
    reg.fit(x_train, y_train)
    print(reg.best_params_)
    print(reg.best_score_)
    return reg

def xgboost_bayes_basic(x, y):
    try:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    except:
        x = list(x)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    xgb_temp = xgb.XGBRegressor()

    reg = BayesSearchCV(
        xgb_temp, {
            "colsample_bytree": Real(0.5, 0.99),
            "max_depth": Integer(40, 55),
            "lambda": Real(0, 0.25),
            "learning_rate": Real(0.1, 0.25),
            "alpha": Real(0, 0.2),
            "eta": Real(0.01, 0.2),
            "gamma": Real(0, 0.1),
            "n_estimators": Integer(500, 5000),
            "objective": ["reg:squarederror"],
            "tree_method": ["gpu_hist"]
        },
        n_iter=50,
        verbose=4, cv=3,
        random_state=0)

    reg.fit(x_train, y_train)

    score = str(mean_squared_error(reg.predict(x_test), y_test))
    print("MSE score:   " + str(score))

    score = str(mean_absolute_error(reg.predict(x_test), y_test))
    print("MAE score:   " + str(score))

    score = str(r2_score(reg.predict(x_test), y_test))
    print("r2 score:   " + str(score))

    return reg

def xgboost_rand(x, y):
    try:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    except:
        x = list(x)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    params = {"objective": ['reg:squarederror'],
              "colsample_bytree": stats.uniform(0.2, 0.8),
              "learning_rate": stats.uniform(0, 0.5),
              "max_depth": stats.uniform(10, 5), "gamma": stats.uniform(0, 0.1),
              "lambda": stats.uniform(0.1, 0.2),
              "alpha": [0.1],
              "eta": [0.0, 0.1],
              "n_estimators": [400, 4000],
              "tree_method": ["gpu_hist"]}

    xgb_temp = xgb.XGBRegressor()
    reg = RandomizedSearchCV(xgb_temp, **params, verbose=0, cv=3)
    reg.fit(x_train, y_train)

    print(reg.best_params_)
    print(reg.best_score_)

    return reg
