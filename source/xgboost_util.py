import time

import numpy as np
import scipy.stats as stats
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from skopt import BayesSearchCV
from skopt.space import Real, Integer

import xgboost as xgb


def xgboost(x, y):
    try:
        x = preprocessing.scale(np.array(x))
        # scaler = preprocessing.StandardScaler().fit_transform(x)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    except:
        x = list(x)
        # x = preprocessing.scale(np.array(x))
        # scaler = preprocessing.StandardScaler().fit_transform(x)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    params = {"objective": "reg:squarederror",
              "colsample_bytree": 0.3,
              "learning_rate": 0.01,
              "max_depth": 40, "gamma": 0.0,
              "lambda": 0.0,
              "alpha": 0.1,
              "eta": 0.0,
              "n_estimators": 5000,
              "tree_method": "gpu_hist"}

    reg = xgb.XGBRegressor(**params)
    est = make_pipeline(StandardScaler(), reg)
    # x_train = scaler.transform(x_train)

    t1 = time.time()
    # non grid
    est.fit(list(x_train), y_train)
    t2 = time.time()

    time_el = t2 - t1
    score = est.score(list(x_train), y_train)
    print("xgboost score:               " + str(score) + " time: " + str(time_el))

    score = est.score(list(x_test), y_test)
    print("xgboost score:               " + str(score) + " time: " + str(time_el))

def xgboost_grid(x, y):

    try:
        x = preprocessing.scale(np.array(x))
        scaler = preprocessing.StandardScaler().fit(x)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    except:
        x = list(x)
        x = preprocessing.scale(np.array(x))
        scaler = preprocessing.StandardScaler().fit(x)
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
    reg = GridSearchCV(xgb_temp, params, verbose=3, cv=3)
    x_train = scaler.transform(x_train)
    reg.fit(x_train, y_train)
    print(reg.best_params_)
    print(reg.best_score_)
    return reg


def xgboost_bayes_basic(x, y):
    try:
        x = preprocessing.scale(np.array(x))
        # scaler = preprocessing.StandardScaler().fit(x)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    except:
        x = list(x)
        x = preprocessing.scale(np.array(x))
        # scaler = preprocessing.StandardScaler().fit(x)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    xgb_temp = xgb.XGBRegressor(objective="reg:squarederror", tree_method="gpu_hist")

    reg = BayesSearchCV(
        xgb_temp, {
            "colsample_bytree": Real(0.2, 0.8),
            "max_depth": Integer(25, 60),
            "lambda": Real(0, 0.4),
            "learning_rate": Real(0.01, 0.15),
            "alpha": Real(0, 0.4),
            "eta": Real(0, 0.4),
            "gamma": Real(0, 0.4),
            "n_estimators": Integer(100, 6000)
        }, n_iter=2000, verbose=2, cv=3, n_jobs=4)

    reg.fit(list(x_train), y_train)
    print(reg.best_params_)
    print(reg.best_score_)
    print("Score on test data: " + str(reg.score(list(x_test), y_test)))
    return reg

def xgboost_rand(x, y):
    try:
        x = preprocessing.scale(np.array(x))
        scaler = preprocessing.StandardScaler().fit(x)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    except:
        x = list(x)
        x = preprocessing.scale(np.array(x))
        scaler = preprocessing.StandardScaler().fit(x)
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
    x_train = scaler.transform(x_train)
    reg.fit(x_train, y_train)

    print(reg.best_params_)
    print(reg.best_score_)
    return reg
