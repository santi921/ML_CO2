import sklearn.utils.fixes
from numpy.ma import MaskedArray
sklearn.utils.fixes.MaskedArray = MaskedArray
import os
import time
from datetime import datetime
import joblib
import numpy as np
import xgboost as xgb
import scipy.stats as stats

import sigopt
from sigopt_sklearn.search import SigOptSearchCV
from sigopt import Connection
from skopt import BayesSearchCV
from skopt.callbacks import DeadlineStopper, CheckpointSaver
from skopt.space import Real, Integer

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV,\
    ShuffleSplit, cross_val_score

def evaluate_model(reg, x, y):

    cv = ShuffleSplit(n_splits=3)
    cv_mse = cross_val_score(reg, x, y, cv=cv, scoring = "neg_mean_squared_error")
    cv_mae = cross_val_score(reg, x, y, cv=cv, scoring = "neg_mean_absolute_error")
    cv_r2 = cross_val_score(reg, x, y, cv=cv, scoring = "r2")
    return (np.mean(cv_mse), np.mean(cv_mae), np.mean(cv_r2))


def xgboost(x, y, scale):
    x = np.array(x)
    y = np.array(y)
    try:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    except:
        x = list(x)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    params = {
        "colsample_bytree": 0.563133210670266,
        "learning_rate": 0.20875083323873022,
        "max_depth": 12, "gamma": 0.00,
        "lambda": 0.16649470140308757,
        "alpha": 0.023794165626311915,
        "eta": 0.0,
        "n_estimators": 3350}


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

    time_to_stop = 60 * 60
    ckpt_loc = "../data/train/bayes/ckpt_bayes_xgboost.pkl"
    checkpoint_callback = CheckpointSaver(ckpt_loc)
    reg.fit(x_train, y_train, callback=[DeadlineStopper(time_to_stop), checkpoint_callback])
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
        n_iter=5000,
        verbose=4, cv=3)

    time_to_stop = 60 * 60 * 47

    now = datetime.now()
    year = now.strftime("%Y")
    month = now.strftime("%m")
    day = now.strftime("%d")
    hour = now.strftime("%H")
    minute = now.strftime("%M")
    sec = now.strftime("%S")

    ckpt_loc = "../data/train/bayes/ckpt_bayes_xgboost" + str(year) + "_"+ str(month) + "_" + str(day) + "_" + \
               str(hour) + "_" + str(minute) + "_" + str(sec) + ".pkl"

    checkpoint_callback = CheckpointSaver(ckpt_loc)
    reg.fit(x_train, y_train, callback=[DeadlineStopper(time_to_stop), checkpoint_callback])

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

def xgboost_bayes_sigopt(x, y):

    try:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    except:
        x = list(x)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    conn = Connection(client_token="BQQYYDTUYJASQCFMUKVLJJEAWAESEKTAHTFKSBHTVBACYTDZ")
    # 328985 - morg, db3, xgboost

    '''
    experiment = conn.experiments().create(
        name="xgboost",
        project="ml_co2_xgboost_morg",
        metrics=[dict(name='nmse', objective='maximize')],
        parameters=[
            dict(name="colsample_bytree", type="double", bounds=dict(min=0.5, max=0.99)),
            dict(name="max_depth", type="int", bounds=dict(min=5, max=28)),
            dict(name="lambda", type="double", bounds=dict(min=0, max=0.25)),
            dict(name="learning_rate", type="double", bounds=dict(min=0.1, max=0.25)),
            dict(name="alpha", type="double", bounds=dict(min=0, max=0.2)),
            dict(name="eta", type="double", bounds=dict(min=0.01, max=0.2)),
            dict(name="gamma", type="double", bounds=dict(min=0, max=0.1)),
            dict(name="n_estimators", type="int", bounds=dict(min=300, max=5000))
        ],
        observation_budget = 5
    )
    '''

    temp_dict = {"objective": "reg:squarederror", "tree_method": "gpu_hist",
         "colsample_bytree": sigopt.get_parameter("colsample_bytree", default = 0.5),
         "max_depth": sigopt.get_parameter("max_depth", default = 10),
         "lambda": sigopt.get_parameter("lambda", default = 0.0),
         "learning_rate": sigopt.get_parameter("learning_rate", default = 0.1),
         "alpha": sigopt.get_parameter("alpha", default = 0.0),
         "eta": sigopt.get_parameter("eta", default =0.01),
         "gamma": sigopt.get_parameter("gamma", default = 0),
         "n_estimators": sigopt.get_parameter("n_estimators", default= 500)}
    print(temp_dict)
    xgb_reg = xgb.XGBRegressor(**temp_dict  )

    (mse, mae, r2) = evaluate_model(xgb_reg, x, y)

    print("Current MSE: " + str(mse))
    print("Current MAE: " + str(mae))
    print("Current R_2: " + str(r2))

    sigopt.log_metric("mse", mse)
    sigopt.log_metric("mae", mae)
    sigopt.log_metric("r2", r2)


    '''
    best_assignments = conn.experiments(experiment.id).best_assignments().fetch().data[0].assignments

    xgb_reg_best = xgb.XGBRegressor({"objective" : "reg:squarederror", "tree_method" : "gpu_hist",
                                    "colsample_bytree" : best_assignments["colsample_bytree"],
                                    "max_depth" : best_assignments["max_depth"],
                                    "lambda" : best_assignments["lambda"],
                                    "learning_rate" : best_assignments["learning_rate"],
                                    "alpha" : best_assignments["alpha"],
                                    "eta": best_assignments["eta"],
                                    "gamma" : best_assignments["gamma"],
                                    "n_estimators" : best_assignments["n_estimators"]})

    #(mse, mae, r2) = evaluate_model(xgb_reg_best)
    #print("Best MSE: " + str(mse))
    #print("Best MAE: " + str(mae))
    #print("Best R_2: " + str(r2))
    '''
    #return xgb_reg_best
