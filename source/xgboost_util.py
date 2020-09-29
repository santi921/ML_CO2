import sklearn.utils.fixes
from numpy.ma import MaskedArray
sklearn.utils.fixes.MaskedArray = MaskedArray

from datetime import datetime
import csv, joblib, time, sigopt
import numpy as np
import xgboost as xgb
import scipy.stats as stats

from sigopt import Connection
from skopt.searchcv import BayesSearchCV
from skopt.callbacks import DeadlineStopper, CheckpointSaver
from skopt.space import Real, Integer

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score,\
    make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV,\
    ShuffleSplit, cross_val_score

csv_loc_rand = "../data/train/rand.csv"
csv_loc_bayes = "../data/train/bayes.csv"


def xgboost(x, y, scale, dict=None):
    
    x = np.array(x)
    y = np.array(y)
    try:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    except:
        x = list(x)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    if(dict == None):
        params = {
            "colsample_bytree": 0.563133210670266,
            "learning_rate": 0.20875083323873022,
            "max_depth": 12, "gamma": 0.00,
            "lambda": 0.16649470140308757,
            "alpha": 0.023794165626311915,
            "eta": 0.0,
            "n_estimators": 3350}
    else:
        params = {}
        params["colsample_bytree"] = dict["colsample_bytree"]
        params["learning_rate"] = dict["learning_rate"]
        params["max_depth"] = dict["max_depth"]
        params["lambda"] = dict["lambda"]
        params["alpha"] = dict["alpha"]
        params["eta"] = dict["eta"]
        params["n_estimators"] = dict["n_estimators"]

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

def xgboost_bayes_basic(x, y, csv_loc= "../data/train/bayes.csv"):

    global csv_loc_bayes
    csv_loc_bayes = csv_loc

    try:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    except:
        x = list(x)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    xgb_temp = xgb.XGBRegressor()
    reg = BayesSearchCV(
        xgb_temp, {
            "colsample_bytree": Real(0.5, 0.99),
            "max_depth": Integer(5, 25),
            "lambda": Real(0, 0.25),
            "learning_rate": Real(0.1, 0.25),
            "alpha": Real(0, 0.2),
            "eta": Real(0, 0.1),
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
    #ckpt_loc = "../data/train/bayes/ckpt_bayes_xgboost" + str(year) + "_"+ str(month) + "_" + str(day) + "_" + \
    #           str(hour) + "_" + str(minute) + "_" + str(sec) + ".pkl"
    #checkpoint_callback = CheckpointSaver(ckpt_loc)
    #reg.fit(x_train, y_train, callback=[DeadlineStopper(time_to_stop), checkpoint_callback])



    custom_scorer = custom_skopt_scorer
    reg.fit(x_train, y_train, callback=[DeadlineStopper(time_to_stop), custom_scorer(x,y)])
    #reg.fit(x_train, y_train, callback=[DeadlineStopper(time_to_stop)])

    score = str(mean_squared_error(reg.predict(x_test), y_test))
    print("MSE score:   " + str(score))
    score = str(mean_absolute_error(reg.predict(x_test), y_test))
    print("MAE score:   " + str(score))
    score = str(r2_score(reg.predict(x_test), y_test))
    print("r2 score:   " + str(score))
    return reg

def xgboost_rand(x, y, csv_loc = "../data/train/rand.csv"):
    global csv_loc_rand
    csv_loc_rand = csv_loc

    try:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    except:
        x = list(x)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    params = {"colsample_bytree": stats.uniform(0.2, 0.8),
              "learning_rate": stats.uniform(0, 0.5),
              "max_depth": stats.randint(5, 25),
              "gamma": stats.uniform(0, 0.1),
              "lambda": stats.uniform(0.1, 0.2),
              "alpha": stats.uniform(0.0, 0.2),
              "eta": stats.uniform(0.0, 0.2),
              "n_estimators": stats.randint(300,2000)}

    xgb_temp = xgb.XGBRegressor(objective = 'reg:squarederror', tree_method= "gpu_hist")
    reg = RandomizedSearchCV(xgb_temp, scoring = custom_sklearn_scorer , param_distributions = params, verbose=3, cv=3)
    #reg = RandomizedSearchCV(xgb_temp , param_distributions = params, verbose=3, cv=3)

    reg.fit(x_train, y_train)

    print(reg.best_params_)
    print(reg.best_score_)

    return reg

def xgboost_bayes_sigopt(x, y):

    params = {"objective": "reg:squarederror", "tree_method": "gpu_hist",
         "colsample_bytree": sigopt.get_parameter("colsample_bytree", default = 0.5),
         "max_depth": sigopt.get_parameter("max_depth", default = 10),
         "lambda": sigopt.get_parameter("lambda", default = 0.0),
         "learning_rate": sigopt.get_parameter("learning_rate", default = 0.1),
         "alpha": sigopt.get_parameter("alpha", default = 0.0),
         "eta": sigopt.get_parameter("eta", default =0.01),
         "gamma": sigopt.get_parameter("gamma", default = 0),
         "n_estimators": sigopt.get_parameter("n_estimators", default= 500)}

    xgb_reg = xgb.XGBRegressor(**params)
    try:
        (mse, mae, r2) = evaluate_model(xgb_reg, x, y)
    except:
        (mse, mae, r2) = evaluate_model(xgb_reg, list(x), y)

    print("Current MSE: " + str(mse))
    print("Current MAE: " + str(mae))
    print("Current R_2: " + str(r2))
    sigopt.log_metric("mse", mse)
    sigopt.log_metric("mae", mae)
    sigopt.log_metric("r2", r2)

def evaluate_model(reg, x, y):

    cv = ShuffleSplit(n_splits=3)
    cv_mse = cross_val_score(reg, x, y, cv=cv, scoring = "neg_mean_squared_error")
    cv_mae = cross_val_score(reg, x, y, cv=cv, scoring = "neg_mean_absolute_error")
    cv_r2 = cross_val_score(reg, x, y, cv=cv, scoring = "r2")
    return np.mean(cv_mse), np.mean(cv_mae), np.mean(cv_r2)



def custom_sklearn_scorer(reg,x,y):

    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    reg.fit(x_train, y_train)

    y_pred = np.array(reg.predict(x_test))

    mean_squared_error = mean_squared_error(y_test, y_pred)
    mean_absolute_error = mean_absolute_error(y_test, y_pred)
    r2_score = r2_score(y_test, y_pred)

    alpha =  reg.get_params()["reg_alpha"]
    lamb = reg.get_params()["reg_lambda"]
    learn_rate = reg.get_params()["learning_rate"]
    colsamples = reg.get_params()["colsample_bytree"]
    n_estimators = reg.get_params()["n_estimators"]
    max_depth = reg.get_params()["max_depth"]
    gamma = reg.get_params()["gamma"]
    eta = reg.get_params()["eta"]
    mse = mean_squared_error
    mae = mean_absolute_error
    r2 = r2_score

    str_csv = str(alpha) + ", " + str(colsamples) + ", " + str(eta) + ", " + str(gamma) + ", " \
              + str(lamb) + ", " + str(learn_rate) + ", " + str(max_depth) + ", " + str(n_estimators) \
              + ", " + str(mse) + ", " + str(mae) + ", " + str(r2) + "\n"

    with open(csv_loc_rand, 'a') as fd:
        fd.write(str_csv)

    return np.mean(mean_squared_error)

class custom_skopt_scorer(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.dir = csv_loc_bayes

    def __call__(self, res):
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        dict = {
            "objective":"reg:squarederror", "tree_method":"gpu_hist",
             "alpha":res["x"][0], "colsample_bytree":res["x"][1],"eta":res["x"][2],
            "gamma":res["x"][3], "lambda": res["x"][4], "learning_rate":res["x"][5],
            "max_depth":res["x"][6],"n_estimators":res["x"][7]
        }
        reg = xgb.XGBRegressor(**dict)

        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, test_size=0.2)
        reg.fit(x_train, y_train)
        y_pred = np.array(reg.predict(x_test))

        alpha = res["x"][0]
        lamb = res["x"][4]
        learn_rate = res["x"][5]
        colsamples = res["x"][1]
        n_estimators = res["x"][7]
        max_depth = res["x"][6]
        gamma = res["x"][3]
        eta = res["x"][2]

        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        str_csv = str(alpha) + ", " + str(colsamples) + ", " + str(eta) + ", " + str(gamma) + ", " \
                  + str(lamb) + ", " + str(learn_rate) + ", " + str(max_depth) + ", " + str(n_estimators) \
                  + ", " + str(mse) + ", " + str(mae) + "," + str(r2) + "\n"
        with open(self.dir, 'a') as fd:
            fd.write(str_csv)

        return 0

