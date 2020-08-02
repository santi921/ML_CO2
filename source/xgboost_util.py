import time
import numpy as np
import xgboost as xgb
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV

def xgboost(x, y):
    try:
        x = preprocessing.scale(np.array(x))
        scaler = preprocessing.StandardScaler().fit(x)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    except:
        x = list(x)
        x = preprocessing.scale(np.array(x))
        scaler = preprocessing.StandardScaler().fit(x)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    params = {"objective": "reg:squarederror",
              "colsample_bytree": 0.5,
              "learning_rate": 0.1,
              "max_depth": 20, "gamma": 0.0,
              "lambda": 0.1,
              "alpha": 0.1,
              "eta": 0.0,
              "n_estimators": 4000,
              "tree_method": "gpu_hist"}

    reg = xgb.XGBRegressor(**params)
    est = make_pipeline(StandardScaler(), reg)
    x_train = scaler.transform(x_train)

    t1 = time.time()
    # non grid
    est.fit(list(x_train), y_train)
    t2 = time.time()
    time_el = t2 - t1
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
              "colsample_bytree": [0.5, 0.75],
              "learning_rate": [0.1, 0.2, 0.3],
              "max_depth": [10, 20, 50], "gamma": [0.0, 0.05, 0.1],
              "lambda": [0.1, 0.2],
              "alpha": [0.1],
              "eta": [0.0, 0.1],
              "n_estimators": [400, 4000],
              "tree_method": ["gpu_hist"]}

    xgb_temp = xgb.XGBRegressor()
    reg = GridSearchCV(xgb_temp, params, verbose=6, cv=3)
    x_train = scaler.transform(x_train)
    reg.fit(x_train, y_train)
    print(reg.best_params_)
    print(reg.best_score_)
