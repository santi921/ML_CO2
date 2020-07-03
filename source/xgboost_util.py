import time

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def xgboost(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # reg = xgb.XGBRegressor(**params)int()
    # params = {"objective": 'reg:squarederror', "colsample_bytree": 0.3, "learning_rate": 0.1,
    #         "max_depth": 50, "alpha": 10, "gamma": 1, "lambda": 1, "n_estimators": 20000}

    reg = GridSearchCV(xgb.XGBRegressor(gamma=1,
                                        alpha=1,
                                        verbose=1,
                                        n_jobs=16), {
                           "max_depth": [10, 20, 30],
                           "n_estimators": [10, 20, 30],
                           "learning_rate": [0.1, 0.2, 0.3]},
    , verbose = 1)

    est = make_pipeline(StandardScaler(), reg)

    t1 = time.time()
    est.fit(list(x_train), y_train)
    t2 = time.time()
    time_el = t2 - t1
    pr
    score = est.score(list(x_test), y_test)
    print("xgboost score:               " + str(score) + " time: " + str(time_el))
