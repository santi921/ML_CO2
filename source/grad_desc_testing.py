import numpy as np
import xgboost as xgb

from utils.selfies_util import smile_to_hot, multiple_smile_to_hot, selfies_to_hot, \
multiple_selfies_to_hot, get_selfie_and_smiles_encodings_for_dataset, compare_equality, \
tanimoto_dist, smiles, selfies

from scipy.integrate import odeint, simpson
from skopt.searchcv import BayesSearchCV
from skopt.space import Real, Integer

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers,regularizers

from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import scale, StandardScaler
from sklearn.ensemble import ExtraTreesRegressor

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
import os

def train_model(x, y):
    encoder = keras.models.load_model("./encoder")
    selfies_list, selfies_alphabet, largest_selfies_len, \
    smiles_list, smiles_alphabet, largest_smiles_len = get_selfie_and_smiles_encodings_for_dataset(x)
    data = multiple_selfies_to_hot(selfies_list, largest_selfies_len, selfies_alphabet)
    data_smiles = multiple_smile_to_hot(smiles_list, largest_smiles_len, smiles_alphabet)
    max_mol_len = data.shape[1]
    alpha_len = data.shape[2]
    len_alphabet_mol = alpha_len * max_mol_len

    data_reshape = data.reshape(data.shape[0], data.shape[1] * data.shape[2], )
    scaler = StandardScaler()
    scaler.fit(data_reshape)
    data_reshape = scaler.transform(data_reshape)
    
    train_ind, test_ind = train_test_split(range(data.shape[0]), test_size=0.15)
    
    data_reshape = scale(data_reshape)
    x_train = data_reshape[train_ind]
    x_test = data_reshape[test_ind]   
    x_encoded_test = encoder.predict(x_test)[0]
    x_encoded_train = encoder.predict(x_train)[0]
    y_train = np.array([y[i] for i in train_ind])
    y_test = np.array([y[i] for i in test_ind])
        
    """
    model = keras.Sequential() # this could be subbed with xgboost
    model.add(keras.layers.Input(shape = (len(x_encoded_test[0]),) ))
    model.add(keras.layers.Dense(1000, kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)))
    model.add(keras.layers.Dense(1000, kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)))
    model.add(keras.layers.Dense(1))
    model.compile(optimizer="Adam", loss="mse", metrics=["mae"])
    #es = keras.callbacks.EarlyStopping(monitor="val_loss", patience = 5)
    model.fit(np.array(x_encoded_train), np.array(y_train), validation_data = (x_encoded_test, y_test), 
              epochs = 100)
    #callbacks = [es]
    """
    xgb_model = xgb.XGBRegressor(n_jobs=-1)
    extra_model = ExtraTreesRegressor(criterion="mse", bootstrap=True)
    params_xgb = {
            "colsample_bytree": Real(0.3, 0.99),
            "max_depth": Integer(5, 100),
            "lambda": Real(0, 0.25),
            "learning_rate": Real(0.001, 0.5, prior='log-uniform'),
            "alpha": Real(0.01, 0.2, prior='log-uniform'),
            "eta": Real(0.01, 0.2, prior='log-uniform'),
            "gamma": Real(0.01, 0.2, prior='log-uniform'),
            "n_estimators": Integer(200, 1500),
            "objective": ["reg:squarederror"],
            "tree_method": ["gpu_hist"]
        }
    params_extra = {
          "n_estimators": Integer(100, 1000),
          "max_depth": Integer(5, 100),
          "min_samples_split": Integer(2,4),
          "min_samples_leaf": Integer(2,4)
          } 
    
    model_extra = BayesSearchCV(extra_model, params_extra, n_iter=20, verbose=0, cv=3) 
    model_xgb = BayesSearchCV(xgb_model, params_xgb, n_iter=20, verbose=0, cv=3) 

    cv_model = model_xgb
    cv_model.fit(np.array(x_encoded_train), np.array(y_train))
    print("best model score:")
    print(model_xgb.best_score_)
    print("best test score: ")
    y_hat_test = cv_model.predict(np.array(x_encoded_test))
    print(r2_score(y_hat_test, y_test))
    print("best train score: ")
    y_hat_train = cv_model.predict(np.array(x_encoded_train))
    print(r2_score(y_hat_train, y_train))
    return cv_model



names, ret_self, homo, homo1, diff = selfies()
homo_cv = train_model(names, homo)
diff_cv = train_model(names, diff)
homo1_cv = train_model(names, homo1)
