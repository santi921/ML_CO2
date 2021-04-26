import joblib, argparse, uuid, sigopt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import DataStructs
from sklearn import preprocessing
from utils.sklearn_utils import *
import matplotlib.pyplot as plt
import seaborn as sns

import selfies as sf
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,regularizers
from utils.selfies_util import selfies,smile_to_hot, \
multiple_smile_to_hot, selfies_to_hot, multiple_selfies_to_hot,\
get_selfie_and_smiles_encodings_for_dataset, compare_equality, tanimoto_dist


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    except RuntimeError as e:
        print(e)

names, ret, homo, homo1, diff = selfies()
selfies_list, selfies_alphabet, largest_selfies_len,\
smiles_list, smiles_alphabet, largest_smiles_len\
= get_selfie_and_smiles_encodings_for_dataset(names)

data = multiple_selfies_to_hot(selfies_list, largest_selfies_len,\
                                       selfies_alphabet)

max_mol_len = data.shape[1]
alpha_len = data.shape[2]
len_alphabet_mol = alpha_len * max_mol_len

#-----------------------------------------------------------------------------
# LSTM Auto encoder


data_reshape = data.reshape(data.shape[0], data.shape[1] * data.shape[2])
x_train = data[0:int(data.shape[0] * 0.8)]
x_test = data[int(data.shape[0] * 0.8):-1]

timesteps = data.shape[1]
input_dim = data.shape[2]
latent_dim = 64


inputs = keras.Input(shape=(timesteps, input_dim))

encoded = layers.LSTM(512,return_sequences=True)(inputs)
encoded = layers.LSTM(128,return_sequences=True)(encoded)
encoder = layers.LSTM(latent_dim,return_sequences=True)(encoded)
# batch norm??
decoded = layers.LSTM(128,return_sequences=True)(encoder)
decoded = layers.LSTM(512,return_sequences=True)(decoded)
decoder = layers.LSTM(input_dim, return_sequences=True)(encoded)
sequence_autoencoder = keras.Model(inputs, decoder)
encoder = keras.Model(inputs, encoded)
sequence_autoencoder.compile(optimizer=keras.optimizers.Adam(),
                             loss = "mse",
                            metrics=[tf.keras.metrics.RootMeanSquaredError()])
sequence_autoencoder.summary()
history = sequence_autoencoder.fit(x_train, x_train, epochs=15, batch_size=512,
                   validation_data=(x_test, x_test))

autoencode_out = sequence_autoencoder.predict(x_test)
autoencode_out_reshape = autoencode_out.reshape((len(x_test), data.shape[1], data.shape[2]))
################### AUTOENCODER TEST SCRIPTS - Full test script
print(np.shape(x_test.reshape(x_test.shape[0], data.shape[1], data.shape[2])))
print(np.shape(autoencode_out_reshape))
compare_equality(x_test.reshape(x_test.shape[0], data.shape[1], data.shape[2]), autoencode_out_reshape,
                 (data.shape[1], data.shape[2]), selfies_alphabet)



x_test_encoded = encoder.predict(x_test)
x_test_encoded_reshape = x_test_encoded.reshape(x_test_encoded.shape[0], x_test_encoded.shape[1] * x_test_encoded.shape[2])
x_train_encoded = encoder.predict(x_train)
x_train_encoded_reshape = x_train_encoded.reshape(x_train_encoded.shape[0], x_train_encoded.shape[1] * x_train_encoded.shape[2])
diff_train = diff[0:int(data.shape[0] * 0.8)]
diff_test = diff[int(data.shape[0] * 0.8):-1]
encode_dim = np.shape(x_test_encoded[0])
x_test_encoded = x_test_encoded.reshape(-1, encode_dim[0], encode_dim[1])
x_train_encoded = x_train_encoded.reshape(-1, encode_dim[0], encode_dim[1])

######################### MODELS #########################
model = keras.Sequential()
dims = np.shape(x_test_encoded)
print(dims)
#model.add(layers.Embedding(input_dim=dims[0], output_dim=dims[1])) input_dim=dims[0], output_dim=dims[1]
model.add(layers.LSTM(128, dropout = 0.25, input_shape = (dims[1], dims[2])))
model.add(layers.Dense(1))
model.summary()
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.002),
        loss = "mse", metrics=[tf.keras.metrics.RootMeanSquaredError()])
history = model.fit(np.array(x_train_encoded), np.array(diff_train),
                    epochs=50,  validation_split=0.15, verbose = 1)

y_pred = model.predict(x_test_encoded)
y_pred_train = model.predict(x_train_encoded)
print(np.array(diff_test).shape)
print(y_pred.shape)
print(r2_score(np.array(y_pred), diff_test))
print(r2_score(np.array(y_pred_train), diff_train))


