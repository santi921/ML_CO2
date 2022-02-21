
from utils.selfies_util import (
    smile_to_hot,
    multiple_smile_to_hot,
    selfies_to_hot,
    multiple_selfies_to_hot,
    get_selfie_and_smiles_encodings_for_dataset,
    compare_equality,
    tanimoto_dist,
    smiles,
    selfies,
)

import tensorflow as tf
import tensorflow.keras as keras
import numpy as np



names, ret_list, homo, homo1, diff = selfies()
print(ret_list[-10:])

(
    selfies_list,
    selfies_alphabet,
    largest_selfies_len,
    smiles_list,
    smiles_alphabet,
    largest_smiles_len,
) = get_selfie_and_smiles_encodings_for_dataset(names)

selfies_alphabet.sort()
print("len of alphabet: " + str(len(selfies_alphabet)))
print("alphabet list: " + str(selfies_alphabet))
print("number of molecules: " + str(len(names)))


data = multiple_selfies_to_hot(selfies_list, largest_selfies_len, selfies_alphabet)
data_smiles = multiple_smile_to_hot(
    smiles_list, largest_smiles_len, smiles_alphabet
)
max_mol_len = data.shape[1]
alpha_len = data.shape[2]
len_alphabet_mol = alpha_len * max_mol_len

data_reshape = data.reshape(
    data.shape[0],
    data.shape[1] * data.shape[2],
)

encoder = keras.models.load_model("./encoder")
decoder = keras.models.load_model("./decoder")


(
    selfies_list,
    selfies_alphabet1,
    largest_selfies_len,
    smiles_list,
    smiles_alphabet,
    largest_smiles_len,
) = get_selfie_and_smiles_encodings_for_dataset(ret_list)
print(len(selfies_alphabet1))



test = selfies_list
(
    selfies_list,
    selfies_alphabet2,
    largest_selfies_len,
    smiles_list,
    smiles_alphabet,
    largest_smiles_len,
) = get_selfie_and_smiles_encodings_for_dataset(ret_list)

test = selfies_list
data = multiple_selfies_to_hot(selfies_list, largest_selfies_len, selfies_alphabet2)
data_reshape = data.reshape(
    data.shape[0],
    data.shape[1] * data.shape[2],
)

print("data converted")
print(np.shape(data_reshape))
encoder.summary()
decoder.summary()

encoded_data = encoder.predict(data_reshape)
code_decode = decoder.predict(encoded_data)
print("prediction step")
compare_equality(data, code_decode, (data.shape[1], data.shape[2]), selfies_alphabet)
