import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers, regularizers

import argparse
import numpy as np
from sklearn.model_selection import train_test_split

from utils.selfies_util import smile_to_hot, multiple_smile_to_hot, selfies_to_hot, \
    multiple_selfies_to_hot, get_selfie_and_smiles_encodings_for_dataset, compare_equality, \
    tanimoto_dist, smiles, selfies

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))

        return (z_mean + tf.exp(z_log_var) * epsilon)


class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker, ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = keras.losses.mse(data, reconstruction)
            # reconstruction_loss = keras.losses.binary_crossentropy(data, reconstruction)
            kl_loss = (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = -0.5 * tf.reduce_mean(kl_loss)
            beta = coef * latent_dim / input_size
            total_loss = reconstruction_loss + beta * kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(), }


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='parameters of vae to train')
    parser.add_argument("--nn", action="store_true", dest="tf_nn_vae", default=False, help="ltsm or nn encoder?")
    parser.add_argument("-beta", action="store", dest="coef", default=0.1, help="coef for beta in vae loss")
    parser.add_argument("-epochs", action="store", dest="epochs", default=100, help="epochs")
    parser.add_argument("-latdim", action="store", dest="latdim", default=100, help="latent dimension")
    parser.add_argument("-ltsm_encode", action="store", dest="ltsm_encode", default=75, help="ltsm encoder dim")
    parser.add_argument("-ltsm_decode", action="store", dest="ltsm_decode", default=75, help="ltsm decode dim")
    parser.add_argument("--nn_encode", action="store", dest="nn_encode", default="100, 100", help="nn encode dim")
    parser.add_argument("--nn_decode", action="store", dest="nn_decode", default="100, 100, 100, 100",
                        help="nn decode dim")
    parser.add_argument("--verbose", action="store_true", dest="verbose", default=False, help="verbose")
    parser.add_argument("--save", action="store_true", dest="save", default=False, help="save trained vae")

    results = parser.parse_args()

    global coef
    global latent_dim
    global input_size
    global timesteps

    verbose = int(results.verbose)
    save = results.save

    coef = float(results.coef)
    tf_nn_vae = results.tf_nn_vae
    epochs = int(results.epochs)
    latent_dim = int(results.latdim)
    ltsm_encode = int(results.ltsm_encode)
    ltsm_decode = int(results.ltsm_decode)
    nn_encode = results.nn_encode
    nn_decode = results.nn_decode


    #files, ret_list = smiles(verbose=verbose)
    #files, ret_list = smiles("../data/smi/DB3/", verbose=verbose)
    #ret_list = []
    '''
    ### This is one option for pulling smis for vae training
    dir = "../data/smi/DB3/actual_smi/"
    files = os.listdir(dir)
    for file in files:
        try:
            with open(dir + file, "rb+") as filehandle:
                ret_list.append(filehandle.readlines()[0].decode("utf-8"))
        except:
            pass
    '''
    # more standard method

    names, ret_self, homo, homo1, diff = selfies()
    print(ret_list[0:10])

    selfies_list, selfies_alphabet, largest_selfies_len, \
    smiles_list, smiles_alphabet, largest_smiles_len = get_selfie_and_smiles_encodings_for_dataset(ret_list)

    selfies_alphabet.sort()
    print("len of alphabet: " + str(len(selfies_alphabet)))
    print("alphabet list: " + str(selfies_alphabet))

    data = multiple_selfies_to_hot(selfies_list, largest_selfies_len, selfies_alphabet)
    data_smiles = multiple_smile_to_hot(smiles_list, largest_smiles_len, smiles_alphabet)
    max_mol_len = data.shape[1]
    alpha_len = data.shape[2]
    len_alphabet_mol = alpha_len * max_mol_len

    data_reshape = data.reshape(data.shape[0], data.shape[1] * data.shape[2], )
    train_ind, test_ind = train_test_split(range(data.shape[0]), test_size=0.15, random_state=11)
    x_train = data_reshape[train_ind]
    x_test = data_reshape[test_ind]

    if (tf_nn_vae == True):

        input_size = data.shape[1] * data.shape[2]
        timesteps = data.shape[1]

        # create Encoder
        inputs = keras.Input(shape=(data.shape[1] * data.shape[2]))
        encode_arr = nn_encode.split(",")
        decode_arr = nn_decode.split(",")
        for ind, temp_layer in enumerate(encode_arr):
            if (ind == 0):
                x = layers.Dense(temp_layer, activation='relu')(inputs)
                x = layers.BatchNormalization()(x)
                x = layers.Dropout(0.25)(x)

            else:
                x = layers.Dense(temp_layer, activation='relu')(x)

        # Sampling Layers
        z_mean = layers.Dense(latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])
        encoder = keras.Model(inputs, [z_mean, z_log_var, z], name="encoder")

        # Create Decoder
        latent_inputs = keras.Input(shape=(latent_dim,))
        for ind, temp_layer in enumerate(decode_arr):
            if (ind == 0):
                decoded = layers.Dense(temp_layer, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4))(latent_inputs)
                decoded = layers.BatchNormalization()(decoded)
                decoded = layers.Dropout(0.25)(decoded)
            else:
                decoded = layers.Dense(temp_layer, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4))(decoded)
                decoded = layers.BatchNormalization()(decoded)
                decoded = layers.Dropout(0.25)(decoded)

        decoded = layers.Dense(data.shape[1] * data.shape[2], activation='sigmoid')(decoded)
        decoder = keras.Model(latent_inputs, decoded, name="decoder")

        vae = VAE(encoder, decoder)  # build vae models
        vae.compile(keras.optimizers.Adam(learning_rate=0.0001))
        decoder.summary()
        encoder.summary()
        es = EarlyStopping(monitor='loss', verbose=0, patience=3)
        history = vae.fit(x_train, epochs=epochs, callbacks=[es], verbose=verbose)

        train_data = x_train.reshape(x_train.shape[0], data.shape[1] * data.shape[2])
        test_data = x_test.reshape(x_test.shape[0], data.shape[1] * data.shape[2])

    else:
        timesteps = data.shape[1]
        input_size = data.shape[2]

        # Create encoder
        inputs = keras.Input(shape=(timesteps, input_size))
        x = layers.LSTM(ltsm_encode, kernel_regularizer=regularizers.l2(0.001), recurrent_regularizer=regularizers.l2(0.001), bias_regularizer=regularizers.l2(0.001))(inputs)

        # Sampling Layers
        z_mean = layers.Dense(latent_dim, name="z_mean")(x)
        z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])
        encoder = keras.Model(inputs, [z_mean, z_log_var, z], name="encoder")

        # Create Decoder
        input_latent = keras.Input(shape=(latent_dim,))
        decoder1 = layers.RepeatVector(timesteps)(input_latent)
        decoder1 = layers.Dropout(rate = 0.10)(decoder1)
        decoder1 = layers.LSTM(ltsm_decode, return_sequences=True, kernel_regularizer=regularizers.l2(0.001), recurrent_regularizer=regularizers.l2(0.001), bias_regularizer=regularizers.l2(0.001))(decoder1)
        decoder1 = layers.Dropout(rate = 0.10)(decoder1)
        decoder1 = layers.TimeDistributed(layers.Dense(input_size))(decoder1)
        decoder = keras.Model(input_latent, decoder1)

        vae = VAE(encoder, decoder)
        vae.compile(keras.optimizers.Adam(learning_rate=0.001))

        es = EarlyStopping(monitor='loss', mode='min', verbose=0, patience=5)
        train_data = x_train.reshape(x_train.shape[0], data.shape[1], data.shape[2])
        test_data = x_test.reshape(x_test.shape[0], data.shape[1], data.shape[2])
        print(type(train_data))
        print(np.shape(train_data))
        encoder.summary()
        decoder.summary()
        history = vae.fit(train_data, train_data, epochs=epochs, verbose=verbose)

    print(type(train_data))
    print(np.shape(train_data))
    encoder_train = vae.encoder.predict(train_data)
    encoder_test = vae.encoder.predict(test_data)
    print(type(encoder_test))
    print(np.shape(encoder_test))

    try:
        code_decode_train = vae.decoder.predict(encoder_train)
        code_decode_test = vae.decoder.predict(encoder_test)
    except:
        code_decode_train = vae.decoder.predict(encoder_train[0])
        code_decode_test = vae.decoder.predict(encoder_test[0])

    encoder.summary()
    decoder.summary()

    print("..............statistics for training dataset..............")
    compare_equality(
        train_data,
        code_decode_train,
        (data.shape[1], data.shape[2]),
        selfies_alphabet)
    print("..............statistics for test dataset..............")
    compare_equality(
        test_data,
        code_decode_test,
        (data.shape[1], data.shape[2]),
        selfies_alphabet)
    print(".................Model Parameters.................")

    if (tf_nn_vae == True):
        print("NN-based VAE")
    else:
        print("LSTM-based VAE")
    if(save == True):
        encoder.save("encoder")
        decoder.save("decoder")

    print("epochs:" + str(epochs))
    print("latent dims:" + str(latent_dim))
    print("beta: " + str(coef))



