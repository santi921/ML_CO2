import sys
import os
import pandas as pd
import numpy as np
import selfies as sf
import argparse, pybel
from matplotlib import pyplot as plt
from scipy.integrate import odeint, simpson
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers,regularizers

from data_utils import smile_to_hot, multiple_smile_to_hot, selfies_to_hot, \
multiple_selfies_to_hot, get_selfie_and_smiles_encodings_for_dataset, compare_equality, \
tanimoto_dist, smiles


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

def PBPKsim(params, dose, species, simTime, adminRoute):
    # parse out estimated parameters
    KpISF = params[0]
    KpCSF = params[1]
    KpT = params[2]
    PSB = params[3]
    PST = params[4]
    CL = params[5]

    # physiological parameters
    # for rat:
    if (species == "Rat"):
        # Volumes(L)
        Vp = 0.00906  # volume of plasma in systemic circulation
        VBv = 0.0000502  # volume of brain plasma
        VCSF = 0.00012  # Volume of brain cerebrospinal fluid, 90-150uL
        VBev = 0.00228 - VBv - VCSF  # Volume of brain extravasculature
        VTv = 0.00789468  # Volume of tissue vasculature
        VTev = 0.28 - Vp - VBv - VCSF - VBev - VTv  # Volume of tissue extravasculature

        # Flows(L/h)
        QCSF = 0.000132  # CSF flow to brain tissue and back from brain tissue
        QB = 0.0653  # Flow to brain vasculature
        QT = 2.8797  # Flow to tissue vasculature
        CLG = 0.00012  # Glymphatic clearance set to zero

        # Brain Permeability Surface Coefficients
        PSBBB = PSB * (0.0155 / 0.018)  # (BBB_SA/Total_SA)
        PSCSF = PSB * (0.0025 / 0.018)  # (BCSFB_SA/Total_SA)

    # for human:
    else:
        # Volumes (L)
        Vp = 3.126  # Volume of plasma in systemic circulation
        VBv = 0.0319  # volume of brain plasma
        VCSF = 0.15  # Volume of brain cerebrospinal fluid, 90-150mL in humans
        VBev = 1.45 - VBv - VCSF  # Volume of brain extravasculature
        VTv = 1.682043  # Volume of tissue vasculature
        VTev = 71 - Vp - VBv - VCSF - VBev - VTv  # Volume of tissue extravasculature

        # Flows (L/h)
        QCSF = 0.01  # CSF flow to brain tissue and back from brain tissue
        QB = 21.453  # Flow to brain vasculature
        QT = 160.46  # Flow to tissue vasculature
        CLG = 0.023  # Glymphatic clearance set to zero

        # Permeability Surface Coefficients
        # uncomment these lines for only one PSB parameter
        PSBBB = PSB * (20 / 27.5)  # (BBB_SA / Total_SA)
        PSCSF = PSB * (7.5 / 27.5)  # (BCSFB_SA / Total_SA)

    # PBPK model structure
    def PBPKmodel(y, t):
        # define outputs
        Cp = y[0]  # concentration in systemic plasma
        CBv = y[1]  # concentration in brain vasculature
        CCSF = y[2]  # concentration in cerebrospinal fluid
        CBev = y[3]  # concentration in brain extravasculature (i.e. the interstitial fluid)
        CTv = y[4]  # concentration in tissue vasculature
        CTev = y[5]  # concentration in tissue extravasculature

        # define ODE equations
        dCpdt = ((QB * (CBv - Cp) + QT * (CTv - Cp) + (CLG * CCSF) - (CL * Cp))) / Vp
        dCBvdt = (QB * (Cp - CBv) + PSCSF * (CCSF / KpCSF - CBv) + PSBBB * (CBev / KpISF - CBv)) / VBv
        dCCSFdt = (PSCSF * (CBv - CCSF / KpCSF) + QCSF * (CBev / (KpISF / KpCSF) - CCSF) - (CLG * CCSF)) / VCSF
        dCBevdt = (PSBBB * (CBv - CBev / KpISF) + QCSF * (CCSF - CBev / (KpISF / KpCSF))) / VBev
        dCTvdt = (QT * (Cp - CTv) + PST * (CTev / KpT - CTv)) / VTv
        dCTevdt = (PST * (CTv - CTev / KpT)) / VTev

        return [dCpdt, dCBvdt, dCCSFdt, dCBevdt, dCTvdt, dCTevdt]

    # initial conditions
    y0 = [dose / Vp, 0, 0, 0, 0, 0]  # dose enters absorption compartment first

    # vector of simulation times
    t = np.arange(0, simTime + 0.01, 0.01)

    # solve PBPK ODEs with the given parameters and initial conditions
    # PBPKmodel function is defined above
    y = odeint(PBPKmodel, y0, t)

    return y

def integrate_pbpk(kpisf, kpcsf, kpt, psb, pst, cl, dose, species="Rat", admin_route=1):
    species = "Rat"  # can change this to human and code will run with new physioligical parameters
    dose = df.iloc[0, 3]  # get drug dose, in umol

    simTime = 24  # simulation length, in hours
    t = np.arange(0, simTime + 0.01, 0.01)  # vector of simulation times
    adminRoute = 1  # 1=Intravenous
    params = [kpisf, kpcsf, kpt, psb, pst, cl]  # parameter vector
    y = PBPKsim(params, dose, species, simTime, admin_route)  # run ODE solver

    # some volumes for later calculations
    if (species == "Rat"):
        VBv = 0.0000502  # volume of brain plasma
        VCSF = 0.00012  # Volume of brain cerebrospinal fluid
        VBev = 0.00228 - VBv - VCSF  # Volume of brain extravasculature
    else:  # human
        VBv = 0.0319  # volume of brain plasma
        VCSF = 0.15  # Volume of brain cerebrospinal fluid, 90-150mL in humans
        VBev = 1.45 - VBv - VCSF  # Volume of brain extravasculature

    Cp = np.array(y[:, 0])  # plasma concentration
    CB = np.array(((y[:, 1] * VBv) + (y[:, 3] * VBev) + (y[:, 2] * VCSF)) / (
                VBv + VBev + VCSF))  # concentration in brain homogenate
    CCSF = np.array(y[:, 2])  # CSF concentration
    CBev = np.array(y[:, 3])  # ISF concentration

    # return simpson(CBev, t)
    # return simpson(CCSF, t)
    # return simpson(Cp, t)

    return simpson(CB, t)

def train_model(epochs=60, latent=75, beta=0.0005, encode=75, decode=75):
    global coef
    global latent_dim
    global input_size
    global timesteps

    verbose = 1
    coef = float(beta)
    epochs = int(epochs)
    latent_dim = int(latent)
    ltsm_encode = int(encode)
    ltsm_decode = int(decode)

    files, ret_list = smiles("./data/smi_qm9/", verbose=verbose)
    selfies_list, selfies_alphabet, largest_selfies_len, \
    smiles_list, smiles_alphabet, largest_smiles_len = get_selfie_and_smiles_encodings_for_dataset(ret_list)

    data = multiple_selfies_to_hot(selfies_list, largest_selfies_len, selfies_alphabet)
    data_smiles = multiple_smile_to_hot(smiles_list, largest_smiles_len, smiles_alphabet)
    max_mol_len = data.shape[1]
    alpha_len = data.shape[2]
    len_alphabet_mol = alpha_len * max_mol_len

    data_reshape = data.reshape(data.shape[0], data.shape[1] * data.shape[2], )
    train_ind, test_ind = train_test_split(range(data.shape[0]), test_size=0.15)
    x_train = data_reshape[train_ind]
    x_test = data_reshape[test_ind]

    timesteps = data.shape[1]
    input_size = data.shape[2]

    # Create encoder
    inputs = keras.Input(shape=(timesteps, input_size))
    x = layers.LSTM(ltsm_encode, activation='relu')(inputs)
    # Sampling Layers
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(inputs, [z_mean, z_log_var, z], name="encoder")

    # Create Decoder
    input_latent = keras.Input(shape=(latent_dim,))
    decoder1 = layers.RepeatVector(timesteps)(input_latent)
    decoder1 = layers.LSTM(ltsm_decode, activation='sigmoid', return_sequences=True)(decoder1)
    decoder1 = layers.TimeDistributed(layers.Dense(input_size))(decoder1)
    decoder = keras.Model(input_latent, decoder1)

    vae = VAE(encoder, decoder)
    vae.compile(keras.optimizers.Adam(learning_rate=0.001))

    es = EarlyStopping(monitor='loss', mode='min', verbose=0, patience=5)
    train_data = x_train.reshape(x_train.shape[0], data.shape[1], data.shape[2])
    test_data = x_test.reshape(x_test.shape[0], data.shape[1], data.shape[2])
    history = vae.fit(train_data, train_data, epochs=epochs, verbose=verbose)

    encoder_train = vae.encoder.predict(train_data)
    encoder_test = vae.encoder.predict(test_data)

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

    print("epochs:" + str(epochs))
    print("latent dims:" + str(latent_dim))
    print("beta: " + str(coef))
    encoder.save("encoder")
    decode.save("decode")
    vae.save("vae")

def cost(individual):
    # TODO once trained ---------------------------------
    kpisf = df.iloc[i, 1]  # get KpISF value
    kpcsf = df.iloc[i, 2]  # get KpCSF value
    kpt = df.iloc[i, 3]  # get KpT value
    psb = df.iloc[i, 4]  # get PSB value
    pst = df.iloc[i, 5]  # get PST value
    cl = df.iloc[i, 6]  # get CL value
    # TODO once trained --------------------------------

    species = "Rat"  # can change this to human and code will run with new physioligical parameters
    dose = df.iloc[0, 3]  # get drug dose, in umol
    admin_route = 1  # 1=Intravenous
    integral = integrate_pbpk(kpisf, kpcsf, kpt, psb, pst, cl, dose, species, admin_route)
    return integral

if __name__=="__main__":
    df = pd.read_excel('../../data/Individual_Params.xlsx')  # individual drug parameters
    for i in range(0, len(df["id"])):
        # parse parameters and information for each drug
        drugID = df.iloc[i, 0]  # get drug ID
        kpisf = df.iloc[i, 1]  # get KpISF value
        kpcsf = df.iloc[i, 2]  # get KpCSF value
        kpt = df.iloc[i, 3]  # get KpT value
        psb = df.iloc[i, 4]  # get PSB value
        pst = df.iloc[i, 5]  # get PST value
        cl = df.iloc[i, 6]  # get CL value
        species = "Rat"  # can change this to human and code will run with new physioligical parameters
        dose = df.iloc[0, 3]  # get drug dose, in umol
        admin_route = 1  # 1=Intravenous

        integral = integrate_pbpk(kpisf, kpcsf, kpt, psb, pst, cl, dose, species, admin_route)
        print(integral)