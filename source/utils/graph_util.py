from turtle import delay
from glob import glob
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import get_file
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Input

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)  


from spektral.data import BatchLoader, Graph, Dataset, Loader, utils, DisjointLoader, MixedLoader, SingleLoader
from spektral.utils import label_to_one_hot, load_sdf, load_csv
#from spektral.utils.io import _parse_header,_parse_bonds_block,_parse_data_fields
#parse_counts_line, parse_atoms_block, parse_properties
from spektral.datasets import QM9
from spektral.data import Dataset, Graph
from spektral.utils import label_to_one_hot, sparse
from spektral.layers import AGNNConv, ECCConv, GlobalSumPool,DiffusionConv, GATConv, GeneralConv, GlobalAttentionPool, GCNConv,CrystalConv, MessagePassing, MinCutPool, GraphMasking
from spektral.models import GeneralGNN, GCN


from sklearn.preprocessing import StandardScaler, MinMaxScaler, scale
from sklearn.metrics import r2_score

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt 
 
from joblib import Parallel, delayed

#from rdkit.Chem import PandasTools, SDMolSupplier, Descriptors
from rdkit import Chem, DataStructs

import numpy as np
import pandas as pd
import selfies as sf
import seaborn as sns

from utils.sklearn_util import *
from utils.selfies_util import *

ATOM_TYPES = [1, 5, 6, 7, 8, 9, 11, 13, 14, 15, 16, 17, 35]
BOND_TYPES = [1, 2, 3]
HEADER_SIZE = 3
NUM_TO_SYMBOL = {
    1: "H",
    2: "He",
    3: "Li",
    4: "Be",
    5: "B",
    6: "C",
    7: "N",
    8: "O",
    9: "F",
    10: "Ne",
    11: "Na",
    12: "Mg",
    13: "Al",
    14: "Si",
    15: "P",
    16: "S",
    17: "Cl",
    18: "Ar",
    19: "K",
    20: "Ca",
    21: "Sc",
    22: "Ti",
    23: "V",
    24: "Cr",
    25: "Mn",
    26: "Fe",
    27: "Co",
    28: "Ni",
    29: "Cu",
    30: "Zn",
    31: "Ga",
    32: "Ge",
    33: "As",
    34: "Se",
    35: "Br",
    36: "Kr",
    37: "Rb",
    38: "Sr",
    39: "Y",
    40: "Zr",
    41: "Nb",
    42: "Mo",
    43: "Tc",
    44: "Ru",
    45: "Rh",
    46: "Pd",
    47: "Ag",
    48: "Cd",
    49: "In",
    50: "Sn",
    51: "Sb",
    52: "Te",
    53: "I",
    54: "Xe",
    55: "Cs",
    56: "Ba",
    57: "La",
    58: "Ce",
    59: "Pr",
    60: "Nd",
    61: "Pm",
    62: "Sm",
    63: "Eu",
    64: "Gd",
    65: "Tb",
    66: "Dy",
    67: "Ho",
    68: "Er",
    69: "Tm",
    70: "Yb",
    71: "Lu",
    72: "Hf",
    73: "Ta",
    74: "W",
    75: "Re",
    76: "Os",
    77: "Ir",
    78: "Pt",
    79: "Au",
    80: "Hg",
    81: "Tl",
    82: "Pb",
    83: "Bi",
    84: "Po",
    85: "At",
    86: "Rn",
    87: "Fr",
    88: "Ra",
    89: "Ac",
    90: "Th",
    91: "Pa",
    92: "U",
    93: "Np",
    94: "Pu",
    95: "Am",
    96: "Cm",
    97: "Bk",
    98: "Cf",
    99: "Es",
    100: "Fm",
    101: "Md",
    102: "No",
    103: "Lr",
    104: "Rf",
    105: "Db",
    106: "Sg",
    107: "Bh",
    108: "Hs",
    109: "Mt",
    110: "Ds",
    111: "Rg",
    112: "Cn",
    113: "Nh",
    114: "Fl",
    115: "Mc",
    116: "Lv",
    117: "Ts",
    118: "Og",
}
SYMBOL_TO_NUM = {v: k for k, v in NUM_TO_SYMBOL.items()}


def atom_to_feature(atom):

        
    atomic_num = label_to_one_hot(atom["atomic_num"], ATOM_TYPES)
    coords = atom["coords"]
    charge = atom["charge"]
    iso = atom["iso"]
    return np.concatenate((atomic_num, coords, [charge, iso]), -1)


def mol_to_adj(mol):
    
    row, col, edge_features = [], [], []
    for bond in mol["bonds"]:
        start, end = bond["start_atom"], bond["end_atom"]
        row += [start, end]
        col += [end, start]
        edge_features += [bond["type"]] * 2

    a, e = sparse.edge_index_to_matrix(
        edge_index=np.array((row, col)).T,
        edge_weight=np.ones_like(row),
        edge_features=label_to_one_hot(edge_features, BOND_TYPES),
    )
    return a, e


def read_mol(mol):
    x = np.array([atom_to_feature(atom) for atom in mol["atoms"]])
    a, e = mol_to_adj(mol)
    return x, a, e


def parse_sdf(sdf):
    #print(sdf)
    sdf_out = {}
    sdf = sdf.split("\n")
    sdf_out["name"], sdf_out["details"], sdf_out["comment"] = _parse_header(sdf)
    sdf_out["n_atoms"], sdf_out["n_bonds"] = _parse_counts_line(sdf)
    sdf_out["atoms"] = _parse_atoms_block(sdf, sdf_out["n_atoms"])
    sdf_out["bonds"] = _parse_bonds_block(sdf, sdf_out["n_atoms"], sdf_out["n_bonds"])
    sdf_out["properties"] = _parse_properties(
        sdf, sdf_out["n_atoms"], sdf_out["n_bonds"]
    )
    sdf_out["data"] = _parse_data_fields(sdf)
    return sdf_out


def _get_atomic_num(symbol):
    return SYMBOL_TO_NUM[symbol.lower().capitalize()]


def _parse_header(sdf):
    try:
        return sdf[0].strip(), sdf[1].strip(), sdf[2].strip()
    except IndexError:
        print(sdf)


def _parse_counts_line(sdf):
    # 12 fields
    # First 11 are 3 characters long
    # Last one is 6 characters long
    # First two give the number of atoms and bonds

    values = sdf[HEADER_SIZE]
    n_atoms = int(values[:3])
    n_bonds = int(values[3:6])

    return n_atoms, n_bonds


def _parse_atoms_block(sdf, n_atoms):
    # The first three fields, 10 characters long each, describe the atom's
    # position in the X, Y, and Z dimensions.
    # After that there is a space, and three characters for an atomic symbol.
    # After the symbol, there are two characters for the mass difference from
    # the monoisotope.
    # Next you have three characters for the charge.
    # There are ten more fields with three characters each, but these are all
    # rarely used.

    start = HEADER_SIZE + 1  # Add 1 for counts line
    stop = start + n_atoms
    values = sdf[start:stop]

    atoms = []
    for i, v in enumerate(values):
        coords = np.array([float(v[pos : pos + 10]) for pos in range(0, 30, 10)])
        atomic_num = _get_atomic_num(v[31:34].strip())
        iso = int(v[34:36])
        charge = int(v[36:39])
        info = np.array([int(v[pos : pos + 3]) for pos in range(39, len(v), 3)])
        atoms.append(
            {
                "index": i,
                "coords": coords,
                "atomic_num": atomic_num,
                "iso": iso,
                "charge": charge,
                "info": info,
            }
        )
    return atoms


def _parse_bonds_block(sdf, n_atoms, n_bonds):
    # The first two fields are the indexes of the atoms included in this bond
    # (starting from 1). The third field defines the type of bond, and the
    # fourth the stereoscopy of the bond.
    # There are a further three fields, with 3 characters each, but these are
    # rarely used and can be left blank.

    start = HEADER_SIZE + n_atoms + 1  # Add 1 for counts line
    stop = start + n_bonds
    values = sdf[start:stop]

    bonds = []
    for v in values:
        start_atom = int(v[:3]) - 1
        end_atom = int(v[3:6]) - 1
        type_ = int(v[6:9])
        stereo = int(v[9:12])
        info = np.array([int(v[pos : pos + 3]) for pos in range(12, len(v), 3)])
        bonds.append(
            {
                "start_atom": start_atom,
                "end_atom": end_atom,
                "type": type_,
                "stereo": stereo,
                "info": info,
            }
        )
    return bonds


def _parse_properties(sdf, n_atoms, n_bonds):
    # TODO This just returns a list of properties.
    # See https://docs.chemaxon.com/display/docs/MDL+MOLfiles%2C+RGfiles%2C+SDfiles%2C+Rxnfiles%2C+RDfiles+formats
    # for documentation.

    start = HEADER_SIZE + n_atoms + n_bonds + 1  # Add 1 for counts line
    stop = sdf.index("M  END")

    return sdf[start:stop]


def _parse_data_fields(sdf):
    # TODO This just returns a list of data fields.

    start = sdf.index("M  END") + 1

    return sdf[start:] if start < len(sdf) else []


def parse_sdf_file(sdf_file, amount=None):
    data = sdf_file.read().split("$$$$\n")
    if data[-1] == "":
        data = data[:-1]
    if amount is not None:
        data = data[:amount]
    output = [parse_sdf(sdf) for sdf in data]  # Parallel execution doesn't help
    return output


def load_sdf(filename, amount=None):
    """
    Load an .sdf file and return a list of molecules in the internal SDF format.
    :param filename: target SDF file
    :param amount: only load the first `amount` molecules from the file
    :return: a list of molecules in the internal SDF format (see documentation).
    """
    print("Reading SDF")
    with open(filename) as f:
        return parse_sdf_file(f, amount=amount)


def parse_sdf_file(sdf_file, amount=None):
    data = sdf_file.read().split("$$$$\n")
    if data[-1] == "":
        data = data[:-1]
    if amount is not None:
        data = data[:amount]
    output = [parse_sdf(sdf) for sdf in data]  # Parallel execution doesn't help
    return output


def load_sdf(filename, amount=None):
    """
    Load an .sdf file and return a list of molecules in the internal SDF format.
    :param filename: target SDF file
    :param amount: only load the first `amount` molecules from the file
    :return: a list of molecules in the internal SDF format (see documentation).
    """
    #print("Reading SDF")
    with open(filename) as f:
        return parse_sdf_file(f, amount=amount)

    
class dataset(Dataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def read(self):
        names, ret, homo, homo1, diff = sdf()
        #print(ret[0]) 
        mean = np.mean(diff)
        std = np.std(diff)
        diff_scale = (diff - mean) / std
        mean = np.mean(homo)
        std = np.std(homo)
        homo_scale = (homo - mean) / std
        #homo_scale = homo # change back

        data_sdf = [parse_sdf(i) for i in ret]
        data = Parallel(n_jobs=1)(delayed(read_mol)(mol) for mol in tqdm(data_sdf, ncols=80))
        x_list, a_list, e_list = list(zip(*data))
        dataset = [Graph(x=x, a=a, e=e, y = y) for x, a, e, y 
                   in zip(x_list, a_list, e_list, homo_scale)]

        return dataset


class dataset_benzo(Dataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def read(self):
        ret = []
        data_graph = []
        df = pd.read_hdf("../data/benzo/compiled.h5")
        sdf_full = glob("../data/benzo/*/input1.sdf")
        
        homo = df["homo"].tolist()
        homo1 = df["homo1"].tolist()
        diff = df['diff'].tolist()
        smiles = df["smiles"].tolist()
        smiles = [Chem.MolToSmiles(Chem.MolFromSmiles(i), canonical=True) for i in smiles]
        
        mean = np.mean(np.array(homo))
        std = np.std(np.array(homo))
        homo_scale = (np.array(homo) - mean) / std
        #homo_scale = np.array(homo)

        with open('../data/benzo/save_smiles.txt') as f: 
            smiles_full_line = f.readlines()

        smiles_files = [i.split(":")[1] for i in smiles_full_line]
        smiles_files = [Chem.MolToSmiles(Chem.MolFromSmiles(i), canonical=True) for i in smiles_files]
        
        for ind in range(len(smiles)):    
            if(smiles[ind] in smiles_files):
                ret.append(pybel.readstring("smi",smiles[ind]).write("sdf"))
        
        data = [parse_sdf(i) for i in ret]

        for mol in tqdm(data, ncols=80):
            try:
                data_graph.append(read_mol(mol))
            except:
                print("failed molecule")
                
                pass
        #data = Parallel(n_jobs=-1)(delayed(read_mol)(mol) for mol in tqdm(data, ncols=80))
        
        x_list, a_list, e_list = list(zip(*data_graph))

        dataset = [Graph(x=x, a=a, e=e, y = y) for x, a, e, y 
                   in zip(x_list, a_list, e_list, homo_scale)]
        
        print(mean)
        return dataset


def partition_dataset(dataset):
    learning_rate = 1e-3  # Learning rate
    epochs = 50  # Number of training epochs
    batch_size = 100 # Batch size
    
    # Train/test split
    idxs = np.random.permutation(len(dataset))
    split = int(0.85 * len(dataset))
    idx_tr, idx_te = np.split(idxs, [split])
    idx_tr = [int(i) for i in idx_tr]
    idx_te = [int(i) for i in idx_te]
    dataset_train = dataset[idx_tr]  
    dataset_test = dataset[idx_te] 
    
    #steps_per_epoch = len(dataset) /  batch_size
    loader = BatchLoader(dataset, epochs = epochs, batch_size = batch_size)

    #steps_per_epoch = len(dataset_train) /  batch_size
    loader_train = BatchLoader(dataset_train, epochs = epochs, batch_size = batch_size)

    #steps_per_epoch = len(dataset_test) /  batch_size
    loader_test = BatchLoader(dataset_test, batch_size = batch_size)

    return loader_train, loader_test, loader


def gnn_model_v1(dataset, loader_train):

    ################################################################################
    # PARAMETERS
    ################################################################################
    learning_rate = 1e-3  # Learning rate
    epochs = 50  # Number of training epochs
    batch_size = 1 # Batch size
    
    # input 
    F = dataset.n_node_features  # Dimension of node features
    S = dataset.n_edge_features  # Dimension of edge features
    n_out = dataset.n_labels     # Dimension of the target

    X_in = Input(shape=(None, F))
    A_in = Input(shape=(None, None))
    E_in = Input(shape=(None, None, S))

    X_1 = ECCConv(64, activation="relu")([X_in, A_in, E_in])
    X_2 = ECCConv(32, activation='relu')([X_1, A_in, E_in])
    output = GlobalSumPool()(X_2)    
    model = Model(inputs=[X_in, A_in, E_in], outputs=output)
    #------------------------------------------------------
    optimizer = Adam(lr=learning_rate)
    model.compile(optimizer=optimizer, loss="mse")
    model.summary()
    es = EarlyStopping(monitor='loss', mode='min', patience = 5)
    model.fit(loader_train.load(),  steps_per_epoch=loader_train.steps_per_epoch, 
          epochs = epochs, callbacks = [es], batch_size=batch_size)

    return model 


def gnn_model_v2(dataset, loader_train):

    ################################################################################
    # PARAMETERS
    ################################################################################
    learning_rate = 1e-3  # Learning rate
    epochs = 50  # Number of training epochs
    batch_size = 1 # Batch size
    
    # input 
    N = max(data.n_nodes for data in dataset)

    F = dataset.n_node_features  # Dimension of node features
    S = dataset.n_edge_features  # Dimension of edge features
    n_out = dataset.n_labels     # Dimension of the target

    X_in = Input(shape=(None, F))
    A_in = Input(shape=(None, None))
    
    X_1 = GraphMasking()([X_in])
    X_2 = GCSConv(32, activation="relu")([X_1, A_in])
    X_3, A_2 = MinCutPool(N // 2)
    X_4 = GCSConv(32, activation="relu")([X_3, A_2])
    output = GlobalSumPool()(X_4)
    output = Dense(n_out, activation='linear')(output)
    model = Model(inputs=[X_in, A_in], outputs=output)
    
    #------------------------------------------------------
    optimizer = Adam(lr=learning_rate)
    model.compile(optimizer=optimizer, loss="mse")
    model.summary()
    es = EarlyStopping(monitor='loss', mode='min', patience = 5)
    model.fit(loader_train.load(),  steps_per_epoch=loader_train.steps_per_epoch, 
          epochs = epochs, callbacks = [es], batch_size=batch_size)


    return model 


def gnn_model_v3(dataset, loader_train):

    ################################################################################
    # PARAMETERS
    ################################################################################
    learning_rate = 1e-3  # Learning rate
    epochs = 50  # Number of training epochs
    batch_size = 1 # Batch size
    
    # input 
    n_out = dataset.n_labels     # Dimension of the target

    model = GeneralGNN(n_out, activation="linear")
    #------------------------------------------------------
    optimizer = Adam(lr=learning_rate)
    model.compile(optimizer=optimizer, loss="mse")
    model.summary()

    es = EarlyStopping(monitor='loss', mode='min', patience = 5)

    model.fit(loader_train.load(),  steps_per_epoch=loader_train.steps_per_epoch, 
          epochs = epochs, callbacks = [es], batch_size=batch_size)

    return model 


def gnn_model_v4(dataset, loader_train):

    ################################################################################
    # PARAMETERS
    ################################################################################
    learning_rate = 1e-3  # Learning rate
    epochs = 50  # Number of training epochs
    batch_size = 1 # Batch size
    
    # input 
    F = dataset.n_node_features  # Dimension of node features
    n_out = dataset.n_labels     # Dimension of the target

    #------------------------------------------------------

    X_in = Input(shape=(None, F))
    A_in = Input(shape=(None, None))
    graph_conv_1 = GraphConv(32,
                        activation='elu',
                        kernel_regularizer=l2(l2_reg),
                        use_bias=True)([X_in, A_in])
    graph_conv_2 = GraphConv(32,
                        activation='elu',
                        kernel_regularizer=l2(l2_reg),
                        use_bias=True)([graph_conv_1, A_in])
    flatten = Flatten()(graph_conv_2)
    fc = Dense(512, activation='relu')(flatten)
    output = Dense(n_out, activation='linear')(fc)
    model = Model(inputs=[X_in, A_in], outputs=output)
    
    #------------------------------------------------------
    optimizer = Adam(lr=learning_rate)
    model.compile(optimizer=optimizer, loss="mse")
    model.summary()
    es = EarlyStopping(monitor='loss', mode='min', patience = 5)
    model.fit(loader_train.load(),  steps_per_epoch=loader_train.steps_per_epoch, 
          epochs = epochs, callbacks = [es], batch_size=batch_size)

    return model 



def gnn_model_v5(dataset, loader_train):

    ################################################################################
    # PARAMETERS
    ################################################################################
    learning_rate = 1e-4  # Learning rate
    epochs = 50  # Number of training epochs
    batch_size = 32 # Batch size
    l2_reg = 5e-4  

    # input 
    F = dataset.n_node_features  # Dimension of node features
    S = dataset.n_edge_features  # Dimension of edge features
    n_out = dataset.n_labels     # Dimension of the target
    
    #------------------------------------------------------
    X_in = Input(shape=(None, F))
    A_in = Input(shape=(None, None))
    gc1 = GraphAttention(32, activation='relu', kernel_regularizer=l2(l2_reg))([X_in, A_in]) 
    gc2 = GraphAttention(64, activation='relu', kernel_regularizer=l2(l2_reg))([gc1, A_in]) 
    pool = GlobalAttentionPool(128)(gc2) 
    output = Dense(n_out, activation='linear')(pool) 
    
    # Build model 
    model = Model(inputs=[X_in, A_in], outputs=output)
    
    #------------------------------------------------------
    optimizer = Adam(lr=learning_rate)
    model.compile(optimizer=optimizer, loss="mse")
    model.summary()
    es = EarlyStopping(monitor='loss', mode='min', patience = 5)
    model.fit(loader_train.load(),  steps_per_epoch=loader_train.steps_per_epoch, 
          epochs = epochs, callbacks = [es], batch_size=batch_size)

    return model 

