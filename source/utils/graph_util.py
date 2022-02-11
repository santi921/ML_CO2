from turtle import delay
from glob import glob
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import get_file
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Input, Dropout

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)  


from spektral.data import BatchLoader, Graph, Dataset, Loader, utils, DisjointLoader, MixedLoader, SingleLoader
from spektral.datasets import QM9

from spektral.utils import label_to_one_hot, load_sdf, load_csv
from spektral.models import GeneralGNN, GCN

from spektral.layers import AGNNConv, ECCConv, GlobalSumPool,DiffusionConv, GATConv, GINConv,\
     GeneralConv, GlobalAttentionPool, GCNConv,CrystalConv, MessagePassing, MinCutPool, GlobalAvgPool
from spektral.layers.convolutional import GCSConv, MessagePassing, GeneralConv, GATConv
from spektral.transforms.normalize_adj import NormalizeAdj
from spektral.layers.pooling import MinCutPool, TopKPool

from sklearn.preprocessing import StandardScaler, MinMaxScaler, scale
from sklearn.metrics import r2_score

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt 
from scipy import sparse as sp

from joblib import Parallel, delayed

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


def sp_matrix_to_sp_tensor(m):
    """
    Converts a Scipy sparse matrix to a SparseTensor.
    The indices of the output are reordered in the canonical row-major ordering, and
    duplicate entries are summed together (which is the default behaviour of Scipy).
    :param x: a Scipy sparse matrix.
    :return: a SparseTensor.
    """
    x = sp.csr_matrix(m)
    
    if len(x.shape) != 2:
        raise ValueError("x must have rank 2")
    row, col, values = sp.find(x)
    out = tf.SparseTensor(
        indices=np.array([row, col]).T, values=values, dense_shape=x.shape
    )
    return tf.sparse.reorder(out)
    
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



class gnn_v5(Model):
    def __init__(self, N):
        super().__init__()
        
        #self.mask = GraphMasking()
        self.conv1 = GCSConv(32, activation="relu")
        self.pool = MinCutPool(N / 2)
        self.conv2 = GCSConv(32, activation="relu")
        #self.global_pool = GlobalSumPool()
        self.dense1 = Dense(1)

    def call(self, inputs):

        x, a, _ = inputs
        #x = self.mask(x)
        x = self.conv1([x, a])
        x_pool, a_pool = self.pool([x, a])
        x_pool = self.conv2([x_pool, a_pool])
        #output = self.global_pool(x_pool)
        output = self.dense1(output)
        return output

class gnn_v2(Model):
    def __init__(self):
        super().__init__()
        #self.masking = GraphMasking()
        self.conv1 = ECCConv(32, activation="relu")
        self.conv2 = ECCConv(32, activation="relu")
        self.global_pool = GlobalSumPool()
        self.dense = Dense(1)

    def call(self, inputs):
        x, a, e = inputs
        #x = self.masking(x)
        x = self.conv1([x, a, e])
        x = self.conv2([x, a, e])
        output = self.global_pool(x)
        output = self.dense(output)
        return output

class gnn_v3(Model):
    def __init__(self, channels = 64, n_layers = 3):
        super().__init__()
        self.conv1 = ECCConv(channels, epsilon=0, mlp_hidden=[channels, channels])
        self.convs = []
        for _ in range(1, n_layers):
            self.convs.append(
                ECCConv(channels, epsilon=0, mlp_hidden=[channels, channels])
            )
        self.pool = GlobalSumPool()
        self.dense1 = Dense(channels, activation="relu")
        self.dropout = Dropout(0.5)
        self.dense2 = Dense(1, activation="linear")

    def call(self, inputs):
        x, a, e = inputs
        #a = sp_matrix_to_sp_tensor(a)
        
        x = self.conv1([x, a, e])
        for conv in self.convs:
            x = conv([x, a, e])
        x = self.pool(x)
        x = self.dense1(x)
        x = self.dropout(x)
        return self.dense2(x)


class gnn_v4(Model):
    def __init__(self):
        super().__init__()
        #self.masking = GraphMasking()
        self.conv1 = GCSConv(32, activation="relu")
        self.conv2 = GCSConv(32, activation="relu")
        self.global_pool = GlobalSumPool()
        self.dense1 = Dense(24)
        self.dense2 = Dense(1)

    def call(self, inputs):
        x, a, e = inputs
        #x = self.masking(x)
        x = self.conv1([x, a])
        x = self.conv2([x, a])
        output = self.global_pool(x)
        output = self.dense1(output)
        output = self.dense2(output)

        return output



class gnn_v1(Model):

    def __init__(self):
        super().__init__()
        self.conv1 = GCSConv(32, activation="relu")
        self.pool1 = TopKPool(ratio=0.5)
        self.conv2 = GCSConv(32, activation="relu")
        self.pool2 = TopKPool(ratio=0.5)
        self.conv3 = GCSConv(32, activation="relu")
        self.global_pool = GlobalAvgPool()
        self.dense = Dense(1, activation="linear")

    def call(self, inputs):
        x, a, i = inputs
        x = self.conv1([x, a])
        x1, a1, i1 = self.pool1([x, a, i])
        x1 = self.conv2([x1, a1])
        x2, a2, i2 = self.pool1([x1, a1, i1])
        x2 = self.conv3([x2, a2])
        output = self.global_pool([x2, i2])
        output = self.dense(output)

        return output