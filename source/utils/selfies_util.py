import sys
import pybel
import os
import numpy as np
import pandas as pd
import selfies as sf
from rdkit import Chem
from rdkit import DataStructs
from selfies import encoder
from utils.helpers import merge_dir_and_data

# worked in python 3
def selfies_to_hot(selfie, largest_selfie_len, alphabet):
    """Go from a single selfies string to a one-hot encoding.
    """

    symbol_to_int = dict((c, i) for i, c in enumerate(alphabet))

    # pad with [nop]
    selfie += '[nop]' * (largest_selfie_len - sf.len_selfies(selfie))

    # integer encode
    symbol_list = sf.split_selfies(selfie)
    integer_encoded = [symbol_to_int[symbol] for symbol in symbol_list]

    # one hot-encode the integer encoded selfie
    onehot_encoded = list()
    for index in integer_encoded:
        letter = [0] * len(alphabet)
        letter[index] = 1
        onehot_encoded.append(letter)

    return integer_encoded, np.array(onehot_encoded)

def multiple_selfies_to_hot(selfies_list, largest_molecule_len, alphabet):
    """Convert a list of selfies strings to a one-hot encoding
    """

    hot_list = []
    for s in selfies_list:
        _, onehot_encoded = selfies_to_hot(s, largest_molecule_len, alphabet)
        hot_list.append(onehot_encoded)
    return np.array(hot_list)

def smile_to_hot(smile, largest_smile_len, alphabet):
    """Go from a single smile string to a one-hot encoding.
    """

    char_to_int = dict((c, i) for i, c in enumerate(alphabet))

    # pad with ' '
    smile += ' ' * (largest_smile_len - len(smile))

    # integer encode input smile
    integer_encoded = [char_to_int[char] for char in smile]

    # one hot-encode input smile
    onehot_encoded = list()
    for value in integer_encoded:
        letter = [0 for _ in range(len(alphabet))]
        letter[value] = 1
        onehot_encoded.append(letter)
    return integer_encoded, np.array(onehot_encoded)

def multiple_smile_to_hot(smiles_list, largest_molecule_len, alphabet):
    """Convert a list of smile strings to a one-hot encoding
    Returned shape (num_smiles x len_of_largest_smile x len_smile_encoding)
    """

    hot_list = []
    for s in smiles_list:
        
        _, onehot_encoded = smile_to_hot(s, largest_molecule_len, alphabet)
        hot_list.append(onehot_encoded)
    return np.array(hot_list)

def get_selfie_and_smiles_encodings_for_dataset(smiles_list):
    """
    Returns encoding, alphabet and length of largest molecule in SMILES and
    SELFIES, given a file containing SMILES molecules.
    input:
        csv file with molecules. Column's name must be 'smiles'.
    output:
        - selfies encoding
        - selfies alphabet
        - longest selfies string
        - smiles encoding (equivalent to file content)
        - smiles alphabet (character based)
        - longest smiles string
    """

    # df = pd.read_csv(file_path)
    # smiles_list = np.asanyarray(df.smiles)
    smiles_alphabet = list(set(''.join(smiles_list)))
    smiles_alphabet.append(' ')  # for padding
    largest_smiles_len = len(max(smiles_list, key=len))

    print('--> Translating SMILES to SELFIES...')
    
    
    selfies_list = list(map(sf.encoder, smiles_list))
         
    all_selfies_symbols = sf.get_alphabet_from_selfies(selfies_list)
    all_selfies_symbols.add('[nop]')
    selfies_alphabet = list(all_selfies_symbols)

    largest_selfies_len = max(sf.len_selfies(s) for s in selfies_list)

    print('Finished translating SMILES to SELFIES.')

    return selfies_list, selfies_alphabet, largest_selfies_len, \
           smiles_list, smiles_alphabet, largest_smiles_len

def compare_equality(x_test, autoencoded_selfies, dim, selfies_alphabet):
    '''
    method that computes the equality in encode-decode performance between a test
    dataset and an encode-decoded dataset
    '''

    test_size = len(x_test)
    count_good = 0
    for i, mol in enumerate(x_test):

        # single point - through vae
        one_hot = np.zeros((dim[0], dim[1]))
        one_hot_true = np.zeros((dim[0], dim[1]))

        for ind, row in enumerate(autoencoded_selfies[i].reshape(dim[0], dim[1])):
            lab_temp = np.argmax(row)
            one_hot[ind][lab_temp] = 1

        # single point - non vae
        for ind, row in enumerate(mol.reshape(dim[0], dim[1])):
            lab_temp = np.argmax(row)
            one_hot_true[ind][lab_temp] = 1

        self_test = sf.encoding_to_selfies(one_hot.tolist(), selfies_alphabet, "one_hot")
        self_true = sf.encoding_to_selfies(one_hot_true.tolist(), selfies_alphabet, "one_hot")

        canonical_smiles = Chem.CanonSmiles(sf.decoder(self_true))
        canonical_autoencoder_smiles = Chem.CanonSmiles(sf.decoder(self_test))
        if(i == 1):
            print("Autoencoded Smiles: " + canonical_autoencoder_smiles)
            print("True Smiles: " + canonical_smiles)

        if (canonical_autoencoder_smiles == canonical_smiles):
            count_good += 1
    print("Percent Reconstructed Molescules: " + str(count_good / test_size))

def tanimoto_dist(x_test, autoencoded_selfies, dim, selfies_alphabet):
    '''
    method that computes the equality in encode-decode performance between a test
    dataset and an encode-decoded dataset
    '''
        
    test_size = len(x_test)
    count_good = 0
    dist = []
    
    for i, mol in enumerate(x_test):
        
        # single point - through vae
        one_hot = np.zeros((dim[0], dim[1]))
        one_hot_true = np.zeros((dim[0], dim[1]))

        for ind, row in enumerate(autoencoded_selfies[i].reshape(dim[0], dim[1])):
            lab_temp = np.argmax(row)
            one_hot[ind][lab_temp] = 1

        # single point - non vae
        for ind, row in enumerate(mol.reshape(dim[0], dim[1])):
            lab_temp = np.argmax(row)
            one_hot_true[ind][lab_temp] = 1

        self_test = sf.encoding_to_selfies(one_hot.tolist(), selfies_alphabet, "one_hot")
        self_true = sf.encoding_to_selfies(one_hot_true.tolist(), selfies_alphabet, "one_hot")

        canonical_smiles = Chem.CanonSmiles(sf.decoder(self_test))
        canonical_autoencoder_smiles = Chem.CanonSmiles(sf.decoder(self_true))
        
        fps1 = Chem.RDKFingerprint(Chem.MolFromSmiles(canonical_smiles))
        fps2 = Chem.RDKFingerprint(Chem.MolFromSmiles(canonical_autoencoder_smiles))
        diff = DataStructs.FingerprintSimilarity(fps1, fps2)
        dist.append(float(diff))        
        
    return np.array(dist)

def get_dataset_stats(smiles_arr):
        """
        Returns encoding, alphabet and length of largest molecule in SMILES and
        SELFIES, given a file containing SMILES molecules.
        input:
            csv file with molecules. Column's name must be 'smiles'.
        output:
            - selfies encoding
            - selfies alphabet
            - longest selfies string
            - smiles encoding (equivalent to file content)
            - smiles alphabet (character based)
            - longest smiles string
        """
        df = pd.read_csv(file_path)
        smiles_list = np.asanyarray(df.smiles)
        smiles_alphabet = list(set(''.join(smiles_list)))
        smiles_alphabet.append(' ')  # for padding
        largest_smiles_len = len(max(smiles_list, key=len))
        print('--> Translating SMILES to SELFIES...')
        selfies_list = list(map(sf.encoder, smiles_list))
        all_selfies_symbols = sf.get_alphabet_from_selfies(selfies_list)
        all_selfies_symbols.add('[nop]')
        selfies_alphabet = list(all_selfies_symbols)
        largest_selfies_len = max(sf.len_selfies(s) for s in selfies_list)
        print('Finished translating SMILES to SELFIES.')
        return selfies_list, selfies_alphabet, largest_selfies_len, \
               smiles_list, smiles_alphabet, largest_smiles_len
        return alphabet, max_len

def selfies(dir="../data/xyz/DB3/"):
    ret = []
    homo = []
    homo1 = []
    diff = []
    names = []

    print("..........converting xyz to smiles.......")
    dir_fl_names, list_to_sort = merge_dir_and_data(dir=dir)
    smiles = []
    rm_ind = []

    for j, i in enumerate(dir_fl_names):
        try:
            mol = next(pybel.readfile("xyz", dir + i))
            smi = mol.write(format="smi")
            smiles.append(smi.split()[0].strip())
            sys.stdout.write("\r %s / " % j + str(len(dir_fl_names)))
            sys.stdout.flush()
        except:
            rm_ind.append(j)

    rm_ind.reverse()
    [dir_fl_names.pop(i) for i in rm_ind]
    [list_to_sort.pop(i) for i in rm_ind]
    print("\n\nsmiles length: " + str(len(smiles)) + "\n\n")
    #---------------------------------------------------------------------------
    for tmp, item in enumerate(smiles):
        try:
            selfies_temp = encoder(item)
            selfies_temp_shift = encoder(smiles[tmp+1])
            selfies_temp_shift_min = encoder(smiles[tmp-1])

            mol = next(pybel.readfile("xyz", dir + list_to_sort[tmp].split(":")[0] + ".xyz"))
            mol_shift = next(pybel.readfile("xyz", dir + list_to_sort[tmp+1].split(":")[0] + ".xyz"))
            mol_shift_min = next(pybel.readfile("xyz", dir + list_to_sort[tmp-1].split(":")[0] + ".xyz"))

            smi = mol.write(format="smi").split()[0].strip()
            smi_shift = mol_shift.write(format="smi").split()[0].strip()
            smi_shift_min = mol_shift_min.write(format="smi").split()[0].strip()

            #selfies_one_hot = selfies_to_hot(selfies_temp,largest_len, alpha)
            #selfies_one_hot_shift = selfies_to_hot(selfies_temp_shift,largest_len, alpha)
            #selfies_one_hot_shift_min = selfies_to_hot(selfies_one_hot_shift_min,largest_len, alpha)

            if (item == smi):
                ret.append(selfies_temp)
                names.append(item)
                homo_temp = float(list_to_sort[tmp].split(":")[1])
                homo1_temp = float(list_to_sort[tmp].split(":")[2])
                homo.append(homo_temp)
                homo1.append(homo1_temp)
                diff.append(homo_temp - homo1_temp)
            else:
                try:
                    if (item == smi_shift):
                        ret.append(selfies_temp_shift)
                        names.append(item)
                        homo_temp = float(list_to_sort[tmp+1].split(":")[1])
                        homo1_temp = float(list_to_sort[tmp+1].split(":")[2])
                        homo.append(homo_temp)
                        homo1.append(homo1_temp)
                        diff.append(homo_temp - homo1_temp)
                    else:
                        pass

                except:
                    try:
                        if ("item" == smi_shift_min):
                            ret.append(selfies_temp_shift_min)
                            names.append(item)
                            homo_temp = float(list_to_sort[tmp + 1].split(":")[1])
                            homo1_temp = float(list_to_sort[tmp + 1].split(":")[2])
                            homo.append(homo_temp)
                            homo1.append(homo1_temp)
                            diff.append(homo_temp - homo1_temp)
                    except:
                        print("failed to match")
                        pass
            sys.stdout.write("\r %s /" % tmp + str(len(dir_fl_names)))
            sys.stdout.flush()
        except:
            pass
    print(len(names))
    print(len(ret))
    print(len(homo))
    print(len(homo1))
    print(len(diff))

    ret = np.array(ret)
    return names, ret, homo, homo1, diff

def sdf(dir="../data/xyz/DB3/"):

    ret = []
    homo = []
    homo1 = []
    diff = []
    names = []

    print("..........converting xyz to smiles.......")
    dir_fl_names, list_to_sort = merge_dir_and_data(dir=dir)
    smiles = []
    rm_ind = []

    for j, i in enumerate(dir_fl_names):
        try:
            mol = next(pybel.readfile("xyz", dir + i))
            smi = mol.write(format="smi")
            smiles.append(smi.split()[0].strip())
            sys.stdout.write("\r %s / " % j + str(len(dir_fl_names)))
            sys.stdout.flush()
        except:
            rm_ind.append(j)

    rm_ind.reverse()
    [dir_fl_names.pop(i) for i in rm_ind]
    [list_to_sort.pop(i) for i in rm_ind]
    print("\n\nsmiles length: " + str(len(smiles)) + "\n\n")

    #---------------------------------------------------------------------------
    for tmp, item in enumerate(smiles):

        try:
            sdf_file = pybel.readstring("smi",item).write("sdf")
            sdf_file_shift = pybel.readstring("smi",smiles[tmp+1]).write("sdf")
            sdf_file_shift_min = pybel.readstring("smi",smiles[tmp-1]).write("sdf")

            mol = next(pybel.readfile("xyz", dir + list_to_sort[tmp].split(":")[0] + ".xyz"))
            mol_shift = next(pybel.readfile("xyz", dir + list_to_sort[tmp+1].split(":")[0] + ".xyz"))
            mol_shift_min = next(pybel.readfile("xyz", dir + list_to_sort[tmp-1].split(":")[0] + ".xyz"))

            smi = mol.write(format="smi").split()[0].strip()
            smi_shift = mol_shift.write(format="smi").split()[0].strip()
            smi_shift_min = mol_shift_min.write(format="smi").split()[0].strip()


            if (item == smi):

                ret.append(sdf_file)
                names.append(item)
                homo_temp = float(list_to_sort[tmp].split(":")[1])
                homo1_temp = float(list_to_sort[tmp].split(":")[2])
                homo.append(homo_temp)
                homo1.append(homo1_temp)
                diff.append(homo_temp - homo1_temp)
            else:
                try:
                    if (item == smi_shift):
                        ret.append(sdf_file_shift)
                        names.append(item)
                        homo_temp = float(list_to_sort[tmp+1].split(":")[1])
                        homo1_temp = float(list_to_sort[tmp+1].split(":")[2])
                        homo.append(homo_temp)
                        homo1.append(homo1_temp)
                        diff.append(homo_temp - homo1_temp)
                    else:
                        pass

                except:
                    try:
                        if ("item" == smi_shift_min):
                            ret.append(sdf_file_shift_min)
                            names.append(item)
                            homo_temp = float(list_to_sort[tmp + 1].split(":")[1])
                            homo1_temp = float(list_to_sort[tmp + 1].split(":")[2])
                            homo.append(homo_temp)
                            homo1.append(homo1_temp)
                            diff.append(homo_temp - homo1_temp)
                    except:
                        print("failed to match")
                        pass
            sys.stdout.write("\r %s /" % tmp + str(len(dir_fl_names)))
            sys.stdout.flush()
        except:
            pass

    print(len(names))
    print(len(ret))
    print(len(homo))
    print(len(homo1))
    print(len(diff))

    ret = np.array(ret)
    return names, ret, homo, homo1, diff

def smiles(dir="../data/xyz/DB3/", verbose = 1):

    dir_str = "ls " + str(dir) + " | sort -d "
    temp = os.popen(dir_str).read()
    temp = str(temp).split()
    ret_list = []
    names = []
    for j, i in enumerate(temp):
        try:
            mol = next(pybel.readfile("xyz", dir + i))
            smi = mol.write(format="smi")
            smi = Chem.CanonSmiles(smi)
            ret_list.append(smi.split()[0].strip())
            names.append(i)
            if(verbose == 1):
                        sys.stdout.write("\r %s / " % j + str(len(temp)))
                        sys.stdout.flush()
        except:
            try:

                f = open(dir + i, "r")
                smi = f.readlines()[0]
                if(len(smi > 5)):
                    ret_list.append(smi)
                    names.append(i)
                if(verbose == 1):
                        sys.stdout.write("\r %s / " % j + str(len(temp)))
                        sys.stdout.flush()
            except:
                pass
    # print(ret_list[0:4])
    return names, ret_list
