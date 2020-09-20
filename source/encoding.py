import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA, TruncatedSVD
import seaborn as sn
import matplotlib.pyplot as plt

if __name__ == "__main__":

    dir_temp = "DB3"
    #des = "morg"
    des = "rdkit"
    #des = "layer"
    #des = "aval"
    scale_x_tf = True

    print("pulling directory: " + dir_temp + " with descriptor: " + des)


    if (dir_temp == "DB3" or dir_temp == "DB2"):
        try:
            print("done processing dataframe")
            str = "../data/desc/" + dir_temp + "/desc_calc_" + dir_temp + "_" + des + ".pkl"
            df = pd.read_pickle(str)
            pkl = 1
        except:
            print("done processing dataframe")
            str = "../data/desc/" + dir_temp + "/desc_calc_" + dir_temp + "_" + des + ".h5"
            df = pd.read_hdf(str)
            pkl = 0
    HOMO = df["HOMO"].to_numpy()
    HOMO_1 = df["HOMO-1"].to_numpy()
    diff = df["diff"].to_numpy()

    if (des == "vae"):
        temp = df["mat"].tolist()
        mat = list([i.flatten() for i in temp])

    elif (des == "auto"):
        temp = df["mat"].tolist()
        mat = list([i.flatten() for i in temp])
    else:
        mat = df["mat"].to_numpy()

    scale_x_tf = True
    if (scale_x_tf == True):
        try:
            mat = preprocessing.scale(np.array(mat))

        except:
            mat = list(mat)
            mat = preprocessing.scale(np.array(mat))

    x_data = df["mat"].to_numpy()
    x_data = np.ndarray.tolist(x_data)


    pca = PCA(n_components = 6)
    pca.fit(x_data)
    print(pca.explained_variance_ratio_.sum())

    svd = TruncatedSVD(n_components=6, n_iter = 10)
    svd.fit(x_data)
    print(svd.explained_variance_ratio_.sum())

    print(type(x_data))
    temp = pd.DataFrame(x_data)

    corr_matrix = temp.corr()
    sn.heatmap(corr_matrix)
    plt.show()

    scale_diff = (np.max(diff) - np.min(diff))
    diff = (diff - np.min(diff)) / scale_diff