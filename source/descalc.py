import argparse
import math
import os

import numpy as np
import pandas as pd


def write_des(des, dir_temp):
    # check if this folder has other folders to traverse, works up to 1 layer deep

    if (des == "aval" or des == "morg" or des == "layer" or des == "rdkit"):
        from helpers import rdk, aval, layer, morgan
        dir = "../data/sdf/" + dir_temp + "/"

    else:
        dir = "../data/xyz/" + dir_temp + "/"

    if (des == "aval"):
        print("...........aval started..........")
        name, mat, homo, homo1, diff = layer(dir)

    elif (des == "morg"):
        print("...........morgan started..........")
        name, mat, homo, homo1, diff = morgan(256, dir)

    elif (des == "layer"):
        print("...........layers started..........")
        name, mat, homo, homo1, diff = layer(dir)

    elif (des == "vae"):
        from vae_util import vae
        print("...........vae started..........")
        name, mat, homo, homo1, diff  = vae(dir)

    elif (des == "self"):
        from selfies_util import selfies
        print("...........selfies started..........")
        name, mat, homo, homo1, diff = selfies(dir)

    elif (des == "auto"):
        from molsimplify_util import full_autocorr
        print("...........autocorrelation started..........")
        name, mat, homo, homo1, diff = full_autocorr(dir)

    #requires a metal in the compound for this desc
    elif (des == "delta"):
        from molsimplify_util import metal_deltametrics
        print("...........deltametrics started..........")
        name, mat, homo, homo1, diff = metal_deltametrics(dir)

    elif (des == "persist"):
        from Persist_util import persistent
        print("...........persistent images started..........")
        name, mat, homo, homo1, diff= persistent(dir)

    else:
        from helpers import rdk
        print("...........rdk started..........")
        name, mat, homo, homo1, diff = rdk(dir)

    if (np.shape(mat)[0] > 70000 and (des == "persist")):
        size = 25000
        chunks = math.ceil(np.shape(mat)[0] / size)

        for i in range(chunks):
            if (i == 0):

                temp_array = np.array(mat[0:size * (i + 1)]).astype("float32")
                temp_dict = {"name": name[0:size * (i + 1)], "mat": temp_array}
                df = pd.DataFrame.from_dict(temp_dict, orient="index")
                df = df.transpose()

            elif (i == chunks - 1):
                temp_array = np.array(mat[size * i:-1]).astype("float32")
                temp_dict = {"name": name[size * i:-1], "mat": temp_array}
                df = pd.DataFrame.from_dict(temp_dict, orient="index")
                df = df.transpose()

            else:
                temp_array = np.array(mat[size * i:size * (i + 1)]).astype("float32")
                temp_dict = {"name": name[size * i:size * (i + 1)], "mat": temp_array}
                df = pd.DataFrame.from_dict(temp_dict, orient="index")
                df = df.transpose()

            filename = "desc_calc_" + dir_temp + "_" + des + "_" + str(i)
            try:
                # print("suppressed")
                df.to_pickle("../data/desc/" + filename + ".pkl")
            except:
                pass

            try:
                # print("supressed")
                df.to_hdf("../data/desc/" + filename + ".h5", key="df", mode='a')
            except:
                pass

    else:
        if (des != "self"):
            mat = np.array(mat).astype("float32")
        mat = list(mat)
        temp_dict = {"name": name, "mat": mat, "HOMO": homo, "HOMO-1": homo1, "diff": diff}
        df = pd.DataFrame.from_dict(temp_dict, orient="index")
        df = df.transpose()
        print(df.head())
        filename = "desc_calc_" + dir_temp + "_" + des
        df.to_pickle("../data/desc/" + filename + ".pkl")
        df.to_hdf("../data/desc/" + filename + ".h5", key="df", mode='a')
        # df_reload = pd.read_pickle("../data/desc/" + filename + ".pkl")
        # df_reload2 = pd.read_hdf("../data/desc/" + filename + ".h5")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='select descriptor, and directory of files')
    parser.add_argument("--des", action='store', dest="desc", default="rdkit", help="select descriptor to convert to")
    parser.add_argument("--dir", action="store", dest="dir", default="DB", help="select directory")

    results = parser.parse_args()
    des = results.desc
    print("parser parsed")
    dir_temp = results.dir
    print("pulled director: " + dir_temp)

    if(des == "all"):
        #first env
        print("......rdk started.....")
        write_des("rdkit", dir_temp)
        print("......aval started.....")
        write_des("aval", dir_temp)
        print("......morgan started.....")
        write_des("morg", dir_temp)
        print("......layer started.....")
        write_des("layer", dir_temp)
        print("......persistent images started.....")
        write_des("persist", dir_temp)

        # print("......autocorrelation started.....")
        # write_des("auto", dir_temp)
        # print("......SELFIES started.....")
        # write_des("self", dir_temp)
        # honestly vae kinda sucks
        # os.system("conda activate py35")
        # os.system("export KERAS_BACKEND=tensorflow")
        # write_des("vae", dir_temp)

    else:
        write_des(des, dir_temp)


