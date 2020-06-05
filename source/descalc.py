


import argparse
import pandas as pd
import os
import numpy as np

def write_des(des, dir_temp):

    if( des == "aval" or des == "morg" or des == "layer" or des == "rdkit"):
        from helpers import rdk, aval, layer, morgan
        dir = "../data/sdf/" + dir_temp + "/"

    else:
        dir = "../data/xyz/" + dir_temp + "/"


    if (des == "aval"):
        print("...........aval started..........")
        name, mat = aval(dir)

    elif (des == "morg"):
        print("...........morgan started..........")
        name, mat = morgan(256, dir)

    elif (des == "layer"):
        print("...........layers started..........")
        name, mat = layer(dir)

    elif (des == "vae"):
        from vae_util import vae
        print("...........vae started..........")
        name, mat = vae(dir)

    elif (des == "self"):
        from selfies_util import selfies
        print("...........selfies started..........")
        name, mat = selfies(dir)

    elif (des == "auto"):
        from molsimplify_util import full_autocorr
        print("...........autocorrelation started..........")
        name, mat = full_autocorr(dir)

    #requires a metal in the compound for this desc
    elif (des == "delta"):
        from molsimplify_util import metal_deltametrics
        print("...........deltametrics started..........")
        name, mat = metal_deltametrics(dir)

    elif (des == "persist"):
        from Persist_util import persistent
        print("...........persistent images started..........")
        name, mat = persistent(dir)

    else:
        from helpers import rdk
        print("...........rdk started..........")
        name, mat = rdk(dir)


    temp_dict = {"name": name, "mat": mat}
    df = pd.DataFrame.from_dict(temp_dict, orient="index")
    df = df.transpose()

    #benchmark these two and csv potentially

    filename = "desc_calc_" + dir_temp + "_" + des

    repo_dir = os.getcwd()[:-6]
    df.to_pickle("../data/desc/" + filename + ".pkl")
    df_reload = pd.read_pickle("../data/desc/" + filename + ".pkl")
    #print(df_reload.head())
    df.to_hdf("../data/desc/" + filename + ".h5", key = "df", mode = 'a')
    df_reload2 = pd.read_hdf("../data/desc/" + filename + ".h5")
    #print(df_reload2.head())
    

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='select descriptor, and directory of files')
    parser.add_argument("--des", action='store', dest="desc", default="rdkit", help="select descriptor to convert to")
    parser.add_argument("--dir", action="store", dest="dir",  default="DB",    help="select directory")

    results = parser.parse_args()
    des = results.desc
    print("parser parsed")
    dir_temp = results.dir
    print("pulled directory")

    if(des == "all"):
        #first env
        print("......rdk started.....")
        write_des("rdk", dir_temp)
        print("......aval started.....")
        write_des("aval", dir_temp)
        print("......morgan started.....")
        write_des("morg", dir_temp)
        print("......layer started.....")
        write_des("layer", dir_temp)
        print("......persistent images started.....")
        write_des("persist", dir_temp)
        print("......autocorrelation started.....")
        write_des("auto", dir_temp)
        print("......SELFIES started.....")
        write_des("self", dir_temp)

        # honestly vae kinda sucks
        #os.system("conda activate py35")
        #os.system("export KERAS_BACKEND=tensorflow")
        #write_des("vae", dir_temp)

    else:
        write_des(des, dir_temp)


