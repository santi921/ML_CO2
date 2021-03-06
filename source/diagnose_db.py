import os
import pandas as pd

def check_file(dir = "../data/desc/ZZ/", dir_orig = "../data/sdf/ZZ/"):


    ls_dir = "ls " + dir_orig
    temp = os.popen(ls_dir).read()
    temp = str(temp).split()
    molecules = len(temp)
    print("original number of molecules: " + str(molecules))

    ls_dir = "ls " + dir
    temp = os.popen(ls_dir).read()
    temp = str(temp).split()
    print("dbs to check " + str(temp))

    for db in temp:
        if(db[-1] == "l"):
            try:
                df_reload = pd.read_pickle("../data/desc/" + filename + ".pkl")
                print(db + " converted: " + str(df_reload.shape[0]))
            except:
                print(db + " isn't working")
        else:
            try:

                df_reload2 = pd.read_hdf(dir + db)
                print(db + " converted: " + str(df_reload2.shape[0]))
            except:
                print(db + " isn't working")

