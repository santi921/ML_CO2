import argparse

import pandas as pd
from sklearn_utils import gradient_boost_reg, \
    random_forest, sk_nn
from xgboost_util import xgboost


def process_input_DB2(dir="DB2", desc="rdkit"):
    try:
        str = "../data/desc/" + dir + "/desc_calc_DB2_" + desc + ".pkl"
        print(str)
        df = pd.read_pickle(str)
        pkl = 1

    except:
        str = "../data/desc/" + dir + "/desc_calc_DB2_" + desc + ".h5"
        df = pd.read_hdf(str)
        pkl = 0

    df["HOMO"] = ""
    df["HOMO-1"] = ""
    df["diff"] = ""
    print(df.head())
    list_to_sort = []
    with open("../data/benzoquinone_DB/DATA_copy") as fp:
        line = fp.readline()
        while line:
            list_to_sort.append(line[0:-2])
            line = fp.readline()
    list_to_sort.sort()

    for i in range(df.copy().shape[0]):
        for j in list_to_sort:

            # print(df["name"].iloc[i][:-4])
            # print(j.split(":")[0])
            if (desc == "auto"):
                if (df["name"].iloc[i].split("/")[-1][:-4] in j.split(";")[0]):
                    temp1 = float(j.split(":")[1])
                    temp2 = float(j.split(":")[2])
                    df["HOMO"].loc[i] = float(j.split(":")[1])
                    df["HOMO-1"].loc[i] = float(j.split(":")[2])
                    df["diff"].loc[i] = temp2 - temp1
                    print(temp2 - temp1)

            if (df["name"].iloc[i][:-4] in j.split(";")[0]):
                temp1 = float(j.split(":")[1])
                temp2 = float(j.split(":")[2])

                print(j)

                df["HOMO"].loc[i] = float(j.split(":")[1])
                df["HOMO-1"].loc[i] = float(j.split(":")[2])
                df["diff"].loc[i] = temp2 - temp1
    if (pkl == 0):
        df.to_hdf(str, key="df", mode='a')
    else:
        df.to_pickle(str)


# TODO : this
def process_input_DB3(dir="DB3", desc="rdkit"):
    try:
        str = "../data/desc/" + dir + "/desc_calc_DB3_" + desc + ".pkl"
        print(str)
        df = pd.read_pickle(str)
        pkl = 1

    except:
        str = "../data/desc/" + dir + "/desc_calc_DB3_" + desc + ".h5"
        df = pd.read_hdf(str)
        pkl = 0

    df["HOMO"] = ""
    df["HOMO-1"] = ""
    df["diff"] = ""

    print(df.head())

    list_to_sort = []
    with open("../data/DATA_DB3") as fp:
        line = fp.readline()
        while line:
            list_to_sort.append(line[0:-2])
            line = fp.readline()
    list_to_sort.sort()

    for i in range(df.copy().shape[0]):
        for j in list_to_sort:

            # print(df["name"].iloc[i][:-4])
            # print(j.split(":")[0])
            if (desc == "auto"):
                if (df["name"].iloc[i].split("/")[-1][:-4] in j.split(";")[0]):
                    temp1 = float(j.split(":")[1])
                    temp2 = float(j.split(":")[2])
                    df["HOMO"].loc[i] = float(j.split(":")[1])
                    df["HOMO-1"].loc[i] = float(j.split(":")[2])
                    df["diff"].loc[i] = temp2 - temp1
                    print(temp2 - temp1)

            if (df["name"].iloc[i][:-4] in j.split(";")[0]):
                temp1 = float(j.split(":")[1])
                temp2 = float(j.split(":")[2])

                print(j)

                df["HOMO"].loc[i] = float(j.split(":")[1])
                df["HOMO-1"].loc[i] = float(j.split(":")[2])
                df["diff"].loc[i] = temp2 - temp1
    if (pkl == 0):
        df.to_hdf(str, key="df", mode='a')
    else:
        df.to_pickle(str)


def process_input_ZZ(dir="ZZ", desc="vae"):
    # todo: process ZZ's
    return 0


def calc(x, y):
    print("in calc")
    random_forest(x, y)
    # sgd(mat, HOMO)
    gradient_boost_reg(x, y)
    xgboost(x, y)
    # random_forest(mat,HOMO)
    # svr(mat, diff)
    sk_nn(x, y)
    # bayesian(mat,diff)
    # kernel(mat,diff)
    # gaussian(mat,diff)
    # process_input_DB2(dir = "DB2",  desc="persist")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='select descriptor, and directory of files')
    parser.add_argument("--des", action='store', dest="desc", default="rdkit", help="select descriptor to convert to")
    parser.add_argument("--dir", action="store", dest="dir", default="DB", help="select directory")

    results = parser.parse_args()
    des = results.desc
    print("parser parsed")
    dir_temp = results.dir
    print("pulled directory: " + dir_temp)

    if (dir_temp == "DB2"):
        try:
            # process_input_DB2()
            print("done processing dataframe")
            str = "../data/desc/" + dir + "/desc_calc_" + dir + "_" + des + ".pkl"
            df = pd.read_pickle(str)
            pkl = 1
        except:
            # process_input_DB2()
            print("done processing dataframe")
            str = "../data/desc/" + dir + "/desc_calc_" + dir + "_" + des + ".h5"
            df = pd.read_hdf(str)
            pkl = 0

            print(df.head())
            print(str)
    if (dir_temp == "DB3"):
        try:
            process_input_DB3()
            print("done processing dataframe")
            str = "../data/desc/" + dir + "/desc_calc_" + dir + "_" + des + ".pkl"
            df = pd.read_pickle(str)
            pkl = 1
        except:
            process_input_DB3()
            print("done processing dataframe")
            str = "../data/desc/" + dir + "/desc_calc_" + dir + "_" + des + ".h5"
            df = pd.read_hdf(str)
            pkl = 0

            print(df.head())
            print(str)

    HOMO = df["HOMO"].to_numpy()
    HOMO_1 = df["HOMO-1"].to_numpy()
    diff = df["diff"].to_numpy()

    if (desc == "vae"):
        temp = df["mat"].tolist()
        mat = list([i.flatten() for i in temp])

    elif (desc == "auto"):
        temp = df["mat"].tolist()
        mat = list([i.flatten() for i in temp])
    else:
        mat = df["mat"].to_numpy()
    print("Using " + desc + " as the descriptor")

    calc(mat, desc="persist")
    calc(mat)
    calc(mat)
    calc(mat)
    calc(mat)
# morgan and layer were the best
# potentially try wider morgan or layer
'''

with open("../data/benzoquinone_DB/DATA_copy") as fp:
    line = fp.readline()
    while line:
        list_to_sort.append(line)
        line = fp.readline()

list_to_sort.sort()
print(list_to_sort[0:4])
#print(sorted[0:4])
'''
'''
        if(line.split(":")[0] == df["name"].iloc[index_search][:-4]):
            #print(line.split(":")[0])
            print("yeet")
            #index_search += 1
        if (index_search < 4):
            #print(df["mat"].iloc[index_search])
            print(df["name"].iloc[index_search][:-4])
            print(line.split(":")[0])
            index_search+=1
            #if (line.split(":")[0][0] != " " and line.split(":")[0][0] != "-"):

        line = fp.readline()

'''

# process_input_DB2(dir = "DB2", desc="aval")
# calc()

'''
df_reload = pd.read_pickle("../data/desc/DB/desc_calc_DB_aval.pkl")
df_y = pd.read_excel("../data/quinones_for_Santiago.xlsx")
x_aval = df_reload["mat"].values
y_lnk_biradical = df_y["lnK_biradical"].values
y_rad_HOMO = df_y["radical HOMO"].values
y_birad_HOMO = df_y["biradical HOMO"].values
y_CO2_HOMO = df_y["biradical_CO2 HOMO"].values
sgd(x_aval, y_lnk_biradical)
sgd(x_aval, y_rad_HOMO)
sgd(x_aval, y_birad_HOMO)
sgd(x_aval, y_CO2_HOMO)
'''