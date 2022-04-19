from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor

import sys
import os
import time
import xgboost as xgb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pybel
from rdkit.Chem import AllChem, DataStructs, SDMolSupplier, Draw
from rdkit.Chem.Draw import IPythonConsole


def morgan(dir, bit_length=256):
    morgan = []
    names = []
    ret_arr = []
    bitInfo_arr = []
    mol_arr = []
    dir = "../../data/sdf/master.sdf"
    temp_str = "ls " + dir
    temp = os.popen(temp_str).read()
    temp = str(temp).split()
    mols = [i for i in SDMolSupplier(dir)]

    for i, suppl in enumerate(mols):
        try:
            bitInfo = {}
            fp = AllChem.GetMorganFingerprintAsBitVect(
                suppl, 2, bitInfo=bitInfo, nBits=int(bit_length)
            )
            arr = np.zeros((1,))
            DataStructs.ConvertToNumpyArray(fp, arr)
            ret_arr.append(arr)
            morgan.append(fp)
            names.append(suppl.GetProp("NAME"))
            bitInfo_arr.append(bitInfo)

            sys.stdout.write("\r %s / " % i + str(len(mols)))
            sys.stdout.flush()
        except:
            pass

    morgan = np.array(morgan)
    print(
        "successfully processed " + str(i) + " out of " + str(len(temp)) + " molecules"
    )
    return names, morgan, ret_arr, bitInfo_arr, mols


def process_input_DB3(name, mat):

    mat = list(mat)
    temp_dict = {"name": name, "mat": mat}
    df = pd.DataFrame.from_dict(temp_dict, orient="index")
    df = df.transpose()
    list_to_sort = []

    with open("../data/DATA_DB3_transform") as fp:
        line = fp.readline()
        while line:
            list_to_sort.append(line[0:-2])
            line = fp.readline()

    list_to_sort.sort()
    print("done processing lines of data file")
    df["HOMO"] = 0
    df["HOMO-1"] = 0
    df["diff"] = 0
    track = 0
    for i in range(df.shape[0]):
        trip = 0
        for j in list_to_sort:
            # print(df["name"].iloc[i][0:-4])
            # print(j.split(":")[0])
            if df["name"].iloc[i][0:-4] in j.split(":")[0] and j[0:2] != "--":
                temp1 = float(j.split(":")[1])
                temp2 = float(j.split(":")[2])
                track += 1
                df["HOMO"].loc[i] = temp1
                df["HOMO-1"].loc[i] = temp2
                df["diff"].loc[i] = temp2 - temp1
                break

    print("done parsing energies")
    return df


def xgboost(x_train, x_test, y_train, y_test, scale, dict=None):

    params = {
        "colsample_bytree": 0.8,
        "learning_rate": 0.1,
        "max_depth": 10,
        "gamma": 0.00,
        "lambda": 0.0,
        "alpha": 0.2,
        "eta": 0.0,
        "n_estimators": 2500,
    }

    reg = xgb.XGBRegressor(
        **params, objective="reg:squarederror", tree_method="gpu_hist"
    )

    params = {
        "max_depth": 20,
        "n_estimators": 500,
        "bootstrap": True,
        "min_samples_leaf": 2,
        "n_jobs": 16,
        "verbose": False,
        "n_jobs": 4,
    }

    reg = RandomForestRegressor(**params)

    t1 = time.time()
    # non grid
    print(y_train)
    reg.fit(x_train, y_train)
    t2 = time.time()

    time_el = t2 - t1
    score = reg.score(x_test, y_test)
    print("xgboost score:               " + str(score) + " time: " + str(time_el))
    score = str(mean_squared_error(reg.predict(x_test), y_test))
    print("MSE score:   " + str(score))
    score = str(mean_absolute_error(reg.predict(x_test), y_test))
    print("MAE score:   " + str(score))
    score = str(r2_score(reg.predict(x_test), y_test))
    print("r2 score:   " + str(score))
    score_mae = mean_absolute_error(reg.predict(x_test), y_test)
    print("scaled MAE")
    print(scale * score_mae)
    return reg


names, morganArr, retArr, bitInfoArr, molArr = morgan(dir=dir, bit_length=1024)

print("input begun with processing dataframe")

# print(names)
df = process_input_DB3(names, morganArr)
# print(df["diff"])

mat = list(df["mat"])
mat = preprocessing.scale(np.array(mat))
scale = np.max(np.array(df["diff"])) - np.min(np.array(df["diff"]))

x = np.array(mat)
y = np.array(np.array(df["diff"]))
indices = range(len(x))
try:
    x_train, x_test, y_train, y_test, indices_train, indices_test = train_test_split(
        x, y, indices, test_size=0.2, random_state=42
    )
except:
    x = list(x)
    x_train, x_test, y_train, y_test, indices_train, indices_test = train_test_split(
        x, y, indices, test_size=0.2, random_state=42
    )

reg = xgboost(x_train, x_test, y_train, y_test, scale)

fimportance = reg.feature_importances_
fimportance_dict = dict(zip(range(1024), fimportance))
sorteddata = sorted(fimportance_dict.items(), key=lambda x: -x[1])
top10feat = [x[0] for x in sorteddata][:10]

plt.hist(sorteddata)
plt.show()
print(top10feat)


testidx = np.argsort(y_test)
print(testidx)
print(testidx.tolist())
slice_conv = tuple(slice(x) for x in testidx)

# print(x_test[testidx][0:4])
# print(reg.predict(x_test[testidx][0].reshape(1,-1)))
# print(reg.predict(x_test[testidx][0:4]))
# print(y_test[0:4])
# print(onbit)
# print(set(onbit))
# print(set(top50feat))
# print(set(onbit) & set(top50feat))

testmols = [molArr[i] for i in testidx]

testmols[3]

slice_conv = tuple(slice(x) for x in testidx)
testmols = molArr[slice(testidx[0])]

test_probe = 10

# print([i for i in testidx])
bitInfo = {}
fp = AllChem.GetMorganFingerprintAsBitVect(testmols[test_probe], 2, bitInfo=bitInfo)
arr = np.zeros((1,))
DataStructs.ConvertToNumpyArray(fp, arr)
onbit = [bit for bit in bitInfo.keys()]

importantonbits = list(set(onbit) & set(top10feat))
tpls = [(testmols[test_probe], x, bitInfo) for x in importantonbits]
Draw.DrawMorganBits(tpls)
