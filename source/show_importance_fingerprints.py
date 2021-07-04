import sys
import os
import base64
import time
import pybel
import pandas as pd
import numpy as np
from itertools import islice
import matplotlib.pyplot as plt
import xgboost as xgb
from PIL import Image
from io import BytesIO
from IPython.display import HTML

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor

from rdkit.Chem import AllChem, DataStructs, SDMolSupplier, Draw, RDConfig, rdBase
from rdkit.Chem.Draw import IPythonConsole

pd.set_option('display.max_colwidth', -1)


def depictBit(bitId,mol,molSize=(450,200)):
    info={}
    fp = AllChem.GetMorganFingerprintAsBitVect(mol,radius=2,nBits=2048,bitInfo=info)
    aid,rad = info[bitId][0]
    return getSubstructDepiction(mol,aid,rad,molSize=molSize)

def get_thumbnail(path):
    path = "\\\\?\\"+path # This "\\\\?\\" is used to prevent problems with long Windows paths
    i = Image.open(path)    
    return i

def image_base64(im):
    if isinstance(im, str):
        im = get_thumbnail(im)
    with BytesIO() as buffer:
        im.save(buffer, 'jpeg')
        return base64.b64encode(buffer.getvalue()).decode()

def image_formatter(im):
    return f'<img src="data:image/jpeg;base64,{image_base64(im)}">'

def View(df):
    css = """<style>
    table { border-collapse: collapse; border: 3px solid #eee; }
    table tr th:first-child { background-color: #eeeeee; color: #333; font-weight: bold }
    table thead th { background-color: #eee; color: #000; }
    tr, th, td { border: 1px solid #ccc; border-width: 1px 0 0 1px; border-collapse: collapse;
    padding: 3px; font-family: monospace; font-size: 10px }</style>
    """
    s  = '<script type="text/Javascript">'
    s += 'var win = window.open("", "Title", "toolbar=no, location=no, directories=no, status=no, menubar=no, scrollbars=yes, resizable=yes, width=780, height=200, top="+(screen.height-400)+", left="+(screen.width-840));'
    s += 'win.document.body.innerHTML = \'' + (df.to_html(formatters={'Image': image_formatter}, escape=False) + css).replace("\n",'\\') + '\';'
    s += '</script>'

    return(HTML(s+css))

def take(n, iterable):
    "Return first n items of the iterable as a list"
    return list(islice(iterable, n))

def xgboost(x_train, x_test, y_train, y_test, scale, dict=None):

    params = {
        "colsample_bytree": 0.5,
        "learning_rate": 0.1,
        "max_depth": 10, "gamma": 0.00,
        "lambda": 0.0,
        "alpha": 0,
        "eta": 0.01,
        "n_estimators": 2000}

    reg = xgb.XGBRegressor(**params, objective="reg:squarederror", tree_method="gpu_hist")


    t1 = time.time()
    #print(y_train)
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
    score = str(r2_score(reg.predict(x_train), y_train))
    print("r2 score:   " + str(score))
    score_mae = mean_absolute_error(reg.predict(x_test), y_test)
    print("scaled MAE")
    print(scale * score_mae)
    return reg

def morgan_bits(bit_length=256, dir="../data/sdf/DB3/", bit=True):
    from utils.helpers import merge_dir_and_data
    morgan = []
    morgan_bit = []
    names = []
    homo = []
    homo1 = []
    diff = []
    bitInfo_arr = []
    ret_arr = []
    mol_arr = []
    count = 0 
    mols = []
    dir_fl_names, list_to_sort = merge_dir_and_data(dir = dir)
    print("files to process: " + str(len(dir_fl_names)))
    
    #---------------------------------------------------------------------------
    for tmp, item in enumerate(dir_fl_names):
        try:
            suppl = SDMolSupplier(dir + item)[0]

            bitInfo = {}
            fp = AllChem.GetMorganFingerprintAsBitVect(suppl, 2, bitInfo=bitInfo,nBits=int(bit_length))
            arr = np.zeros((1,))
            DataStructs.ConvertToNumpyArray(fp, arr)
            fp_bit = AllChem.GetMorganFingerprintAsBitVect(suppl, int(2), nBits=int(bit_length))

            morgan_bit.append(fp_bit)
            names.append(item)
            homo_temp = float(list_to_sort[tmp].split(":")[1])
            homo1_temp = float(list_to_sort[tmp].split(":")[2])
            homo.append(homo_temp)
            homo1.append(homo1_temp)
            diff.append(homo_temp - homo1_temp)
            ret_arr.append(arr)
            morgan.append(fp)
            bitInfo_arr.append(bitInfo)
            mols.append(suppl)
            count += 1 
        except:
            pass
        
    morgan = np.array(morgan)
    print("successfully processed " + str(count) + " out of " + str(len(mols)) + " molecules")
    return mols, names, morgan, ret_arr, bitInfo_arr, homo, homo1, diff

molArr, names, morganArr, retArr, bitInfo_arr, homo, homo1, diff\
= morgan_bits(dir = "../data/sdf/DB3/",bit_length = 1024)
print("input begun with processing dataframe")


x = morganArr
y = homo1
scale = np.max(y) - np.min(y)

#y = np.array(np.array(df["HOMO"])) # selected target
indices = range(len(x))
try:
    x_train, x_test, y_train, y_test, indices_train, indices_test = train_test_split(x, y,indices, test_size=0.2, random_state=42)
except:
    x = list(x)
    x_train, x_test, y_train, y_test, indices_train, indices_test = train_test_split(x, y,indices, test_size=0.2, random_state=42)

reg = xgboost(x_train, x_test, y_train, y_test, scale)


fimportance = reg.feature_importances_
fimportance_dict = dict(zip(range(1024), fimportance))
sorteddata = sorted(fimportance_dict.items(), key=lambda x: -x[1])
top10feat = [x[0] for x in sorteddata][:10]
top15feat = [x[0] for x in sorteddata][:15]
top25feat = [x[0] for x in sorteddata][:25]
top50feat = [x[0] for x in sorteddata][:50]


print("Table of Feature Importance and Weight:")
for i in take(15, sorteddata):
    print(i[0], i[1])
    
testidx = np.argsort(y_test)
slice_conv = tuple(slice(x) for x in testidx)
testmols = [molArr[i] for i in testidx]

#output for single molecule
tpls = []
#testmols = testmols[0:10]
test_probe = 1

for i in range(len(testmols)):
    bitInfo={}
    fp = AllChem.GetMorganFingerprintAsBitVect(testmols[i], 2, bitInfo=bitInfo)
    arr = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(fp, arr)
    onbit = [bit for bit in bitInfo.keys()]
    importantonbits = list(set(onbit) & set(top15feat))
    
    #append to this array 
    if (importantonbits != []):
        tpls.append([(testmols[test_probe], x, bitInfo) for x in importantonbits])
    
img = []
rows = []
for frag in tpls: 
    try:
        fp = frag[0][1]
        img = Draw.DrawMorganBit(frag[0][0], frag[0][1], frag[0][2])
        rows.append([fp, img])
        df = pd.DataFrame(rows, columns=("FP", "Image"))
    except:
        pass
print("Dimension of Bit Array")
print(np.shape(df))
html = df.sort_values("FP").to_html(formatters={'Image': image_formatter}, escape=False)
html_out = open("reshomo1.html", "w")
html_out.write(html)
html_out.close()
#View(df)

