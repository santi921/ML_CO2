import numpy as np 
import scipy as sp
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt


#NOTE there are some features that simply do not have the values 
#for the entire table
features = ["ortho","unsubstituted","halogenated","conjugation","sterics",\
"proton donating","heteroatoms","positive charge"]
out = ["REDOX Q1-/Q2-","pKa_biradical","lnK_biradical", "biradical_CO2 HOMO"]
out_extended = ["REDOX Q1-/Q2-","pKa_biradical","lnK_biradical", "biradical_CO2 HOMO",\
 "FreeE biradical","FreeE radical", "FreeE biradical+co2", "FreeE biradical+proton"\
 , "radical HOMO"]
df = pd.read_excel("quinones.xlsx")

#print(df.head)

del df["Unnamed: 24"]
del df["Unnamed: 25"]
del df["Unnamed: 26"]
del df["CO2 in DMF"]

#input vectors
x = df.loc[:, features].values

#output vector 
y_redox = df.loc[:, out[0]].values
y_pka = df.loc[:, out[1]].values
y_lnk = df.loc[:, out[2]].values
y_homo = df.loc[:, out[3]].values
y_tot = df.loc[:,out].values
y_extended = df.loc[:,out_extended].values
# one-hot 
x.astype("float64")
y_extended.astype("float64")
full_table = np.concatenate((x,y_extended),axis=1).astype("float64")


from sklearn import neighbors

n_samples = 157
outliers_fraction = 0.1 
clusters_separation = [0,1,2]
clf = neighbors.LocalOutlierFactor(novelty=True)
clf.fit(full_table)

# Use this to compare with other descriptors we gen 
# -erate later. This table gives what may be outliers
# We could compare with chemical distance via other 
# metrics

print(clf.negative_outlier_factor_ > -1.5)


# regress via multivariate linear,
# bayesian, gd, huberregessor(applies linear loss to outliers)
# knn regressor 
# NN