from Element_PI import VariancePersistv1
import os, glob
import pandas as pd
import numpy as np

valid_orb_files = glob.glob("./benzo/*/orbital_en")
valid_xyz_files = glob.glob("./benzo/*/*xyz")

valid_count = 0
pixelsx = 60
pixelsy = 60
spread = 0.28
Max = 2.5

homo_arr = []
homo1_arr = []
mat_arr = []
smi_arr = []

for i in valid_orb_files:
    valid = False
    ind_temp = 0
    with open(i) as f:
        temp_lines = f.readlines()
    # print(i.split("/")[2])
    if len(temp_lines) > 1:

        for ind, j in enumerate(temp_lines):
            if (j.split()[1] == "1.0000" or j.split()[1] == "2.0000"):
                valid = True
                ind_temp = ind
                with open(i.split("/")[0] + "/" + i.split("/")[1] + "/" + i.split("/")[2] + "/mol.smi") as g:
                    smi = g.readlines()

        if (valid):
            xyz_str = i.split("/")[0] + "/" + i.split("/")[1] + "/" + i.split("/")[2] + "/input1.xyz"
            try:
                temp_persist = VariancePersistv1(
                    xyz_str,
                    pixelx=pixelsx, pixely=pixelsy,
                    myspread=spread, myspecs={"maxBD": Max, "minBD": -.10}, showplot=False)
                smiles = smi[0]
                homo_1 = float(temp_lines[ind_temp - 1].split()[2])

                homo = float(temp_lines[ind_temp].split()[2])
                valid_count += 1

                smi_arr.append(smiles)
                homo_arr.append(homo)
                homo1_arr.append(homo_1)
                mat_arr.append(temp_persist)

            except:
                pass

diff_arr = homo1_arr - homo_arr
df = pd.DataFrame()
df["smiles"] = smi_arr
df['homo'] = homo_arr
df['homo1'] = homo1_arr
df['mat'] = mat_arr
df['diff'] = diff_arr
homo_arr = np.array(homo_arr)
homo1_arr = np.array(homo1_arr)
df.to_hdf('./benzo/compiled.h5', key='s')
