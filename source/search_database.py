from tqdm import tqdm
from utils.helpers import napht_check, quinone_check, anthra_check


napth_list = []
quinone_list = []
anthra_list = []

with open('../data/version.smi') as f: 
    lines = f.readlines()
smiles = [i.split()[0] for i in lines]

for i in tqdm(smiles): 
    try: 
        napth_tf = napht_check(i)
    except: 
        napth_tf = False
    try: 
        quinone_tf =  quinone_check(i)
    except: 
        quinone_tf = False
    try: 
        anthra_tf = anthra_check(i)
    except: 
        anthra_tf = False

    if napth_tf:
        print(i) 
        napth_list.append(i)
    if quinone_tf: 
        print(i)
        quinone_list.append(i)
    if anthra_tf: 
        print(i)
        anthra_list.append(i)

print("anthra count: " + str(len(anthra_list)))
print("quinone count: " + str(len(quinone_list)))
print("napth count: " + str(len(napth_list)))

with open('../data/commercial_anthra.smi', 'w') as f: 
    for element in anthra_list:
        f.write(element + "\n")
with open('../data/commercial_quinone.smi', 'w') as f: 
    for element in quinone_list:
        f.write(element + "\n")
with open('../data/commercial_napth.smi', 'w') as f: 
    for element in napth_list:
        f.write(element + "\n")
