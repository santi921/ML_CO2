# gen_files
import os
import random
from tqdm import tqdm
import numpy as np
import stat
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Descriptors import NumValenceElectrons, NumRadicalElectrons

import threading

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

master_string = "C1(C)=C(C)C(=O)C(C)=C(C)C1=O"
master_list = ["C1(", ")=C(", ")C(=O)C(", ")=C(", ")C1=O"]

thiol = 'S'
thio_me = 'SC'
sulfoxide = 'S(=O)C'
sulfone = 'S(=O)(=O)C'
sulfonate = 'S(=O)(=O)O'
sulfonate_me = 'S(=O)(=O)OC'
phenyl_sulfulnate = 'S(=O)(=O)c1ccccc1'
ng = [thiol, thio_me, sulfoxide, sulfone, sulfonate, sulfonate_me, phenyl_sulfulnate]
pg = ['C#N', 'CC(=O)O', 'F', 'N', 'CN', 'CC(O)(O)O', 'C(F)(F)F', 'CC(N)=O', 'O', 'CNC', 'COC=O', 'Oc1ccccc1',
      '[N+](=O)[O-]', 'CO', 'Cl', 'C', 'CC', 'CC(C)C', 'Br', 'Nc1ccccc1', 'Cc1ccccc1', 'c1ccccc1']
dono_list = ['C#N', 'CC(=O)O', 'N', 'CC(O)(O)O', 'CC(N)=O', 'O', 'CNC', 'COC=O', 'Oc1ccccc1', 'CO', 'Nc1ccccc1',
             'Cc1ccccc1', 'C', 'CC', 'CC(C)C', 'c1ccccc1']


class ThreadWithResult(threading.Thread):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs={}, *, daemon=None):
        def function():
            self.result = gen_folder_files(*args, **kwargs)
        super().__init__(group=group, target=function, name=name, daemon=daemon)



def gen_variant_smiles_v1():
    ng_single = [i+"_z"+"_z"+"_z" for i in ng]
    ng_doubles = []
    ng_triple = []
    ng_quad = []

    for i in ng:
        for j in ng: 
            ng_doubles.append(i + "_" + j + "_z_" + "z")
            ng_doubles.append(i + "_z_" + j + "_z")

    for i in ng:
        for j in ng: 
            for k in ng:
                ng_triple.append(i + "_" + j + "_" + k + "_z")

    for i in ng:
        for j in ng: 
            for k in ng:
                for l in ng:
                    ng_quad.append(i + "_" + j + "_" + k + "_" + l)

    # generate smiles
    raw_smiles = [list_to_smiles(i) for i in ng_single]
    [raw_smiles.append(list_to_smiles(i)) for i in ng_doubles]
    [raw_smiles.append(list_to_smiles(i)) for i in ng_triple]
    [raw_smiles.append(list_to_smiles(i)) for i in ng_quad]

    bi_sub_smiles = set(sample_inter(5000, 1, 2))
    tri_sub_smiles = sample_inter(8000, 1, 1)
    [tri_sub_smiles.append(i) for i in sample_inter(10000, 1, 2)]
    tri_sub_smiles = set(tri_sub_smiles)
    quad_sub_smiles = sample_inter(5000, 1, 0)
    [quad_sub_smiles.append(i) for i in sample_inter(10000,2,0)]
    [quad_sub_smiles.append(i) for i in sample_inter(10000,3,0)]
    quad_sub_smiles = set(quad_sub_smiles)

    # merge sets
    [raw_smiles.append(list_to_smiles(i)) for i in bi_sub_smiles]
    [raw_smiles.append(list_to_smiles(i)) for i in tri_sub_smiles]
    [raw_smiles.append(list_to_smiles(i)) for i in quad_sub_smiles]


    canon_smiles = []
    track_charge = []

    fail = 0 
    for i in tqdm(raw_smiles):
        mol_temp = Chem.MolFromSmiles(i)
        try:
            canon_smiles.append(Chem.MolToSmiles(mol_temp, canonical=True))

        except:
            fail += 1
            pass
    print("fail rate was: [ " + str(fail) + " / " + str(len(raw_smiles)) + " ]")

    final_smiles = list(set(canon_smiles))    
    print(len(final_smiles))
    return final_smiles


def sample_inter(samples, n_new_group = 1, n_empty = 0, benzo = True, anthra = False, napth = False):
    ret_list = []
    out_list = []

    for i in range(samples):
        ng_arr = []
        pg_arr = []
        if (n_empty > 0):
            empty_arr = ["z" for i in range(n_empty)]
        else:
            empty_arr = []

        if (benzo == True):
            for i in range(n_new_group):
                ng_arr.append(random.choice(ng))
            for i in range(4 - n_new_group - n_empty):
                pg_arr.append(random.choice(pg))

        if (anthra == True):
            list_a = []
            list_b = []
            list_a.append(random.choice(ng + pg))
            list_a.append(random.choice(ng + pg))
            list_b.append(random.choice(dono_list))
            list_b.append(random.choice(dono_list))

        if (napth == True):
            for i in range(4 - n_empty):
                ng_arr.append(random.choice(dono_list))

        if (napth == True or benzo == True):
            [out_list.append(i) for i in ng_arr]
            [out_list.append(i) for i in pg_arr]
            [out_list.append(i) for i in empty_arr]
            random.shuffle(out_list)

        if (anthra == True):
            out_list = list_a + list_b
            empty_list = []
            if(n_empty == 1):
                empty_list = list(np.random.choice([0, 1, 2, 3], size=n_empty, replace=False))
            if(n_empty>1):
                empty_list = np.random.choice([0, 1, 2, 3], size=n_empty, replace=False)

            [out_list.append("z") for i in empty_list]

        o1 = out_list[0]
        m1 = out_list[1]
        m2 = out_list[2]
        o2 = out_list[3]
        str_final = o1 + "_" + m1 + "_" + m2 + "_" + o2
        ret_list.append(str_final)

    return ret_list


def list_to_smiles(list_combin, benzo=True, anthra=False, napth=False):
    benzo_list = ["C1(", ")=C(", ")C(=O)C(", ")=C(", ")C1=O"]
    anthraquinone_list = ['c1cc(', ')c2c(c(', ')1)C(=O)c3c(', ')ccc(', ')c3C2=O']
    naphtoquinone_list = ['O=C1C(', ')=C(', ')C(=O)C2=C(', ')C=CC(', ')=C12']

    if (benzo == True):
        master_list = benzo_list
    if (anthra ==True):
        master_list = anthraquinone_list
    if (napth == True):
        master_list = naphtoquinone_list

    list_values = list_combin.split("_")
    ring_sub = []
    shift = 3
    for ind, i in enumerate(list_values):
        out_res = np.array([int(char.isdigit()) for char in i])
        digit = np.max(out_res)
        if (digit > 0):
            shift += 1
            list_values[ind] = list_values[ind].replace('1', str(shift), 2)
    # removes para to account for no functionalization
    if (list_values[0] == 'z'):
        master_list[0] = master_list[0][0:-1]
        master_list[1] = master_list[1][1:]
    if (list_values[1] == 'z'):
        master_list[1] = master_list[1][0:-1]
        master_list[2] = master_list[2][1:]
    if (list_values[2] == 'z'):
        master_list[2] = master_list[2][0:-1]
        master_list[3] = master_list[3][1:]
    if (list_values[3] == 'z'):
        master_list[-2] = master_list[-2][0:-1]
        master_list[-1] = master_list[-1][1:]

    shift = 0  # inserts functionalization into list
    if (list_values[0] != 'z'):
        master_list.insert(1, list_values[0])
        shift += 1
    if (list_values[1] != 'z'):
        master_list.insert(2 + shift, list_values[1])
        shift += 1
    if (list_values[2] != 'z'):
        master_list.insert(3 + shift, list_values[2])
        shift += 1
    if (list_values[3] != 'z'):
        master_list.insert(4 + shift, list_values[3])
    return ''.join(master_list)


def gen_variant_smiles(benzo=True, anthra=False, napth=False):
    raw_smiles_benzo = []
    raw_smiles_anthra = []
    raw_smiles_napth = []

    if(benzo == True):
        # -------------------------------- benzo list --------------------------------

        ng_single = [i + "_z" + "_z" + "_z" for i in ng+pg]
        ng_doubles = []
        ng_triple = []
        ng_quad = []

        for i in ng:
            for j in ng:
                ng_doubles.append(i + "_" + j + "_z_" + "z")
                ng_doubles.append(i + "_z_" + j + "_z")

        for i in ng:
            for j in ng:
                for k in ng:
                    ng_triple.append(i + "_" + j + "_" + k + "_z")

        for i in ng:
            for j in ng:
                for k in ng:
                    for l in ng:
                        ng_quad.append(i + "_" + j + "_" + k + "_" + l)
        # generate smiles
        raw_smiles_benzo = [list_to_smiles(i) for i in ng_single]
        [raw_smiles_benzo.append(list_to_smiles(i)) for i in ng_doubles]
        [raw_smiles_benzo.append(list_to_smiles(i)) for i in ng_triple]
        [raw_smiles_benzo.append(list_to_smiles(i)) for i in ng_quad]



        bi_sub_smiles = set(sample_inter(2000, 1, 2))
        tri_sub_smiles = sample_inter(5000, 1, 1)
        [tri_sub_smiles.append(i) for i in sample_inter(5000, 1, 2)]
        tri_sub_smiles = set(tri_sub_smiles)
        quad_sub_smiles = sample_inter(1000, 1, 0)
        [quad_sub_smiles.append(i) for i in sample_inter(5000, 2, 0)]
        [quad_sub_smiles.append(i) for i in sample_inter(5000, 3, 0)]
        quad_sub_smiles = set(quad_sub_smiles)
        # merge sets
        [raw_smiles_benzo.append(list_to_smiles(i)) for i in bi_sub_smiles]
        [raw_smiles_benzo.append(list_to_smiles(i)) for i in tri_sub_smiles]
        [raw_smiles_benzo.append(list_to_smiles(i)) for i in quad_sub_smiles]

    # -------------------------------- anthra list --------------------------------
    if (anthra == True):
        anthra_meta_double = []
        anthra_double_ortho_opposite = []
        anthra_double_ortho_same = []
        tri = []
        tetra = []
        anthra_single = [i + "_z" + "_z" + "_z" for i in dono_list]

        for i in dono_list:
            for j in dono_list:
                anthra_meta_double.append(i + "_" + j + "_z_" + "z")
                anthra_double_ortho_opposite.append(i + "_z_" + j + "_z")
                anthra_double_ortho_same.append(i + "_z_z_" + j)

        # triple -------> set of triply substituted
        [tri.append(i) for i in sample_inter(5000, n_new_group=0, n_empty=1, benzo=False, anthra=True, napth=False)]

        # tetra --------> set of tetra substituted
        [tetra.append(i) for i in sample_inter(5000, n_new_group=0, n_empty=0, benzo=False, anthra=True, napth=False)]

        [raw_smiles_anthra.append(list_to_smiles(i, anthra = True, benzo= False)) for i in set(anthra_single)]
        [raw_smiles_anthra.append(list_to_smiles(i, anthra = True, benzo= False)) for i in set(anthra_meta_double)]
        [raw_smiles_anthra.append(list_to_smiles(i, anthra = True, benzo= False)) for i in set(anthra_double_ortho_opposite)]
        [raw_smiles_anthra.append(list_to_smiles(i, anthra = True, benzo= False)) for i in set(anthra_double_ortho_same)]
        [raw_smiles_anthra.append(list_to_smiles(i, anthra = True, benzo= False)) for i in set(tri)]
        [raw_smiles_anthra.append(list_to_smiles(i, anthra = True, benzo= False)) for i in set(tetra)]



    # -------------------------------- naphto list --------------------------------
    if(napth == True):
        tri = []
        tetra = []

        napthto_benzo_single = ["z_z_" + i + "_z" for i in dono_list]
        napthto_quinone_single = [i + "_z_z_z" for i in ng + pg]

        # double -------> set of doubly substituted
        napthto_benzo_double = []
        for i in dono_list:
            for j in dono_list:
                napthto_benzo_double.append("z_z" + "_" + i + "_" + j)

        napthto_quinone_double = []
        for i in ng + pg:
            for j in ng + pg:
                napthto_quinone_double.append(i + "_" + j + "_z_z")

        napthto_double_meta = []
        napthto_double_ortho = []
        for i in ng + pg:
            for j in dono_list:
                napthto_double_meta.append(i + "_z_" + j + "_z")
                napthto_double_ortho.append(i + "_z_z_" + j)

        # triple -------> set of triply substituted
        [tri.append(i) for i in sample_inter(8000, n_new_group=0, n_empty=1, benzo=False, anthra=False, napth=True)]
        # tetra --------> set of tetra substituted
        [tetra.append(i) for i in sample_inter(8000, n_new_group=0, n_empty=0, benzo=False, anthra=False, napth=True)]


        [raw_smiles_napth.append(list_to_smiles(i, anthra=True, napth=False)) for i in set(napthto_benzo_single)]
        [raw_smiles_napth.append(list_to_smiles(i, anthra=True, napth=False)) for i in set(napthto_quinone_single)]
        [raw_smiles_napth.append(list_to_smiles(i, anthra=True, napth=False)) for i in set(napthto_benzo_double)]
        [raw_smiles_napth.append(list_to_smiles(i, anthra=True, napth=False)) for i in set(napthto_double_meta)]
        [raw_smiles_napth.append(list_to_smiles(i, anthra = True, napth = False)) for i in set(napthto_double_ortho)]
        [raw_smiles_napth.append(list_to_smiles(i, anthra = True, napth = False)) for i in set(tri)]
        [raw_smiles_napth.append(list_to_smiles(i, anthra = True, napth = False)) for i in set(tetra)]

    # -------------------------------- -------------------------------- -------------------------------- --------------------------------


    raw_smiles = raw_smiles_benzo
    [raw_smiles.append(i) for i in raw_smiles_napth]
    [raw_smiles.append(i) for i in raw_smiles_anthra]

    canon_smiles = []
    fail = 0
    for i in tqdm(raw_smiles):
        mol_temp = Chem.MolFromSmiles(i)
        canon_smiles.append(Chem.MolToSmiles(mol_temp, canonical=True))
        try:
            canon_smiles.append(Chem.MolToSmiles(mol_temp, canonical=True))

        except:
            fail += 1
            pass
    print("fail rate was: [ " + str(fail) + " / " + str(len(raw_smiles)) + " ]")

    final_smiles = list(set(canon_smiles))
    print(len(final_smiles))
    return final_smiles


def gen_folder_files(dir_save='./sulphur', benzo=True, napth=True, anthra=True):
    if (not os.path.isdir(dir_save)):
        os.mkdir(dir_save)

    final_smiles = gen_variant_smiles(benzo = benzo, napth = napth, anthra = anthra)
    # for use by controller.py
    textfile = open(dir_save + "/save_smiles.txt", "w")
    for ind, element in enumerate(final_smiles):
        textfile.write(str(ind) + ":" + element + "\n")
    textfile.close()

    failed_translation = 0

    for ind, i in enumerate(tqdm(final_smiles)):
        mol_temp = Chem.MolFromSmiles(i)
        mol_temp = Chem.AddHs(mol_temp)
        try:
            AllChem.EmbedMolecule(mol_temp)
            AllChem.MMFFOptimizeMolecule(mol_temp)
            txt_xyz = Chem.MolToMolBlock(mol_temp)
            succ = True
        except:
            failed_translation += 1
            succ = False

        if (succ):
            charge = -2
            electron_count = 0
            electron_count = NumValenceElectrons(mol_temp)
            print(electron_count)
            if (electron_count % 2 == 1):
                multi = 2
            else:
                multi = 1
            file1 = dir_save + '/' + str(ind) + "/input1.xyz"
            file2 = dir_save + "/" + str(ind) + "/input2.in"
            file_exe_local = dir_save + "/" + str(ind) + "/launch1.sh"
            file_exe_hpc = dir_save + "/" + str(ind) + "/launch2.sh"
            file_smiles = dir_save + "/" + str(ind) + "/mol.smi"

            if (not os.path.isdir(dir_save + "/" + str(ind))):
                os.mkdir(dir_save + "/" + str(ind))

            if (not os.path.isdir(file1)):
                with open(file1, 'w') as f1:
                    num_atoms = int(txt_xyz.split("\n")[3].split()[0])
                    f1.write(str(num_atoms)+"\n\n")
                    for j in range(num_atoms):
                        line_xyz = txt_xyz.split("\n")[4 + j]
                        x, y, z, element = line_xyz.split()[0], line_xyz.split()[1], line_xyz.split()[2], \
                                           line_xyz.split()[3]
                        f1.write(element + "\t" + x + "\t" + y + "\t" + z + "\n")

                        if(element == 'r' or element == "R"):
                            print(element + "\t" + x + "\t" + y + "\t" + z + "\n")
                    f1.write('*\n')
                st = os.stat(file1)
                os.chmod(file1, st.st_mode | stat.S_IEXEC| stat.S_IREAD)


            if (not os.path.isdir(file2)):
                with open(file2, 'w') as f2:

                    f2.write("!RIJCOSX PBE0 DEF2-SVP def2/j D3BJ TIGHTSCF CPCM(DMF) Hcore \n")
                    f2.write("* xyzfile " + str(int(charge)) + " " + str(int(multi)) + " " + dir_save + "/" + str(ind) + "/input1.xyz\n")
               # st = os.stat(file1)
               # os.chmod(file1, st.st_mode | stat.S_IEXEC| stat.S_IREAD)

            if (not os.path.isdir(file_exe_local)):
                with open(file_exe_local, 'w') as f_exe:
                    f_exe.write("#!/bin/sh\n")
                    f_exe.write("conda activate orca\n")
                    f_exe.write("ulimit -s unlimited\n")
                    f_exe.write("export OMP_STACKSIZE=4\n")
                    f_exe.write("xtb " + " input1.xyz -T 4 -opt\n")
                    f_exe.write("crest" +  " xtbopt.xyz --niceprint -T 64 --noreftopo -gff\n")

                    #f_exe.write("xtb " + dir_save + "/" + str(ind) + "/input1.xyz -T 4 -opt\n")
                    #f_exe.write("crest" + dir_save + "/" + str(ind) +  "/xtbopt.xyz --niceprint -T 64 --noreftopo -gff\n")
                    #f_exe.write("./orca "  + dir_save + "/" + str(ind) + "/input1.in > " + dir_save + "/" + str(ind)  + "/out_temp\n")
                st = os.stat(file_exe_local)
                os.chmod(file_exe_local, st.st_mode | stat.S_IEXEC| stat.S_IREAD)

            if (not os.path.isdir(file_exe_hpc)):
                with open(file_exe_hpc, 'w') as f_exe:
                    f_exe.write("#!/bin/sh\n")
                    f_exe.write("./orca "  + dir_save + "/" + str(ind) + "crest_best.xyz > " + dir_save + "/" + str(ind)  + "/out\n")
                    f_exe.write("grep '2.0000' " + dir_save + "/" + str(ind)  + "/out | grep -Ev 's6|Mayer|Sum|Startup|Fraction' > " + dir_save + "/" + str(ind) + "/orbital_en\n")
                    f_exe.write("grep '1.0000' " + dir_save + "/" + str(ind)  + "/out | grep -Ev 's6|Mayer|Sum|Startup|Fraction' > " + dir_save + "/" + str(ind) + "/orbital_en")


                st = os.stat(file_exe_hpc)
                os.chmod(file_exe_hpc, st.st_mode | stat.S_IEXEC | stat.S_IREAD)
            if (not os.path.isdir(file_smiles)):
                with open(file_smiles, 'w') as f_sm:
                    f_sm.write(i)

        mol_list = [Chem.MolFromSmiles(i) for i in final_smiles[100:120]]
        img = Draw.MolsToGridImage(mol_list, molsPerRow=5, subImgSize=(400, 400))
        img.save('output.png')


#while(True):
t1 = ThreadWithResult(target=gen_folder_files, kwargs={"dir_save" : "./benzo", "benzo" : True, "napth" : False, "anthra" : False})
t2 = ThreadWithResult(target=gen_folder_files, kwargs={"dir_save" : './anthra', "benzo" : False, "napth" : True, "anthra" : False})
t3 = ThreadWithResult(target=gen_folder_files, kwargs={"dir_save" : './napth', "benzo" : False, "napth" : False, "anthra" : True})
#t1.start()
#t2.start()
t3.start()
#t1.join()
#t2.join()
t3.join()

