import sys
from Element_PI import VariancePersistv1
from utils.helpers import merge_dir_and_data


def persistent(dir="../data/xyz/", pixelsx=60, pixelsy=60, spread=0.28, Max=2.5):

    persist = []
    names = []
    homo = []
    homo1 = []
    diff = []

    dir_fl_names, list_to_sort = merge_dir_and_data(dir = dir)
    print(len(dir_fl_names))
    print(len(list_to_sort))

    #---------------------------------------------------------------------------
    for tmp, item in enumerate(dir_fl_names):
        temp = dir + str(item)
        try:

            temp_persist = VariancePersistv1(
                temp.format(temp + str(1)),
                pixelx=pixelsx, pixely=pixelsy,
                myspread=spread, myspecs={"maxBD": Max, "minBD": -.10}, showplot=False)
            if (item[0:-4] == list_to_sort[tmp].split(":")[0] ):
                persist.append(temp_persist)
                names.append(item)
                homo_temp = float(list_to_sort[tmp].split(":")[1])
                homo1_temp = float(list_to_sort[tmp].split(":")[2])
                homo.append(homo_temp)
                homo1.append(homo1_temp)
                diff.append(homo_temp - homo1_temp)
            else:
                try:
                    if (item[0:-4] == list_to_sort[tmp+1].split(":")[0]):
                        persist.append(temp_persist)
                        names.append(item)
                        homo_temp = float(list_to_sort[tmp+1].split(":")[1])
                        homo1_temp = float(list_to_sort[tmp+1].split(":")[2])
                        homo.append(homo_temp)
                        homo1.append(homo1_temp)
                        diff.append(homo_temp - homo1_temp)
                except:
                    print(list_to_sort[tmp].split(":")[0], item[0:-4])
                    pass
            sys.stdout.write("\r %s /" % tmp + str(len(dir_fl_names)))
            sys.stdout.flush()
        except:
            pass
    return names, persist, homo, homo1, diff
