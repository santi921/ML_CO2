import os
import sys

from Element_PI import VariancePersistv1


def persistent(dir="../data/xyz/", pixelsx=150, pixelsy=150, spread=0.28, Max=2.5):
    ls_dir = "ls " + str(dir) + " | sort -d"
    temp = os.popen(ls_dir).read()
    temp = str(temp).split()
    persist = []
    names = []

    for j, item in enumerate(temp):
        temp_file = dir + str(item)

        # print(temp_file)
        try:
            temp_persist = VariancePersistv1(
                temp_file.format(temp_file + str(1)),
                pixelx=pixelsx, pixely=pixelsy,
                myspread=spread, myspecs={"maxBD": Max, "minBD": -.10}, showplot=False)

            # print(temp_persist)
            persist.append(temp_persist)
            names.append(item)
            sys.stdout.write("\r %s / " % j + str(len(temp)))
            sys.stdout.flush()

        except:
            # print("error")
            # sys.stdout.write("\r error")
            # sys.stdout.flush()
            pass
    print("\n Number of files processed: " + str(len(persist)))
    return names, persist


