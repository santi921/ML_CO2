from Element_PI import VariancePersist
from Element_PI import VariancePersistv1
import numpy as np
import os

def persistent(dir="../data/xyz/", pixelsx = 150, pixelsy = 150, spread = 0.28, Max = 2.5):

    ls_dir = "ls " + dir
    temp = os.popen(ls_dir).read()
    temp = str(temp).split()
    persist = []
    names = []
    samples = len(temp)
    print(samples)

    #this one is incorrectly implemented
    #persist=np.zeros((samples,pixelsx*pixelsy))


    for item in temp:
        temp_file = dir + str(item)

        try:
            temp_persist = VariancePersistv1(
                temp_file.format(item + str(1)),
                pixelx=pixelsx, pixely=pixelsy,
                myspread=spread, myspecs={"maxBD": Max, "minBD": -.10}, showplot=False)
            persist.append(temp_persist)
            names.append(item)

        except:
            pass
    print("Number of files processed: " + str(len(persist)))
    return names, persist


