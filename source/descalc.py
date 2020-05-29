


import argparse
#import vae_util this requires the vae enviroment


if __name__=="__main__":
    print("past importing")
    parser = argparse.ArgumentParser(description='select descriptor, and directory of files')
    parser.add_argument("--des", action='store', dest="desc", default="rdkit", help="select descriptor to convert to")
    parser.add_argument("--dir", action="store", dest="dir",  default="DB",    help="select directory")

    results = parser.parse_args()
    des = results.desc
    dir = results.dir

    if( des == "aval" or des == "morg" or des == "layer" or des == "rdkit"):
        from helpers import rdk, aval, layer, morgan
        dir = "../data/sdf/" + dir + "/"

    else:
        dir = "../data/xyz/" + dir + "/"


    if (des == "aval"):
        name, mat = aval(dir)

    elif (des == "morg"):
        name, mat = morgan(256, dir)


    elif (des == "layer"):
        name, mat = layer(dir)

    elif (des == "vae"):
        from vae_util import vae
        name, mat = vae(dir)

    elif (des == "self"):
        from selfies_util import selfies
        name, mat = selfies(dir)

    elif (des == "auto"):
        from molsimplify_util import full_autocorr
        name, mat = full_autocorr(dir)

    #requires a metal in the compound for this desc
    elif (des == "delta"):
        from molsimplify_util import metal_deltametrics
        name, mat = metal_deltametrics(dir)

    else:
        name, mat = rdk(dir)


    # todo convert name and the matrices into pandas/json
