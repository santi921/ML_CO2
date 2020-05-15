



if __name__=="__main__":

    parser = argparse.ArgumentParser(description='select descriptor, and directory of files')
    parser.add_argument("--sgd", action='store_true', dest='sgd', default=False, help="increase output printing")
    parser.add_argument("-epochs", action="store", dest="epochs", default=50, help="epochs")
    results = parser.parse_args()
