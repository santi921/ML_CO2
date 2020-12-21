from helpers import process_input_DB3
import multiprocessing as mp

des = ["auto","rdkit", "layer", "persist","aval", "morg", "vae", "self"]
#process_input_DB3(desc="auto", dir="DB3")
process_input_DB3(desc="persist", dir="DB3")
#process_input_DB3(desc="vae", dir="DB3")
#process_input_DB3(desc="self", dir="DB3")

#process_input_DB3(desc="rdkit", dir="DB3")
#process_input_DB3(desc="aval", dir="DB3")
#process_input_DB3(desc="morg", dir="DB3")
#process_input_DB3(desc="layer", dir="DB3")