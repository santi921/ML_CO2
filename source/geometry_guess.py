# https://arxiv.org/abs/1904.00314
# dl4hchem-geometry

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.platform import gfile
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io
from tensorflow.python.platform import gfile

#working directories for these files
loc = os.getcwd() + "/geo/"
output_fld  = './'
output_model_file = 'model.pb'
temp_str    = "../../dl4chem-geometry/neuralnet_model_best.ckpt.meta"


def save():
    sess = tf.compat.v1.Session()
    tf.compat.v1.disable_eager_execution()
    #save
    best = tf.compat.v1.train.import_meta_graph(temp_str)
    object_methods = [method_name for method_name in dir(best) if callable(getattr(best, method_name))]
    best.restore(sess, "../../dl4chem-geometry/neuralnet_model_best.ckpt")
    best.save(sess,"trial file")

def print_graph_params():
    sess = tf.compat.v1.Session()
    tf.compat.v1.disable_eager_execution()

    best = tf.compat.v1.train.import_meta_graph(temp_str)
    best.restore(sess, "../../dl4chem-geometry/neuralnet_model_best.ckpt")

    #Generate the list of network parameters
    all_nodes   = [n for n in tf.compat.v1.get_default_graph().as_graph_def().node]
    all_ops     = tf.compat.v1.get_default_graph().get_operations()
    #get graph and label nodes
    graph = tf.compat.v1.get_default_graph()
    input_graph_def = graph.as_graph_def()
    output_node_names = "update_GRUpriorZ/rnn/TensorArray_1"
    output_graph_def = graph_util.convert_variables_to_constants(sess,input_graph_def , output_node_names.split(","))
    graph_io.write_graph(output_graph_def, output_fld, output_model_file, as_text=False)
    for i in all_nodes:
        print(i)

def load_graph():
    sess = tf.compat.v1.Session()
    tf.compat.v1.disable_eager_execution()
    #this works for generating the graph, not the whole model
    with gfile.FastGFile("./geo/model.pb", 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.compat.v1.import_graph_def(graph_def, name='')
        tf.compat.v1.saved_model.load(sess,[tf.compat.v1.saved_model.tag_constants.SERVING], graph_def)

    graph_nodes=[n for n in graph_def.node]
    names = []
    for t in graph_nodes:
           names.append(t.name)
    print(names)

def load_tf1( save = False):
    sess = tf.compat.v1.Session()
    tf.compat.v1.disable_eager_execution()
    # regenerate as a saver
    saver = tf.compat.v1.train.import_meta_graph("./geo/model.meta")
    saver.restore(sess,"./geo/model")

    if (save == True):
        saver.save(sess, "saved")

    with gfile.FastGFile("./geo/model.pb", 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.compat.v1.import_graph_def(graph_def, name='')
        tf.compat.v1.saved_model.load(sess,[tf.compat.v1.saved_model.tag_constants.TRAINING], graph_def)

#def load_tf2():
#    new = tf.keras.models.load_model(loc)
#    print(new)
#    new.fit()
#    new.summary()

#load_graph()
load_tf1()
