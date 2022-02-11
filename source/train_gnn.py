import os
import numpy as np
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt 

from sklearn.preprocessing import StandardScaler, MinMaxScaler, scale
from sklearn.metrics import r2_score

from spektral.utils.io import *
from spektral.models import GeneralGNN
from spektral.data import DisjointLoader
from utils.graph_util import *
from utils.sklearn_util import *
from utils.selfies_util import *


if __name__ == "__main__":
    #dataset = dataset()
    #loader_train, loader_test, loader = partition_dataset(dataset)
    #dataset = dataset_benzo()
    #loader_benzo = BatchLoader(dataset, batch_size = 100)
    
    from spektral.datasets import QM9
    dataset = QM9(amount = 1000)
    
    print("..........partition dataset..........")
    loader_train, loader_test, loader = partition_dataset(dataset)

    N = max(data.n_nodes for data in dataset)
    F = dataset.n_node_features  # Dimension of node features
    S = dataset.n_edge_features  # Dimension of edge features
    batch_size = 128
    epochs = 10
    

    print("..........   pull model   ...........")
    model = 3
    if (model == 1): # fails
      model = gnn_v1() 
    if (model == 2): # works
      model = gnn_v2() 
    if (model == 3): # works
      model = gnn_v3() 
    if (model == 4): # works
      model = gnn_v4()
    if (model == 5): # fails
      model = gnn_v5(N)

    optimizer = Adam(1e-4)
    print("..........compile model..........")
    model.compile(optimizer=optimizer, loss="mse")
    
    es = EarlyStopping(monitor='loss', mode='min', patience = 5)
    model.fit(loader_train.load(),  steps_per_epoch=loader_train.steps_per_epoch, 
          epochs = epochs, callbacks = [es], batch_size=batch_size, verbose= 1)
    model.summary()
    
    print("..........testing model..........")
    #----------------- testing segment -----------------
    model_loss = model.evaluate(loader_test.load(), steps=loader_test.steps_per_epoch)
    print("Done. Test loss: {}".format(model_loss))
    y_test = [loader_test.dataset[i]["y"] for i in range(len(loader_test.dataset))]
    y_test_pred = model.predict(loader_test.load(), verbose = 1, steps=loader_test.steps_per_epoch)
    print(r2_score(y_test_pred, y_test))

    benzo_set = dataset_benzo()
    loader_benzo = BatchLoader(benzo_set, batch_size = 100)
    y_test = [loader_test.dataset[i]["y"] for i in range(len(loader_test.dataset))]
    y_test_pred = model.predict(loader_benzo.load(), verbose = 1, steps=loader_benzo.steps_per_epoch)
    print(r2_score(y_test_pred, y_test))

