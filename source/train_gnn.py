import os
import numpy as np
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt 

from sklearn.preprocessing import StandardScaler, MinMaxScaler, scale
from sklearn.metrics import r2_score

from spektral.utils.io import *
from utils.graph_util import *
from utils.sklearn_util import *
from utils.selfies_util import *


if __name__ == "__main__":
    #dataset = dataset()
    model = 3
    #loader_train, loader_test, loader = partition_dataset(dataset)
    dataset = dataset_benzo()
    #loader_benzo = BatchLoader(benzo_set, batch_size = 100)
    loader_train, loader_test, loader = partition_dataset(dataset)

    
    if (model == 1):
      model = gnn_model_v1(dataset, loader_train) # working
    if (model == 2):
      model = gnn_model_v2(dataset, loader_train) # not working
    if (model == 3):
      model = gnn_model_v3(dataset, loader_train) # not working
    if (model == 4):
      model = gnn_model_v4(dataset, loader_train)
    if (model == 5):
      model = gnn_model_v5(dataset, loader_train)
    


    #----------------- testing segment -----------------
    model_loss = model.evaluate(loader_test.load(), steps=loader_test.steps_per_epoch)
    print("Done. Test loss: {}".format(model_loss))
    y_test = [loader_test.dataset[i]["y"] for i in range(len(loader_test))]
    y_test_pred = model.predict(loader_test.load(), verbose = 1, steps=loader_test.steps_per_epoch)
    print(r2_score(y_test_pred, y_test))

    benzo_set = dataset_benzo()
    loader_benzo = BatchLoader(benzo_set, batch_size = 100)
    y_test = [loader_test.dataset[i]["y"] for i in range(len(loader_test))]
    y_test_pred = model.predict(loader_benzo.load(), verbose = 1, steps=loader_benzo.steps_per_epoch)
    print(r2_score(y_test_pred, y_test))

