# %%
import numpy as np
import sys
sys.path.append('../')

import torch
from torch import nn, optim
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR, ExponentialLR
from tqdm.autonotebook import tqdm

from src.owkin_project.pipelines.classification_pipeline.model import *
from src.owkin_project.pipelines.classification_pipeline.utils import *
from src.owkin_project.pipelines.classification_pipeline.datamanager import *
from src.owkin_project.pipelines.classification_pipeline.train import *
from src.owkin_project.pipelines.classification_pipeline.MIL import *

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler

from sklearn import metrics as mtx
from sklearn import model_selection as ms


import random
random.seed(10000)

from typing import Dict
from itertools import chain
import time

import logging
import warnings
warnings.filterwarnings("ignore")

%load_ext kedro.extras.extensions.ipython
%reload_ext kedro.extras.extensions.ipython
catalog = catalog 

logging.basicConfig(filename='log.txt',
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)
logger = logging.getLogger(__name__)
# %%

def create_new_bags(X, y, n_neg, n_pos):
    """combine features from different bags to create new instances

    Args:
        n_neg: nb of new negative bags to create
        n_pos: nb of new positive bags to create
    """
    def flatten(list_of_lists):
        return list(chain(*list_of_lists))
    
    negative_idx = np.where(y==0)[0]
    positive_idx = np.where(y==1)[0]
    
    negative_X = X[negative_idx]
    positive_X = X[positive_idx]
    
    new_instances_negative = negative_X[flatten([random.sample(range(0, len(negative_X)), 1000) for i in range(n_neg)])]
    new_instances_positive = positive_X[flatten([random.sample(range(0, len(positive_X)), 1000) for i in range(n_pos)])]
    
    augmented_instances = np.concatenate([X, new_instances_negative, new_instances_positive])
    augmented_lables = np.concatenate([y, np.zeros(n_neg * 1000), np.ones(n_pos * 1000)])
    
    return augmented_instances, augmented_lables


# %%
if __name__ == '__main__':
    X_train = catalog.load('X_train')
    y_train = catalog.load('y_train')

    X_eval = catalog.load('X_eval')
    y_eval = catalog.load('y_eval')
    
    
    # params:
    n_features = X_train.shape[-1] #catalog.load('params:umap.umap_kwargs.n_components')
    train_batch_size = catalog.load('params:train_batch_size')
    eval_batch_size = catalog.load('params:val_batch_size')
    batch_size = catalog.load('params:batch_size')
    n_instances_train = catalog.load('params:train.n_instances')

    hyperparameters = catalog.load('params:train.hyperparameters')

    # %%
    n_train, n_test = 1000, 1000
    n_neg, n_pos = 0, 0
    #X_train, y_train = create_new_bags(X_train, y_train, n_neg=n_neg, n_pos=n_pos)
    
    scaler = fit_scaler(X_train)
    train_data = get_dataset(X_train, y_train, n_instances=n_instances_train, scaler=scaler)
    train_loader = get_dataloader(train_data, batch_size=train_batch_size)

    device = 'cpu'
    # %%
    # model:
    input_size = 2048
    feature_size = 128
    output_class = 1
    feature_extractor = nn.Sequential(
                nn.Linear(input_size, feature_size),
                nn.ReLU()
            )
    i_classifier = IClassifier(feature_extractor, feature_size, output_class)
    b_classifier = BClassifier(feature_size, output_class)
    model = MILNet(i_classifier, b_classifier)
    
    X = torch.randn((3,100,input_size))
    classes, bag_prediction, _, _ = model(X)
    # %%
    X, y = next(iter(train_loader))
    #feats, c = i_classifier(X)
    #prediction_bag, A, B = b_classifier(feats, c)

    classes, bag_prediction, _, _ = model(X)
    # %%
    train(model, train_loader, train_loader, hyperparameters=hyperparameters)


# %%
