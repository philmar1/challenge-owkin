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
def shuffleX(X, n=1000):
    n_features = X.shape[-1]
    X = X[[np.random.permutation(np.arange(i*n,(i+1)*n)) for i in range(0, len(X)//n)]]
    return X.reshape(-1, n_features)

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

def calculate_metric(metric_fn, true_y, pred_y):
    return metric_fn(true_y, pred_y)
    
def print_scores(p, r, f1, a):
    # just an utility printing function
    for name, scores in zip(("precision", "recall", "F1", "accuracy"), (p, r, f1, a)):
        print(f"\t{name.rjust(14, ' ')}: {scores:.4f}")        

# Model & others        
def weighted_loss(weight):
    bce_loss = torch.nn.BCELoss(reduction='none')
    def loss(y_pred, y_true):
        intermediate_loss = bce_loss(y_pred, y_true)
        return torch.mean(weight * y_true * intermediate_loss + (1 - y_true) * intermediate_loss)
    return loss

def warmup_lr(lr: float, start_lr: float, end_lr: float, epoch_max: int)        :
    step_lr = (end_lr - start_lr)/epoch_max
    lr += step_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def set_lr(lr: float):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_optimizer(model, lr: float):
    return optim.Adam(model.parameters(), lr=lr)

def train_epoch(model, optimizer, loss_function, dataloader):
    model.train()
    
    progress = tqdm(enumerate(dataloader), desc="Loss: ") #, total=train_batches)
    total_loss = 0
    for i, data in progress: 
        X, y = data[0].to(device), data[1].to(device)
        y = y.type(torch.FloatTensor)
        y = y.reshape(-1,1)
            
        # training step for single batch
        model.zero_grad() 
            
        outputs = model(X) 
        loss = loss_function(outputs, y) 
        loss.backward() 
        optimizer.step() 

        # getting training quality data
        current_loss = loss.item()
        total_loss += current_loss

        # updating progress bar
        progress.set_description("Loss: {:.4f}, learning_rate : {}".format(total_loss/(i+1), lr))
        
    return model, total_loss/(i+1) #current_loss    
    
def eval_epoch(model, loss_function, dataloader):
    # releasing unceseccary memory in GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    total_loss = 0
    y_pred, y_true = [], []
    precision, recall, f1, accuracy = [], [], [], []
        
    # set model to evaluating (testing)
    model.eval()
    progress = tqdm(enumerate(dataloader), desc="Loss: ")
    with torch.no_grad():
        for i, data in progress:
            X, y = data[0].to(device), data[1].to(device)
            y = y.type(torch.FloatTensor)
            y = y.reshape(-1,1)
            outputs = model(X) 
            prediced_classes = outputs.detach().round()
            total_loss += loss_function(outputs, y)
            y_pred.extend(prediced_classes.reshape(-1).tolist())
            y_true.extend(y.reshape(-1).tolist())
                    
        # calculate P/R/F1/A metrics for batch
        for acc, metric in zip((precision, recall, f1, accuracy), 
                                (precision_score, recall_score, f1_score, accuracy_score)):
            acc.append(metric(y_true, y_pred))
                
    return total_loss/i, precision[0], recall[0], f1[0], accuracy[0]
    
def train(model, optimizer, train_loader, eval_loader, hyperarameters: Dict):
    # Get training parameters
    epochs = hyperarameters['epochs']
    positive_weight = hyperarameters['positive_weight']
    gamma = hyperarameters['gamma']
    warmup_epoch = hyperarameters['warmup_epoch']
    start_lr = hyperarameters['start_lr']
    end_lr = hyperarameters['end_lr']
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    loss_function = torch.nn.BCELoss(reduction='mean') #weighted_loss(weight=positive_weight)
    
    train_batches = len(train_loader)
    eval_batches = len(eval_loader)
    start_ts = time.time()
    
    losses_train, losses_val = [], []
    for epoch in range(epochs):
        lr = optimizer.param_groups[0]['lr']
        if epoch < warmup_epoch:
            warmup_lr(lr, start_lr, end_lr, warmup_epoch)
        else:
            set_lr(lr * gamma)
        
        model, train_loss = train_epoch(model, optimizer, loss_function, train_loader)
        val_loss, precision, recall, f1, accuracy = eval_epoch(model, loss_function, eval_loader)
        print(f"Epoch {epoch + 1}/{epochs}, lr {lr:.7f}, training loss: {train_loss/train_batches}, validation loss: {val_loss/eval_batches}")
        
        print_scores(precision, recall, f1, accuracy)
        losses_train.append(train_loss/train_batches) # for plotting learning curve
        losses_val.append(val_loss/train_batches) # for plotting learning curve
        
    print(f"Training time: {time.time()-start_ts}s")

    return model

class MILDataset(Dataset):
    def __init__(self, X, y = None, n_instances = 1000, scaler = None):
        n_bags, n_features = len(X)//n_instances, X.shape[-1]
        self.X = X.reshape(n_bags, n_instances, n_features) # create n bags of 1000 instances of dim = n_features
        self.y = y[[i*n_instances for i in range(len(y)//n_instances)]]
        self.scaler = scaler
        if scaler is not None:
            self.X = scaler.transform(self.X)
            
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        features = self.X[idx,:,:]
        target = None
        if self.y is not None:
            target = self.y[idx]
        return features, target 


# %%
if __name__ == '__main__':
    X_train = catalog.load('X_train')
    y_train = catalog.load('y_train')
    indexs_train = catalog.load('indexs_train')

    X_eval = catalog.load('X_eval')
    y_eval = catalog.load('y_eval')
    indexs_eval = catalog.load('indexs_eval')
    
    
    # params:
    n_features = X_train.shape[-1] #catalog.load('params:umap.umap_kwargs.n_components')
    train_batch_size = catalog.load('params:train_batch_size')
    eval_batch_size = catalog.load('params:val_batch_size')
    batch_size = catalog.load('params:batch_size')

    # %%
    scaler = StandardScaler()
    scaler.fit(X_train)
    device = "cpu"
    # %%
    n_train, n_test = 1000, 1000
    n_neg, n_pos = 0, 0
    #X_train = shuffleX(X_train)
    #X_train, y_train = create_new_bags(X_train, y_train, n_neg=n_neg, n_pos=n_pos)

    train_data = MILDataset(X_train, y_train, n_instances=n_train, scaler=None)
    eval_data = MILDataset(X_eval, y_eval, n_instances=n_test, scaler=None)
    train_loader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True)
    eval_loader = DataLoader(eval_data, batch_size=eval_batch_size, shuffle=True)
    # %%
    # model:
    hyperarameters = {'start_lr': 0.0000001,
                    'end_lr': 0.0001,
                    'warmup_epoch': 10,
                    'epochs': 40,
                    'positive_weight': 1.1,
                    'gamma': 0.9,
                    }
    n_features = 2048
    out_features = 128
    aggregator = AttentionModule(input_dim=n_features, embed_dim=out_features, att='gated')
    classifier = NN_classifier(input_dim=n_features, hidden_layers_size = [out_features])
    model = MIL_NN(classifier=classifier, aggregator=aggregator, transformers_first=False)

    optimizer = optim.Adam(model.parameters(), lr=hyperarameters['start_lr'])
    scheduler = ExponentialLR(optimizer, gamma=0.9, verbose=True)

    # %%
    train(model, optimizer, train_loader, eval_loader, hyperarameters=hyperarameters)

