"""
This is a boilerplate pipeline 'classification_pipeline'
generated using Kedro 0.18.4
"""

import numpy as np
from tqdm import tqdm

def get_train_eval_IDs(metadata, eval_pct: float = 0.33):
    # keep only patientID and center to split train and eval sets
    IDs = metadata[['Patient ID','Center ID']].drop_duplicates().values
    n = len(IDs)
    
    # Get train IDs
    selected_IDs = np.random.randint(0, n, int(eval_pct * n))
    eval_IDs = IDs[selected_IDs] 
    train_IDs = np.delete(IDs, selected_IDs, axis=0)
    
    return eval_IDs, train_IDs    
    
def split_train_eval_old(X, y, indexs, eval_IDs, train_IDs):
    # Get train indexs
    train_indexs = np.isin(indexs[:,[0,2]], train_IDs).min(axis=1)
    eval_indexs = np.isin(indexs[:,[0,2]], eval_IDs).min(axis=1)
    
    X_train, y_train, indexs_train = X[train_indexs], y[train_indexs], indexs[train_indexs]    
    X_eval, y_eval, indexs_eval = X[eval_indexs], y[eval_indexs], indexs[eval_indexs]    
    
    return X_train, y_train, indexs_train, X_eval, y_eval, indexs_eval

def split_train_eval(data, indexs, eval_IDs, train_IDs):
    # Get train indexs
    train_indexs = np.isin(indexs[:,[0,2]], train_IDs).min(axis=1)
    eval_indexs = np.isin(indexs[:,[0,2]], eval_IDs).min(axis=1)
    
    data_train, data_eval = data[train_indexs], data[eval_indexs]
    
    return data_train, data_eval
