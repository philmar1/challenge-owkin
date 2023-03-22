import logging
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

def calculate_metric(metric_fn, true_y, pred_y):
    print(metric_fn(true_y, pred_y))
    return metric_fn(true_y, pred_y)
    
def print_scores(p, r, f1, a):
    # just an utility printing function
    for name, scores in zip(("precision", "recall", "F1", "accuracy"), (p, r, f1, a)):
        logger.info(f"\t{name.rjust(14, ' ')}: {scores:.4f}")        

def get_train_eval_IDs(metadata, eval_pct: float = 0.33):
    # keep only patientID and center to split train and eval sets
    IDs = metadata[['Patient ID','Center ID']].drop_duplicates().values
    n = len(IDs)
    
    # Get train IDs
    selected_IDs = np.random.randint(0, n, int(eval_pct * n))
    eval_IDs = IDs[selected_IDs] 
    train_IDs = np.delete(IDs, selected_IDs, axis=0)
    
    return eval_IDs, train_IDs    

def split_train_eval(data, indexs, eval_IDs, train_IDs):
    # Get train indexs
    train_indexs = np.isin(indexs[:,[0,2]], train_IDs).min(axis=1)
    eval_indexs = np.isin(indexs[:,[0,2]], eval_IDs).min(axis=1)
    
    data_train, data_eval = data[train_indexs], data[eval_indexs]
    
    return data_train, data_eval

def shuffleX(X):
    n = 1000
    n_features = X.shape[-1]
    """X_shuffled = X[[np.random.permutation(np.arange(i*n,(i+1)*n)) for i in range(0, len(X)//n)]]
    return X_shuffled.reshape(-1, n_features)"""
    return X

def scale(X, scaler = None):
    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(X)
        return scaler, scaler.transform(X)
    return scaler.transform(X)

def fit_scaler(X):
    scaler = StandardScaler()
    scaler.fit(X)
    return scaler
    