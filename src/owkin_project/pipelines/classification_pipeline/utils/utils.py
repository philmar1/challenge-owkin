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

"""def scale(X, scaler = None):
    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(X)
        return scaler, scaler.transform(X)
    return scaler.transform(X)

def fit_scaler(X):
    scaler = StandardScaler()
    scaler.fit(X)
    return scaler"""
    
class CenterStandardScaler():
    def __init__(self) -> None:
        self.scalers = {}
        self.all_centers = []
        
    def fit(self, X, indexs):
        """ Fit Scaler for each of the center 

        Args:
            X (n_slices, n_features): features for each slice
            indexs (n_slices, 3): indexs of each slice : (personID, sliceID, centerID)
        """
        self.all_centers = np.unique(indexs[:,-1]).tolist()
        self.scalers = {centerID: StandardScaler() for centerID in self.all_centers}
        for centerID in self.all_centers:
            center_mask = (indexs[:,-1] == centerID)
            X_masked = X[center_mask]
            self.scalers[centerID].fit(X_masked)
            
        logger.info('Fitted X for centers: {}'.format(self.all_centers))
            
    def transform(self, X, indexs):
        """ Use fitted scalers to transform new data. If new center is encountered,
        fit a new scaler

        Args:
            X (n_slices, n_features): features for each slice
            indexs (n_slices, 3): indexs of each slice : (personID, sliceID, centerID)
        """
        old_centers = self.all_centers.copy()
        all_centers = np.unique(indexs[:,-1]).tolist()
        new_centers = [centerID for centerID in all_centers if centerID not in old_centers]
        for centerID in all_centers:
            center_mask = (indexs[:,-1] == centerID)
            X_masked = X[center_mask]
            if centerID in self.all_centers:    
                X_masked = self.scalers[centerID].transform(X_masked)
            else:
                self.scalers[centerID] = StandardScaler()
                self.scalers[centerID].fit(X_masked)
                X_masked = self.scalers[centerID].transform(X_masked)
            # duplicate center_mask into n_features columns 
            #center_mask = np.tile(center_mask, (X.shape[1],1)).transpose(1,0)   
            #X = np.where(center_mask, X_masked, X)
            X[center_mask] = X_masked    
        logger.info('Transformed X for centers: {}, find new centers:{}'.format(old_centers, new_centers))
        return X
 
def scale(X, indexs, scaler = None):
    if scaler is None:
        scaler = CenterStandardScaler()
        scaler.fit(X, indexs)
        return scaler, scaler.transform(X, indexs)
    return scaler.transform(X, indexs)
    
def fit_scaler(X, indexs):
    scaler = CenterStandardScaler()
    scaler.fit(X, indexs)
    return scaler