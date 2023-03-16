"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.18.4
"""

import numpy as np
import pandas as pd
from os.path import join
from tqdm import tqdm
from typing import Dict

import umap

def concatenate_datasets(parameters:Dict):
    """Create datasets with all image features with associated labels

    Returns:
        X (n_sample_ID * n_images_perID, features)
        y (n_sample_ID * n_images_perID): target for each image
        indexs (n_sample_ID * n_images_perID, 2): Sample_ID, Patient ID for each image
    """
    features_dir = parameters['features_dir']
    mode = parameters['mode']
    metadata_path = parameters['metadata_path']
    target_path = parameters['target_path']

    metadata = pd.read_csv(metadata_path)
    targets = pd.read_csv(target_path)
    metadata = metadata.merge(targets, on="Sample ID")

    X, y, indexs = [], [], []
    
    print('metadata:',metadata)    
    for sample, label, center, patient in tqdm(metadata[["Sample ID", "Target", "Center ID", "Patient ID"]].values): 
                
        _features = np.load(join(features_dir, sample))
        _, features = _features[:, :3], _features[:, 3:]
        
        X.extend(features)
        indexs.extend([(patient, sample, center)] * len(features))
        y.extend([label] * len(features))
        
    # convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    indexs = np.array(indexs)
    
    return X, y, indexs


def concatenate_datasets_test(parameters:Dict):
    """Create datasets with all image features with associated labels

    Returns:
        X (n_sample_ID * n_images_perID, features)
        y (n_sample_ID * n_images_perID): target for each image
        indexs (n_sample_ID * n_images_perID, 2): Sample_ID, Patient ID for each image
    """
    features_dir = parameters['features_dir']
    mode = parameters['mode']
    metadata_path = parameters['metadata_path']
    metadata = pd.read_csv(metadata_path)
    
    X, indexs = [], []
    
    print('metadata:',metadata)    
    for sample, center, patient in tqdm(metadata[["Sample ID", "Center ID", "Patient ID"]].values): 
        _features = np.load(join(features_dir, sample))
        _, features = _features[:, :3], _features[:, 3:]
        
        X.extend(features)
        indexs.extend([(patient, sample, center)] * len(features))
        
    # convert to numpy arrays
    X = np.array(X)
    indexs = np.array(indexs)
    
    return X, _, indexs

def umap_fit(X: np.array, umap_kwargs: Dict = None):
    mapper = umap.UMAP(**umap_kwargs)
    mapper.fit(X)
    return mapper

def umap_transform(mapper: umap.UMAP, X: np.array):
    X_emb = mapper.transform(X)
    return X_emb