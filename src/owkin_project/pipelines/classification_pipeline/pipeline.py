"""
This is a boilerplate pipeline 'classification_pipeline'
generated using Kedro 0.18.4
"""

from kedro.pipeline import Pipeline, node, pipeline
from .train import *
from .utils.utils import split_train_eval, get_train_eval_IDs, shuffleX
from .utils.datamanager import *
from .models.DualStream import get_model

def split_train_eval_pipeline(**kwargs) -> Pipeline:

    return pipeline([node(
        func=get_train_eval_IDs,
        inputs=['train_metadata'],
        outputs= ['eval_IDs', 'train_IDs'],
        name='split_train_eval_emb'
    ),
                    node(
        func=split_train_eval,
        inputs=['concatenated_datasets', 'concatenated_indexs', 'eval_IDs', 'train_IDs'],
        outputs= ['X_train', 'X_eval'],
        name='split_train_eval_X'
    ),
                    node(
        func=split_train_eval,
        inputs=['concatenated_targets', 'concatenated_indexs', 'eval_IDs', 'train_IDs'],
        outputs= ['y_train', 'y_eval'],
        name='split_train_eval_y'
    ),
                    node(
        func=split_train_eval,
        inputs=['concatenated_indexs', 'concatenated_indexs', 'eval_IDs', 'train_IDs'],
        outputs= ['indexs_train', 'indexs_eval'],
        name='split_train_eval_indexs'
    ),
                    node(
        func=split_train_eval,
        inputs=['y_clustered', 'concatenated_indexs', 'eval_IDs', 'train_IDs'],
        outputs= ['y_clustered_train', 'y_clustered_eval'],
        name='split_train_eval_yclustered'
    )
                    ])
    
def training_pipeline(**kwargs) -> Pipeline:
    
    return pipeline([node(
        func=shuffleX,
        inputs='X_train',
        outputs='X_train_shuffled',
        name='shuffleX'
    ),
                     node(
        func=fit_scaler,
        inputs='X_train',
        outputs='scaler',
        name='scale_train'
    ),
                     node(
        func=get_dataset,
        inputs=['X_train_shuffled',
                'y_train',
                'params:train.n_instances',
                'scaler'],
        outputs='train_dataset',
        name='get_dataset_train'
    ),
                     
                     node(
        func=get_dataset,
        inputs=['X_eval',
                'y_eval',
                'params:eval.n_instances',
                'scaler'],
        outputs='eval_dataset',
        name='get_dataset_eval'
    ),
                     
                     node(
        func=get_dataloader,
        inputs=['train_dataset',
                'params:train.train_batch_size'],
        outputs='train_dataloader',
        name='get_dataloader_train'
    ),
                     
                     node(
        func=get_dataloader,
        inputs=['eval_dataset',
                'params:train.eval_batch_size'],
        outputs='eval_dataloader',
        name='get_dataloader_eval'
    ),
                     node(
        func=get_model,
        inputs=['params:dual_stream_model.input_dim', 
                'params:dual_stream_model.embed_dim', 
                'params:dual_stream_model.dropout', 
                'params:dual_stream_model.passing_v',
                'params:dual_stream_model.transformers_first'],
        outputs='raw_model',
        name='get_model'
    ),
                     node(
        func=train,
        inputs=['raw_model',
                'train_dataloader',
                'eval_dataloader', 
                'params:train.hyperparameters'],
        outputs='model',
        name='train'
    )
    ])
    
def create_pipeline(**kwargs) -> Pipeline:
    basic_pipeline = pipeline([])
    return basic_pipeline