"""
This is a boilerplate pipeline 'classification_pipeline'
generated using Kedro 0.18.4
"""

from kedro.pipeline import Pipeline, node, pipeline
from .train import *
from .utils.utils import split_train_eval, get_train_eval_IDs, shuffleX
from .utils.datamanager import *
from .models.DualStream import get_model
from .models.Classifier import get_instance_classifier

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
    
def prepare_data_pipeline(**kwargs) -> Pipeline:
    return pipeline([node(
        func=fit_scaler,
        inputs=['X_train', 'indexs_train'],
        outputs='scaler',
        name='fit_scaler'
    ),
                     node(
        func=scale,
        inputs=['X_train', 'indexs_train', 'scaler'],
        outputs='X_train_scaled',
        name='scale_train'
    ),
                     node(
        func=scale,
        inputs=['X_eval', 'indexs_eval', 'scaler'],
        outputs='X_eval_scaled',
        name='scale_eval'
    ),
                     node(
        func=shuffleX,
        inputs='X_train_scaled',
        outputs='X_train_shuffled',
        name='shuffleX'
    )
    ])
    
def training_instance_classifier_pipeline(**kwargs) -> Pipeline:
    return pipeline([ node(
        func=get_InstanceDataset,
        inputs=['X_train_shuffled',
                'y_clustered_train'], ## TODO: PUT Y_CLUSTERED
        outputs='train_instance_dataset',
        name='get_instance_dataset_train'
    ),
                     
                     node(
        func=get_InstanceDataset,
        inputs=['X_eval',
                'y_clustered_eval'],
        outputs='eval_instance_dataset',
        name='get_instance_dataset_eval'
    ),
                     
                     node(
        func=get_dataloader,
        inputs=['train_instance_dataset',
                'params:instance_classifier_train.train_batch_size'],
        outputs='train_instance_dataloader',
        name='get_instance_dataloader_train'
    ),
                     
                     node(
        func=get_dataloader,
        inputs=['eval_instance_dataset',
                'params:instance_classifier_train.eval_batch_size'],
        outputs='eval_instance_dataloader',
        name='get_instance_dataloader_eval'
    ),
                     node(
        func=get_instance_classifier,
        inputs=['params:instance_classifier.input_dim',
                'params:instance_classifier.cl_hidden_layers_size',
                'params:instance_classifier.dropout',
                'params:instance_classifier.end_activation'],
        outputs='raw_instance_classifier',
        name='get_instance_classifier'
    ),
                     node(
        func=train,
        inputs=['raw_instance_classifier',
                'train_instance_dataloader',
                'eval_instance_dataloader', 
                'params:train.hyperparameters'],
        outputs='instance_classifier',
        name='train_instance_classifier'                  
    )
                     ])

def training_pipeline(**kwargs) -> Pipeline:
    return pipeline([ node(
        func=get_MILDataset,
        inputs=['X_train_shuffled',
                'y_train',
                'params:train.n_instances'],
                #'scaler'],
        outputs='train_dataset',
        name='get_dataset_train'
    ),
                     
                     node(
        func=get_MILDataset,
        inputs=['X_eval',
                'y_eval',
                'params:eval.n_instances'],
                #'scaler'],
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
                'params:dual_stream_model.transformers_first',
                'params:dual_stream_model.instance_classifier'],
                #'instance_classifier'],
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