"""
This is a boilerplate pipeline 'classification_pipeline'
generated using Kedro 0.18.4
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import split_train_eval, get_train_eval_IDs

def create_pipeline(**kwargs) -> Pipeline:
    old = pipeline([node(
        func=split_train_eval,
        inputs=['X_emb', 'concatenated_targets', 'concatenated_indexs', 'train_metadata'],
        outputs= ['X_train_emb', 'y_train_emb', 'indexs_train_emb', 'X_eval_emb', 'y_eval_emb', 'indexs_eval_emb'],
        name='split_train_eval_emb'
    ),
                    node(
        func=split_train_eval,
        inputs=['concatenated_datasets', 'concatenated_targets', 'concatenated_indexs', 'train_metadata'],
        outputs= ['X_train', 'y_train', 'indexs_train', 'X_eval', 'y_eval', 'indexs_eval'],
        name='split_train_eval'
    )])

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