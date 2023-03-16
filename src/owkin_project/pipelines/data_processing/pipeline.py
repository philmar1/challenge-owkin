"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.18.4
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import concatenate_datasets, concatenate_datasets_test, umap_fit, umap_transform

def create_pipeline(**kwargs) -> Pipeline:
    basic_pipeline = pipeline([])
    return basic_pipeline
    
        
def concatenate_pipeline(**kwargs) -> Pipeline:
    concatenate_datasets_pipeline = pipeline([
        node(
            func=concatenate_datasets,
            inputs=['params:concatenate_datasets_pipeline.parameters'],
            outputs=['concatenated_datasets', 'concatenated_targets', 'concatenated_indexs'],
            name='concatenate_datasets'
        )
    ])
    
    concatenate_datasets_pipeline_test = pipeline([
        node(
            func=concatenate_datasets_test,
            inputs=['params:concatenate_datasets_pipeline_test.parameters'],
            outputs=['concatenated_datasets_test', 'concatenated_targets_test', 'concatenated_indexs_test'],
            name='concatenate_datasets_test'
        )
    ])

    return concatenate_datasets_pipeline, concatenate_datasets_pipeline_test


def embedding_pipeline(**kwargs) -> Pipeline:
    umap_embedding_pipeline = pipeline([
        node(
            func=umap_fit,
            inputs=['concatenated_datasets', 'params:umap.umap_kwargs'],
            outputs='umap_mapper',
            name='umap_embedding'
        ),
        node(
            func=umap_transform,
            inputs=['umap_mapper', 'concatenated_datasets_train'],
            outputs='X_emb_train',
            name='umap_transform_train'
        ),
        node(
            func=umap_transform,
            inputs=['umap_mapper', 'concatenated_datasets_test'],
            outputs='X_emb_test',
            name='umap_transform_test'
        )
        
    ])
    
    return umap_embedding_pipeline