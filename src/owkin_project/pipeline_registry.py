"""Project pipelines."""
from typing import Dict

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline

from owkin_project.pipelines import data_processing as dp 
from owkin_project.pipelines import classification_pipeline as cp 

def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    pipelines = find_pipelines()
    pipelines["__default__"] = sum(pipelines.values())
    
    # Concatenate pipeline
    data_processing_pipeline, data_processing_pipeline_test = dp.concatenate_pipeline()
    pipelines["data_processing_train"] = data_processing_pipeline
    pipelines["data_processing_test"] = data_processing_pipeline_test
    pipelines["data_processing"] = data_processing_pipeline + data_processing_pipeline_test
    
    # Embed pipeline
    umap_embedding_pipeline = dp.embedding_pipeline()
    pipelines["umap_embedding_pipeline"] = umap_embedding_pipeline
    
    # Model pipeline
    split_train_eval_pipeline = cp.create_pipeline()
    pipelines['split_train_eval_pipeline'] = split_train_eval_pipeline
    
    return pipelines
