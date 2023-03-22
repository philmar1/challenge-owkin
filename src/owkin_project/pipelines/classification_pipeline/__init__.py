"""
This is a boilerplate pipeline 'classification_pipeline'
generated using Kedro 0.18.4
"""

from .pipeline import create_pipeline, split_train_eval_pipeline, training_pipeline

__all__ = ["create_pipeline", "split_train_eval_pipeline", "training_pipeline"]

__version__ = "0.1"
