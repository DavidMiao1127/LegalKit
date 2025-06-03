from .base import BaseModel
from .local import LocalModel
from .huggingface import HuggingfaceModel
from .api import APIModel

MODEL_REGISTRY = {
    "local": LocalModel,
    "hf": HuggingfaceModel,
    "api": APIModel
}

def build_model(model_type: str, **kwargs) -> BaseModel:
    """
    Factory method to instantiate a model by type string.

    Args:
        model_type: One of 'local', 'hf', 'api'
        kwargs: Arguments to pass to model constructor

    Returns:
        An instance of a subclass of BaseModel
    """
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {model_type}")
    return MODEL_REGISTRY[model_type](**kwargs)