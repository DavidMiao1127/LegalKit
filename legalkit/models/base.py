from abc import ABC, abstractmethod

class BaseModel(ABC):
    """Abstract base class for all models."""

    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text from the given prompt.
        """
        pass

    def __repr__(self):
        return f"<{self.__class__.__name__} model='{self.model_name}'>"