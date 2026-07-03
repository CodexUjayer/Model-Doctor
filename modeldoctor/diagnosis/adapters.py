import abc
from typing import Any, Optional

class ComplexityAdapter(abc.ABC):
    """Abstract adapter for retrieving complexity metrics from models."""
    def __init__(self, model: Any):
        self.model = model
        
    @abc.abstractmethod
    def parameter_count(self) -> Optional[int]:
        pass
        
    @abc.abstractmethod
    def depth(self) -> Optional[int]:
        pass

class GenericTreeAdapter(ComplexityAdapter):
    """Adapter for sklearn-like tree models."""
    def parameter_count(self) -> Optional[int]:
        return None
        
    def depth(self) -> Optional[int]:
        if hasattr(self.model, "get_depth"):
            return self.model.get_depth()
        if hasattr(self.model, "max_depth"):
            return self.model.max_depth
        return None

def get_adapter(model: Any) -> ComplexityAdapter:
    # A real implementation would inspect the class and return the specific adapter.
    # We fallback to GenericTreeAdapter for now since it's the main focus of Overfitting.
    return GenericTreeAdapter(model)
