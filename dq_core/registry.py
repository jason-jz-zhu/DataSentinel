"""Plugin registry system for extensible components."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type
import importlib
import inspect


class PluginRegistry:
    """Registry for managing plugins."""
    
    def __init__(self, base_class: Type[ABC]):
        self.base_class = base_class
        self.plugins: Dict[str, Type[ABC]] = {}
    
    def register(self, name: str, plugin_class: Type[ABC]) -> None:
        """Register a plugin."""
        if not issubclass(plugin_class, self.base_class):
            raise TypeError(
                f"{plugin_class} must be a subclass of {self.base_class}"
            )
        self.plugins[name] = plugin_class
    
    def get(self, name: str) -> Optional[Type[ABC]]:
        """Get a plugin by name."""
        return self.plugins.get(name)
    
    def list(self) -> List[str]:
        """List all registered plugin names."""
        return list(self.plugins.keys())
    
    def create(self, name: str, **kwargs) -> Any:
        """Create an instance of a plugin."""
        plugin_class = self.get(name)
        if not plugin_class:
            raise ValueError(f"Plugin '{name}' not found")
        return plugin_class(**kwargs)
    
    def auto_register(self, module_path: str) -> None:
        """Auto-register all plugins from a module."""
        module = importlib.import_module(module_path)
        
        for name, obj in inspect.getmembers(module):
            if (
                inspect.isclass(obj)
                and issubclass(obj, self.base_class)
                and obj != self.base_class
                and not inspect.isabstract(obj)
            ):
                plugin_name = getattr(obj, "name", name.lower())
                self.register(plugin_name, obj)


class BaseAdapter(ABC):
    """Base class for data adapters."""
    
    @abstractmethod
    def connect(self) -> None:
        """Connect to the data source."""
        pass
    
    @abstractmethod
    def validate(self, dataset: Any, checks: List[Any]) -> Any:
        """Run validation checks on the dataset."""
        pass
    
    @abstractmethod
    def profile(self, dataset: Any) -> Any:
        """Profile the dataset."""
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Close the connection."""
        pass


class BaseCheck(ABC):
    """Base class for data quality checks."""
    
    name: str = ""
    
    @abstractmethod
    def run(self, data: Any, **params) -> Any:
        """Run the check on the data."""
        pass
    
    @abstractmethod
    def validate_params(self, **params) -> bool:
        """Validate check parameters."""
        pass


class BaseSink(ABC):
    """Base class for result sinks."""
    
    @abstractmethod
    def write(self, result: Any) -> None:
        """Write the result to the sink."""
        pass
    
    @abstractmethod
    def read(self, query: Dict[str, Any]) -> List[Any]:
        """Read results from the sink."""
        pass


class BaseDetector(ABC):
    """Base class for anomaly detectors."""
    
    @abstractmethod
    def fit(self, data: Any) -> None:
        """Fit the detector on historical data."""
        pass
    
    @abstractmethod
    def detect(self, data: Any) -> Any:
        """Detect anomalies in the data."""
        pass


# Create global registries
adapter_registry = PluginRegistry(BaseAdapter)
check_registry = PluginRegistry(BaseCheck)
sink_registry = PluginRegistry(BaseSink)
detector_registry = PluginRegistry(BaseDetector)


def register_adapter(name: str):
    """Decorator to register an adapter."""
    def decorator(cls):
        adapter_registry.register(name, cls)
        return cls
    return decorator


def register_check(name: str):
    """Decorator to register a check."""
    def decorator(cls):
        check_registry.register(name, cls)
        cls.name = name
        return cls
    return decorator


def register_sink(name: str):
    """Decorator to register a sink."""
    def decorator(cls):
        sink_registry.register(name, cls)
        return cls
    return decorator


def register_detector(name: str):
    """Decorator to register a detector."""
    def decorator(cls):
        detector_registry.register(name, cls)
        return cls
    return decorator