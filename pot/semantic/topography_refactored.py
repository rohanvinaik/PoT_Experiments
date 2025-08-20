"""
Refactored topography module with modular design.
This module fixes context window thrashing by splitting functionality into focused classes.
"""

import json
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class TopographyConfig:
    """Configuration for topographical mapping"""
    n_components: int = 2
    metric: str = "cosine"
    method: str = "umap"
    random_state: int = 42
    n_neighbors: int = 15
    min_dist: float = 0.1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TopographyConfig':
        """Create from dictionary"""
        return cls(**data)


class Persistable(ABC):
    """
    Base class for objects that can be saved and loaded.
    This consolidates the duplicate save/load implementations.
    """
    
    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """Get object state for serialization"""
        pass
    
    @abstractmethod
    def set_state(self, state: Dict[str, Any]) -> None:
        """Set object state from deserialization"""
        pass
    
    def save(self, path: Union[str, Path]) -> None:
        """
        Save object to file.
        This is the consolidated save method used by all subclasses.
        
        Args:
            path: Path to save file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        state = self.get_state()
        
        # Determine format from extension
        if path.suffix == '.json':
            # JSON format for simple data
            with open(path, 'w') as f:
                json.dump(state, f, indent=2, default=str)
        else:
            # Pickle for complex data
            with open(path, 'wb') as f:
                pickle.dump(state, f)
        
        logger.info(f"Saved {self.__class__.__name__} to {path}")
    
    def load(self, path: Union[str, Path]) -> None:
        """
        Load object from file.
        This is the consolidated load method used by all subclasses.
        
        Args:
            path: Path to load file
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        # Determine format from extension
        if path.suffix == '.json':
            with open(path, 'r') as f:
                state = json.load(f)
        else:
            with open(path, 'rb') as f:
                state = pickle.load(f)
        
        self.set_state(state)
        logger.info(f"Loaded {self.__class__.__name__} from {path}")


class BaseTopography(Persistable):
    """
    Base class for all topography implementations.
    """
    
    def __init__(self, config: Optional[TopographyConfig] = None):
        """
        Initialize base topography.
        
        Args:
            config: Topography configuration
        """
        self.config = config or TopographyConfig()
        self.is_fitted = False
        self.embeddings = None
        self.projection = None
    
    @abstractmethod
    def fit(self, embeddings: np.ndarray) -> 'BaseTopography':
        """Fit the topography to embeddings"""
        pass
    
    @abstractmethod
    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Transform embeddings to topographical space"""
        pass
    
    def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Fit and transform in one step"""
        self.fit(embeddings)
        return self.transform(embeddings)
    
    def get_state(self) -> Dict[str, Any]:
        """Get state for serialization"""
        state = {
            'config': self.config.to_dict(),
            'is_fitted': self.is_fitted,
            'embeddings_shape': self.embeddings.shape if self.embeddings is not None else None,
            'projection_shape': self.projection.shape if self.projection is not None else None
        }
        
        # Add numpy arrays if they exist
        if self.embeddings is not None:
            state['embeddings'] = self.embeddings.tolist()
        if self.projection is not None:
            state['projection'] = self.projection.tolist()
        
        return state
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """Set state from deserialization"""
        self.config = TopographyConfig.from_dict(state['config'])
        self.is_fitted = state['is_fitted']
        
        # Restore numpy arrays
        if 'embeddings' in state and state['embeddings'] is not None:
            self.embeddings = np.array(state['embeddings'])
        if 'projection' in state and state['projection'] is not None:
            self.projection = np.array(state['projection'])


class UMAPTopography(BaseTopography):
    """
    UMAP-based topographical mapping.
    Refactored from the original implementation around line 200.
    """
    
    def __init__(self, config: Optional[TopographyConfig] = None):
        """
        Initialize UMAP topography.
        
        Args:
            config: Configuration with UMAP parameters
        """
        super().__init__(config)
        self.model = None
    
    def fit(self, embeddings: np.ndarray) -> 'UMAPTopography':
        """
        Fit UMAP model to embeddings.
        
        Args:
            embeddings: Input embeddings
            
        Returns:
            Self for chaining
        """
        try:
            import umap
            
            self.model = umap.UMAP(
                n_components=self.config.n_components,
                n_neighbors=self.config.n_neighbors,
                min_dist=self.config.min_dist,
                metric=self.config.metric,
                random_state=self.config.random_state
            )
            
            self.embeddings = embeddings
            self.projection = self.model.fit_transform(embeddings)
            self.is_fitted = True
            
        except ImportError:
            raise ImportError("UMAP not installed. Run: pip install umap-learn")
        
        return self
    
    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Transform embeddings using fitted UMAP.
        
        Args:
            embeddings: Input embeddings
            
        Returns:
            Projected embeddings
        """
        if not self.is_fitted or self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        return self.model.transform(embeddings)
    
    def get_state(self) -> Dict[str, Any]:
        """Get state including UMAP model"""
        state = super().get_state()
        
        # Add UMAP-specific state
        if self.model is not None:
            # UMAP models can be pickled
            import pickle
            import base64
            model_bytes = pickle.dumps(self.model)
            state['model'] = base64.b64encode(model_bytes).decode('utf-8')
        
        return state
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """Set state including UMAP model"""
        super().set_state(state)
        
        # Restore UMAP model
        if 'model' in state and state['model'] is not None:
            import pickle
            import base64
            model_bytes = base64.b64decode(state['model'].encode('utf-8'))
            self.model = pickle.loads(model_bytes)


class TSNETopography(BaseTopography):
    """
    t-SNE based topographical mapping.
    Refactored from the original implementation around line 240.
    """
    
    def __init__(self, config: Optional[TopographyConfig] = None):
        """
        Initialize t-SNE topography.
        
        Args:
            config: Configuration with t-SNE parameters
        """
        super().__init__(config)
        # t-SNE doesn't support separate fit/transform
        self.perplexity = 30.0
        self.learning_rate = 200.0
    
    def fit(self, embeddings: np.ndarray) -> 'TSNETopography':
        """
        Fit t-SNE to embeddings.
        Note: t-SNE doesn't support transform on new data.
        
        Args:
            embeddings: Input embeddings
            
        Returns:
            Self for chaining
        """
        from sklearn.manifold import TSNE
        
        model = TSNE(
            n_components=self.config.n_components,
            perplexity=self.perplexity,
            learning_rate=self.learning_rate,
            random_state=self.config.random_state,
            metric=self.config.metric
        )
        
        self.embeddings = embeddings
        self.projection = model.fit_transform(embeddings)
        self.is_fitted = True
        
        return self
    
    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Transform not supported for t-SNE.
        
        Args:
            embeddings: Input embeddings
            
        Returns:
            Raises NotImplementedError
        """
        raise NotImplementedError("t-SNE doesn't support transform on new data. Use fit_transform instead.")


class PCATopography(BaseTopography):
    """
    PCA-based topographical mapping.
    """
    
    def __init__(self, config: Optional[TopographyConfig] = None):
        """
        Initialize PCA topography.
        
        Args:
            config: Configuration
        """
        super().__init__(config)
        self.model = None
    
    def fit(self, embeddings: np.ndarray) -> 'PCATopography':
        """
        Fit PCA to embeddings.
        
        Args:
            embeddings: Input embeddings
            
        Returns:
            Self for chaining
        """
        from sklearn.decomposition import PCA
        
        self.model = PCA(
            n_components=self.config.n_components,
            random_state=self.config.random_state
        )
        
        self.embeddings = embeddings
        self.projection = self.model.fit_transform(embeddings)
        self.is_fitted = True
        
        return self
    
    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Transform embeddings using fitted PCA.
        
        Args:
            embeddings: Input embeddings
            
        Returns:
            Projected embeddings
        """
        if not self.is_fitted or self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        return self.model.transform(embeddings)
    
    def get_state(self) -> Dict[str, Any]:
        """Get state including PCA components"""
        state = super().get_state()
        
        if self.model is not None:
            state['components'] = self.model.components_.tolist()
            state['mean'] = self.model.mean_.tolist()
            state['explained_variance'] = self.model.explained_variance_.tolist()
        
        return state
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """Set state including PCA components"""
        super().set_state(state)
        
        if 'components' in state:
            from sklearn.decomposition import PCA
            
            # Create a new PCA model and set its attributes
            self.model = PCA(n_components=self.config.n_components)
            self.model.components_ = np.array(state['components'])
            self.model.mean_ = np.array(state['mean'])
            self.model.explained_variance_ = np.array(state['explained_variance'])
            self.model.n_components_ = len(state['components'])


class TopographyFactory:
    """Factory for creating topography instances"""
    
    _topographies = {
        'umap': UMAPTopography,
        'tsne': TSNETopography,
        'pca': PCATopography
    }
    
    @classmethod
    def create(cls, method: str, config: Optional[TopographyConfig] = None) -> BaseTopography:
        """
        Create a topography instance.
        
        Args:
            method: Topography method ('umap', 'tsne', 'pca')
            config: Configuration
            
        Returns:
            Topography instance
        """
        if method not in cls._topographies:
            raise ValueError(f"Unknown topography method: {method}")
        
        if config is None:
            config = TopographyConfig(method=method)
        
        return cls._topographies[method](config)
    
    @classmethod
    def register(cls, name: str, topography_class: type):
        """Register a new topography type"""
        if not issubclass(topography_class, BaseTopography):
            raise TypeError(f"{topography_class} must inherit from BaseTopography")
        cls._topographies[name] = topography_class
    
    @classmethod
    def list_available(cls) -> List[str]:
        """List available topography methods"""
        return list(cls._topographies.keys())


class TopographicalProjector(Persistable):
    """
    High-level projector that manages multiple topography methods.
    Refactored from around line 1578 in the original file.
    """
    
    def __init__(self, default_method: str = "umap"):
        """
        Initialize topographical projector.
        
        Args:
            default_method: Default projection method
        """
        self.default_method = default_method
        self.topographies = {}
        self.current_method = default_method
    
    def add_method(self, method: str, config: Optional[TopographyConfig] = None):
        """Add a topography method"""
        self.topographies[method] = TopographyFactory.create(method, config)
    
    def fit(self, embeddings: np.ndarray, method: Optional[str] = None) -> 'TopographicalProjector':
        """
        Fit projector to embeddings.
        
        Args:
            embeddings: Input embeddings
            method: Method to use (None = default)
            
        Returns:
            Self for chaining
        """
        method = method or self.default_method
        
        if method not in self.topographies:
            self.add_method(method)
        
        self.topographies[method].fit(embeddings)
        self.current_method = method
        
        return self
    
    def transform(self, embeddings: np.ndarray, method: Optional[str] = None) -> np.ndarray:
        """
        Transform embeddings.
        
        Args:
            embeddings: Input embeddings
            method: Method to use (None = current)
            
        Returns:
            Projected embeddings
        """
        method = method or self.current_method
        
        if method not in self.topographies:
            raise ValueError(f"Method {method} not fitted")
        
        return self.topographies[method].transform(embeddings)
    
    def fit_transform(self, embeddings: np.ndarray, method: Optional[str] = None) -> np.ndarray:
        """Fit and transform in one step"""
        self.fit(embeddings, method)
        return self.transform(embeddings, method)
    
    def get_state(self) -> Dict[str, Any]:
        """Get state for serialization"""
        state = {
            'default_method': self.default_method,
            'current_method': self.current_method,
            'topographies': {}
        }
        
        for method, topo in self.topographies.items():
            state['topographies'][method] = topo.get_state()
        
        return state
    
    def set_state(self, state: Dict[str, Any]) -> None:
        """Set state from deserialization"""
        self.default_method = state['default_method']
        self.current_method = state['current_method']
        self.topographies = {}
        
        for method, topo_state in state['topographies'].items():
            topo = TopographyFactory.create(method)
            topo.set_state(topo_state)
            self.topographies[method] = topo


# Convenience functions
def create_topography(method: str = "umap", **kwargs) -> BaseTopography:
    """
    Create a topography instance with custom configuration.
    
    Args:
        method: Topography method
        **kwargs: Configuration parameters
        
    Returns:
        Topography instance
    """
    config = TopographyConfig(method=method, **kwargs)
    return TopographyFactory.create(method, config)


def project_embeddings(embeddings: np.ndarray,
                       method: str = "umap",
                       n_components: int = 2) -> np.ndarray:
    """
    Quick projection of embeddings.
    
    Args:
        embeddings: Input embeddings
        method: Projection method
        n_components: Number of components
        
    Returns:
        Projected embeddings
    """
    config = TopographyConfig(method=method, n_components=n_components)
    topography = TopographyFactory.create(method, config)
    return topography.fit_transform(embeddings)