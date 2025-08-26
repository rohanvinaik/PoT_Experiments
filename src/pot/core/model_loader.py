"""
Unified Model Loader with Local and HuggingFace Hub Support

This module provides a flexible interface for loading models from either:
1. Local filesystem
2. HuggingFace Hub (with optional authentication)
"""

import os
import logging
from pathlib import Path
from typing import Optional, Tuple, Union, Dict, Any
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ModelSource(Enum):
    """Source for model loading"""
    LOCAL = "local"
    HUGGINGFACE = "huggingface"
    AUTO = "auto"  # Try local first, then HuggingFace

@dataclass
class ModelConfig:
    """Configuration for model loading"""
    name: str
    source: ModelSource = ModelSource.AUTO
    local_base_path: Optional[str] = None
    cache_dir: Optional[str] = None
    use_auth_token: Optional[str] = None
    torch_dtype: Optional[str] = "auto"  # auto, float16, float32, bfloat16
    device_map: Optional[str] = None  # auto, cpu, cuda, mps
    trust_remote_code: bool = False
    revision: Optional[str] = None  # specific commit/branch for HF models

class UnifiedModelLoader:
    """
    Unified model loader supporting both local and HuggingFace models
    
    Examples:
        # Load from local filesystem
        loader = UnifiedModelLoader(local_base="/path/to/models")
        model, tokenizer = loader.load("gpt2", source=ModelSource.LOCAL)
        
        # Load from HuggingFace Hub
        loader = UnifiedModelLoader()
        model, tokenizer = loader.load("gpt2", source=ModelSource.HUGGINGFACE)
        
        # Auto-detect (try local first, fallback to HF)
        loader = UnifiedModelLoader(local_base="/path/to/models")
        model, tokenizer = loader.load("gpt2")  # uses AUTO mode
    """
    
    def __init__(
        self,
        local_base: Optional[str] = None,
        cache_dir: Optional[str] = None,
        default_source: ModelSource = ModelSource.AUTO,
        use_auth_token: Optional[str] = None
    ):
        """
        Initialize the unified model loader
        
        Args:
            local_base: Base directory for local models
            cache_dir: Cache directory for downloaded models
            default_source: Default source when not specified
            use_auth_token: HuggingFace auth token for private models
        """
        self.local_base = Path(local_base) if local_base else None
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/huggingface")
        self.default_source = default_source
        self.use_auth_token = use_auth_token or os.environ.get("HF_TOKEN")
        
        # Common model name mappings for local paths
        self.local_model_mappings = {
            "gpt2": "gpt2",
            "gpt2-medium": "gpt2-medium",
            "gpt2-large": "gpt2-large",
            "gpt2-xl": "gpt2-xl",
            "distilgpt2": "distilgpt2",
            "mistral": "mistral_for_colab",
            "mistral-7b": "mistral_for_colab",
            "zephyr": "zephyr-7b-beta-final",
            "zephyr-7b": "zephyr-7b-beta-final",
            "llama2-7b": "llama-2-7b-hf",
            "llama2-7b-chat": "llama-2-7b-chat-hf",
            "falcon-7b": "falcon-7b",
            "vicuna-7b": "vicuna-7b-v1.5",
            "phi-2": "phi-2"
        }
        
        # HuggingFace model ID mappings
        self.hf_model_mappings = {
            "gpt2": "gpt2",
            "gpt2-medium": "gpt2-medium",
            "gpt2-large": "gpt2-large", 
            "gpt2-xl": "gpt2-xl",
            "distilgpt2": "distilgpt2",
            "mistral": "mistralai/Mistral-7B-v0.1",
            "mistral-7b": "mistralai/Mistral-7B-v0.1",
            "mistral-instruct": "mistralai/Mistral-7B-Instruct-v0.2",
            "zephyr": "HuggingFaceH4/zephyr-7b-beta",
            "zephyr-7b": "HuggingFaceH4/zephyr-7b-beta",
            "llama2-7b": "meta-llama/Llama-2-7b-hf",
            "llama2-7b-chat": "meta-llama/Llama-2-7b-chat-hf",
            "falcon-7b": "tiiuae/falcon-7b",
            "falcon-7b-instruct": "tiiuae/falcon-7b-instruct",
            "vicuna-7b": "lmsys/vicuna-7b-v1.5",
            "phi-2": "microsoft/phi-2",
            "pythia-70m": "EleutherAI/pythia-70m",
            "pythia-1.4b": "EleutherAI/pythia-1.4b",
            "opt-125m": "facebook/opt-125m",
            "opt-1.3b": "facebook/opt-1.3b",
            "bloom-560m": "bigscience/bloom-560m",
            "bloom-1b7": "bigscience/bloom-1b7"
        }
    
    def _check_local_model(self, model_name: str) -> Optional[Path]:
        """Check if model exists locally"""
        if not self.local_base:
            return None
            
        # Try direct name first
        model_path = self.local_base / model_name
        if model_path.exists() and (model_path / "config.json").exists():
            return model_path
            
        # Try mapped name
        mapped_name = self.local_model_mappings.get(model_name.lower())
        if mapped_name:
            model_path = self.local_base / mapped_name
            if model_path.exists() and (model_path / "config.json").exists():
                return model_path
                
        return None
    
    def _get_hf_model_id(self, model_name: str) -> str:
        """Get HuggingFace model ID from name"""
        # Check if it's already a full HF model ID
        if "/" in model_name:
            return model_name
            
        # Try mapping
        return self.hf_model_mappings.get(model_name.lower(), model_name)
    
    def load(
        self,
        model_name: str,
        source: Optional[ModelSource] = None,
        config: Optional[ModelConfig] = None,
        **kwargs
    ) -> Tuple[Any, Any]:
        """
        Load model and tokenizer
        
        Args:
            model_name: Model name or path
            source: Source to load from (overrides default)
            config: Optional ModelConfig for advanced settings
            **kwargs: Additional arguments passed to model/tokenizer constructors
            
        Returns:
            Tuple of (model, tokenizer)
        """
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
        except ImportError as e:
            raise ImportError("Transformers and torch required for model loading") from e
        
        source = source or self.default_source
        config = config or ModelConfig(name=model_name, source=source)
        
        # Determine actual source
        actual_path = None
        if source == ModelSource.LOCAL:
            actual_path = self._check_local_model(model_name)
            if not actual_path:
                raise ValueError(f"Model {model_name} not found locally at {self.local_base}")
                
        elif source == ModelSource.AUTO:
            # Try local first
            actual_path = self._check_local_model(model_name)
            if not actual_path:
                logger.info(f"Model {model_name} not found locally, trying HuggingFace Hub")
                source = ModelSource.HUGGINGFACE
            else:
                source = ModelSource.LOCAL
                
        # Load based on final source
        if source == ModelSource.LOCAL and actual_path:
            logger.info(f"Loading model from local path: {actual_path}")
            model_path = str(actual_path)
        else:
            # Load from HuggingFace
            model_path = self._get_hf_model_id(model_name)
            logger.info(f"Loading model from HuggingFace: {model_path}")
        
        # Prepare loading arguments
        load_kwargs = {
            "cache_dir": config.cache_dir or self.cache_dir,
            "trust_remote_code": config.trust_remote_code
        }
        
        # Add auth token if needed
        if source == ModelSource.HUGGINGFACE and (config.use_auth_token or self.use_auth_token):
            load_kwargs["use_auth_token"] = config.use_auth_token or self.use_auth_token
            
        # Add revision if specified
        if config.revision and source == ModelSource.HUGGINGFACE:
            load_kwargs["revision"] = config.revision
        
        # Handle torch dtype
        if config.torch_dtype:
            if config.torch_dtype == "auto":
                load_kwargs["torch_dtype"] = "auto"
            elif config.torch_dtype == "float16":
                load_kwargs["torch_dtype"] = torch.float16
            elif config.torch_dtype == "bfloat16":
                load_kwargs["torch_dtype"] = torch.bfloat16
            elif config.torch_dtype == "float32":
                load_kwargs["torch_dtype"] = torch.float32
                
        # Handle device map
        if config.device_map:
            load_kwargs["device_map"] = config.device_map
            
        # Merge with user kwargs
        load_kwargs.update(kwargs)
        
        # Load tokenizer
        logger.info(f"Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, **{
            k: v for k, v in load_kwargs.items() 
            if k not in ["torch_dtype", "device_map"]
        })
        
        # Set pad token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        # Load model
        logger.info(f"Loading model...")
        model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
        
        logger.info(f"Model and tokenizer loaded successfully")
        return model, tokenizer
    
    def list_available_models(self) -> Dict[str, Dict[str, Any]]:
        """
        List all available models (local and known HuggingFace)
        
        Returns:
            Dict with model information
        """
        available = {
            "local": {},
            "huggingface": {}
        }
        
        # List local models
        if self.local_base and self.local_base.exists():
            for model_dir in self.local_base.iterdir():
                if model_dir.is_dir() and (model_dir / "config.json").exists():
                    try:
                        import json
                        with open(model_dir / "config.json") as f:
                            config = json.load(f)
                        
                        # Try to determine model size
                        hidden_size = config.get("hidden_size", config.get("d_model", 0))
                        n_layers = config.get("num_hidden_layers", config.get("num_layers", 0))
                        
                        available["local"][model_dir.name] = {
                            "path": str(model_dir),
                            "architecture": config.get("model_type", "unknown"),
                            "hidden_size": hidden_size,
                            "num_layers": n_layers
                        }
                    except Exception as e:
                        logger.warning(f"Failed to read config for {model_dir}: {e}")
        
        # Add known HuggingFace models
        available["huggingface"] = {
            name: {"model_id": model_id, "requires_auth": "meta-llama" in model_id}
            for name, model_id in self.hf_model_mappings.items()
        }
        
        return available


def create_model_loader(
    prefer_local: bool = True,
    local_base: Optional[str] = None,
    use_hf_token: bool = False
) -> UnifiedModelLoader:
    """
    Convenience function to create a model loader with common settings
    
    Args:
        prefer_local: Whether to prefer local models over HuggingFace
        local_base: Base directory for local models (auto-detected if None)
        use_hf_token: Whether to use HuggingFace token from environment
        
    Returns:
        Configured UnifiedModelLoader instance
    """
    # Auto-detect common local model directories
    if local_base is None:
        possible_dirs = [
            os.path.expanduser("~/LLM_Models"),
            os.path.expanduser("~/models"),
            "./models",
            "../models"
        ]
        
        for dir_path in possible_dirs:
            if os.path.exists(dir_path):
                local_base = dir_path
                logger.info(f"Auto-detected local model directory: {local_base}")
                break
    
    # Set default source based on preference
    default_source = ModelSource.AUTO if prefer_local else ModelSource.HUGGINGFACE
    
    # Get HF token if requested
    hf_token = os.environ.get("HF_TOKEN") if use_hf_token else None
    
    return UnifiedModelLoader(
        local_base=local_base,
        default_source=default_source,
        use_auth_token=hf_token
    )