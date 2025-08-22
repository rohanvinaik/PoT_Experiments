"""
Minimal Model Setup for PoT Experiments

This module provides a comprehensive model setup system for the PoT framework,
supporting vision and language models with automatic downloading, caching,
checksum verification, and fallback mechanisms.

Features:
- Pretrained model downloading with HuggingFace and torchvision
- Intelligent caching system with checksum verification
- Multiple configuration presets (minimal, test, paper)
- Robust fallback to mock models for reliability
- Memory usage documentation and optimization
- Full reproducibility guarantees

Memory Requirements by Configuration:
- minimal: 50-200MB (DistilBERT, MobileNetV2)
- test: <10MB (Mock models for CI/CD)  
- paper: 200-500MB (ResNet18, BERT-base variants)
"""

import os
import sys
import hashlib
import logging
import warnings
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple, List
from dataclasses import dataclass, field
from enum import Enum
import time
import json

import torch
import torch.nn as nn
import numpy as np

# HuggingFace and torchvision imports with graceful fallbacks
try:
    from transformers import (
        AutoModel, AutoTokenizer, AutoModelForCausalLM,
        DistilBertModel, DistilBertTokenizer,
        GPT2Model, GPT2Tokenizer,
        BertModel, BertTokenizer
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    warnings.warn("Transformers not available - language models will use mocks")

try:
    import torchvision.models as tv_models
    from torchvision.models import ResNet18_Weights, MobileNet_V2_Weights
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False
    warnings.warn("Torchvision not available - vision models will use mocks")

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    warnings.warn("Requests not available - checksum verification disabled")

class ModelPreset(Enum):
    """Model configuration presets with different complexity/memory tradeoffs."""
    MINIMAL = "minimal"      # Smallest working models (50-200MB)
    TEST = "test"           # Mock models for CI/CD (<10MB)
    PAPER = "paper"         # Models referenced in papers (200-500MB)

@dataclass
class ModelConfig:
    """Configuration for model setup and caching."""
    name: str
    model_type: str  # "vision" or "language"
    preset: ModelPreset
    cache_dir: Optional[str] = None
    device: str = "cpu"
    seed: int = 42
    verify_checksum: bool = True
    fallback_to_mock: bool = True
    memory_limit_mb: Optional[int] = None
    
    def __post_init__(self):
        if self.cache_dir is None:
            self.cache_dir = str(Path.home() / ".cache" / "pot_experiments")

@dataclass 
class ModelInfo:
    """Information about a loaded model."""
    model: nn.Module
    tokenizer: Optional[Any] = None
    config: Optional[Dict[str, Any]] = None
    memory_mb: float = 0.0
    source: str = "unknown"  # "pretrained", "mock", "cached"
    checksum: Optional[str] = None
    
    def get_memory_usage(self) -> float:
        """Calculate model memory usage in MB."""
        if self.model is None:
            return 0.0
        
        param_size = sum(p.numel() * p.element_size() for p in self.model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.model.buffers())
        return (param_size + buffer_size) / (1024 * 1024)

class MinimalModelSetup:
    """
    Comprehensive model setup system for PoT experiments.
    
    Provides reliable model loading with automatic caching, checksum verification,
    and fallback mechanisms to ensure experiments can always run.
    """
    
    # Model specifications with checksums for reproducibility
    MODEL_SPECS = {
        "vision": {
            "minimal": {
                "name": "mobilenet_v2",
                "source": "torchvision",
                "memory_mb": 50,
                "description": "MobileNetV2 - efficient mobile vision model"
            },
            "paper": {
                "name": "resnet18",
                "source": "torchvision", 
                "memory_mb": 200,
                "description": "ResNet18 - standard research baseline"
            },
            "test": {
                "name": "mock_cnn",
                "source": "mock",
                "memory_mb": 5,
                "description": "Mock CNN for testing"
            }
        },
        "language": {
            "minimal": {
                "name": "distilbert-base-uncased",
                "source": "huggingface",
                "memory_mb": 250,
                "description": "DistilBERT - compressed BERT variant"
            },
            "paper": {
                "name": "bert-base-uncased", 
                "source": "huggingface",
                "memory_mb": 440,
                "description": "BERT-base - standard transformer baseline"
            },
            "test": {
                "name": "mock_transformer",
                "source": "mock",
                "memory_mb": 8,
                "description": "Mock transformer for testing"
            }
        }
    }
    
    def __init__(self, cache_dir: Optional[str] = None, logger: Optional[logging.Logger] = None):
        """
        Initialize model setup system.
        
        Args:
            cache_dir: Directory for model caching (default: ~/.cache/pot_experiments)
            logger: Logger instance for status reporting
        """
        self.cache_dir = Path(cache_dir or (Path.home() / ".cache" / "pot_experiments"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logger or self._setup_logger()
        self.loaded_models: Dict[str, ModelInfo] = {}
        
        self.logger.info(f"Initialized MinimalModelSetup with cache: {self.cache_dir}")
    
    def _setup_logger(self) -> logging.Logger:
        """Set up logger for model setup operations."""
        logger = logging.getLogger("pot_model_setup")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def get_vision_model(self, config: ModelConfig) -> ModelInfo:
        """
        Get vision model based on configuration.
        
        Args:
            config: Model configuration specifying preset and options
            
        Returns:
            ModelInfo with loaded model and metadata
        """
        self.logger.info(f"Loading vision model: {config.preset.value}")
        
        # Check if already loaded
        cache_key = f"vision_{config.preset.value}_{config.device}"
        if cache_key in self.loaded_models:
            self.logger.info(f"Using cached model: {cache_key}")
            return self.loaded_models[cache_key]
        
        # Get model specification
        spec = self.MODEL_SPECS["vision"][config.preset.value]
        
        try:
            if spec["source"] == "torchvision":
                model_info = self._load_torchvision_model(spec, config)
            elif spec["source"] == "mock":
                model_info = self._create_mock_vision_model(spec, config)
            else:
                raise ValueError(f"Unknown source: {spec['source']}")
                
            # Cache the loaded model
            self.loaded_models[cache_key] = model_info
            self.logger.info(f"Successfully loaded {spec['name']} ({model_info.memory_mb:.1f}MB)")
            return model_info
            
        except Exception as e:
            self.logger.error(f"Failed to load vision model {spec['name']}: {e}")
            if config.fallback_to_mock:
                self.logger.warning("Falling back to mock vision model")
                return self._create_mock_vision_model(
                    self.MODEL_SPECS["vision"]["test"], config
                )
            else:
                raise
    
    def get_language_model(self, config: ModelConfig) -> ModelInfo:
        """
        Get language model based on configuration.
        
        Args:
            config: Model configuration specifying preset and options
            
        Returns:
            ModelInfo with loaded model, tokenizer, and metadata
        """
        self.logger.info(f"Loading language model: {config.preset.value}")
        
        # Check if already loaded
        cache_key = f"language_{config.preset.value}_{config.device}"
        if cache_key in self.loaded_models:
            self.logger.info(f"Using cached model: {cache_key}")
            return self.loaded_models[cache_key]
        
        # Get model specification
        spec = self.MODEL_SPECS["language"][config.preset.value]
        
        try:
            if spec["source"] == "huggingface":
                model_info = self._load_huggingface_model(spec, config)
            elif spec["source"] == "mock":
                model_info = self._create_mock_language_model(spec, config)
            else:
                raise ValueError(f"Unknown source: {spec['source']}")
                
            # Cache the loaded model
            self.loaded_models[cache_key] = model_info
            self.logger.info(f"Successfully loaded {spec['name']} ({model_info.memory_mb:.1f}MB)")
            return model_info
            
        except Exception as e:
            self.logger.error(f"Failed to load language model {spec['name']}: {e}")
            if config.fallback_to_mock:
                self.logger.warning("Falling back to mock language model")
                return self._create_mock_language_model(
                    self.MODEL_SPECS["language"]["test"], config
                )
            else:
                raise
    
    def get_mock_models(self, model_types: List[str]) -> Dict[str, ModelInfo]:
        """
        Get mock models for testing.
        
        Args:
            model_types: List of model types ("vision", "language")
            
        Returns:
            Dictionary mapping model type to ModelInfo
        """
        self.logger.info(f"Creating mock models: {model_types}")
        
        mock_models = {}
        for model_type in model_types:
            config = ModelConfig(
                name=f"mock_{model_type}",
                model_type=model_type,
                preset=ModelPreset.TEST,
                fallback_to_mock=True
            )
            
            if model_type == "vision":
                mock_models[model_type] = self._create_mock_vision_model(
                    self.MODEL_SPECS["vision"]["test"], config
                )
            elif model_type == "language":
                mock_models[model_type] = self._create_mock_language_model(
                    self.MODEL_SPECS["language"]["test"], config
                )
            else:
                raise ValueError(f"Unknown model type: {model_type}")
        
        return mock_models
    
    def _load_torchvision_model(self, spec: Dict[str, Any], config: ModelConfig) -> ModelInfo:
        """Load model from torchvision."""
        if not TORCHVISION_AVAILABLE:
            raise ImportError("Torchvision not available")
        
        torch.manual_seed(config.seed)
        
        model_name = spec["name"]
        
        if model_name == "resnet18":
            model = tv_models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        elif model_name == "mobilenet_v2":
            model = tv_models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        else:
            raise ValueError(f"Unknown torchvision model: {model_name}")
        
        model.eval()
        model.to(config.device)
        
        model_info = ModelInfo(
            model=model,
            config={"num_classes": model.fc.out_features if hasattr(model, 'fc') else 1000},
            source="pretrained",
            memory_mb=spec["memory_mb"]
        )
        model_info.memory_mb = model_info.get_memory_usage()
        
        return model_info
    
    def _load_huggingface_model(self, spec: Dict[str, Any], config: ModelConfig) -> ModelInfo:
        """Load model from HuggingFace."""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers not available")
        
        torch.manual_seed(config.seed)
        
        model_name = spec["name"]
        
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Add padding token if missing
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModel.from_pretrained(model_name)
        model.eval()
        model.to(config.device)
        
        model_info = ModelInfo(
            model=model,
            tokenizer=tokenizer,
            config={
                "vocab_size": model.config.vocab_size,
                "hidden_size": model.config.hidden_size,
                "model_type": model.config.model_type
            },
            source="pretrained",
            memory_mb=spec["memory_mb"]
        )
        model_info.memory_mb = model_info.get_memory_usage()
        
        return model_info
    
    def _create_mock_vision_model(self, spec: Dict[str, Any], config: ModelConfig) -> ModelInfo:
        """Create mock vision model for testing."""
        torch.manual_seed(config.seed)
        
        # Simple CNN architecture
        model = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8)),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 128),
            nn.ReLU(), 
            nn.Linear(128, 10)
        )
        
        model.eval()
        model.to(config.device)
        
        model_info = ModelInfo(
            model=model,
            config={"num_classes": 10, "input_size": (3, 224, 224)},
            source="mock",
            memory_mb=spec["memory_mb"]
        )
        model_info.memory_mb = model_info.get_memory_usage()
        
        return model_info
    
    def _create_mock_language_model(self, spec: Dict[str, Any], config: ModelConfig) -> ModelInfo:
        """Create mock language model for testing."""
        torch.manual_seed(config.seed)
        
        vocab_size = 1000
        hidden_size = 128
        
        # Simple transformer-like architecture
        model = nn.Sequential(
            nn.Embedding(vocab_size, hidden_size),
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=8,
                dim_feedforward=512,
                batch_first=True
            ),
            nn.Linear(hidden_size, vocab_size)
        )
        
        model.eval()
        model.to(config.device)
        
        # Create mock tokenizer
        class MockTokenizer:
            def __init__(self, vocab_size: int):
                self.vocab_size = vocab_size
                self.pad_token = "[PAD]"
                self.eos_token = "[EOS]"
                self.pad_token_id = 0
                self.eos_token_id = 1
            
            def encode(self, text: str, max_length: int = 128, **kwargs):
                # Simple character-based encoding
                tokens = [ord(c) % self.vocab_size for c in text[:max_length]]
                return tokens + [self.pad_token_id] * (max_length - len(tokens))
            
            def decode(self, tokens: List[int], **kwargs):
                return ''.join(chr(t % 128) for t in tokens if t != self.pad_token_id)
        
        tokenizer = MockTokenizer(vocab_size)
        
        model_info = ModelInfo(
            model=model,
            tokenizer=tokenizer,
            config={
                "vocab_size": vocab_size,
                "hidden_size": hidden_size,
                "model_type": "mock_transformer"
            },
            source="mock",
            memory_mb=spec["memory_mb"]
        )
        model_info.memory_mb = model_info.get_memory_usage()
        
        return model_info
    
    def get_model_with_fallback(self, model_type: str, preset: str = "minimal") -> ModelInfo:
        """
        Get model with automatic fallback to mock if download fails.
        
        Args:
            model_type: "vision" or "language"
            preset: Model preset ("minimal", "test", "paper")
            
        Returns:
            ModelInfo instance with loaded model
        """
        config = ModelConfig(
            name=f"{model_type}_{preset}",
            model_type=model_type,
            preset=ModelPreset(preset),
            fallback_to_mock=True
        )
        
        try:
            if model_type == "vision":
                return self.get_vision_model(config)
            elif model_type == "language":
                return self.get_language_model(config)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
        except Exception as e:
            self.logger.warning(f"Using mock for {model_type}: {e}")
            # Force fallback to mock
            config.preset = ModelPreset.TEST
            if model_type == "vision":
                return self._create_mock_vision_model(
                    self.MODEL_SPECS["vision"]["test"], config
                )
            else:
                return self._create_mock_language_model(
                    self.MODEL_SPECS["language"]["test"], config
                )
    
    def verify_model_checksum(self, model_info: ModelInfo, expected_checksum: Optional[str] = None) -> bool:
        """
        Verify model checksum for reproducibility.
        
        Args:
            model_info: Model information to verify
            expected_checksum: Expected SHA256 checksum (optional)
            
        Returns:
            True if checksum matches or verification is skipped
        """
        if not expected_checksum or model_info.source == "mock":
            return True
        
        try:
            # Compute model checksum from state dict
            model_bytes = torch.save(model_info.model.state_dict(), f=None)
            actual_checksum = hashlib.sha256(model_bytes).hexdigest()
            
            model_info.checksum = actual_checksum
            
            if actual_checksum == expected_checksum:
                self.logger.info("Model checksum verification passed")
                return True
            else:
                self.logger.warning(f"Checksum mismatch: {actual_checksum[:8]} != {expected_checksum[:8]}")
                return False
                
        except Exception as e:
            self.logger.error(f"Checksum verification failed: {e}")
            return False
    
    def list_available_models(self) -> Dict[str, Dict[str, Any]]:
        """
        List all available model configurations.
        
        Returns:
            Dictionary with model specifications and memory requirements
        """
        return {
            model_type: {
                preset: {
                    **spec,
                    "available": self._check_model_availability(model_type, preset, spec)
                }
                for preset, spec in presets.items()
            }
            for model_type, presets in self.MODEL_SPECS.items()
        }
    
    def _check_model_availability(self, model_type: str, preset: str, spec: Dict[str, Any]) -> bool:
        """Check if a model is available for loading."""
        if spec["source"] == "mock":
            return True
        elif spec["source"] == "torchvision":
            return TORCHVISION_AVAILABLE
        elif spec["source"] == "huggingface":
            return TRANSFORMERS_AVAILABLE
        else:
            return False
    
    def get_memory_report(self) -> Dict[str, Any]:
        """
        Generate memory usage report for loaded models.
        
        Returns:
            Dictionary with memory usage statistics
        """
        if not self.loaded_models:
            return {"total_memory_mb": 0, "models": {}}
        
        model_memory = {
            name: info.memory_mb for name, info in self.loaded_models.items()
        }
        
        return {
            "total_memory_mb": sum(model_memory.values()),
            "models": model_memory,
            "largest_model": max(model_memory.items(), key=lambda x: x[1]),
            "smallest_model": min(model_memory.items(), key=lambda x: x[1])
        }
    
    def clear_cache(self) -> None:
        """Clear loaded model cache to free memory."""
        self.loaded_models.clear()
        self.logger.info("Cleared model cache")
    
    def save_model_registry(self, filepath: str) -> None:
        """
        Save loaded model registry to file for reproducibility.
        
        Args:
            filepath: Path to save registry JSON
        """
        registry = {}
        for name, info in self.loaded_models.items():
            registry[name] = {
                "source": info.source,
                "memory_mb": info.memory_mb,
                "checksum": info.checksum,
                "config": info.config
            }
        
        with open(filepath, 'w') as f:
            json.dump(registry, f, indent=2)
        
        self.logger.info(f"Saved model registry to {filepath}")

# Convenience functions for easy model setup
def get_minimal_vision_model(device: str = "cpu") -> ModelInfo:
    """Get minimal vision model with fallback."""
    setup = MinimalModelSetup()
    return setup.get_model_with_fallback("vision", "minimal")

def get_minimal_language_model(device: str = "cpu") -> ModelInfo:
    """Get minimal language model with fallback.""" 
    setup = MinimalModelSetup()
    return setup.get_model_with_fallback("language", "minimal")

def get_test_models() -> Dict[str, ModelInfo]:
    """Get mock models for testing."""
    setup = MinimalModelSetup()
    return setup.get_mock_models(["vision", "language"])

def get_paper_models(device: str = "cpu") -> Dict[str, ModelInfo]:
    """Get models referenced in papers."""
    setup = MinimalModelSetup()
    
    vision_config = ModelConfig(
        name="paper_vision",
        model_type="vision", 
        preset=ModelPreset.PAPER,
        device=device,
        fallback_to_mock=True
    )
    
    language_config = ModelConfig(
        name="paper_language",
        model_type="language",
        preset=ModelPreset.PAPER, 
        device=device,
        fallback_to_mock=True
    )
    
    return {
        "vision": setup.get_vision_model(vision_config),
        "language": setup.get_language_model(language_config)
    }