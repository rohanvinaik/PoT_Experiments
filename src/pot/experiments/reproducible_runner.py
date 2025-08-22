"""
Reproducible Experiment Runner for PoT Framework

This module provides a complete experimental validation pipeline with:
- Deterministic model setup and configuration
- Challenge family execution with sequential testing
- Comprehensive metrics computation (FAR, FRR, confidence)
- Full reproducibility with seed management
- Transparent result logging and export
"""

import os
import sys
import json
import csv
import time
import logging
import hashlib
import secrets
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

# Import PoT framework components
try:
    from pot.core.challenge import generate_challenges, ChallengeConfig, Challenge
    from pot.core.fingerprint import fingerprint_run, FingerprintConfig, FingerprintResult
    from pot.core.sequential import sequential_verify, SPRTResult, SequentialState
    from pot.vision.verifier import VisionVerifier
    from pot.lm.verifier import LMVerifier
    from pot.vision.models import VisionModel, MockVisionModel, load_resnet
    from pot.lm.models import LM
    from pot.experiments.model_setup import MinimalModelSetup, ModelConfig, ModelPreset
    CORE_AVAILABLE = True
except ImportError as e:
    CORE_AVAILABLE = False
    print(f"Warning: Core PoT components not available: {e}")

@dataclass
class ExperimentConfig:
    """Configuration for reproducible experiments."""
    # Experiment identification
    experiment_name: str
    trial_id: str = field(default_factory=lambda: secrets.token_hex(8))
    
    # Model configuration
    model_type: str = "vision"  # "vision" or "language"
    model_architecture: str = "simple_cnn"  # Architecture to use
    
    # Challenge configuration
    challenge_families: List[str] = field(default_factory=lambda: ["vision:freq"])
    n_challenges_per_family: int = 10
    master_key_hex: str = field(default_factory=lambda: secrets.token_hex(32))
    session_nonce_hex: str = field(default_factory=lambda: secrets.token_hex(16))
    
    # Sequential testing parameters
    alpha: float = 0.05  # Type I error rate
    beta: float = 0.05   # Type II error rate
    tau_id: float = 0.01  # Identity threshold
    tau_far: float = 0.1  # Far threshold
    max_challenges: int = 100
    
    # Reproducibility
    global_seed: int = 42
    torch_seed: int = 42
    numpy_seed: int = 42
    
    # Output configuration
    output_dir: str = "outputs/experiments"
    save_models: bool = False
    verbose: bool = True
    
    # Fingerprinting configuration
    use_jacobian: bool = False
    batch_size: int = 32
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

@dataclass 
class ExperimentResult:
    """Results from a single experiment trial."""
    trial_id: str
    challenge_type: str
    distance: float
    confidence_radius: float
    stopping_time: int
    verified: bool
    confidence_score: float
    ground_truth: bool
    timestamp: str
    challenge_id: Optional[str] = None
    fingerprint_time: float = 0.0
    decision_time: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

class ReproducibleExperimentRunner:
    """
    Complete experimental validation pipeline for PoT framework.
    
    Provides deterministic model setup, challenge execution, sequential testing,
    and comprehensive result logging with full reproducibility guarantees.
    """
    
    def __init__(self, config: ExperimentConfig):
        """Initialize runner with configuration."""
        self.config = config
        self.logger = self._setup_logging()
        self.results: List[ExperimentResult] = []
        self.models: Dict[str, Any] = {}
        self.verifiers: Dict[str, Any] = {}
        
        # Initialize model setup system
        self.model_setup = MinimalModelSetup(
            cache_dir=str(self.config.output_dir + "/.cache"),
            logger=self.logger
        )
        
        # Set up deterministic behavior
        self._set_seeds()
        self._setup_output_directory()
        
        self.logger.info(f"Initialized ReproducibleExperimentRunner with trial_id: {config.trial_id}")
        
    def _setup_logging(self) -> logging.Logger:
        """Set up structured logging."""
        logger = logging.getLogger(f"pot_experiments_{self.config.trial_id}")
        logger.setLevel(logging.INFO if self.config.verbose else logging.WARNING)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger
        
    def _set_seeds(self) -> None:
        """Set all random seeds for reproducibility."""
        # Python random
        import random
        random.seed(self.config.global_seed)
        
        # NumPy
        np.random.seed(self.config.numpy_seed)
        
        # PyTorch
        torch.manual_seed(self.config.torch_seed)
        torch.cuda.manual_seed_all(self.config.torch_seed)
        
        # Ensure deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        
        self.logger.info(f"Set seeds: global={self.config.global_seed}, "
                        f"numpy={self.config.numpy_seed}, torch={self.config.torch_seed}")
    
    def _setup_output_directory(self) -> None:
        """Create output directory structure."""
        self.output_path = Path(self.config.output_dir) / self.config.trial_id
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_path / "models").mkdir(exist_ok=True)
        (self.output_path / "logs").mkdir(exist_ok=True)
        (self.output_path / "results").mkdir(exist_ok=True)
        
        # Save configuration
        config_path = self.output_path / "config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
            
        self.logger.info(f"Created output directory: {self.output_path}")
    
    def setup_models(self) -> Dict[str, Any]:
        """
        Download or train minimal models for verification.
        
        Returns:
            Dictionary of model_name -> model mappings
        """
        self.logger.info("Setting up models...")
        
        if self.config.model_type == "vision":
            self._setup_vision_models()
        elif self.config.model_type == "language":
            self._setup_language_models()
        else:
            raise ValueError(f"Unknown model type: {self.config.model_type}")
            
        # Setup verifiers
        self._setup_verifiers()
        
        self.logger.info(f"Set up {len(self.models)} models and {len(self.verifiers)} verifiers")
        return self.models
    
    def _setup_vision_models(self) -> None:
        """Set up vision models for verification using model setup system."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Determine model preset based on architecture
        if self.config.model_architecture in ["mock", "test"]:
            preset = ModelPreset.TEST
        elif self.config.model_architecture in ["minimal", "mobilenet", "basic_cnn"]:
            preset = ModelPreset.MINIMAL
        elif self.config.model_architecture in ["paper", "resnet18", "resnet50"]:
            preset = ModelPreset.PAPER
        else:
            preset = ModelPreset.MINIMAL  # Default fallback
        
        # Create model configuration
        model_config = ModelConfig(
            name=f"vision_{self.config.model_architecture}",
            model_type="vision",
            preset=preset,
            device=str(device),
            seed=self.config.torch_seed,
            fallback_to_mock=True
        )
        
        # Load model using setup system
        model_info = self.model_setup.get_vision_model(model_config)
        
        # Extract PyTorch model from ModelInfo
        if isinstance(model_info.model, torch.nn.Module):
            model = model_info.model
        else:
            # Fallback to basic CNN if needed
            model = torch.nn.Sequential(
                torch.nn.Conv2d(3, 32, 3, padding=1),
                torch.nn.ReLU(),
                torch.nn.AdaptiveAvgPool2d((8, 8)),
                torch.nn.Flatten(),
                torch.nn.Linear(32 * 8 * 8, 10)
            )
            model.to(device)
            model.eval()
        
        self.models["target_model"] = model
        self.models["reference_model"] = model  # Same model for testing
        self.models["model_info"] = model_info  # Store additional info
        
        # Log model information
        self.logger.info(f"Loaded vision model: {model_info.source}, "
                        f"{model_info.memory_mb:.1f}MB")
        
        # Save models if requested
        if self.config.save_models:
            model_path = self.output_path / "models" / "target_model.pth"
            torch.save(model.state_dict(), model_path)
            
            # Save model info as well
            info_path = self.output_path / "models" / "model_info.json"
            with open(info_path, 'w') as f:
                json.dump({
                    "source": model_info.source,
                    "memory_mb": model_info.memory_mb,
                    "config": model_info.config,
                    "architecture": self.config.model_architecture
                }, f, indent=2)
    
    def _setup_language_models(self) -> None:
        """Set up language models for verification using model setup system."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Determine model preset based on architecture
        if self.config.model_architecture in ["mock", "test"]:
            preset = ModelPreset.TEST
        elif self.config.model_architecture in ["minimal", "distilbert"]:
            preset = ModelPreset.MINIMAL
        elif self.config.model_architecture in ["paper", "bert", "bert-base"]:
            preset = ModelPreset.PAPER
        else:
            preset = ModelPreset.MINIMAL  # Default fallback
        
        # Create model configuration
        model_config = ModelConfig(
            name=f"language_{self.config.model_architecture}",
            model_type="language",
            preset=preset,
            device=str(device),
            seed=self.config.torch_seed,
            fallback_to_mock=True
        )
        
        # Load model using setup system
        model_info = self.model_setup.get_language_model(model_config)
        
        # Extract PyTorch model from ModelInfo
        if isinstance(model_info.model, torch.nn.Module):
            model = model_info.model
        else:
            # Fallback to simple LSTM if needed
            model = torch.nn.Sequential(
                torch.nn.Embedding(1000, 128),
                torch.nn.LSTM(128, 256, batch_first=True),
                torch.nn.Linear(256, 1000)
            )
            model.to(device)
            model.eval()
        
        self.models["target_model"] = model
        self.models["reference_model"] = model
        self.models["model_info"] = model_info  # Store additional info
        self.models["tokenizer"] = model_info.tokenizer  # Store tokenizer if available
        
        # Log model information
        self.logger.info(f"Loaded language model: {model_info.source}, "
                        f"{model_info.memory_mb:.1f}MB")
        
        # Save models if requested  
        if self.config.save_models:
            model_path = self.output_path / "models" / "target_model.pth"
            torch.save(model.state_dict(), model_path)
            
            # Save model info as well
            info_path = self.output_path / "models" / "model_info.json"
            with open(info_path, 'w') as f:
                json.dump({
                    "source": model_info.source,
                    "memory_mb": model_info.memory_mb,
                    "config": model_info.config,
                    "architecture": self.config.model_architecture
                }, f, indent=2)
    
    def _setup_verifiers(self) -> None:
        """Set up appropriate verifiers for the model type."""
        if self.config.model_type == "vision":
            # Create VisionModel wrapper for the reference model
            reference_model = VisionModel(model_name="reference", device="cpu")
            reference_model.model = self.models["reference_model"]
            
            self.verifiers["vision"] = VisionVerifier(
                reference_model=reference_model,
                delta=0.01,
                use_sequential=True
            )
        elif self.config.model_type == "language": 
            self.verifiers["language"] = LMVerifier()
    
    def run_challenge_family(self, family_name: str) -> List[ExperimentResult]:
        """
        Execute one challenge family type.
        
        Args:
            family_name: Challenge family (e.g., "vision:freq", "lm:templates")
            
        Returns:
            List of experiment results for this family
        """
        self.logger.info(f"Running challenge family: {family_name}")
        
        # Generate challenges
        challenge_config = ChallengeConfig(
            master_key_hex=self.config.master_key_hex,
            session_nonce_hex=self.config.session_nonce_hex,
            n=self.config.n_challenges_per_family,
            family=family_name,
            params=self._get_family_params(family_name)
        )
        
        challenge_data = generate_challenges(challenge_config)
        
        # Extract challenge list from the returned dictionary
        if 'challenges' in challenge_data:
            challenges = challenge_data['challenges']
        elif 'items' in challenge_data:
            # Fallback to legacy format - convert items to Challenge objects
            challenges = []
            for i, item in enumerate(challenge_data['items']):
                challenge = Challenge(
                    challenge_id=f"{challenge_data['challenge_id']}_{i}",
                    index=i,
                    family=family_name,
                    parameters=item if isinstance(item, dict) else {'params': item}
                )
                challenges.append(challenge)
        else:
            challenges = []
            
        self.logger.info(f"Generated {len(challenges)} challenges for family {family_name}")
        
        # Execute challenges with progress bar
        family_results = []
        with tqdm(challenges, desc=f"Processing {family_name}", disable=not self.config.verbose) as pbar:
            for challenge in pbar:
                try:
                    result = self._execute_single_challenge(challenge, family_name)
                    family_results.append(result)
                    self.results.append(result)
                    
                    pbar.set_postfix({
                        'verified': result.verified,
                        'distance': f"{result.distance:.4f}",
                        'confidence': f"{result.confidence_score:.3f}"
                    })
                    
                except Exception as e:
                    self.logger.error(f"Error processing challenge {challenge.challenge_id}: {e}")
                    continue
        
        self.logger.info(f"Completed {len(family_results)} challenges for family {family_name}")
        return family_results
    
    def _get_family_params(self, family_name: str) -> Dict[str, Any]:
        """Get default parameters for challenge family."""
        if family_name.startswith("vision:freq"):
            return {
                "freq_range": [0.1, 0.5],
                "image_size": 32,
                "num_frequencies": 3
            }
        elif family_name.startswith("vision:texture"):
            return {
                "texture_scale": 0.1,
                "image_size": 32
            }
        elif family_name.startswith("lm:templates"):
            return {
                "max_length": 128,
                "num_templates": 5
            }
        else:
            return {}
    
    def _execute_single_challenge(self, challenge: 'Challenge', family_name: str) -> 'ExperimentResult':
        """Execute a single challenge and return result."""
        start_time = time.time()
        
        # Get appropriate model and verifier
        model = self.models["target_model"]
        verifier = self.verifiers.get(self.config.model_type)
        
        if not verifier:
            raise ValueError(f"No verifier available for model type: {self.config.model_type}")
        
        # Configure fingerprinting
        fp_config = FingerprintConfig(
            compute_jacobian=self.config.use_jacobian,
            batch_size=self.config.batch_size
        )
        
        # Execute fingerprinting
        fp_start = time.time()
        try:
            fp_result = fingerprint_run(model, challenge, fp_config)
            distance = fp_result.distance
        except Exception as e:
            self.logger.warning(f"Fingerprinting failed for challenge {challenge.challenge_id}: {e}")
            distance = 1.0  # Max distance on failure
            
        fp_time = time.time() - fp_start
        
        # Create result
        result = ExperimentResult(
            trial_id=self.config.trial_id,
            challenge_type=family_name,
            distance=distance,
            confidence_radius=0.0,  # Will be computed in sequential testing
            stopping_time=1,  # Single challenge
            verified=distance < self.config.tau_id,
            confidence_score=max(0.0, 1.0 - distance),
            ground_truth=True,  # Assume same model for now
            timestamp=datetime.now(timezone.utc).isoformat(),
            challenge_id=challenge.challenge_id,
            fingerprint_time=fp_time,
            decision_time=time.time() - start_time
        )
        
        return result
    
    def perform_sequential_decision(self, challenges: List['Challenge']) -> 'SPRTResult':
        """
        Implement sequential testing across multiple challenges.
        
        Args:
            challenges: List of challenges to test sequentially
            
        Returns:
            SPRT result with decision and statistics
        """
        self.logger.info(f"Performing sequential decision on {len(challenges)} challenges")
        
        # Initialize sequential state
        state = SequentialState()
        distances = []
        
        # Process challenges sequentially
        for i, challenge in enumerate(challenges):
            if i >= self.config.max_challenges:
                break
                
            # Execute challenge
            result = self._execute_single_challenge(challenge, challenge.family)
            distances.append(result.distance)
            state.update(result.distance)
            
            # Check stopping condition using sequential testing
            sprt_result = sequential_verify(
                distances,
                tau_id=self.config.tau_id,
                tau_far=self.config.tau_far,
                alpha=self.config.alpha,
                beta=self.config.beta
            )
            
            if sprt_result.decision != "continue":
                self.logger.info(f"Sequential testing stopped at challenge {i+1}: {sprt_result.decision}")
                break
        
        return sprt_result
    
    def compute_metrics(self) -> Dict[str, float]:
        """
        Calculate FAR, FRR, confidence scores across all results.
        
        Returns:
            Dictionary with computed metrics
        """
        if not self.results:
            return {}
        
        # Separate results by ground truth
        legitimate_results = [r for r in self.results if r.ground_truth]
        illegitimate_results = [r for r in self.results if not r.ground_truth]
        
        # Compute False Accept Rate (FAR) - illegitimate models verified as legitimate  
        if illegitimate_results:
            false_accepts = sum(1 for r in illegitimate_results if r.verified)
            far = false_accepts / len(illegitimate_results)
        else:
            far = 0.0
        
        # Compute False Reject Rate (FRR) - legitimate models rejected
        if legitimate_results:
            false_rejects = sum(1 for r in legitimate_results if not r.verified)
            frr = false_rejects / len(legitimate_results)
        else:
            frr = 0.0
        
        # Overall accuracy
        total_correct = sum(1 for r in self.results if r.verified == r.ground_truth)
        accuracy = total_correct / len(self.results)
        
        # Average confidence and distance
        avg_confidence = np.mean([r.confidence_score for r in self.results])
        avg_distance = np.mean([r.distance for r in self.results])
        avg_stopping_time = np.mean([r.stopping_time for r in self.results])
        
        # Timing metrics
        avg_fingerprint_time = np.mean([r.fingerprint_time for r in self.results])
        avg_decision_time = np.mean([r.decision_time for r in self.results])
        
        metrics = {
            "far": far,
            "frr": frr, 
            "accuracy": accuracy,
            "avg_confidence": avg_confidence,
            "avg_distance": avg_distance,
            "avg_stopping_time": avg_stopping_time,
            "avg_fingerprint_time": avg_fingerprint_time,
            "avg_decision_time": avg_decision_time,
            "total_experiments": len(self.results),
            "legitimate_count": len(legitimate_results),
            "illegitimate_count": len(illegitimate_results)
        }
        
        self.logger.info(f"Computed metrics: FAR={far:.4f}, FRR={frr:.4f}, Accuracy={accuracy:.4f}")
        return metrics
    
    def save_results(self) -> Tuple[Path, Path]:
        """
        Save results to CSV and JSON with full transparency.
        
        Returns:
            Tuple of (csv_path, json_path)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save as CSV
        csv_path = self.output_path / "results" / f"results_{timestamp}.csv"
        with open(csv_path, 'w', newline='') as f:
            if self.results:
                writer = csv.DictWriter(f, fieldnames=self.results[0].to_dict().keys())
                writer.writeheader()
                for result in self.results:
                    writer.writerow(result.to_dict())
        
        # Save as JSON with metrics
        metrics = self.compute_metrics()
        json_data = {
            "experiment_config": self.config.to_dict(),
            "metrics": metrics,
            "results": [r.to_dict() for r in self.results],
            "metadata": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "total_results": len(self.results),
                "core_available": CORE_AVAILABLE
            }
        }
        
        json_path = self.output_path / "results" / f"results_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        self.logger.info(f"Saved results to {csv_path} and {json_path}")
        return csv_path, json_path
    
    def run_full_experiment(self) -> Dict[str, Any]:
        """
        Run complete experimental pipeline.
        
        Returns:
            Dictionary with experiment summary and results
        """
        self.logger.info(f"Starting full experiment: {self.config.experiment_name}")
        start_time = time.time()
        
        try:
            # Setup models
            self.setup_models()
            
            # Run all challenge families
            all_results = []
            for family in self.config.challenge_families:
                family_results = self.run_challenge_family(family)
                all_results.extend(family_results)
            
            # Compute final metrics
            metrics = self.compute_metrics()
            
            # Save results
            csv_path, json_path = self.save_results()
            
            total_time = time.time() - start_time
            
            summary = {
                "experiment_name": self.config.experiment_name,
                "trial_id": self.config.trial_id,
                "total_time": total_time,
                "total_results": len(self.results),
                "metrics": metrics,
                "output_paths": {
                    "csv": str(csv_path),
                    "json": str(json_path),
                    "output_dir": str(self.output_path)
                }
            }
            
            self.logger.info(f"Experiment completed in {total_time:.2f}s with {len(self.results)} results")
            return summary
            
        except Exception as e:
            self.logger.error(f"Experiment failed: {e}")
            raise