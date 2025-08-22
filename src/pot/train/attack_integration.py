"""
Integration module for attack resistance evaluation during training.

This module provides functions to periodically evaluate attack resistance
during model training and integrate with existing training pipelines.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import json
import time
from typing import Dict, Any, Optional, List, Callable
import logging
from datetime import datetime
import numpy as np

# Import attack components
from pot.core.attack_suites import (
    AttackRunner, 
    StandardAttackSuite,
    AttackConfig
)
from pot.core.defenses import IntegratedDefenseSystem, DefenseConfig
from pot.eval.attack_benchmarks import AttackBenchmark
from pot.security.proof_of_training import ProofOfTraining

logger = logging.getLogger(__name__)


class AttackResistanceMonitor:
    """
    Monitor attack resistance during training.
    
    This class integrates with training loops to periodically evaluate
    model robustness against attacks.
    """
    
    def __init__(self,
                 evaluation_frequency: int = 10,
                 attack_suite: str = 'quick',
                 save_results: bool = True,
                 output_dir: str = 'attack_monitoring',
                 device: str = 'cpu',
                 verbose: bool = True):
        """
        Initialize attack resistance monitor.
        
        Args:
            evaluation_frequency: Evaluate every N epochs
            attack_suite: Attack suite to use (quick, standard, comprehensive)
            save_results: Whether to save evaluation results
            output_dir: Directory for saving results
            device: Device for evaluation
            verbose: Print progress information
        """
        self.evaluation_frequency = evaluation_frequency
        self.attack_suite = attack_suite
        self.save_results = save_results
        self.output_dir = Path(output_dir)
        self.device = device
        self.verbose = verbose
        
        # Create output directory
        if self.save_results:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.attack_runner = AttackRunner(device=device, verbose=verbose)
        self.attack_configs = self._get_attack_configs()
        
        # Tracking
        self.evaluation_history = []
        self.robustness_scores = []
        self.best_robustness = 0.0
        self.best_epoch = 0
        
    def _get_attack_configs(self) -> List[AttackConfig]:
        """Get attack configurations based on suite type."""
        if self.attack_suite == 'quick':
            return [
                AttackConfig(
                    name="quick_distillation",
                    attack_type="distillation",
                    budget={'queries': 100, 'compute_time': 10},
                    strength='weak',
                    success_metrics={'accuracy_drop': 0.1},
                    parameters={'temperature': 3.0, 'epochs': 3}
                ),
                AttackConfig(
                    name="quick_pruning",
                    attack_type="compression",
                    budget={'queries': 100, 'compute_time': 10},
                    strength='weak',
                    success_metrics={'accuracy_drop': 0.2},
                    parameters={'pruning_rate': 0.3, 'quantization_bits': 32}
                )
            ]
        elif self.attack_suite == 'standard':
            suite = StandardAttackSuite()
            return suite.get_all_configs()[:5]  # Subset for training
        else:  # comprehensive
            suite = StandardAttackSuite()
            return suite.get_all_configs()
    
    def evaluate(self, 
                model: nn.Module,
                epoch: int,
                val_loader: DataLoader,
                verifier: Optional[Any] = None) -> Dict[str, Any]:
        """
        Evaluate attack resistance.
        
        Args:
            model: Model to evaluate
            epoch: Current training epoch
            val_loader: Validation data loader
            verifier: Optional verifier for PoT
            
        Returns:
            Evaluation results dictionary
        """
        if epoch % self.evaluation_frequency != 0:
            return {}
        
        if self.verbose:
            print(f"\n[Epoch {epoch}] Evaluating attack resistance...")
        
        start_time = time.time()
        
        # Run attacks
        results = []
        for config in self.attack_configs:
            result = self.attack_runner.run_single_attack(
                model, config, val_loader
            )
            results.append(result)
        
        # Calculate metrics
        metrics = self.attack_runner.calculate_metrics(results)
        robustness = self.attack_runner.calculate_robustness_score(results)
        
        # Run verification if available
        verification_result = None
        if verifier:
            try:
                verification_result = verifier.verify(model)
            except Exception as e:
                logger.warning(f"Verification failed: {e}")
        
        # Prepare evaluation summary
        evaluation = {
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(),
            'robustness_score': robustness,
            'attack_success_rate': metrics['success_rate'],
            'total_attacks': metrics['total_attacks'],
            'successful_attacks': metrics['successful_attacks'],
            'evaluation_time': time.time() - start_time,
            'verification': verification_result,
            'detailed_results': results
        }
        
        # Update tracking
        self.evaluation_history.append(evaluation)
        self.robustness_scores.append(robustness)
        
        if robustness > self.best_robustness:
            self.best_robustness = robustness
            self.best_epoch = epoch
        
        # Save results
        if self.save_results:
            self._save_evaluation(evaluation, epoch)
        
        # Print summary
        if self.verbose:
            print(f"  Robustness: {robustness:.1f}/100")
            print(f"  Attack success rate: {metrics['success_rate']:.1%}")
            print(f"  Best robustness: {self.best_robustness:.1f} (epoch {self.best_epoch})")
        
        return evaluation
    
    def _save_evaluation(self, evaluation: Dict, epoch: int):
        """Save evaluation results."""
        # Save detailed results
        result_file = self.output_dir / f"eval_epoch_{epoch}.json"
        with open(result_file, 'w') as f:
            json.dump(evaluation, f, indent=2, default=str)
        
        # Update summary
        summary_file = self.output_dir / "summary.json"
        summary = {
            'latest_epoch': epoch,
            'best_robustness': self.best_robustness,
            'best_epoch': self.best_epoch,
            'robustness_history': self.robustness_scores,
            'total_evaluations': len(self.evaluation_history)
        }
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get evaluation summary."""
        return {
            'best_robustness': self.best_robustness,
            'best_epoch': self.best_epoch,
            'total_evaluations': len(self.evaluation_history),
            'average_robustness': np.mean(self.robustness_scores) if self.robustness_scores else 0,
            'robustness_trend': self._calculate_trend()
        }
    
    def _calculate_trend(self) -> str:
        """Calculate robustness trend."""
        if len(self.robustness_scores) < 2:
            return 'insufficient_data'
        
        recent = self.robustness_scores[-3:]
        earlier = self.robustness_scores[-6:-3] if len(self.robustness_scores) >= 6 else self.robustness_scores[:len(self.robustness_scores)//2]
        
        if not earlier:
            return 'insufficient_data'
        
        recent_avg = np.mean(recent)
        earlier_avg = np.mean(earlier)
        
        if recent_avg > earlier_avg * 1.1:
            return 'improving'
        elif recent_avg < earlier_avg * 0.9:
            return 'degrading'
        else:
            return 'stable'
    
    def should_early_stop(self, patience: int = 5) -> bool:
        """
        Check if training should stop based on robustness.
        
        Args:
            patience: Number of evaluations without improvement
            
        Returns:
            True if should stop
        """
        if len(self.evaluation_history) < patience:
            return False
        
        recent_best = max(self.robustness_scores[-patience:])
        return recent_best <= self.best_robustness


class TrainingIntegration:
    """
    Integration utilities for existing training pipelines.
    """
    
    @staticmethod
    def create_attack_callback(monitor: AttackResistanceMonitor) -> Callable:
        """
        Create callback function for training loops.
        
        Args:
            monitor: Attack resistance monitor
            
        Returns:
            Callback function
        """
        def callback(model: nn.Module, epoch: int, val_loader: DataLoader, **kwargs):
            """Callback to evaluate attack resistance."""
            return monitor.evaluate(model, epoch, val_loader, kwargs.get('verifier'))
        
        return callback
    
    @staticmethod
    def integrate_with_pytorch_lightning(monitor: AttackResistanceMonitor):
        """
        Create PyTorch Lightning callback.
        
        Args:
            monitor: Attack resistance monitor
            
        Returns:
            Lightning callback
        """
        try:
            import pytorch_lightning as pl
            
            class AttackResistanceCallback(pl.Callback):
                def __init__(self, monitor):
                    self.monitor = monitor
                
                def on_validation_epoch_end(self, trainer, pl_module):
                    """Evaluate at end of validation epoch."""
                    evaluation = self.monitor.evaluate(
                        pl_module,
                        trainer.current_epoch,
                        trainer.val_dataloaders[0]
                    )
                    
                    # Log metrics
                    if evaluation:
                        trainer.logger.log_metrics({
                            'robustness_score': evaluation['robustness_score'],
                            'attack_success_rate': evaluation['attack_success_rate']
                        }, step=trainer.current_epoch)
            
            return AttackResistanceCallback(monitor)
            
        except ImportError:
            logger.warning("PyTorch Lightning not available")
            return None
    
    @staticmethod
    def integrate_with_transformers(monitor: AttackResistanceMonitor):
        """
        Create Hugging Face Transformers callback.
        
        Args:
            monitor: Attack resistance monitor
            
        Returns:
            Transformers callback
        """
        try:
            from transformers import TrainerCallback
            
            class AttackResistanceCallback(TrainerCallback):
                def __init__(self, monitor):
                    self.monitor = monitor
                
                def on_evaluate(self, args, state, control, model, **kwargs):
                    """Evaluate after validation."""
                    if state.global_step % args.eval_steps == 0:
                        # Create simple data loader from eval dataset
                        eval_dataloader = kwargs.get('eval_dataloader')
                        if eval_dataloader:
                            evaluation = self.monitor.evaluate(
                                model,
                                state.epoch,
                                eval_dataloader
                            )
                            
                            # Log to wandb if available
                            if state.is_world_process_zero and args.report_to == ['wandb']:
                                import wandb
                                wandb.log({
                                    'robustness_score': evaluation['robustness_score'],
                                    'attack_success_rate': evaluation['attack_success_rate']
                                })
            
            return AttackResistanceCallback(monitor)
            
        except ImportError:
            logger.warning("Transformers not available")
            return None
    
    @staticmethod
    def add_to_training_loop(train_func: Callable,
                            monitor: AttackResistanceMonitor) -> Callable:
        """
        Decorator to add attack monitoring to training loop.
        
        Args:
            train_func: Original training function
            monitor: Attack resistance monitor
            
        Returns:
            Wrapped training function
        """
        def wrapped_train(*args, **kwargs):
            # Extract model and epoch from args/kwargs
            model = kwargs.get('model', args[0] if args else None)
            epoch = kwargs.get('epoch', 0)
            val_loader = kwargs.get('val_loader', None)
            
            # Run original training
            result = train_func(*args, **kwargs)
            
            # Evaluate attack resistance
            if model and val_loader:
                monitor.evaluate(model, epoch, val_loader)
            
            return result
        
        return wrapped_train


def evaluate_attack_resistance(model: nn.Module,
                              verifier: Any,
                              epoch: int,
                              val_loader: DataLoader,
                              config: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Standalone function to evaluate attack resistance during training.
    
    Args:
        model: Model to evaluate
        verifier: PoT verifier
        epoch: Current epoch
        val_loader: Validation data
        config: Configuration dictionary
        
    Returns:
        Evaluation results
    """
    config = config or {}
    
    # Only evaluate at specified frequency
    eval_frequency = config.get('attack_eval_frequency', 10)
    if epoch % eval_frequency != 0:
        return {}
    
    print(f"\n[Epoch {epoch}] Evaluating attack resistance...")
    
    # Create temporary monitor
    monitor = AttackResistanceMonitor(
        evaluation_frequency=1,  # Always evaluate when called
        attack_suite=config.get('attack_suite', 'quick'),
        save_results=config.get('save_results', True),
        output_dir=config.get('output_dir', 'attack_monitoring'),
        device=config.get('device', 'cpu'),
        verbose=config.get('verbose', True)
    )
    
    # Run evaluation
    return monitor.evaluate(model, epoch, val_loader, verifier)


def log_attack_metrics(results: Dict[str, Any], 
                       epoch: int,
                       logger: Optional[Any] = None):
    """
    Log attack evaluation metrics.
    
    Args:
        results: Evaluation results
        epoch: Current epoch
        logger: Optional logger (tensorboard, wandb, etc.)
    """
    if not results:
        return
    
    # Console logging
    print(f"[Epoch {epoch}] Attack Resistance Metrics:")
    print(f"  Robustness: {results.get('robustness_score', 0):.1f}/100")
    print(f"  Success rate: {results.get('attack_success_rate', 0):.1%}")
    
    # TensorBoard logging
    if logger and hasattr(logger, 'add_scalar'):
        logger.add_scalar('attack/robustness', results['robustness_score'], epoch)
        logger.add_scalar('attack/success_rate', results['attack_success_rate'], epoch)
    
    # WandB logging
    try:
        import wandb
        if wandb.run:
            wandb.log({
                'attack_robustness': results['robustness_score'],
                'attack_success_rate': results['attack_success_rate'],
                'epoch': epoch
            })
    except ImportError:
        pass


class CheckpointManager:
    """
    Manage checkpoints based on attack resistance.
    """
    
    def __init__(self, 
                checkpoint_dir: str = 'checkpoints',
                save_best_only: bool = True,
                metric: str = 'robustness_score'):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory for checkpoints
            save_best_only: Only save best model
            metric: Metric to track for best model
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_best_only = save_best_only
        self.metric = metric
        self.best_value = 0.0
        
    def save_checkpoint(self,
                       model: nn.Module,
                       epoch: int,
                       evaluation: Dict[str, Any],
                       optimizer: Optional[Any] = None):
        """
        Save checkpoint based on evaluation.
        
        Args:
            model: Model to save
            epoch: Current epoch
            evaluation: Attack evaluation results
            optimizer: Optional optimizer state
        """
        current_value = evaluation.get(self.metric, 0)
        
        if self.save_best_only and current_value <= self.best_value:
            return
        
        # Update best value
        if current_value > self.best_value:
            self.best_value = current_value
            
            # Save best model
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
                'robustness_score': evaluation.get('robustness_score', 0),
                'attack_success_rate': evaluation.get('attack_success_rate', 0),
                'evaluation': evaluation
            }, best_path)
            
            print(f"Saved best model with {self.metric}: {current_value:.3f}")
        
        if not self.save_best_only:
            # Save regular checkpoint
            checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
                'evaluation': evaluation
            }, checkpoint_path)


# Example integration functions for common frameworks

def integrate_with_standard_training(model: nn.Module,
                                    train_loader: DataLoader,
                                    val_loader: DataLoader,
                                    epochs: int,
                                    optimizer: torch.optim.Optimizer,
                                    criterion: nn.Module,
                                    device: str = 'cpu',
                                    attack_config: Optional[Dict] = None):
    """
    Example integration with standard PyTorch training loop.
    
    Args:
        model: Model to train
        train_loader: Training data
        val_loader: Validation data
        epochs: Number of epochs
        optimizer: Optimizer
        criterion: Loss function
        device: Device for training
        attack_config: Attack evaluation configuration
    """
    # Initialize attack monitor
    monitor = AttackResistanceMonitor(
        evaluation_frequency=attack_config.get('frequency', 10) if attack_config else 10,
        attack_suite=attack_config.get('suite', 'quick') if attack_config else 'quick',
        device=device
    )
    
    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager()
    
    for epoch in range(epochs):
        # Training loop
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        
        # Validation loop
        model.eval()
        val_loss = 0
        correct = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        
        val_loss /= len(val_loader)
        accuracy = correct / len(val_loader.dataset)
        
        print(f'Epoch {epoch}: Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}')
        
        # Evaluate attack resistance
        evaluation = monitor.evaluate(model, epoch, val_loader)
        
        # Save checkpoint based on robustness
        if evaluation:
            checkpoint_manager.save_checkpoint(model, epoch, evaluation, optimizer)
        
        # Early stopping based on robustness
        if monitor.should_early_stop(patience=5):
            print("Early stopping due to no robustness improvement")
            break
    
    # Print final summary
    summary = monitor.get_summary()
    print(f"\nTraining Complete!")
    print(f"Best Robustness: {summary['best_robustness']:.1f} at epoch {summary['best_epoch']}")
    print(f"Robustness Trend: {summary['robustness_trend']}")


if __name__ == "__main__":
    # Example usage
    print("Attack Integration Module")
    print("Use AttackResistanceMonitor for training integration")
    print("Use evaluate_attack_resistance() for standalone evaluation")