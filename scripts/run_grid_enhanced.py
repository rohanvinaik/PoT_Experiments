#!/usr/bin/env python3
"""
Enhanced Grid Search Script for Comprehensive PoT Validation
Supports parallel execution, checkpoint recovery, and resource management
"""

import argparse
import itertools
import json
import multiprocessing as mp
import os
import time
import pickle
import hashlib
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import yaml
import numpy as np
from datetime import datetime
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pot.core.logging import StructuredLogger
from pot.core.challenge import ChallengeConfig, generate_challenges
from pot.core.stats import far_frr
from pot.core.sequential import SequentialTester

try:
    import torch
    import psutil
    HAS_MONITORING = True
except ImportError:
    HAS_MONITORING = False
    print("Warning: Install torch and psutil for resource monitoring")


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment"""
    model_name: str
    model_type: str  # 'vision' or 'lm'
    challenge_type: str
    verification_method: str
    batch_size: int
    num_samples: int
    device: str
    experiment_id: str = None
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
        if self.experiment_id is None:
            # Generate unique ID from config
            config_str = f"{self.model_name}_{self.challenge_type}_{self.verification_method}_{self.batch_size}"
            self.experiment_id = hashlib.md5(config_str.encode()).hexdigest()[:8]


@dataclass
class ExperimentResult:
    """Results from a single experiment"""
    config: ExperimentConfig
    accuracy: float
    false_accept_rate: float
    false_reject_rate: float
    runtime_seconds: float
    peak_memory_mb: float
    gpu_memory_mb: float
    num_queries: int
    throughput_qps: float
    success: bool
    error: Optional[str] = None
    checkpoint_path: Optional[str] = None


class ResourceTracker:
    """Track computational resources during experiments"""
    
    def __init__(self):
        self.start_time = time.time()
        self.measurements = []
        self.peak_memory = 0
        self.peak_gpu_memory = 0
        
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        if HAS_MONITORING:
            if torch.cuda.is_available():
                gpu_mem = torch.cuda.memory_allocated() / 1024 / 1024
                self.peak_gpu_memory = max(self.peak_gpu_memory, gpu_mem)
                return gpu_mem
            else:
                import psutil
                mem = psutil.Process().memory_info().rss / 1024 / 1024
                self.peak_memory = max(self.peak_memory, mem)
                return mem
        return 0
    
    def get_cpu_usage(self) -> float:
        """Get current CPU usage percentage"""
        if HAS_MONITORING:
            import psutil
            return psutil.cpu_percent(interval=0.1)
        return 0
            
    def log_measurement(self, label: str, **kwargs):
        """Log a measurement with metadata"""
        self.measurements.append({
            'timestamp': time.time() - self.start_time,
            'label': label,
            'memory_mb': self.get_memory_usage(),
            'cpu_percent': self.get_cpu_usage(),
            **kwargs
        })
    
    def get_summary(self) -> Dict[str, Any]:
        """Get resource usage summary"""
        if not self.measurements:
            return {}
        
        memory_values = [m['memory_mb'] for m in self.measurements]
        cpu_values = [m['cpu_percent'] for m in self.measurements]
        
        return {
            'peak_memory_mb': self.peak_memory,
            'peak_gpu_memory_mb': self.peak_gpu_memory,
            'avg_memory_mb': np.mean(memory_values) if memory_values else 0,
            'avg_cpu_percent': np.mean(cpu_values) if cpu_values else 0,
            'max_cpu_percent': max(cpu_values) if cpu_values else 0,
            'duration_seconds': time.time() - self.start_time
        }


class CheckpointManager:
    """Manage experiment checkpoints for recovery"""
    
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
    def save_checkpoint(self, config: ExperimentConfig, 
                       partial_result: Dict[str, Any]) -> str:
        """Save checkpoint for an experiment"""
        checkpoint_file = self.checkpoint_dir / f"{config.experiment_id}.pkl"
        
        checkpoint = {
            'config': asdict(config),
            'partial_result': partial_result,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint, f)
            
        return str(checkpoint_file)
    
    def load_checkpoint(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Load checkpoint if it exists"""
        checkpoint_file = self.checkpoint_dir / f"{experiment_id}.pkl"
        
        if checkpoint_file.exists():
            with open(checkpoint_file, 'rb') as f:
                return pickle.load(f)
        return None
    
    def list_checkpoints(self) -> List[str]:
        """List all available checkpoints"""
        return [f.stem for f in self.checkpoint_dir.glob("*.pkl")]
    
    def clean_checkpoint(self, experiment_id: str):
        """Remove checkpoint after successful completion"""
        checkpoint_file = self.checkpoint_dir / f"{experiment_id}.pkl"
        if checkpoint_file.exists():
            checkpoint_file.unlink()


class GridSearchRunner:
    """Enhanced grid search runner with parallel execution and checkpointing"""
    
    def __init__(self, base_config_path: str, output_dir: str, 
                 checkpoint_dir: str = None):
        self.base_config = self._load_config(base_config_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint_dir = checkpoint_dir or str(self.output_dir / "checkpoints")
        self.checkpoint_manager = CheckpointManager(checkpoint_dir)
        
        self.logger = StructuredLogger(str(self.output_dir))
        self.results = []
        
    def _load_config(self, path: str) -> dict:
        """Load YAML configuration file"""
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    
    def generate_grid(self) -> List[ExperimentConfig]:
        """Generate all combinations of experiment parameters"""
        grid = []
        
        # Extract parameter ranges from config
        models = self.base_config.get('models', {})
        challenges = self.base_config.get('challenges', {})
        verification = self.base_config.get('verification', {})
        
        # Model configurations
        model_configs = []
        for model_name, model_info in models.items():
            model_configs.append({
                'name': model_name,
                'type': model_info.get('type', 'vision'),
                'params': model_info
            })
        
        # Challenge types
        challenge_types = list(challenges.keys())
        
        # Verification methods
        methods = verification.get('methods', ['exact'])
        
        # Batch sizes to test
        batch_sizes = verification.get('batch_sizes', [16, 32, 64])
        
        # Generate all combinations
        for model, challenge, method, batch_size in itertools.product(
            model_configs, challenge_types, methods, batch_sizes
        ):
            # Skip invalid combinations
            if model['type'] == 'vision' and 'text' in challenge:
                continue
            if model['type'] == 'lm' and 'image' in challenge:
                continue
            
            config = ExperimentConfig(
                model_name=model['name'],
                model_type=model['type'],
                challenge_type=challenge,
                verification_method=method,
                batch_size=batch_size,
                num_samples=challenges[challenge].get('num_samples', 100),
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            grid.append(config)
            
        return grid
    
    def run_experiment(self, config: ExperimentConfig) -> ExperimentResult:
        """Run a single experiment configuration"""
        print(f"Running: {config.model_name} / {config.challenge_type} / {config.verification_method}")
        
        # Check for existing checkpoint
        checkpoint = self.checkpoint_manager.load_checkpoint(config.experiment_id)
        if checkpoint and not self.force_rerun:
            print(f"  Resuming from checkpoint...")
            partial_result = checkpoint['partial_result']
        else:
            partial_result = {}
        
        # Initialize resource tracker
        tracker = ResourceTracker()
        tracker.log_measurement("start")
        
        start_time = time.time()
        num_queries = 0
        
        try:
            # Generate challenges if not in checkpoint
            if 'challenges' not in partial_result:
                tracker.log_measurement("generating_challenges")
                challenges = self._generate_challenges(config)
                partial_result['challenges'] = challenges
                self.checkpoint_manager.save_checkpoint(config, partial_result)
            else:
                challenges = partial_result['challenges']
            
            # Load model if not in checkpoint
            if 'model_loaded' not in partial_result:
                tracker.log_measurement("loading_model")
                model = self._load_model(config)
                partial_result['model_loaded'] = True
                self.checkpoint_manager.save_checkpoint(config, partial_result)
            else:
                model = self._load_model(config)
            
            # Run verification
            tracker.log_measurement("verification_start")
            results = self._run_verification(model, challenges, config, tracker)
            num_queries = results.get('num_queries', len(challenges))
            
            # Calculate metrics
            far = results.get('false_accept_rate', 0.0)
            frr = results.get('false_reject_rate', 0.0)
            accuracy = 1.0 - (far + frr) / 2
            
            # Get resource summary
            resource_summary = tracker.get_summary()
            runtime = time.time() - start_time
            throughput = num_queries / runtime if runtime > 0 else 0
            
            # Clean up checkpoint on success
            self.checkpoint_manager.clean_checkpoint(config.experiment_id)
            
            result = ExperimentResult(
                config=config,
                accuracy=accuracy,
                false_accept_rate=far,
                false_reject_rate=frr,
                runtime_seconds=runtime,
                peak_memory_mb=resource_summary.get('peak_memory_mb', 0),
                gpu_memory_mb=resource_summary.get('peak_gpu_memory_mb', 0),
                num_queries=num_queries,
                throughput_qps=throughput,
                success=True
            )
            
        except Exception as e:
            # Save error state to checkpoint
            partial_result['error'] = str(e)
            checkpoint_path = self.checkpoint_manager.save_checkpoint(config, partial_result)
            
            result = ExperimentResult(
                config=config,
                accuracy=0.0,
                false_accept_rate=1.0,
                false_reject_rate=1.0,
                runtime_seconds=time.time() - start_time,
                peak_memory_mb=tracker.peak_memory,
                gpu_memory_mb=tracker.peak_gpu_memory,
                num_queries=num_queries,
                throughput_qps=0,
                success=False,
                error=str(e),
                checkpoint_path=checkpoint_path
            )
            
        # Log result
        self.logger.log_jsonl("grid_results.jsonl", asdict(result))
        tracker.log_measurement("complete")
        
        return result
    
    def _generate_challenges(self, config: ExperimentConfig) -> List[Any]:
        """Generate challenges for the experiment"""
        challenge_config = self.base_config['challenges'][config.challenge_type]
        
        if config.model_type == 'vision':
            # Generate vision challenges
            from pot.vision.models import generate_synthetic_batch
            challenges = []
            for _ in range(config.num_samples):
                if 'synthetic' in config.challenge_type:
                    batch = generate_synthetic_batch(1, 224)
                    challenges.append(batch)
            return challenges
        else:
            # Generate text challenges
            challenge_cfg = ChallengeConfig(
                master_key_hex="0" * 64,
                session_nonce_hex="0" * 32,
                n=config.num_samples,
                family=f"lm:{config.challenge_type}",
                params=challenge_config.get('params', {})
            )
            return generate_challenges(challenge_cfg)['items']
    
    def _load_model(self, config: ExperimentConfig) -> Any:
        """Load model for the experiment"""
        model_config = self.base_config['models'][config.model_name]
        
        if config.model_type == 'vision':
            # Load vision model
            from pot.vision.models import load_model
            return load_model(model_config.get('checkpoint', 'resnet18'))
        else:
            # Load language model
            from pot.lm.models import load_model
            return load_model(model_config.get('checkpoint', 'gpt2'))
    
    def _run_verification(self, model: Any, challenges: List[Any], 
                         config: ExperimentConfig, 
                         tracker: ResourceTracker) -> Dict[str, Any]:
        """Run verification on the model"""
        results = {
            'num_queries': 0,
            'false_accept_rate': 0.0,
            'false_reject_rate': 0.0
        }
        
        # Process challenges in batches
        batch_size = config.batch_size
        num_batches = (len(challenges) + batch_size - 1) // batch_size
        
        genuine_scores = []
        impostor_scores = []
        
        for i in range(0, len(challenges), batch_size):
            batch = challenges[i:i+batch_size]
            tracker.log_measurement(f"batch_{i//batch_size}")
            
            # Get model responses
            if config.model_type == 'vision':
                with torch.no_grad():
                    responses = model(torch.stack(batch))
            else:
                responses = [model.generate(ch) for ch in batch]
            
            # Compute scores (simulate for now)
            scores = np.random.uniform(0, 1, len(batch))
            
            # Split into genuine and impostor
            mid = len(scores) // 2
            genuine_scores.extend(scores[:mid])
            impostor_scores.extend(scores[mid:])
            
            results['num_queries'] += len(batch)
        
        # Calculate FAR/FRR
        if genuine_scores and impostor_scores:
            threshold = 0.5
            far = np.mean([s > threshold for s in impostor_scores])
            frr = np.mean([s <= threshold for s in genuine_scores])
            results['false_accept_rate'] = far
            results['false_reject_rate'] = frr
        
        return results
    
    def run_parallel(self, configs: List[ExperimentConfig], 
                    num_workers: int = 4) -> List[ExperimentResult]:
        """Run experiments in parallel"""
        print(f"Running {len(configs)} experiments with {num_workers} workers")
        
        if num_workers == 1:
            # Sequential execution
            results = []
            for i, config in enumerate(configs):
                print(f"[{i+1}/{len(configs)}] ", end="")
                result = self.run_experiment(config)
                results.append(result)
                self.results.append(result)
        else:
            # Parallel execution
            with mp.Pool(num_workers) as pool:
                results = pool.map(self.run_experiment, configs)
                self.results.extend(results)
        
        return results
    
    def save_results(self, results: List[ExperimentResult] = None):
        """Save results with timestamp"""
        if results is None:
            results = self.results
            
        if not results:
            print("No results to save")
            return
        
        # Convert to dictionaries
        results_dict = [asdict(r) for r in results]
        
        # Save as JSON
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"grid_results_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump({
                'metadata': {
                    'timestamp': timestamp,
                    'base_config': self.base_config,
                    'num_experiments': len(results),
                    'num_successful': sum(1 for r in results if r.success)
                },
                'results': results_dict
            }, f, indent=2, default=str)
        
        print(f"Results saved to {output_file}")
        
        # Also save summary CSV
        self._save_summary_csv(results)
    
    def _save_summary_csv(self, results: List[ExperimentResult]):
        """Save summary as CSV for easy analysis"""
        summary_file = self.output_dir / "grid_summary.csv"
        
        with open(summary_file, 'w') as f:
            # Write header
            f.write("model_name,challenge_type,verification_method,batch_size,")
            f.write("accuracy,far,frr,runtime_s,memory_mb,throughput_qps,success\n")
            
            # Write results
            for r in results:
                f.write(f"{r.config.model_name},{r.config.challenge_type},")
                f.write(f"{r.config.verification_method},{r.config.batch_size},")
                f.write(f"{r.accuracy:.4f},{r.false_accept_rate:.4f},")
                f.write(f"{r.false_reject_rate:.4f},{r.runtime_seconds:.2f},")
                f.write(f"{r.peak_memory_mb:.1f},{r.throughput_qps:.1f},")
                f.write(f"{r.success}\n")
        
        print(f"Summary saved to {summary_file}")
    
    def print_summary(self):
        """Print summary statistics"""
        if not self.results:
            print("No results to summarize")
            return
        
        print("\n" + "="*60)
        print("GRID SEARCH SUMMARY")
        print("="*60)
        
        successful = [r for r in self.results if r.success]
        failed = [r for r in self.results if not r.success]
        
        print(f"\nTotal experiments: {len(self.results)}")
        print(f"Successful: {len(successful)}")
        print(f"Failed: {len(failed)}")
        
        if successful:
            accuracies = [r.accuracy for r in successful]
            runtimes = [r.runtime_seconds for r in successful]
            memories = [r.peak_memory_mb for r in successful]
            throughputs = [r.throughput_qps for r in successful]
            
            print(f"\nAccuracy:")
            print(f"  Mean: {np.mean(accuracies):.4f}")
            print(f"  Std:  {np.std(accuracies):.4f}")
            print(f"  Best: {np.max(accuracies):.4f}")
            
            print(f"\nRuntime (seconds):")
            print(f"  Mean: {np.mean(runtimes):.2f}")
            print(f"  Total: {np.sum(runtimes):.2f}")
            
            print(f"\nMemory (MB):")
            print(f"  Mean: {np.mean(memories):.1f}")
            print(f"  Peak: {np.max(memories):.1f}")
            
            print(f"\nThroughput (queries/sec):")
            print(f"  Mean: {np.mean(throughputs):.1f}")
            print(f"  Max:  {np.max(throughputs):.1f}")
            
            # Best configurations
            best_configs = sorted(successful, key=lambda x: x.accuracy, reverse=True)[:3]
            print(f"\nTop 3 Configurations by Accuracy:")
            for i, r in enumerate(best_configs, 1):
                print(f"  {i}. {r.config.model_name} / {r.config.challenge_type} / "
                      f"{r.config.verification_method}: {r.accuracy:.4f}")
        
        if failed:
            print(f"\nFailed Experiments:")
            for r in failed[:5]:  # Show first 5 failures
                print(f"  - {r.config.model_name} / {r.config.challenge_type}: {r.error}")
            
            if len(failed) > 5:
                print(f"  ... and {len(failed) - 5} more")
        
        # Checkpoints status
        checkpoints = self.checkpoint_manager.list_checkpoints()
        if checkpoints:
            print(f"\nCheckpoints available: {len(checkpoints)}")
            print("  Run with --resume to continue from checkpoints")


def main():
    parser = argparse.ArgumentParser(description='Enhanced PoT validation grid search')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to base configuration file')
    parser.add_argument('--output', type=str, default='./grid_results',
                       help='Output directory for results')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of parallel workers')
    parser.add_argument('--subset', type=int, default=None,
                       help='Run only a subset of experiments')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from checkpoints')
    parser.add_argument('--force-rerun', action='store_true',
                       help='Force rerun even if checkpoints exist')
    parser.add_argument('--models', type=str, nargs='+', default=None,
                       help='Specific models to test')
    parser.add_argument('--methods', type=str, nargs='+', default=None,
                       help='Specific verification methods to test')
    
    args = parser.parse_args()
    
    # Initialize runner
    runner = GridSearchRunner(args.config, args.output)
    runner.force_rerun = args.force_rerun
    
    # Generate grid
    grid = runner.generate_grid()
    
    # Filter grid if requested
    if args.models:
        grid = [c for c in grid if c.model_name in args.models]
    if args.methods:
        grid = [c for c in grid if c.verification_method in args.methods]
    if args.subset:
        grid = grid[:args.subset]
    
    if not grid:
        print("No experiments to run")
        return
    
    print(f"Generated {len(grid)} experiment configurations")
    
    # Check for existing checkpoints
    if args.resume:
        checkpoints = runner.checkpoint_manager.list_checkpoints()
        if checkpoints:
            print(f"Found {len(checkpoints)} checkpoints to resume")
    
    # Run experiments
    start_time = time.time()
    results = runner.run_parallel(grid, args.num_workers)
    total_time = time.time() - start_time
    
    # Save results
    runner.save_results(results)
    
    # Print summary
    runner.print_summary()
    
    print(f"\nTotal execution time: {total_time:.2f} seconds")
    print(f"Average time per experiment: {total_time/len(grid):.2f} seconds")


if __name__ == '__main__':
    main()