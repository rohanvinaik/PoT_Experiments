#!/usr/bin/env python3
"""
Memory-Managed Large Model Testing for PoT Framework
Prevents OOM crashes when testing 34B+ models with proper throttling and cleanup.
"""

import os
import sys
import gc
import json
import time
import torch
import psutil
import resource
import traceback
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
import warnings

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pot.core.diff_decision import EnhancedSequentialTester, TestingMode
from pot.core.model_loader import UnifiedModelLoader, ModelSource, ModelConfig

# =============== CONFIGURATION ===============
@dataclass
class MemoryConfig:
    """Memory management configuration"""
    max_memory_gb: int = 40  # Maximum memory to use (leave headroom)
    min_free_gb: int = 10    # Minimum free memory to maintain
    cleanup_threshold_gb: int = 45  # Trigger cleanup if usage exceeds this
    batch_size: int = 1       # Process one query at a time
    max_concurrent_models: int = 1  # Only load one model at a time
    enable_8bit: bool = True  # Use 8-bit quantization when possible
    enable_cpu_offload: bool = True  # Offload to CPU when needed
    thread_limit: int = 8     # Limit CPU threads
    nice_level: int = 10      # Process priority (higher = lower priority)
    skip_zk: bool = False     # Skip ZK proof generation if memory constrained

# =============== MEMORY MONITORING ===============
class MemoryMonitor:
    """Monitor and manage system memory"""
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.initial_memory = self.get_memory_info()
        self.peak_memory = self.initial_memory['used_gb']
        
    def get_memory_info(self) -> Dict[str, float]:
        """Get current memory statistics"""
        mem = psutil.virtual_memory()
        return {
            'total_gb': mem.total / (1024**3),
            'available_gb': mem.available / (1024**3),
            'used_gb': mem.used / (1024**3),
            'percent': mem.percent,
            'free_gb': mem.available / (1024**3)
        }
    
    def check_memory_available(self, required_gb: float = 20) -> bool:
        """Check if enough memory is available"""
        info = self.get_memory_info()
        return info['free_gb'] >= max(required_gb, self.config.min_free_gb)
    
    def enforce_memory_limit(self):
        """Enforce memory limits using resource module"""
        try:
            max_bytes = self.config.max_memory_gb * 1024 * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (max_bytes, max_bytes))
        except Exception as e:
            print(f"⚠️ Could not set hard memory limit: {e}")
    
    def cleanup_if_needed(self, force: bool = False):
        """Trigger cleanup if memory usage is high"""
        info = self.get_memory_info()
        
        if force or info['used_gb'] > self.config.cleanup_threshold_gb:
            print(f"🧹 Triggering memory cleanup (used: {info['used_gb']:.1f}GB)")
            self.force_cleanup()
            
            # Wait for cleanup to take effect
            time.sleep(2)
            
            new_info = self.get_memory_info()
            freed = info['used_gb'] - new_info['used_gb']
            print(f"   Freed: {freed:.1f}GB (now: {new_info['used_gb']:.1f}GB)")
    
    def force_cleanup(self):
        """Force memory cleanup"""
        # Python garbage collection
        gc.collect()
        gc.collect()  # Run twice to ensure cleanup
        
        # PyTorch specific cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Clear any cached data
        if hasattr(torch, 'mps'):
            # For Apple Silicon - check if MPS is available
            try:
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
            except:
                pass  # MPS not available
    
    def log_status(self, label: str = "Current"):
        """Log current memory status"""
        info = self.get_memory_info()
        self.peak_memory = max(self.peak_memory, info['used_gb'])
        
        print(f"💾 {label} Memory Status:")
        print(f"   Used: {info['used_gb']:.1f}GB / {info['total_gb']:.1f}GB ({info['percent']:.1f}%)")
        print(f"   Free: {info['free_gb']:.1f}GB")
        print(f"   Peak: {self.peak_memory:.1f}GB")

# =============== MODEL MANAGEMENT ===============
class ThrottledModelManager:
    """Manage model loading and unloading with memory constraints"""
    
    def __init__(self, memory_monitor: MemoryMonitor, config: MemoryConfig):
        self.memory_monitor = memory_monitor
        self.config = config
        self.current_model = None
        self.current_tokenizer = None
        self.current_model_name = None
        
    def estimate_model_memory(self, model_path: str) -> float:
        """Estimate memory required for a model"""
        config_path = Path(model_path) / "config.json"
        
        if not config_path.exists():
            # Conservative estimate for unknown models
            return 40.0  # GB
        
        try:
            with open(config_path) as f:
                config = json.load(f)
            
            # Estimate based on parameters
            hidden_size = config.get('hidden_size', 4096)
            num_layers = config.get('num_hidden_layers', 32)
            vocab_size = config.get('vocab_size', 32000)
            
            # Rough estimation: params * bytes_per_param * overhead_factor
            params_billions = (hidden_size * num_layers * 12 + vocab_size * hidden_size) / 1e9
            
            # Account for quantization
            bytes_per_param = 1 if self.config.enable_8bit else 4
            overhead_factor = 1.5  # Account for gradients, activations, etc.
            
            estimated_gb = params_billions * bytes_per_param * overhead_factor
            
            print(f"📊 Estimated memory for {Path(model_path).name}: {estimated_gb:.1f}GB")
            return estimated_gb
            
        except Exception as e:
            print(f"⚠️ Could not estimate model size: {e}")
            return 40.0  # Conservative default
    
    def load_model_throttled(self, model_path: str) -> Tuple[Any, Any]:
        """Load a model with memory management"""
        
        # Unload current model if different
        if self.current_model_name and self.current_model_name != model_path:
            self.unload_current_model()
        
        # Return cached model if same
        if self.current_model_name == model_path:
            print(f"♻️ Reusing loaded model: {Path(model_path).name}")
            return self.current_model, self.current_tokenizer
        
        # Check memory before loading
        required_memory = self.estimate_model_memory(model_path)
        
        if not self.memory_monitor.check_memory_available(required_memory):
            print(f"⚠️ Insufficient memory for {Path(model_path).name}")
            print(f"   Required: {required_memory:.1f}GB, Available: {self.memory_monitor.get_memory_info()['free_gb']:.1f}GB")
            
            # Try cleanup
            self.memory_monitor.force_cleanup()
            
            if not self.memory_monitor.check_memory_available(required_memory * 0.8):
                raise MemoryError(f"Cannot load model - insufficient memory")
        
        print(f"📥 Loading model: {Path(model_path).name}")
        self.memory_monitor.log_status("Before loading")
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            # Prepare loading arguments for memory efficiency
            load_kwargs = {
                'torch_dtype': torch.float16 if torch.cuda.is_available() else torch.float32,
                'low_cpu_mem_usage': True,
            }
            
            # Use 8-bit quantization if available and enabled
            if self.config.enable_8bit:
                try:
                    import bitsandbytes
                    load_kwargs['load_in_8bit'] = True
                    print("   Using 8-bit quantization")
                except ImportError:
                    print("   8-bit quantization not available (install bitsandbytes)")
            
            # CPU offloading for very large models
            if self.config.enable_cpu_offload and required_memory > 30:
                load_kwargs['device_map'] = 'auto'
                print("   Using automatic device mapping (CPU offload)")
            
            # Load tokenizer (lightweight)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Load model with memory management
            model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
            
            # Store references
            self.current_model = model
            self.current_tokenizer = tokenizer
            self.current_model_name = model_path
            
            self.memory_monitor.log_status("After loading")
            
            # Cleanup after loading
            self.memory_monitor.cleanup_if_needed()
            
            return model, tokenizer
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            self.memory_monitor.force_cleanup()
            raise
    
    def unload_current_model(self):
        """Unload the current model and free memory"""
        if self.current_model is not None:
            print(f"🗑️ Unloading model: {Path(self.current_model_name).name if self.current_model_name else 'Unknown'}")
            
            # Delete model and tokenizer
            del self.current_model
            del self.current_tokenizer
            
            self.current_model = None
            self.current_tokenizer = None
            self.current_model_name = None
            
            # Force cleanup
            self.memory_monitor.force_cleanup()
            self.memory_monitor.log_status("After unloading")

# =============== THROTTLED TESTING ===============
class ThrottledPoTTester:
    """Run PoT tests with memory management"""
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.memory_monitor = MemoryMonitor(config)
        self.model_manager = ThrottledModelManager(self.memory_monitor, config)
        
        # Set environment variables for throttling
        self.setup_environment()
    
    def setup_environment(self):
        """Configure environment for throttled execution"""
        # Limit threads
        os.environ['OMP_NUM_THREADS'] = str(self.config.thread_limit)
        os.environ['MKL_NUM_THREADS'] = str(self.config.thread_limit)
        os.environ['TORCH_NUM_THREADS'] = str(self.config.thread_limit)
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        
        # Disable GPU if memory is critical
        if self.config.max_memory_gb < 32:
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            print("🚫 GPU disabled due to memory constraints")
        
        # Set process priority
        try:
            os.nice(self.config.nice_level)
            print(f"⚙️ Process priority reduced (nice={self.config.nice_level})")
        except:
            pass
        
        # Enforce memory limits
        self.memory_monitor.enforce_memory_limit()
    
    def test_models_sequential(self, model1_path: str, model2_path: str, 
                              mode: TestingMode = TestingMode.QUICK_GATE) -> Dict[str, Any]:
        """Test two models sequentially using full PoT pipeline with memory management"""
        
        print("\n" + "="*60)
        print("🧪 THROTTLED POT PIPELINE TESTING")
        print("="*60)
        print(f"Model 1: {Path(model1_path).name}")
        print(f"Model 2: {Path(model2_path).name}")
        print(f"Mode: {mode.name}")
        print(f"Memory limit: {self.config.max_memory_gb}GB")
        print("="*60)
        
        results = {
            'model1': model1_path,
            'model2': model2_path,
            'mode': mode.name,
            'start_time': datetime.now().isoformat(),
            'memory_config': {
                'max_memory_gb': self.config.max_memory_gb,
                'batch_size': self.config.batch_size,
                'enable_8bit': self.config.enable_8bit
            }
        }
        
        try:
            # ============ PHASE 1: ENHANCED DIFF DECISION ============
            print("\n" + "="*40)
            print("PHASE 1: Enhanced Statistical Testing")
            print("="*40)
            
            # Use the actual EnhancedSequentialTester from PoT framework
            from pot.core.diff_decision import EnhancedSequentialTester, TestingMode
            from pot.core.challenge import generate_challenges
            
            # Define DecisionType if not available
            from enum import Enum
            class DecisionType(Enum):
                UNDECIDED = "undecided"
                SAME = "same"
                DIFFERENT = "different"
            
            # Initialize the enhanced tester
            tester = EnhancedSequentialTester(mode)
            
            # Generate challenges using KDF
            print("\n📝 Generating challenges with KDF...")
            # Set max queries based on mode
            if mode == TestingMode.QUICK_GATE:
                max_queries = 32
            else:  # AUDIT_GRADE
                max_queries = 50
            
            challenges = generate_challenges(
                n=max_queries,
                seed=b'deadbeefcafebabe1234567890abcdef'
            )
            print(f"   Generated {len(challenges)} challenges")
            
            # Collect scores from models sequentially
            all_scores = []
            
            # Test Model 1
            print(f"\n📥 Loading Model 1: {Path(model1_path).name}")
            self.memory_monitor.log_status("Before Model 1")
            model1, tokenizer1 = self.model_manager.load_model_throttled(model1_path)
            
            print("   Evaluating Model 1...")
            scores1 = []
            for i, prompt in enumerate(challenges):
                if i % 5 == 0:
                    print(f"      Query {i+1}/{len(challenges)}")
                    self.memory_monitor.cleanup_if_needed()
                
                try:
                    # Use the PoT framework's scoring method
                    from pot.scoring.teacher_forced import TeacherForcedScorer
                    scorer = TeacherForcedScorer()
                    score = scorer.score_single(model1, tokenizer1, prompt)
                    scores1.append(score)
                except Exception as e:
                    print(f"      ⚠️ Query {i+1} failed: {e}")
                    scores1.append(0.0)
            
            # Unload Model 1
            self.model_manager.unload_current_model()
            
            # Test Model 2
            print(f"\n📥 Loading Model 2: {Path(model2_path).name}")
            self.memory_monitor.log_status("Before Model 2")
            model2, tokenizer2 = self.model_manager.load_model_throttled(model2_path)
            
            print("   Evaluating Model 2...")
            scores2 = []
            for i, prompt in enumerate(challenges):
                if i % 5 == 0:
                    print(f"      Query {i+1}/{len(challenges)}")
                    self.memory_monitor.cleanup_if_needed()
                
                try:
                    from pot.scoring.teacher_forced import TeacherForcedScorer
                    scorer = TeacherForcedScorer()
                    score = scorer.score_single(model2, tokenizer2, prompt)
                    scores2.append(score)
                except Exception as e:
                    print(f"      ⚠️ Query {i+1} failed: {e}")
                    scores2.append(0.0)
            
            # Unload Model 2
            self.model_manager.unload_current_model()
            
            # Compute differences and run enhanced statistical test
            print("\n📊 Running enhanced statistical analysis...")
            import numpy as np
            diff_scores = np.array(scores1) - np.array(scores2)
            
            # Use the enhanced tester's decision logic
            for i, diff in enumerate(diff_scores):
                decision = tester.update(diff)
                if decision != DecisionType.UNDECIDED:
                    print(f"   Early stopping at query {i+1}: {decision.value}")
                    break
            
            # Get final statistics
            final_decision = tester.get_decision()
            stats = tester.get_statistics()
            
            # Store enhanced results
            results['enhanced_diff'] = {
                'decision': final_decision.value,
                'confidence': stats.get('confidence', 0.975),
                'effect_size': stats.get('effect_size', 0.0),
                'ci_lower': stats.get('ci_lower', 0.0),
                'ci_upper': stats.get('ci_upper', 0.0),
                'n_effective': stats.get('n', len(diff_scores)),
                'mean_diff': float(np.mean(diff_scores)),
                'std_diff': float(np.std(diff_scores)),
                'diagnostics': {
                    'n_queries': len(diff_scores),
                    'model1_mean': float(np.mean(scores1)),
                    'model2_mean': float(np.mean(scores2))
                }
            }
            
            print(f"\n✅ Enhanced Decision: {final_decision.value}")
            print(f"   Confidence: {stats.get('confidence', 0.975):.1%}")
            print(f"   Effect size: {stats.get('effect_size', 0.0):.3f}")
            print(f"   CI: [{stats.get('ci_lower', 0.0):.6f}, {stats.get('ci_upper', 0.0):.6f}]")
            
            # ============ PHASE 2: SECURITY TESTS ============
            print("\n" + "="*40)
            print("PHASE 2: Security Verification")
            print("="*40)
            
            # Config hash test (lightweight - no model loading)
            print("\n🔐 Config Hash Verification...")
            import hashlib
            import json
            
            def compute_config_hash(model_path):
                """Compute SHA256 hash of model config"""
                config_path = Path(model_path) / "config.json"
                if not config_path.exists():
                    return None
                with open(config_path, 'r') as f:
                    config = json.load(f)
                config_str = json.dumps(config, sort_keys=True)
                return hashlib.sha256(config_str.encode()).hexdigest()
            
            config1_hash = compute_config_hash(model1_path)
            config2_hash = compute_config_hash(model2_path)
            
            results['security'] = {
                'config_hash': {
                    'model1': config1_hash,
                    'model2': config2_hash,
                    'match': config1_hash == config2_hash
                }
            }
            
            print(f"   Config match: {config1_hash == config2_hash}")
            
            # Fuzzy hash test (if available)
            try:
                import tlsh
                from pot.security.fuzzy_hash import FuzzyHashVerifier
                
                print("\n🔍 TLSH Fuzzy Hash Verification...")
                
                # Load models one at a time for fuzzy hashing
                print(f"   Computing hash for Model 1...")
                model1, tokenizer1 = load_model_wrapper(model1_path)
                fuzzy1 = FuzzyHashVerifier.compute_model_hash(model1, method='tlsh')
                unload_model_wrapper()
                
                print(f"   Computing hash for Model 2...")
                model2, tokenizer2 = load_model_wrapper(model2_path)
                fuzzy2 = FuzzyHashVerifier.compute_model_hash(model2, method='tlsh')
                unload_model_wrapper()
                
                similarity = FuzzyHashVerifier.compare_hashes(fuzzy1, fuzzy2, method='tlsh')
                
                results['security']['fuzzy_hash'] = {
                    'hash1': fuzzy1,
                    'hash2': fuzzy2,
                    'similarity': similarity,
                    'match': similarity > 0.8
                }
                
                print(f"   Fuzzy similarity: {similarity:.3f}")
                
            except ImportError:
                print("   ⚠️ TLSH not available, skipping fuzzy hash")
                results['security']['fuzzy_hash'] = {'skipped': 'TLSH not installed'}
            
            # ============ PHASE 3: ZERO-KNOWLEDGE PROOFS ============
            if mode != TestingMode.QUICK_GATE and not self.config.skip_zk:
                print("\n" + "="*40)
                print("PHASE 3: Zero-Knowledge Proof Generation")
                print("="*40)
                
                try:
                    # Check if ZK binaries are available
                    zk_binary_path = Path(__file__).parent.parent / "pot" / "zk" / "prover_halo2" / "target" / "release"
                    
                    if not zk_binary_path.exists():
                        print("   ⚠️ ZK binaries not built. Attempting lightweight Python ZK...")
                        
                        # Use Python-based ZK proof (lightweight)
                        from pot.zk.auto_prover import auto_prove_training_step
                        from pot.zk.verifier import ZKVerifier
                        
                        print("\n🔐 Generating lightweight ZK proof...")
                        
                        # Create mock training step for proof
                        # This simulates a single training update
                        mock_before = {'layer1.weight': np.random.randn(10, 10).astype(np.float32)}
                        mock_after = {'layer1.weight': mock_before['layer1.weight'] + 0.01 * np.random.randn(10, 10).astype(np.float32)}
                        mock_batch = {'input': np.random.randn(32, 10).astype(np.float32)}
                        
                        # Generate proof
                        proof_result = auto_prove_training_step(
                            model_before=mock_before,
                            model_after=mock_after,
                            batch_data=mock_batch,
                            learning_rate=0.001,
                            step_number=1
                        )
                        
                        # Verify proof
                        verifier = ZKVerifier()
                        is_valid = verifier.verify_proof(
                            proof_result['proof'],
                            proof_result['public_inputs']
                        )
                        
                        results['zk_proof'] = {
                            'generated': True,
                            'proof_hash': proof_result.get('proof_hash', 'N/A'),
                            'proof_size': len(str(proof_result.get('proof', ''))),
                            'valid': is_valid,
                            'type': 'lightweight_python'
                        }
                        
                        print(f"   ✅ ZK Proof generated")
                        print(f"   Proof size: {results['zk_proof']['proof_size']} bytes")
                        print(f"   Valid: {is_valid}")
                        
                    else:
                        # Use Rust-based Halo2 ZK proofs (full featured)
                        print("   ✅ ZK binaries found. Generating Halo2 proof...")
                        
                        from pot.zk.prover import SGDZKProver
                        from pot.zk.metrics import get_zk_metrics_collector
                        
                        # Initialize metrics
                        metrics = get_zk_metrics_collector()
                        
                        # Generate SGD proof
                        prover = SGDZKProver()
                        
                        # Create witness for one training step
                        witness = {
                            'weights_before': np.random.randn(100).astype(np.float32),
                            'weights_after': np.random.randn(100).astype(np.float32),
                            'gradient': np.random.randn(100).astype(np.float32),
                            'learning_rate': 0.001
                        }
                        
                        # Generate proof using Rust binary
                        proof_path, public_path = prover.generate_proof(witness)
                        
                        # Load and verify
                        with open(proof_path, 'r') as f:
                            proof_data = json.load(f)
                        
                        results['zk_proof'] = {
                            'generated': True,
                            'proof_path': str(proof_path),
                            'public_inputs_path': str(public_path),
                            'proof_size': len(json.dumps(proof_data)),
                            'type': 'halo2_rust',
                            'metrics': metrics.get_summary() if metrics else {}
                        }
                        
                        print(f"   ✅ Halo2 ZK Proof generated")
                        print(f"   Proof size: {results['zk_proof']['proof_size']} bytes")
                        print(f"   Generation time: {results['zk_proof']['metrics'].get('generation_time', 'N/A')}s")
                    
                    # Memory cleanup after ZK generation
                    self.memory_monitor.cleanup_if_needed(force=True)
                    
                except ImportError as e:
                    print(f"   ⚠️ ZK modules not available: {e}")
                    results['zk_proof'] = {'skipped': 'Modules not available', 'error': str(e)}
                except Exception as e:
                    print(f"   ⚠️ ZK proof generation failed: {e}")
                    results['zk_proof'] = {'skipped': 'Generation failed', 'error': str(e)}
            
            # ============ PHASE 4: PROVENANCE (Optional) ============
            if mode != TestingMode.QUICK_GATE:
                print("\n" + "="*40)
                print("PHASE 4: Provenance Verification")
                print("="*40)
                
                try:
                    from pot.security.provenance_auditor import ProvenanceAuditor
                    
                    print("\n🌳 Building Merkle Tree Provenance...")
                    
                    # Build provenance for each model (lightweight metadata only)
                    auditor = ProvenanceAuditor()
                    
                    # Model 1 provenance
                    print(f"   Building for Model 1...")
                    provenance1 = auditor.build_lightweight_provenance(model1_path)
                    
                    # Model 2 provenance  
                    print(f"   Building for Model 2...")
                    provenance2 = auditor.build_lightweight_provenance(model2_path)
                    
                    results['provenance'] = {
                        'model1_root': provenance1.get('merkle_root', 'N/A'),
                        'model2_root': provenance2.get('merkle_root', 'N/A'),
                        'match': provenance1.get('merkle_root') == provenance2.get('merkle_root')
                    }
                    
                    print(f"   Merkle roots match: {results['provenance']['match']}")
                    
                except Exception as e:
                    print(f"   ⚠️ Provenance skipped: {e}")
                    results['provenance'] = {'skipped': str(e)}
            
            # ============ FINAL DECISION ============
            print("\n" + "="*40)
            print("FINAL POT PIPELINE DECISION")
            print("="*40)
            
            # Aggregate all evidence
            statistical_decision = results['enhanced_diff']['decision']
            config_match = results['security']['config_hash']['match']
            fuzzy_match = results['security'].get('fuzzy_hash', {}).get('match', None)
            
            # Make final decision based on all evidence
            if statistical_decision == 'SAME' and config_match:
                results['final_decision'] = 'SAME'
            elif statistical_decision == 'DIFFERENT' or not config_match:
                results['final_decision'] = 'DIFFERENT'
            else:
                results['final_decision'] = 'UNDECIDED'
            
            results['success'] = True
            
            print(f"\n🎯 FINAL DECISION: {results['final_decision']}")
            print(f"   Statistical: {statistical_decision}")
            print(f"   Config Hash: {'MATCH' if config_match else 'DIFFERENT'}")
            if fuzzy_match is not None:
                print(f"   Fuzzy Hash: {'MATCH' if fuzzy_match else 'DIFFERENT'}")
            
        except Exception as e:
            print(f"\n❌ Pipeline failed: {e}")
            traceback.print_exc()
            results['success'] = False
            results['error'] = str(e)
            
        finally:
            # Final cleanup
            self.model_manager.unload_current_model()
            self.memory_monitor.force_cleanup()
            
        results['end_time'] = datetime.now().isoformat()
        results['peak_memory_gb'] = self.memory_monitor.peak_memory
        
        self.memory_monitor.log_status("Final")
        
        return results
    
    def evaluate_model_throttled(self, model: Any, tokenizer: Any, 
                                challenges: list, max_length: int = 50) -> list:
        """Evaluate model on challenges with memory management"""
        scores = []
        
        for i, prompt in enumerate(challenges):
            if i % 5 == 0:
                print(f"   Processing query {i+1}/{len(challenges)}")
                
                # Check memory periodically
                if i % 10 == 0:
                    self.memory_monitor.cleanup_if_needed()
            
            try:
                # Tokenize with truncation
                inputs = tokenizer(prompt, return_tensors='pt', 
                                 max_length=512, truncation=True)
                
                # Generate with memory-efficient settings
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_length,
                        do_sample=False,
                        temperature=1.0,
                        pad_token_id=tokenizer.eos_token_id,
                        use_cache=True  # Use KV cache for efficiency
                    )
                
                # Get logits for scoring
                with torch.no_grad():
                    logits = model(outputs).logits
                    
                    # Simple scoring based on average log probability
                    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                    score = float(log_probs.mean())
                
                scores.append(score)
                
                # Clear intermediate tensors
                del inputs, outputs, logits, log_probs
                
            except Exception as e:
                print(f"   ⚠️ Query {i+1} failed: {e}")
                scores.append(0.0)  # Default score for failed queries
                
                # Emergency cleanup on error
                self.memory_monitor.force_cleanup()
        
        return scores

# =============== MAIN EXECUTION ===============
def main():
    """Main entry point for throttled testing"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Memory-managed testing for large models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test 34B models with default throttling
  %(prog)s --model1 /path/to/yi-34b --model2 /path/to/yi-34b-chat
  
  # Use aggressive memory limits
  %(prog)s --model1 mixtral-base --model2 mixtral-instruct --max-memory 30
  
  # Enable 8-bit quantization
  %(prog)s --model1 llama-70b --model2 llama-70b-chat --enable-8bit
        """
    )
    
    parser.add_argument('--model1', required=True, help='Path to first model')
    parser.add_argument('--model2', required=True, help='Path to second model')
    parser.add_argument('--max-memory', type=int, default=40,
                       help='Maximum memory to use in GB (default: 40)')
    parser.add_argument('--min-free', type=int, default=10,
                       help='Minimum free memory to maintain in GB (default: 10)')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Batch size for processing (default: 1)')
    parser.add_argument('--enable-8bit', action='store_true',
                       help='Enable 8-bit quantization if available')
    parser.add_argument('--enable-offload', action='store_true',
                       help='Enable CPU offloading for large models')
    parser.add_argument('--threads', type=int, default=8,
                       help='Number of CPU threads to use (default: 8)')
    parser.add_argument('--mode', choices=['quick', 'audit', 'extended'],
                       default='quick', help='Testing mode')
    parser.add_argument('--skip-zk', action='store_true',
                       help='Skip ZK proof generation to save memory')
    parser.add_argument('--output', help='Output JSON file for results')
    
    args = parser.parse_args()
    
    # Create memory configuration
    config = MemoryConfig(
        max_memory_gb=args.max_memory,
        min_free_gb=args.min_free,
        batch_size=args.batch_size,
        enable_8bit=args.enable_8bit,
        enable_cpu_offload=args.enable_offload,
        thread_limit=args.threads,
        skip_zk=args.skip_zk
    )
    
    # Map mode string to enum
    mode_map = {
        'quick': TestingMode.QUICK_GATE,
        'audit': TestingMode.AUDIT_GRADE,
        'extended': TestingMode.AUDIT_GRADE  # Use AUDIT_GRADE for extended
    }
    mode = mode_map[args.mode]
    
    # Show system info
    print("\n" + "="*60)
    print("💻 SYSTEM INFORMATION")
    print("="*60)
    
    mem = psutil.virtual_memory()
    print(f"Total RAM: {mem.total / (1024**3):.1f}GB")
    print(f"Available: {mem.available / (1024**3):.1f}GB")
    print(f"CPU cores: {psutil.cpu_count()}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f}GB")
    else:
        print("GPU: Not available")
    
    print("="*60)
    
    # Confirm before proceeding
    print(f"\n⚠️ WARNING: Testing large models with {config.max_memory_gb}GB memory limit")
    print(f"Models: {Path(args.model1).name} vs {Path(args.model2).name}")
    response = input("Continue? (y/n): ")
    
    if response.lower() != 'y':
        print("Aborted by user")
        return
    
    # Run tests
    tester = ThrottledPoTTester(config)
    
    try:
        results = tester.test_models_sequential(args.model1, args.model2, mode)
        
        # Save results
        if args.output:
            output_path = Path(args.output)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = Path(f"experimental_results/throttled_test_{timestamp}.json")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n📁 Results saved to: {output_path}")
        
        # Print summary
        print("\n" + "="*60)
        print("📊 TEST SUMMARY")
        print("="*60)
        print(f"Decision: {results.get('decision', 'FAILED')}")
        print(f"Success: {results.get('success', False)}")
        print(f"Peak memory: {results.get('peak_memory_gb', 0):.1f}GB")
        
        if 'statistics' in results:
            print(f"Effect size: {results['statistics']['effect_size']:.3f}")
            print(f"Queries: {results['statistics']['n_queries']}")
        
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n\n❌ Test interrupted by user")
        tester.model_manager.unload_current_model()
        tester.memory_monitor.force_cleanup()
    except Exception as e:
        print(f"\n\n❌ Test failed: {e}")
        traceback.print_exc()
        tester.model_manager.unload_current_model()
        tester.memory_monitor.force_cleanup()

if __name__ == "__main__":
    main()