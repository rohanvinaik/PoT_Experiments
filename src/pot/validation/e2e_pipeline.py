"""
End-to-End Validation Pipeline for Proof-of-Training Framework

This module orchestrates the complete verification flow from pre-commitment
through evidence bundle generation and optional ZK proof creation.
"""

import os
import time
import json
import hashlib
import tracemalloc
import psutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
import numpy as np# Import core PoT modules with proper fallback handling
try:
    # Try absolute imports from src path first
    from src.pot.core.diff_decision import (
        TestingMode,
        EnhancedSequentialTester,
        DiffDecisionConfig,
        DifferenceVerifier,
        create_enhanced_verifier
    )
    from src.pot.core.challenge import (
        ChallengeConfig, 
        generate_challenges,
        Challenge
    )
    from src.pot.lm.verifier import LMVerifier
    from src.pot.lm.models import LM
    from src.pot.validation.reporting import ReportGenerator
    from src.pot.experiments.metrics_calculator import MetricsCalculator
    from src.pot.experiments.result_validator import ValidationReport
except ImportError:
    try:
        # Fallback to pot.* imports for different package layouts
        from pot.core.diff_decision import (
            TestingMode,
            EnhancedSequentialTester,
            DiffDecisionConfig,
            DifferenceVerifier,
            create_enhanced_verifier
        )
        from pot.core.challenge import (
            ChallengeConfig,
            generate_challenges,
            Challenge
        )
        from pot.lm.verifier import LMVerifier
        from pot.lm.models import LM
        from pot.validation.reporting import ReportGenerator
        from pot.experiments.metrics_calculator import MetricsCalculator
        from pot.experiments.result_validator import ValidationReport
    except ImportError:
        # If both fail, provide informative error
        import sys
        print("Error: Unable to import PoT modules. Please ensure the package is properly installed.", file=sys.stderr)
        print("Try running: pip install -e . from the project root directory.", file=sys.stderr)
        raise  # Fix: Changed from "raises import LM" to just "raise"


class VerificationMode(Enum):
    """Verification mode for the pipeline"""
    LOCAL_WEIGHTS = "local_weights"
    API_BLACK_BOX = "api_black_box"
    HYBRID = "hybrid"


class PipelineStage(Enum):
    """Pipeline execution stages"""
    INITIALIZATION = "initialization"
    PRE_COMMIT = "pre_commit"
    CHALLENGE_GENERATION = "challenge_generation"
    MODEL_LOADING = "model_loading"
    VERIFICATION = "verification"
    EVIDENCE_GENERATION = "evidence_generation"
    REPORTING = "reporting"
    ZK_PROOF = "zk_proof"
    COMPLETED = "completed"


@dataclass
class StageMetrics:
    """Metrics for a single pipeline stage"""
    stage: PipelineStage
    start_time: float
    end_time: float = 0.0
    memory_peak_mb: float = 0.0
    memory_delta_mb: float = 0.0
    cpu_percent: float = 0.0
    query_count: int = 0
    ci_progression: List[Tuple[float, float]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time if self.end_time else 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'stage': self.stage.value,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration': self.duration,
            'memory_peak_mb': self.memory_peak_mb,
            'memory_delta_mb': self.memory_delta_mb,
            'cpu_percent': self.cpu_percent,
            'query_count': self.query_count,
            'ci_progression': self.ci_progression,
            'errors': self.errors,
            'metadata': self.metadata
        }


@dataclass
class PipelineConfig:
    """Configuration for the validation pipeline"""
    testing_mode: TestingMode = TestingMode.AUDIT_GRADE
    verification_mode: VerificationMode = VerificationMode.LOCAL_WEIGHTS
    enable_zk_proof: bool = False
    enable_memory_tracking: bool = True
    enable_cpu_tracking: bool = True
    output_dir: Path = Path("outputs/validation_reports")
    dry_run: bool = False
    benchmark_mode: bool = False
    max_queries: int = 400
    hmac_key: Optional[str] = None
    verbose: bool = True
    
    def __post_init__(self):
        """Ensure output directory exists"""
        self.output_dir = Path(self.output_dir)
        
        # Generate HMAC key if not provided
        if not self.hmac_key:
            self.hmac_key = hashlib.sha256(os.urandom(32)).hexdigest()


class PipelineOrchestrator:
    """
    Orchestrates the complete verification flow for PoT framework
    
    This class manages the full verification lifecycle, tracking metrics
    at each stage and generating comprehensive reports.
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize the pipeline orchestrator
        
        Args:
            config: Pipeline configuration, uses defaults if not provided
        """
        self.config = config or PipelineConfig()
        self.current_stage = PipelineStage.INITIALIZATION
        self.stage_metrics: Dict[PipelineStage, StageMetrics] = {}
        self.run_id = f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.evidence_bundle = {}
        self.process = psutil.Process()
        
        # Initialize memory tracking if enabled
        if self.config.enable_memory_tracking:
            tracemalloc.start()
    
    def _log(self, message: str, level: str = "INFO"):
        """Internal logging method"""
        if self.config.verbose:
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            print(f"[{timestamp}] [{level}] {message}")
    
    def _start_stage(self, stage: PipelineStage) -> StageMetrics:
        """Start tracking a new pipeline stage"""
        self._log(f"Starting stage: {stage.value}")
        self.current_stage = stage
        
        metrics = StageMetrics(
            stage=stage,
            start_time=time.time()
        )
        
        if self.config.enable_memory_tracking:
            metrics.metadata['memory_start'] = self.process.memory_info().rss / 1024 / 1024
        
        return metrics
    
    def _end_stage(self, metrics: StageMetrics):
        """End tracking for the current stage"""
        metrics.end_time = time.time()
        
        if self.config.enable_memory_tracking:
            memory_end = self.process.memory_info().rss / 1024 / 1024
            metrics.memory_peak_mb = memory_end
            metrics.memory_delta_mb = memory_end - metrics.metadata.get('memory_start', 0)
        
        if self.config.enable_cpu_tracking:
            metrics.cpu_percent = self.process.cpu_percent()
        
        self.stage_metrics[metrics.stage] = metrics
        self._log(f"Completed stage: {metrics.stage.value} (duration: {metrics.duration:.2f}s)")
    
    def pre_commit_challenges(self, n_challenges: int = 32) -> Dict[str, Any]:
        """
        Pre-commit challenge generation phase
        
        Args:
            n_challenges: Number of challenges to generate
            
        Returns:
            Dictionary containing challenge seeds and commitment
        """
        metrics = self._start_stage(PipelineStage.PRE_COMMIT)
        
        try:
            # Generate challenge seeds using HMAC
            seeds = []
            for i in range(n_challenges):
                seed = compute_hmac_seed(self.config.hmac_key, self.run_id, i)
                seeds.append(seed)
            
            # Create commitment
            commitment = hashlib.sha256(
                json.dumps(seeds, sort_keys=True).encode()
            ).hexdigest()
            
            result = {
                'n_challenges': n_challenges,
                'hmac_key': self.config.hmac_key,
                'run_id': self.run_id,
                'commitment': commitment,
                'seeds': seeds if not self.config.dry_run else seeds[:2]  # Limit in dry-run
            }
            
            metrics.metadata.update(result)
            self.evidence_bundle['pre_commit'] = result
            
            return result
            
        except Exception as e:
            metrics.errors.append(str(e))
            raise
        finally:
            self._end_stage(metrics)
    
    def generate_challenges(self, seeds: List[str]) -> List[Dict[str, Any]]:
        """Generate cryptographic challenges from the commit seeds"""

        metrics = self._start_stage(PipelineStage.CHALLENGE_GENERATION)

        try:
            # Derive session nonce deterministically from run_id
            session_nonce = hashlib.sha256(self.run_id.encode()).hexdigest()[:32]

            cfg = ChallengeConfig(
                master_key_hex=self.config.hmac_key,
                session_nonce_hex=session_nonce,
                n=len(seeds),
                family="lm:templates",
                params={},
            )

            generated = generate_challenges(cfg)# Build challenges list with seeds for full traceability
            challenges = []

            # Ensure we have matching seeds and challenges
            if seeds and len(seeds) >= len(generated["challenges"]):
                # Use zip for clean iteration when we have enough seeds
                for seed, ch in zip(seeds, generated["challenges"]):
                    challenges.append({
                        "id": ch.challenge_id,
                        "seed": seed,
                        "prompt": ch.parameters.get("prompt", ""),
                        "metadata": ch.parameters,
                    })
            else:
                # Fallback: handle case where seeds might be missing or insufficient
                for i, ch in enumerate(generated["challenges"]):
                    challenge_dict = {
                        "id": ch.challenge_id,
                        "prompt": ch.parameters.get("prompt", ""),
                        "metadata": ch.parameters,
                    }
                    # Add seed if available
                    if seeds and i < len(seeds):
                        challenge_dict["seed"] = seeds[i]
                    challenges.append(challenge_dict)

            metrics.metadata["n_challenges"] = len(challenges)
            self.evidence_bundle["challenges"] = challenges

            return challenges

        except Exception as e:
            metrics.errors.append(str(e))
            raise
        finally:
            self._end_stage(metrics)
    
    def load_models(
        self,
        ref_model_path: str,
        cand_model_path: str
    ) -> Tuple[Any, Any]:
        """
        Load reference and candidate models
        
        Args:
            ref_model_path: Path or identifier for reference model
            cand_model_path: Path or identifier for candidate model
            
        Returns:
            Tuple of (reference_model, candidate_model)
        """
        metrics = self._start_stage(PipelineStage.MODEL_LOADING)
        
        try:
            report: Optional[Dict[str, Any]] = None
            if self.config.dry_run:
                # Return simple mock objects in dry run mode
                self._log("Dry run: Using mock models")
                ref_model = type("MockModel", (), {"name": ref_model_path})()
                cand_model = type("MockModel", (), {"name": cand_model_path})()
            else:
                if self.config.verification_mode == VerificationMode.LOCAL_WEIGHTS:
                    # Load real local models via LM wrapper
                    self._log(
                        f"Loading local models: {ref_model_path}, {cand_model_path}"
                    )
                    ref_model = LM(ref_model_path, device="cpu")
                    cand_model = LM(cand_model_path, device="cpu")
                else:
                    # API mode - create simple client objects
                    self._log("Connecting to API endpoints")
                    ref_model = type("APIModel", (), {"endpoint": ref_model_path})()
                    cand_model = type("APIModel", (), {"endpoint": cand_model_path})()
            
            metrics.metadata['ref_model'] = str(ref_model_path)
            metrics.metadata['cand_model'] = str(cand_model_path)
            
            return ref_model, cand_model
            
        except Exception as e:
            metrics.errors.append(str(e))
            raise
        finally:
            self._end_stage(metrics)
    
    def run_verification(
        self,
        ref_model: Any,
        cand_model: Any,
        challenges: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Run the main verification process
        
        Args:
            ref_model: Reference model
            cand_model: Candidate model
            challenges: List of challenges to run
            
        Returns:
            Verification results dictionary
        """
        metrics = self._start_stage(PipelineStage.VERIFICATION)
        
        try:
            report: Optional[Dict[str, Any]] = None
            if self.config.dry_run:
                # Simulate verification for dry run
                self._log("Dry run: Simulating verification")
                time.sleep(0.5)
                result = {
                    "decision": "SAME",
                    "confidence": 0.99,
                    "n_queries": min(10, len(challenges)),
                    "ci_progression": [],
                    "effect_size": 0.0,
                }
            else:
                # Prepare prompt generator from challenge list
                from itertools import cycle
                prompts = [c["prompt"] for c in challenges]
                prompt_cycle = cycle(prompts)

                def prompt_generator() -> str:
                    return next(prompt_cycle)

                # Scoring function using fuzzy distance between model outputs
                lm_verifier = LMVerifier(ref_model, delta=0.01, use_sequential=False)

                def score_fn(ref, cand, prompt, K=32):
                    ref_out = ref.generate(prompt, max_new_tokens=64)
                    cand_out = cand.generate(prompt, max_new_tokens=64)
                    return lm_verifier.compute_output_distance(ref_out, cand_out, method="fuzzy")

                verifier = create_enhanced_verifier(
                    score_fn,
                    prompt_generator,
                    mode=self.config.testing_mode,
                    n_max=min(len(prompts), self.config.max_queries),
                )
                report = verifier.verify_difference(ref_model, cand_model, verbose=self.config.verbose)

                result = {
                    "decision": report["results"]["decision"],
                    "confidence": 1.0 - verifier.cfg.alpha,
                    "n_queries": report["results"]["n_used"],
                    "ci_progression": [],
                    "effect_size": report["results"]["mean"],
                }

            metrics.query_count = result["n_queries"]
            metrics.ci_progression = result.get("ci_progression", [])
            metrics.metadata["decision"] = result["decision"]

            # Store detailed verification info in evidence bundle
            self.evidence_bundle["verification"] = report if report is not None else result

            return result

        except Exception as e:
            metrics.errors.append(str(e))
            raise
        finally:
            self._end_stage(metrics)
    
    def generate_evidence_bundle(self) -> Dict[str, Any]:
        """
        Generate comprehensive evidence bundle
        
        Returns:
            Evidence bundle dictionary
        """
        metrics = self._start_stage(PipelineStage.EVIDENCE_GENERATION)
        
        try:
            # Compile all evidence
            # Convert config to dictionary, handling Path objects
            config_dict = {}
            if hasattr(self.config, '__dataclass_fields__'):
                for field_name in self.config.__dataclass_fields__:
                    value = getattr(self.config, field_name)
                    if isinstance(value, Path):
                        config_dict[field_name] = str(value)
                    elif hasattr(value, 'value'):  # Enum
                        config_dict[field_name] = value.value
                    else:
                        config_dict[field_name] = value
            else:
                config_dict = {'config': str(self.config)}
            
            bundle = {
                'run_id': self.run_id,
                'timestamp': datetime.now().isoformat(),
                'config': config_dict,
                'pre_commit': self.evidence_bundle.get('pre_commit', {}),
                'challenges': self.evidence_bundle.get('challenges', []),
                'verification': self.evidence_bundle.get('verification', {}),
                'metrics': {
                    stage.value: metrics.to_dict() 
                    for stage, metrics in self.stage_metrics.items()
                },
                'environment': {
                    'platform': os.uname().sysname if hasattr(os, 'uname') else 'unknown',
                    'python_version': '.'.join(map(str, os.sys.version_info[:3])),
                    'cpu_count': psutil.cpu_count(),
                    'memory_total_gb': psutil.virtual_memory().total / 1024 / 1024 / 1024
                }
            }
            
            # Calculate bundle hash
            bundle_json = json.dumps(bundle, sort_keys=True, indent=2)
            bundle['hash'] = hashlib.sha256(bundle_json.encode()).hexdigest()
            
            # Save to file
            bundle_path = self.config.output_dir / f"evidence_bundle_{self.run_id}.json"
            with open(bundle_path, 'w') as f:
                json.dump(bundle, f, indent=2)
            
            self._log(f"Evidence bundle saved to: {bundle_path}")
            metrics.metadata['bundle_path'] = str(bundle_path)
            metrics.metadata['bundle_hash'] = bundle['hash']
            
            return bundle
            
        except Exception as e:
            metrics.errors.append(str(e))
            raise
        finally:
            self._end_stage(metrics)
    
    def generate_zk_proof(self, evidence_bundle: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Generate ZK proof for the verification process
        
        Args:
            evidence_bundle: Evidence bundle to prove
            
        Returns:
            ZK proof dictionary or None if disabled
        """
        if not self.config.enable_zk_proof:
            self._log("ZK proof generation disabled")
            return None
        
        metrics = self._start_stage(PipelineStage.ZK_PROOF)
        
        try:
            if self.config.dry_run:
                # Mock ZK proof for dry run
                self._log("Dry run: Generating mock ZK proof")
                proof = {
                    'proof_type': 'halo2',
                    'circuit': 'verification_proof',
                    'public_inputs': {
                        'bundle_hash': evidence_bundle['hash'],
                        'decision': evidence_bundle['verification']['decision'],
                        'n_queries': evidence_bundle['verification']['n_queries']
                    },
                    'proof': 'mock_proof_' + hashlib.sha256(
                        json.dumps(evidence_bundle, sort_keys=True).encode()
                    ).hexdigest()[:32],
                    'verification_key': 'mock_vk_' + os.urandom(16).hex()
                }
            else:
                # Generate actual ZK proof
                try:
                    from ..zk.auto_prover import AutoProver
                    prover = AutoProver()
                    proof = prover.generate_verification_proof(evidence_bundle)
                except ImportError:
                    self._log("ZK module not available, skipping proof generation")
                    return None
            
            # Save proof
            proof_path = self.config.output_dir / f"zk_proof_{self.run_id}.json"
            with open(proof_path, 'w') as f:
                json.dump(proof, f, indent=2)
            
            self._log(f"ZK proof saved to: {proof_path}")
            metrics.metadata['proof_path'] = str(proof_path)
            
            return proof
            
        except Exception as e:
            metrics.errors.append(str(e))
            self._log(f"ZK proof generation failed: {e}", level="WARNING")
            return None
        finally:
            self._end_stage(metrics)
    
    def run_complete_pipeline(
        self,
        ref_model_path: str,
        cand_model_path: str,
        n_challenges: int = 32
    ) -> Dict[str, Any]:
        """
        Run the complete end-to-end validation pipeline
        
        Args:
            ref_model_path: Path or identifier for reference model
            cand_model_path: Path or identifier for candidate model
            n_challenges: Number of challenges to generate
            
        Returns:
            Complete pipeline results dictionary
        """
        self._log(f"Starting E2E validation pipeline (run_id: {self.run_id})")
        self._log(f"Configuration: {self.config.testing_mode}, {self.config.verification_mode}")
        
        try:
            # Stage 1: Pre-commit challenges
            pre_commit = self.pre_commit_challenges(n_challenges)
            
            # Stage 2: Generate challenges
            challenges = self.generate_challenges(pre_commit['seeds'])
            
            # Stage 3: Load models
            ref_model, cand_model = self.load_models(ref_model_path, cand_model_path)
            
            # Stage 4: Run verification
            verification = self.run_verification(ref_model, cand_model, challenges)
            
            # Stage 5: Generate evidence bundle
            evidence_bundle = self.generate_evidence_bundle()
            
            # Stage 6: Generate ZK proof (optional)
            zk_proof = self.generate_zk_proof(evidence_bundle)
            
            # Mark pipeline as completed
            self.current_stage = PipelineStage.COMPLETED
            
            # Compile final results
            results = {
                'run_id': self.run_id,
                'success': True,
                'decision': verification['decision'],
                'confidence': verification.get('confidence', 0),
                'n_queries': verification['n_queries'],
                'evidence_bundle_path': str(self.config.output_dir / f"evidence_bundle_{self.run_id}.json"),
                'zk_proof_path': str(self.config.output_dir / f"zk_proof_{self.run_id}.json") if zk_proof else None,
                'total_duration': sum(m.duration for m in self.stage_metrics.values()),
                'peak_memory_mb': max((m.memory_peak_mb for m in self.stage_metrics.values()), default=0),
                'stage_metrics': {
                    stage.value: metrics.to_dict()
                    for stage, metrics in self.stage_metrics.items()
                }
            }
            
            # Save final results
            results_path = self.config.output_dir / f"pipeline_results_{self.run_id}.json"
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            # Generate HTML report
            try:
                report_generator = ReportGenerator(self.config.output_dir)
                report_path = report_generator.generate_html_report(
                    pipeline_results=results,
                    evidence_bundle=evidence_bundle
                )
                results['html_report_path'] = str(report_path)
                self._log(f"HTML report generated: {report_path}")
            except Exception as e:
                self._log(f"Warning: Failed to generate HTML report: {e}", level="WARNING")
            
            self._log(f"Pipeline completed successfully!")
            self._log(f"Decision: {results['decision']} (confidence: {results['confidence']:.3f})")
            self._log(f"Results saved to: {results_path}")
            
            return results
            
        except Exception as e:
            self._log(f"Pipeline failed: {e}", level="ERROR")
            error_results = {
                'run_id': self.run_id,
                'success': False,
                'error': str(e),
                'stage_failed': self.current_stage.value,
                'stage_metrics': {
                    stage.value: metrics.to_dict()
                    for stage, metrics in self.stage_metrics.items()
                }
            }
            
            # Save error results
            error_path = self.config.output_dir / f"pipeline_error_{self.run_id}.json"
            with open(error_path, 'w') as f:
                json.dump(error_results, f, indent=2)
            
            # Generate summary report even for errors
            try:
                summary_path = self.config.output_dir / f"summary_{self.run_id}.json"
                summary = {
                    'run_id': self.run_id,
                    'status': 'failed',
                    'error': str(e),
                    'stage_failed': self.current_stage.value,
                    'timestamp': datetime.now().isoformat()
                }
                with open(summary_path, 'w') as f:
                    json.dump(summary, f, indent=2)
            except:
                pass  # Don't fail on summary generation
            
            raise
        
        finally:
            # Clean up memory tracking
            if self.config.enable_memory_tracking:
                tracemalloc.stop()
    
    def benchmark_pipeline(
        self,
        ref_model_path: str,
        cand_model_path: str,
        n_runs: int = 3
    ) -> Dict[str, Any]:
        """
        Run pipeline multiple times for benchmarking
        
        Args:
            ref_model_path: Path or identifier for reference model
            cand_model_path: Path or identifier for candidate model
            n_runs: Number of benchmark runs
            
        Returns:
            Benchmark results dictionary
        """
        self._log(f"Starting benchmark mode ({n_runs} runs)")
        
        benchmark_results = []
        
        for i in range(n_runs):
            self._log(f"Benchmark run {i+1}/{n_runs}")
            # Reset for new run
            self.run_id = f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}"
            self.stage_metrics = {}
            self.evidence_bundle = {}
            
            try:
                result = self.run_complete_pipeline(ref_model_path, cand_model_path)
                benchmark_results.append(result)
            except Exception as e:
                self._log(f"Benchmark run {i+1} failed: {e}", level="WARNING")
                benchmark_results.append({'error': str(e)})
        
        # Calculate aggregate statistics
        successful_runs = [r for r in benchmark_results if r.get('success')]
        
        if successful_runs:
            avg_duration = np.mean([r['total_duration'] for r in successful_runs])
            avg_memory = np.mean([r['peak_memory_mb'] for r in successful_runs])
            avg_queries = np.mean([r['n_queries'] for r in successful_runs])
            
            benchmark_summary = {
                'n_runs': n_runs,
                'n_successful': len(successful_runs),
                'avg_duration_seconds': avg_duration,
                'avg_peak_memory_mb': avg_memory,
                'avg_queries': avg_queries,
                'individual_runs': benchmark_results
            }
        else:
            benchmark_summary = {
                'n_runs': n_runs,
                'n_successful': 0,
                'all_failed': True,
                'individual_runs': benchmark_results
            }
        
        # Save benchmark results
        benchmark_path = self.config.output_dir / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(benchmark_path, 'w') as f:
            json.dump(benchmark_summary, f, indent=2)
        
        self._log(f"Benchmark results saved to: {benchmark_path}")
        
        return benchmark_summary


# Helper function for compute_hmac_seed if not available
def compute_hmac_seed(key: str, run_id: str, index: int) -> str:
    """Compute HMAC seed for challenge generation"""
    import hmac
    message = f"{run_id}_{index}".encode()
    return hmac.new(key.encode(), message, hashlib.sha256).hexdigest()


if __name__ == "__main__":
    # Example usage
    config = PipelineConfig(
        testing_mode=TestingMode.QUICK_GATE,
        verification_mode=VerificationMode.LOCAL_WEIGHTS,
        dry_run=True,
        verbose=True
    )
    
    orchestrator = PipelineOrchestrator(config)
    results = orchestrator.run_complete_pipeline(
        ref_model_path="gpt2",
        cand_model_path="distilgpt2",
        n_challenges=10
    )
    
    print(f"\nPipeline Results:")
    print(f"  Decision: {results['decision']}")
    print(f"  Total Duration: {results['total_duration']:.2f}s")
    print(f"  Peak Memory: {results['peak_memory_mb']:.2f} MB")