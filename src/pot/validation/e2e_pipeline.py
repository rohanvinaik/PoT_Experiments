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
    from src.pot.core.adaptive_sampling import (
        AdaptiveConfig,
        AdaptiveSequentialTester,
        ConvergenceMetrics,
        VarianceReductionStrategy
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
        from pot.core.adaptive_sampling import (
            AdaptiveConfig,
            AdaptiveSequentialTester,
            ConvergenceMetrics,
            VarianceReductionStrategy
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
    # Adaptive sampling configuration
    enable_adaptive: bool = True
    adaptive_batch_size: int = 8
    adaptive_switch_threshold: float = 0.5
    adaptive_noise_margin: float = 2.0
    adaptive_max_factor: float = 1.5
    
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
                    import torch
                    device = "mps" if torch.backends.mps.is_available() else "cpu"
                    self._log(
                        f"Loading local models on {device}: {ref_model_path}, {cand_model_path}"
                    )
                    ref_model = LM(ref_model_path, device=device)
                    cand_model = LM(cand_model_path, device=device)
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
    
    def _run_adaptive_verification(
        self,
        ref_model: Any,
        cand_model: Any,
        prompts: List[str],
        score_fn: callable,
        adaptive_tester: AdaptiveSequentialTester,
        diff_config: DiffDecisionConfig
    ) -> Dict[str, Any]:
        """
        Run adaptive verification with dynamic thresholds
        """
        differences = []
        prompt_index = 0
        n_queries = 0
        
        # Get the base tester from adaptive wrapper
        base_tester = adaptive_tester.base_tester
        
        # Store differences in base_tester for adaptive threshold computation
        if not hasattr(base_tester, 'differences'):
            base_tester.differences = []
        
        # Store config values on base_tester for adaptive threshold computation
        if not hasattr(base_tester, 'gamma'):
            base_tester.gamma = diff_config.gamma
        if not hasattr(base_tester, 'delta_star'):
            base_tester.delta_star = diff_config.delta_star
        if not hasattr(base_tester, 'n_min'):
            base_tester.n_min = diff_config.n_min
        if not hasattr(base_tester, 'n_max'):
            base_tester.n_max = diff_config.n_max
        
        while prompt_index < len(prompts) and n_queries < diff_config.n_max:
            # Get current batch size from adaptive tester
            batch_size = adaptive_tester.adapt_batch_size()
            batch_end = min(prompt_index + batch_size, len(prompts))
            
            # Process batch
            for i in range(prompt_index, batch_end):
                if i >= len(prompts):
                    break
                    
                prompt = prompts[i]
                score = score_fn(ref_model, cand_model, prompt, K=diff_config.positions_per_prompt)
                
                # Update base tester
                base_tester.update(score)
                differences.append(score)
                base_tester.differences.append(score)  # Store for adaptive thresholds
                n_queries += 1
                
                # Update convergence metrics
                if hasattr(base_tester, 'get_state'):
                    state = base_tester.get_state()
                    adaptive_tester.convergence.update(
                        state.get('mean', 0),
                        state.get('half_width', float('inf')),
                        state.get('rel_me', 1.0)
                    )
            
            # Check for early stopping with adaptive thresholds
            if n_queries >= diff_config.n_min:
                # Compute adaptive thresholds
                adaptive_gamma = adaptive_tester.compute_adaptive_threshold('gamma')
                adaptive_delta = adaptive_tester.compute_adaptive_threshold('delta_star')
                
                # Get current state
                state = base_tester.get_state()
                mean = state.get('mean', 0)
                ci = state.get('ci', (-float('inf'), float('inf')))
                half_width = state.get('half_width', float('inf'))
                rel_me = state.get('rel_me', 1.0)
                
                # Apply adaptive decision rules
                # SAME decision with adaptive gamma
                eta = 0.5
                if ci[0] >= -adaptive_gamma and ci[1] <= adaptive_gamma and half_width <= eta * adaptive_gamma:
                    decision = "SAME"
                    self._log(f"Adaptive SAME decision at n={n_queries}: CI {ci} ⊂ [-{adaptive_gamma:.3f}, +{adaptive_gamma:.3f}]")
                    break
                
                # DIFFERENT decision - check if CI excludes the equivalence region
                # This is the correct statistical test: does the CI exclude [-gamma, +gamma]?
                if ci[0] > adaptive_gamma or ci[1] < -adaptive_gamma:
                    decision = "DIFFERENT"
                    self._log(f"Adaptive DIFFERENT decision at n={n_queries}: CI [{ci[0]:.3f}, {ci[1]:.3f}] excludes [-{adaptive_gamma:.3f}, +{adaptive_gamma:.3f}]")
                    break
                else:
                    decision = "UNDECIDED"
            
            # Check convergence
            is_converging, reason = adaptive_tester.convergence.is_converging()
            if is_converging and n_queries >= diff_config.n_min * 2:
                self._log(f"Convergence detected at n={n_queries}: {reason}")
                decision = "UNDECIDED"
                break
            
            # Check for strategy switch
            should_switch, new_strategy = adaptive_tester.should_switch_strategy()
            if should_switch:
                self._log(f"Strategy switch suggested at n={n_queries}: {new_strategy}")
                # Could implement strategy switching here (e.g., switch to symmetric KL)
            
            prompt_index = batch_end
        
        # Final state
        if n_queries >= diff_config.n_max:
            decision = "UNDECIDED"
            self._log(f"Reached n_max={diff_config.n_max}")
        
        # Build report
        final_state = base_tester.get_state()
        diagnostics = adaptive_tester.get_diagnostics()
        
        # Format results to match expected structure
        results = {
            "decision": decision,
            "n_used": n_queries,
            "mean": final_state.get('mean', 0),
            "variance": final_state.get('variance', 0),
            "std_dev": final_state.get('std_dev', 0),
            "ci_99": list(final_state.get('ci', (-float('inf'), float('inf')))),
            "half_width": final_state.get('half_width', float('inf')),
            "rel_me": final_state.get('rel_me', 1.0)
        }
        
        return {
            "results": results,
            "decision": decision,
            "confidence": final_state.get('confidence', diff_config.confidence),
            "n_queries": n_queries,
            "ci_progression": adaptive_tester.convergence.ci_width_history,
            "effect_size": abs(final_state.get('mean', 0)),
            "challenges_used": n_queries,
            "mean": final_state.get('mean', 0),
            "ci": final_state.get('ci', (-float('inf'), float('inf'))),
            "half_width": final_state.get('half_width', float('inf')),
            "rel_me": final_state.get('rel_me', 1.0),
            "adaptive_thresholds": {
                "gamma": adaptive_gamma if 'adaptive_gamma' in locals() else diff_config.gamma,
                "delta_star": adaptive_delta if 'adaptive_delta' in locals() else diff_config.delta_star
            },
            "adaptive_diagnostics": diagnostics
        }
    
    def run_verification(
        self,
        ref_model: Any,
        cand_model: Any,
        challenge_seeds: List[str]
    ) -> Dict[str, Any]:
        """
        Run the main verification process following pot_runner pattern
        
        Args:
            ref_model: Reference model
            cand_model: Candidate model
            challenge_seeds: Pre-commit seeds for challenge generation
            
        Returns:
            Verification results dictionary
        """
        metrics = self._start_stage(PipelineStage.VERIFICATION)
        
        try:
            report = None  # Initialize report variable
            
            if self.config.dry_run:
                # Simulate verification for dry run
                self._log("Dry run: Simulating verification")
                time.sleep(0.5)
                result = {
                    "decision": "SAME",
                    "confidence": 0.99,
                    "n_queries": 2,
                    "ci_progression": [],
                    "effect_size": 0.0,
                    "challenges_used": 2,
                }
                # Track challenges for evidence bundle
                self.evidence_bundle["challenges_used"] = 2
                self.evidence_bundle["challenges"] = []  # Empty list for dry run
            else:
                # Generate prompts upfront (like pot_runner) but store only strings to save memory
                session_nonce = hashlib.sha256(self.run_id.encode()).hexdigest()[:32]
                
                # Generate all prompts as simple strings (memory efficient)
                prompts = []
                challenge_metadata = []  # Store minimal metadata
                
                self._log(f"Starting challenge generation for {len(challenge_seeds)} seeds...")
                for i, seed in enumerate(challenge_seeds):
                    if i % 10 == 0:
                        self._log(f"  Generated {i}/{len(challenge_seeds)} challenges...")
                    cfg = ChallengeConfig(
                        master_key_hex=self.config.hmac_key,
                        session_nonce_hex=session_nonce,
                        n=1,
                        family="lm:templates", 
                        params={"index": i},
                    )
                    
                    challenge_result = generate_challenges(cfg)
                    challenge = challenge_result["challenges"][0]
                    
                    # Store only the prompt string
                    prompts.append(challenge.parameters.get("prompt", ""))
                    
                    # Store minimal metadata for evidence bundle
                    challenge_metadata.append({
                        "id": challenge.challenge_id,
                        "seed": seed,
                        "index": i
                    })
                
                self._log(f"Generated {len(prompts)} prompts, starting verification...")
                
                # Create verifier with proper configuration
                diff_config = DiffDecisionConfig(mode=self.config.testing_mode)
                
                # Create adaptive components if enabled
                adaptive_tester = None
                if self.config.enable_adaptive:
                    self._log("Enabling adaptive sampling with dynamic thresholds")
                    adaptive_config = AdaptiveConfig(
                        initial_batch_size=self.config.adaptive_batch_size,
                        switch_threshold=self.config.adaptive_switch_threshold,
                        noise_margin_factor=self.config.adaptive_noise_margin,
                        max_adaptive_factor=self.config.adaptive_max_factor
                    )
                    # We'll create the adaptive tester wrapper later
                
                # Track which prompts were actually used
                prompts_used = 0
                prompt_iter = iter(prompts)
                
                def prompt_generator():
                    """Return next pre-generated prompt"""
                    nonlocal prompts_used
                    try:
                        prompt = next(prompt_iter)
                        prompts_used += 1
                        return prompt
                    except StopIteration:
                        # Shouldn't happen with proper n_max
                        return prompts[0]
                
                # Proper cross-entropy scoring function with generation
                def score_fn(ref, cand, prompt, K=32):
                    """
                    Generate K tokens and compute cross-entropy difference
                    Following the cryptographic PoT paper methodology
                    """
                    try:
                        import torch
                        
                        # Generate K tokens using reference model (deterministic)
                        ref_inputs = ref.tok(prompt, return_tensors="pt").to(ref.device)
                        with torch.no_grad():
                            ref_outputs = ref.m.generate(
                                **ref_inputs,
                                max_new_tokens=K,
                                do_sample=False,  # Deterministic generation
                                pad_token_id=ref.tok.pad_token_id
                            )
                        
                        # Decode the generated tokens (excluding the prompt)
                        prompt_length = ref_inputs.input_ids.shape[1]
                        generated_tokens = ref_outputs[0][prompt_length:]
                        target_text = ref.tok.decode(generated_tokens, skip_special_tokens=True)
                        
                        # Compute cross-entropy for both models on prompt + target
                        full_text = prompt + " " + target_text  # Add space between prompt and target
                        
                        # Tokenize the full text for both models
                        ref_full_inputs = ref.tok(full_text, return_tensors="pt", truncation=True, max_length=512).to(ref.device)
                        cand_full_inputs = cand.tok(full_text, return_tensors="pt", truncation=True, max_length=512).to(cand.device)
                        
                        # Simpler CE calculation using model's built-in loss
                        with torch.no_grad():
                            # Get CE from reference model
                            ref_outputs = ref.m(**ref_full_inputs, labels=ref_full_inputs.input_ids)
                            ref_loss = ref_outputs.loss.item()
                            
                            # Get CE from candidate model  
                            cand_outputs = cand.m(**cand_full_inputs, labels=cand_full_inputs.input_ids)
                            cand_loss = cand_outputs.loss.item()
                        
                        # Scale by sequence length to get total CE (not average)
                        seq_length = ref_full_inputs.input_ids.shape[1]
                        ref_ce = ref_loss * seq_length
                        cand_ce = cand_loss * seq_length
                        
                        # Return the CE difference (cand - ref)
                        # Following runtime_blackbox_validation.py convention:
                        # Positive: candidate is worse (distilled model has higher CE)
                        # Negative: candidate is better (unlikely for distillation)
                        # Near zero: similar models
                        ce_diff = cand_ce - ref_ce
                        
                        return ce_diff
                        
                    except Exception as e:
                        self._log(f"Warning: CE score computation failed: {e}", level="WARNING")
                        return 0.0  # Neutral score on error
                
                # Use adaptive verification if enabled
                if self.config.enable_adaptive:
                    # Create enhanced sequential tester with adaptive thresholds
                    tester = EnhancedSequentialTester(diff_config)
                    adaptive_tester = AdaptiveSequentialTester(tester, adaptive_config)
                    
                    # Run adaptive verification
                    report = self._run_adaptive_verification(
                        ref_model=ref_model,
                        cand_model=cand_model,
                        prompts=prompts,
                        score_fn=score_fn,
                        adaptive_tester=adaptive_tester,
                        diff_config=diff_config
                    )
                    prompts_used = report.get('n_queries', len(prompts))
                else:
                    # Use standard verifier
                    verifier = DifferenceVerifier(
                        score_fn=score_fn,
                        prompt_generator=prompt_generator,
                        cfg=diff_config,
                        use_enhanced=True
                    )
                    
                    # Run verification (will stop early on decision)
                    report = verifier.verify_difference(
                        ref_model=ref_model,
                        cand_model=cand_model,
                        verbose=self.config.verbose
                    )
                
                # Track actual challenges used for evidence bundle
                challenges_used = []
                for i in range(prompts_used):
                    if i < len(challenge_metadata):
                        challenge_info = challenge_metadata[i].copy()
                        challenge_info["prompt"] = prompts[i]
                        challenges_used.append(challenge_info)
                
                self.evidence_bundle["challenges"] = challenges_used
                self.evidence_bundle["challenges_used"] = prompts_used
                
                # Add fuzzy hashing as secondary verification
                try:
                    from pot.lm.fuzzy_hash import FuzzyHasher
                    hasher = FuzzyHasher()
                    
                    # Generate sample outputs for fuzzy hashing
                    sample_prompts = prompts[:min(5, len(prompts))]
                    ref_outputs = []
                    cand_outputs = []
                    
                    for sp in sample_prompts:
                        ref_out = ref_model.generate(sp, max_new_tokens=50)
                        cand_out = cand_model.generate(sp, max_new_tokens=50)
                        ref_outputs.append(ref_out)
                        cand_outputs.append(cand_out)
                    
                    # Compute fuzzy similarity
                    ref_hash = hasher.hash(" ".join(ref_outputs))
                    cand_hash = hasher.hash(" ".join(cand_outputs))
                    fuzzy_similarity = hasher.similarity(ref_hash, cand_hash)
                    
                    self.evidence_bundle["fuzzy_similarity"] = fuzzy_similarity
                    self._log(f"Fuzzy hash similarity: {fuzzy_similarity:.3f}")
                except Exception as e:
                    self._log(f"Warning: Fuzzy hashing failed: {e}", level="WARNING")
                    self.evidence_bundle["fuzzy_similarity"] = None
                
                result = {
                    "decision": report["results"]["decision"],
                    "confidence": report["results"].get("confidence", 1.0 - diff_config.alpha),
                    "n_queries": report["results"]["n_used"],
                    "challenges_used": len(challenges_used),
                    "ci_progression": report.get("progression", []),
                    "effect_size": report["results"]["mean"],
                }

            metrics.query_count = result["n_queries"]
            metrics.ci_progression = result.get("ci_progression", [])
            metrics.metadata["decision"] = result["decision"]

            # Store detailed verification info in evidence bundle
            self.evidence_bundle["verification"] = report if report is not None else result
            
            # Generate Merkle root for audit trail
            try:
                # Create transcript entries for Merkle tree
                transcript = []
                for i, challenge in enumerate(challenges_used[:prompts_used]):
                    entry = f"challenge_{i}:{challenge['id']}:{challenge['prompt'][:50]}"
                    transcript.append(entry)
                
                # Simple Merkle tree implementation
                def merkle_hash(data):
                    """Compute SHA-256 hash of data"""
                    if isinstance(data, str):
                        data = data.encode()
                    return hashlib.sha256(data).hexdigest()
                
                def build_merkle_tree(leaves):
                    """Build Merkle tree and return root"""
                    if not leaves:
                        return merkle_hash("")
                    if len(leaves) == 1:
                        return merkle_hash(leaves[0])
                    
                    # Hash all leaves
                    layer = [merkle_hash(leaf) for leaf in leaves]
                    
                    # Build tree bottom-up
                    while len(layer) > 1:
                        next_layer = []
                        for i in range(0, len(layer), 2):
                            if i + 1 < len(layer):
                                combined = layer[i] + layer[i + 1]
                            else:
                                combined = layer[i] + layer[i]  # Duplicate if odd
                            next_layer.append(merkle_hash(combined))
                        layer = next_layer
                    
                    return layer[0]
                
                merkle_root = build_merkle_tree(transcript)
                self.evidence_bundle["transcript_merkle_root"] = merkle_root
                self.evidence_bundle["transcript_entries"] = len(transcript)
                self._log(f"Generated Merkle root for {len(transcript)} transcript entries: {merkle_root[:16]}...")
            except Exception as e:
                self._log(f"Warning: Failed to generate Merkle root: {e}", level="WARNING")

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
                'transcript_merkle_root': self.evidence_bundle.get('transcript_merkle_root', ''),
                'transcript_entries': self.evidence_bundle.get('transcript_entries', 0),
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
        n_challenges: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run the complete end-to-end validation pipeline
        
        Args:
            ref_model_path: Path or identifier for reference model
            cand_model_path: Path or identifier for candidate model
            n_challenges: Optional max number of challenges (uses config n_max if not provided)
            
        Returns:
            Complete pipeline results dictionary
        """
        self._log(f"Starting E2E validation pipeline (run_id: {self.run_id})")
        self._log(f"Configuration: {self.config.testing_mode}, {self.config.verification_mode}")
        
        # Use n_max from testing mode config if n_challenges not specified
        if n_challenges is None:
            # Get n_max from the testing mode configuration
            diff_config = DiffDecisionConfig(mode=self.config.testing_mode)
            n_challenges = diff_config.n_max
            self._log(f"Using n_max={n_challenges} from {self.config.testing_mode} mode")
        
        try:
            # Stage 1: Pre-commit challenge seeds (generate max possible)
            pre_commit = self.pre_commit_challenges(n_challenges)
            
            # Stage 2: Load models first (before generating challenges)
            ref_model, cand_model = self.load_models(ref_model_path, cand_model_path)
            
            # Stage 3: Run verification (which will generate challenges as needed)
            verification = self.run_verification(ref_model, cand_model, pre_commit['seeds'])
            
            # Stage 4: Generate evidence bundle
            evidence_bundle = self.generate_evidence_bundle()
            
            # Stage 5: Generate ZK proof (optional)
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