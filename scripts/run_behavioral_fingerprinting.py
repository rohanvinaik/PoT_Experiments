#!/usr/bin/env python3
"""
Behavioral Fingerprinting Experiments

Tests the variance-based relationship inference to discriminate between:
- IDENTICAL: Same model weights
- SAME_ARCH_DIFF_SCALE: Scale variants (e.g., Pythia-70M vs 160M)
- DISTILLED: Student-teacher distillation (e.g., GPT-2 vs DistilGPT-2)
- DIFFERENT_ARCHITECTURE: Different model families (e.g., GPT-2 vs Pythia)
- SAME_ARCH_FINE_TUNED: Fine-tuned variants

Uses the actual PoT scoring infrastructure (delta cross-entropy) for accurate
relationship classification based on the paper's Section 7: "Behavioral
Fingerprinting: Beyond Binary Decisions"

Thresholds (from paper):
- SAME (identical): |X_n| < 0.001 with high confidence
- RELATED_TRAINING: 1 <= |X_n| < 5 (continued pre-training)
- DIFFERENT_TRAINING: 5 <= |X_n| < 10 (distillation)
- DIFFERENT_ARCH: |X_n| >= 10 (different architectures)
"""

import os
import sys
import json
import logging
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict, field
import gc

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from pot.core.enhanced_diff_decision import (
    EnhancedDiffTester,
    ModelRelationship,
    VarianceSignature,
    create_enhanced_tester,
)
from pot.core.diff_decision import TestingMode, DiffDecisionConfig
from pot.scoring.diff_scorer import CorrectedDifferenceScorer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class FingerprintResult:
    """Result of a behavioral fingerprinting experiment"""
    experiment_id: str
    category: str
    ref_model: str
    cand_model: str
    expected_relationship: str
    actual_relationship: str
    relationship_match: bool
    traditional_decision: str
    relationship_confidence: float
    mean_effect: float
    variance: float
    cv: float
    n_queries: int
    duration_seconds: float
    inference_explanation: str
    variance_signature: Dict[str, float] = field(default_factory=dict)


class BehavioralFingerprintingPipeline:
    """Pipeline for running behavioral fingerprinting experiments using PoT infrastructure"""

    def __init__(
        self,
        testing_mode: TestingMode = TestingMode.AUDIT_GRADE,
        device: str = "auto",
        output_dir: Optional[Path] = None,
        k_positions: int = 64,  # Match paper's K=64 positions
    ):
        self.testing_mode = testing_mode
        self.device = self._determine_device(device)
        self.output_dir = output_dir or Path("experimental_results/behavioral_fingerprinting")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.k_positions = k_positions

        # Initialize scorer (PoT's actual scoring infrastructure)
        self.scorer = CorrectedDifferenceScorer(
            epsilon=1e-10,
            min_vocab_overlap=0.8,
            vocab_mismatch_behavior="adapt",
            allow_extended_vocabularies=True
        )

        # Standard prompts for consistent testing
        self.prompts = [
            "The quick brown fox jumps over the lazy dog.",
            "In a world where artificial intelligence has become",
            "The most important thing about machine learning is",
            "Scientists discovered a new species of deep-sea",
            "The future of technology lies in the development of",
            "Once upon a time in a distant land there lived",
            "The principles of quantum computing suggest that",
            "According to recent research on climate change",
            "The history of human civilization shows that",
            "When analyzing complex systems we must consider",
            "The fundamental laws of physics dictate that",
            "In the realm of natural language processing",
        ]

        logger.info(f"Pipeline initialized: mode={testing_mode.name}, device={self.device}")

    def _determine_device(self, device: str) -> str:
        """Determine the best available device"""
        if device != "auto":
            return device
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def load_model(self, model_name: str) -> Tuple[Any, Any]:
        """Load model and tokenizer"""
        logger.info(f"Loading model: {model_name}")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Use float32 for MPS/CPU (float16 causes numerical precision issues on MPS)
        # Only use float16 for CUDA where it's properly supported
        dtype = torch.float16 if self.device == "cuda" else torch.float32

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto" if self.device == "cuda" else None,
            low_cpu_mem_usage=True,
        )

        if self.device == "mps":
            model = model.to(self.device)

        model.eval()
        return model, tokenizer

    def get_logits_for_prompt(
        self,
        model: Any,
        tokenizer: Any,
        prompt: str,
    ) -> torch.Tensor:
        """Get model logits for a prompt at K positions"""

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding=True
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits  # [batch, seq_len, vocab_size]

        return logits

    def compute_delta_ce_score(
        self,
        ref_model: Any,
        ref_tokenizer: Any,
        cand_model: Any,
        cand_tokenizer: Any,
        prompt: str,
    ) -> float:
        """Compute delta cross-entropy score for a single prompt"""

        try:
            # Get logits from both models
            ref_logits = self.get_logits_for_prompt(ref_model, ref_tokenizer, prompt)
            cand_logits = self.get_logits_for_prompt(cand_model, cand_tokenizer, prompt)

            # Handle sequence length mismatch (different tokenizers produce different lengths)
            ref_seq_len = ref_logits.shape[1]
            cand_seq_len = cand_logits.shape[1]

            if ref_seq_len != cand_seq_len:
                # Align to the shorter sequence length
                min_seq_len = min(ref_seq_len, cand_seq_len)
                ref_logits = ref_logits[:, :min_seq_len, :]
                cand_logits = cand_logits[:, :min_seq_len, :]
                logger.debug(f"Aligned sequences: {ref_seq_len} vs {cand_seq_len} -> {min_seq_len}")

            # Compute delta cross-entropy using PoT scorer
            score = self.scorer.delta_ce_abs(ref_logits, cand_logits)

            return score

        except Exception as e:
            logger.warning(f"Error computing score: {e}")
            return float('nan')

    def run_experiment(
        self,
        experiment_id: str,
        category: str,
        ref_model_name: str,
        cand_model_name: str,
        expected_relationship: str,
        max_queries: int = 200,
    ) -> FingerprintResult:
        """Run a single behavioral fingerprinting experiment"""

        logger.info(f"\n{'='*60}")
        logger.info(f"Experiment: {experiment_id}")
        logger.info(f"Category: {category}")
        logger.info(f"Reference: {ref_model_name}")
        logger.info(f"Candidate: {cand_model_name}")
        logger.info(f"Expected: {expected_relationship}")
        logger.info(f"{'='*60}")

        start_time = time.time()

        # Load models
        try:
            ref_model, ref_tokenizer = self.load_model(ref_model_name)

            if ref_model_name == cand_model_name:
                # Self-consistency: use same model instance
                cand_model, cand_tokenizer = ref_model, ref_tokenizer
                same_model = True
            else:
                cand_model, cand_tokenizer = self.load_model(cand_model_name)
                same_model = False

        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            return FingerprintResult(
                experiment_id=experiment_id,
                category=category,
                ref_model=ref_model_name,
                cand_model=cand_model_name,
                expected_relationship=expected_relationship,
                actual_relationship="ERROR",
                relationship_match=False,
                traditional_decision="ERROR",
                relationship_confidence=0.0,
                mean_effect=0.0,
                variance=0.0,
                cv=0.0,
                n_queries=0,
                duration_seconds=time.time() - start_time,
                inference_explanation=f"Model loading error: {e}",
            )

        # Create enhanced tester for relationship inference
        tester = create_enhanced_tester(self.testing_mode)

        query_count = 0
        decision_info = None

        while query_count < max_queries:
            # Cycle through prompts
            prompt = self.prompts[query_count % len(self.prompts)]

            try:
                # Compute delta cross-entropy score
                score = self.compute_delta_ce_score(
                    ref_model, ref_tokenizer,
                    cand_model, cand_tokenizer,
                    prompt
                )

                if np.isnan(score):
                    continue

                # Update tester with score
                tester.update(score)
                query_count += 1

                # Log progress
                if query_count % 10 == 0:
                    signature = tester.compute_variance_signature()
                    logger.info(
                        f"  Query {query_count}: mean={signature.mean_effect:.4f}, "
                        f"var={signature.variance:.6f}, cv={signature.cv:.2f}"
                    )

                # Check if we should stop
                should_stop, info = tester.should_stop()
                if should_stop:
                    decision_info = info
                    break

            except Exception as e:
                logger.warning(f"Query {query_count} failed: {e}")
                continue

        # Get final summary if no early stopping
        if decision_info is None:
            decision_info = tester.get_summary()

        duration = time.time() - start_time

        # Compute variance signature for direct relationship inference
        signature = tester.compute_variance_signature()

        # Direct relationship inference based on statistics
        # This overrides the base tester's classification for consistency
        mean_abs = abs(signature.mean_effect)
        cv = signature.cv
        var = signature.variance

        # IDENTICAL: Both mean and variance are effectively zero
        if mean_abs < 1e-6 and var < 1e-10:
            direct_relationship = "IDENTICAL"
        elif mean_abs < 1e-4 and var < 1e-6:
            direct_relationship = "IDENTICAL"
        elif mean_abs < 0.001:
            direct_relationship = "NEAR_CLONE"
        elif mean_abs > 3.0:
            # Very large mean = different architecture
            direct_relationship = "DIFFERENT_ARCHITECTURE"
        elif np.isfinite(cv):
            if mean_abs > 0.3 and cv < 0.3:
                direct_relationship = "DISTILLED"
            elif mean_abs > 0.1 and cv < 1.5:
                direct_relationship = "SAME_ARCHITECTURE_DIFFERENT_SCALE"
            elif mean_abs > 0.1 and cv > 1.5:
                direct_relationship = "DIFFERENT_ARCHITECTURE"
            elif 0.01 < mean_abs < 0.3 and cv < 0.5:
                direct_relationship = "SAME_ARCHITECTURE_FINE_TUNED"
            else:
                direct_relationship = "INCONCLUSIVE"
        else:
            direct_relationship = "INCONCLUSIVE"

        # Use direct relationship if tester returned INCONCLUSIVE
        tester_relationship = decision_info.get("relationship", "INCONCLUSIVE")
        actual_relationship = direct_relationship if tester_relationship == "INCONCLUSIVE" else tester_relationship

        traditional_decision = decision_info.get(
            "traditional_decision",
            decision_info.get("decision", "UNDECIDED")
        )

        # Compute variance signature
        signature = tester.compute_variance_signature()

        # Determine if relationship classification matches expected
        relationship_match = self._check_relationship_match(
            expected_relationship,
            actual_relationship
        )

        # Build result
        result = FingerprintResult(
            experiment_id=experiment_id,
            category=category,
            ref_model=ref_model_name,
            cand_model=cand_model_name,
            expected_relationship=expected_relationship,
            actual_relationship=actual_relationship,
            relationship_match=relationship_match,
            traditional_decision=traditional_decision,
            relationship_confidence=decision_info.get(
                "relationship_confidence",
                tester._compute_relationship_confidence(signature)
            ),
            mean_effect=signature.mean_effect,
            variance=signature.variance,
            cv=signature.cv,
            n_queries=tester.n,
            duration_seconds=duration,
            inference_explanation=decision_info.get(
                "inference_explanation",
                decision_info.get("inference_basis", "")
            ),
            variance_signature={
                "normalized_variance": signature.normalized_variance,
                "stability_score": signature.stability_score,
                "variance_ratio": signature.variance_ratio,
            }
        )

        logger.info(f"\nResult: {actual_relationship}")
        logger.info(f"Traditional: {traditional_decision}")
        logger.info(f"Match: {'YES' if relationship_match else 'NO'}")
        logger.info(f"Queries: {tester.n}, Time: {duration:.1f}s")
        logger.info(f"Explanation: {result.inference_explanation}")

        # Cleanup
        del ref_model
        if not same_model:
            del cand_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return result

    def _check_relationship_match(self, expected: str, actual: str) -> bool:
        """Check if relationship classification matches expected"""

        # Normalize relationship names
        expected_norm = expected.upper().replace("-", "_").replace(" ", "_")
        actual_norm = actual.upper().replace("-", "_").replace(" ", "_")

        # Direct match
        if expected_norm == actual_norm:
            return True

        # Equivalent mappings
        equivalences = {
            "IDENTICAL": ["IDENTICAL", "SAME", "NEAR_CLONE"],
            "SAME": ["IDENTICAL", "SAME", "NEAR_CLONE"],
            "NEAR_CLONE": ["IDENTICAL", "SAME", "NEAR_CLONE"],
            "SAME_ARCH_DIFF_SCALE": [
                "SAME_ARCH_DIFF_SCALE",
                "SAME_ARCHITECTURE_DIFFERENT_SCALE",
                "SCALE"
            ],
            "DISTILLED": ["DISTILLED", "DISTILLATION"],
            "DIFFERENT_ARCH": [
                "DIFFERENT_ARCH",
                "DIFFERENT_ARCHITECTURE",
                "ARCHITECTURE"
            ],
            "SAME_ARCH_FINE_TUNED": [
                "SAME_ARCH_FINE_TUNED",
                "SAME_ARCHITECTURE_FINE_TUNED",
                "FINE_TUNED",
                "FINETUNED"
            ],
        }

        expected_matches = equivalences.get(expected_norm, [expected_norm])
        return actual_norm in [m.upper() for m in expected_matches]


def define_experiments() -> List[Dict[str, Any]]:
    """Define behavioral fingerprinting experiments for small models"""

    return [
        # Self-consistency (IDENTICAL)
        {
            "id": "self_gpt2",
            "category": "self_consistency",
            "ref": "gpt2",
            "cand": "gpt2",
            "expected": "IDENTICAL",
        },
        {
            "id": "self_pythia160m",
            "category": "self_consistency",
            "ref": "EleutherAI/pythia-160m",
            "cand": "EleutherAI/pythia-160m",
            "expected": "IDENTICAL",
        },

        # Distillation detection (DISTILLED)
        {
            "id": "distill_gpt2",
            "category": "distillation",
            "ref": "gpt2",
            "cand": "distilgpt2",
            "expected": "DISTILLED",
        },

        # Scale variants (SAME_ARCH_DIFF_SCALE)
        {
            "id": "scale_pythia_70m_160m",
            "category": "scale",
            "ref": "EleutherAI/pythia-70m",
            "cand": "EleutherAI/pythia-160m",
            "expected": "SAME_ARCH_DIFF_SCALE",
        },
        {
            "id": "scale_pythia_160m_410m",
            "category": "scale",
            "ref": "EleutherAI/pythia-160m",
            "cand": "EleutherAI/pythia-410m",
            "expected": "SAME_ARCH_DIFF_SCALE",
        },
        {
            "id": "scale_gpt2_gpt2medium",
            "category": "scale",
            "ref": "gpt2",
            "cand": "gpt2-medium",
            "expected": "SAME_ARCH_DIFF_SCALE",
        },

        # Different architectures (DIFFERENT_ARCH)
        {
            "id": "arch_gpt2_pythia",
            "category": "architecture",
            "ref": "gpt2",
            "cand": "EleutherAI/pythia-160m",
            "expected": "DIFFERENT_ARCH",
        },
        {
            "id": "arch_gpt2_neo",
            "category": "architecture",
            "ref": "gpt2",
            "cand": "EleutherAI/gpt-neo-125m",
            "expected": "DIFFERENT_ARCH",
        },
        {
            "id": "arch_pythia_neo",
            "category": "architecture",
            "ref": "EleutherAI/pythia-160m",
            "cand": "EleutherAI/gpt-neo-125m",
            "expected": "DIFFERENT_ARCH",
        },
    ]


def generate_report(results: List[FingerprintResult], output_dir: Path) -> str:
    """Generate a detailed report of behavioral fingerprinting results"""

    report_lines = []
    report_lines.append("# Behavioral Fingerprinting Results")
    report_lines.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Summary statistics
    total = len(results)
    correct = sum(1 for r in results if r.relationship_match)
    accuracy = correct / total * 100 if total > 0 else 0

    report_lines.append("## Summary\n")
    report_lines.append("| Metric | Value |")
    report_lines.append("|--------|-------|")
    report_lines.append(f"| Total Experiments | {total} |")
    report_lines.append(f"| Correct Classifications | {correct} |")
    report_lines.append(f"| **Classification Accuracy** | **{accuracy:.1f}%** |")
    report_lines.append(f"| Avg Queries | {np.mean([r.n_queries for r in results]):.1f} |")
    report_lines.append(f"| Avg Time | {np.mean([r.duration_seconds for r in results]):.1f}s |")
    report_lines.append(f"| Avg Confidence | {np.mean([r.relationship_confidence for r in results]):.2%} |\n")

    # Results by category
    report_lines.append("## Results by Category\n")
    categories = sorted(set(r.category for r in results))

    report_lines.append("| Category | Experiments | Correct | Accuracy | Avg Mean Effect | Avg CV |")
    report_lines.append("|----------|-------------|---------|----------|-----------------|--------|")

    for cat in categories:
        cat_results = [r for r in results if r.category == cat]
        cat_correct = sum(1 for r in cat_results if r.relationship_match)
        cat_acc = cat_correct / len(cat_results) * 100 if cat_results else 0
        avg_mean = np.mean([r.mean_effect for r in cat_results])
        avg_cv = np.mean([r.cv for r in cat_results if not np.isinf(r.cv)])

        report_lines.append(
            f"| {cat.replace('_', ' ').title()} | {len(cat_results)} | "
            f"{cat_correct} | {cat_acc:.1f}% | {avg_mean:.4f} | {avg_cv:.2f} |"
        )
    report_lines.append("")

    # Detailed results
    report_lines.append("## Detailed Results\n")
    report_lines.append(
        "| Experiment | Category | Expected | Actual | Match | "
        "Mean Effect | Variance | CV | Queries |"
    )
    report_lines.append(
        "|------------|----------|----------|--------|-------|"
        "-------------|----------|-----|---------|"
    )

    for r in results:
        match_mark = "YES" if r.relationship_match else "NO"
        cv_str = f"{r.cv:.2f}" if not np.isinf(r.cv) else "inf"
        report_lines.append(
            f"| {r.experiment_id} | {r.category} | {r.expected_relationship} | "
            f"{r.actual_relationship} | {match_mark} | {r.mean_effect:.4f} | "
            f"{r.variance:.6f} | {cv_str} | {r.n_queries} |"
        )
    report_lines.append("")

    # Relationship inference explanations
    report_lines.append("## Inference Explanations\n")
    for r in results:
        status = "CORRECT" if r.relationship_match else "INCORRECT"
        report_lines.append(f"### {r.experiment_id} ({status})")
        report_lines.append(f"- **Expected:** {r.expected_relationship}")
        report_lines.append(f"- **Actual:** {r.actual_relationship}")
        report_lines.append(f"- **Traditional Decision:** {r.traditional_decision}")
        report_lines.append(f"- **Confidence:** {r.relationship_confidence:.2%}")
        report_lines.append(f"- **Explanation:** {r.inference_explanation}\n")

    # Variance signature analysis
    report_lines.append("## Variance Signature Reference\n")
    report_lines.append(
        "Based on Section 7 of the PoT paper, behavioral fingerprinting uses "
        "variance signatures to classify model relationships:\n"
    )
    report_lines.append("| Relationship | Mean Effect Range | Variance Pattern | CV Range |")
    report_lines.append("|--------------|-------------------|------------------|----------|")
    report_lines.append("| IDENTICAL | < 1e-6 | Minimal | N/A |")
    report_lines.append("| NEAR_CLONE | < 0.001 | Low | Low |")
    report_lines.append("| SAME_ARCH_DIFF_SCALE | 0.001 - 0.5 | Moderate | < 2.0 |")
    report_lines.append("| SAME_ARCH_FINE_TUNED | 0.01 - 0.1 | Low | Low |")
    report_lines.append("| DISTILLED | > 0.5 | Low | < 1.0 |")
    report_lines.append("| DIFFERENT_ARCH | > 0.1 | High | > 2.0 |")

    report_text = "\n".join(report_lines)

    # Save report
    report_path = output_dir / "BEHAVIORAL_FINGERPRINTING_REPORT.md"
    with open(report_path, "w") as f:
        f.write(report_text)

    logger.info(f"Report saved to: {report_path}")
    return report_text


def main():
    """Run behavioral fingerprinting experiments"""

    print("\n" + "=" * 70)
    print("BEHAVIORAL FINGERPRINTING EXPERIMENTS")
    print("Discriminating model relationships via variance signatures")
    print("Based on PoT Paper Section 7: Beyond Binary Decisions")
    print("=" * 70 + "\n")

    # Set up output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"experimental_results/behavioral_fingerprinting/run_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize pipeline
    pipeline = BehavioralFingerprintingPipeline(
        testing_mode=TestingMode.AUDIT_GRADE,
        output_dir=output_dir,
    )

    # Define experiments
    experiments = define_experiments()

    logger.info(f"Running {len(experiments)} behavioral fingerprinting experiments")
    logger.info(f"Output directory: {output_dir}")

    # Run experiments
    results = []
    for i, exp in enumerate(experiments):
        logger.info(f"\n[{i+1}/{len(experiments)}] Starting {exp['id']}")

        result = pipeline.run_experiment(
            experiment_id=exp["id"],
            category=exp["category"],
            ref_model_name=exp["ref"],
            cand_model_name=exp["cand"],
            expected_relationship=exp["expected"],
        )
        results.append(result)

        # Save intermediate results
        with open(output_dir / "intermediate_results.json", "w") as f:
            json.dump([asdict(r) for r in results], f, indent=2)

        # Brief cooldown between experiments
        if i < len(experiments) - 1:
            time.sleep(2)

    # Generate report
    report = generate_report(results, output_dir)
    print("\n" + report)

    # Save final results
    with open(output_dir / "final_results.json", "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)

    # Print summary
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)

    correct = sum(1 for r in results if r.relationship_match)
    total = len(results)
    print(f"\nRelationship Classification Accuracy: {correct}/{total} ({correct/total*100:.1f}%)")
    print(f"Results saved to: {output_dir}")

    return results


if __name__ == "__main__":
    results = main()
