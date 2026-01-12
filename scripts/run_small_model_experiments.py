#!/usr/bin/env python3
"""
Small-Model Publication Experiments

Runs experiments only on models <3B parameters to avoid memory issues.
This covers the core verification capabilities without requiring 7B+ models.
"""

import argparse
import json
import os
import sys
import time
import logging
import gc
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pot.validation.e2e_pipeline import (
    PipelineOrchestrator,
    PipelineConfig,
    TestingMode,
    VerificationMode
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class ExperimentResult:
    """Result from a single experiment"""
    experiment_id: str
    category: str
    ref_model: str
    cand_model: str
    expected_decision: str
    actual_decision: str
    confidence: float
    queries_used: int
    duration_seconds: float
    correct: bool
    error: Optional[str] = None


# Small models only (<3B parameters)
SMALL_MODEL_EXPERIMENTS = [
    # Self-Consistency Tests (SAME expected)
    {
        "id": "self_gpt2",
        "category": "self_consistency",
        "ref": "gpt2",
        "cand": "gpt2",
        "expected": "SAME",
        "description": "GPT-2 (124M) self-consistency"
    },
    {
        "id": "self_pythia160m",
        "category": "self_consistency",
        "ref": "EleutherAI/pythia-160m",
        "cand": "EleutherAI/pythia-160m",
        "expected": "SAME",
        "description": "Pythia-160M self-consistency"
    },
    {
        "id": "self_gpt2medium",
        "category": "self_consistency",
        "ref": "gpt2-medium",
        "cand": "gpt2-medium",
        "expected": "SAME",
        "description": "GPT-2 Medium (355M) self-consistency"
    },

    # Distillation Detection (DIFFERENT expected)
    {
        "id": "distill_gpt2",
        "category": "distillation",
        "ref": "gpt2",
        "cand": "distilgpt2",
        "expected": "DIFFERENT",
        "description": "GPT-2 vs DistilGPT-2 (distillation)"
    },

    # Scale Variation (DIFFERENT expected)
    {
        "id": "scale_pythia_70m_160m",
        "category": "scale",
        "ref": "EleutherAI/pythia-70m",
        "cand": "EleutherAI/pythia-160m",
        "expected": "DIFFERENT",
        "description": "Pythia 70M vs 160M (2.3x scale)"
    },
    {
        "id": "scale_pythia_160m_410m",
        "category": "scale",
        "ref": "EleutherAI/pythia-160m",
        "cand": "EleutherAI/pythia-410m",
        "expected": "DIFFERENT",
        "description": "Pythia 160M vs 410M (2.6x scale)"
    },
    {
        "id": "scale_gpt2_gpt2medium",
        "category": "scale",
        "ref": "gpt2",
        "cand": "gpt2-medium",
        "expected": "DIFFERENT",
        "description": "GPT-2 (124M) vs GPT-2 Medium (355M)"
    },
    {
        "id": "scale_gpt_neo",
        "category": "scale",
        "ref": "EleutherAI/gpt-neo-125m",
        "cand": "EleutherAI/gpt-neo-1.3B",
        "expected": "DIFFERENT",
        "description": "GPT-Neo 125M vs 1.3B (10x scale)"
    },

    # Architecture Comparison (DIFFERENT expected)
    {
        "id": "arch_gpt2_pythia",
        "category": "architecture",
        "ref": "gpt2",
        "cand": "EleutherAI/pythia-160m",
        "expected": "DIFFERENT",
        "description": "GPT-2 (124M) vs Pythia-160M (different arch)"
    },
    {
        "id": "arch_gpt2_neo",
        "category": "architecture",
        "ref": "gpt2",
        "cand": "EleutherAI/gpt-neo-125m",
        "expected": "DIFFERENT",
        "description": "GPT-2 (124M) vs GPT-Neo (125M) (different arch)"
    },
    {
        "id": "arch_pythia_neo",
        "category": "architecture",
        "ref": "EleutherAI/pythia-160m",
        "cand": "EleutherAI/gpt-neo-125m",
        "expected": "DIFFERENT",
        "description": "Pythia (160M) vs GPT-Neo (125M) (different arch)"
    },
]


def run_experiment(exp: Dict, mode: TestingMode, output_dir: Path) -> ExperimentResult:
    """Run a single experiment"""
    exp_id = exp["id"]

    logger.info(f"\n{'='*60}")
    logger.info(f"Starting experiment: {exp_id}")
    logger.info(f"  {exp['description']}")
    logger.info(f"  Ref: {exp['ref']}")
    logger.info(f"  Cand: {exp['cand']}")
    logger.info(f"  Expected: {exp['expected']}")
    logger.info(f"{'='*60}")

    start_time = time.time()

    try:
        config = PipelineConfig(
            testing_mode=mode,
            verification_mode=VerificationMode.LOCAL_WEIGHTS,
            output_dir=Path(output_dir / exp_id),
            enable_zk_proof=False,
            enable_sharding=False,
            max_memory_percent=80,
        )

        orchestrator = PipelineOrchestrator(config)
        result = orchestrator.run_complete_pipeline(
            ref_model_path=exp["ref"],
            cand_model_path=exp["cand"]
        )

        duration = time.time() - start_time

        decision = result.get("decision", "UNDECIDED")
        confidence = result.get("confidence", 0.0)
        queries = result.get("n_queries", 0)

        correct = (decision == exp["expected"])

        logger.info(f"Result: {decision} (expected {exp['expected']}) - {'CORRECT' if correct else 'INCORRECT'}")
        logger.info(f"Confidence: {confidence:.4f}, Queries: {queries}, Duration: {duration:.1f}s")

        # Force garbage collection
        gc.collect()

        return ExperimentResult(
            experiment_id=exp_id,
            category=exp["category"],
            ref_model=exp["ref"],
            cand_model=exp["cand"],
            expected_decision=exp["expected"],
            actual_decision=decision,
            confidence=confidence,
            queries_used=queries,
            duration_seconds=duration,
            correct=correct
        )

    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"Experiment {exp_id} failed: {e}")

        return ExperimentResult(
            experiment_id=exp_id,
            category=exp["category"],
            ref_model=exp["ref"],
            cand_model=exp["cand"],
            expected_decision=exp["expected"],
            actual_decision="ERROR",
            confidence=0.0,
            queries_used=0,
            duration_seconds=duration,
            correct=False,
            error=str(e)
        )


def main():
    parser = argparse.ArgumentParser(description='Run small-model publication experiments')
    parser.add_argument('--mode', type=str, default='audit',
                        choices=['quick', 'audit'],
                        help='Testing mode (quick=97.5%%, audit=99%%)')
    parser.add_argument('--output-dir', type=str,
                        default='experimental_results/publication_small',
                        help='Output directory')
    parser.add_argument('--skip-completed', action='store_true',
                        help='Skip already completed experiments')

    args = parser.parse_args()

    mode = {
        'quick': TestingMode.QUICK_GATE,
        'audit': TestingMode.AUDIT_GRADE
    }[args.mode]

    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\n{'='*60}")
    logger.info("SMALL-MODEL PUBLICATION EXPERIMENTS")
    logger.info(f"{'='*60}")
    logger.info(f"Total experiments: {len(SMALL_MODEL_EXPERIMENTS)}")
    logger.info(f"Testing mode: {args.mode} ({mode})")
    logger.info(f"Output directory: {output_dir}")

    results = []

    for i, exp in enumerate(SMALL_MODEL_EXPERIMENTS, 1):
        logger.info(f"\n[{i}/{len(SMALL_MODEL_EXPERIMENTS)}] Running experiment...")

        result = run_experiment(exp, mode, output_dir)
        results.append(result)

        # Save intermediate results
        intermediate_path = output_dir / "intermediate_results.json"
        with open(intermediate_path, 'w') as f:
            json.dump([asdict(r) for r in results], f, indent=2)

        # Cooldown between experiments
        if i < len(SMALL_MODEL_EXPERIMENTS):
            logger.info("Cooldown: 10 seconds...")
            gc.collect()
            time.sleep(10)

    # Save final results
    final_results = {
        "metadata": {
            "timestamp": timestamp,
            "mode": args.mode,
            "total_experiments": len(SMALL_MODEL_EXPERIMENTS),
            "completed": len(results)
        },
        "results": [asdict(r) for r in results]
    }

    results_path = output_dir / "publication_results.json"
    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=2)

    # Calculate statistics
    correct = sum(1 for r in results if r.correct)
    errors = sum(1 for r in results if r.error)

    logger.info(f"\n{'='*60}")
    logger.info("EXPERIMENT SUITE COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Accuracy: {correct/len(results)*100:.1f}%")
    logger.info(f"Correct: {correct}/{len(results)}")
    logger.info(f"Errors: {errors}")
    logger.info(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
