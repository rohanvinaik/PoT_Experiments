#!/usr/bin/env python3
"""
Publication-Quality PoT Experiment Suite

This script runs a comprehensive set of experiments suitable for academic publication,
testing the Proof-of-Training framework across multiple model categories:

1. Self-consistency tests (SAME expected)
2. Distillation detection (DIFFERENT expected)
3. Fine-tuning detection (DIFFERENT expected)
4. Scale variation detection (DIFFERENT expected)
5. Architecture comparison (DIFFERENT expected)

Models capped at ~25B parameters for practical runtime.
"""

import argparse
import json
import os
import sys
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import gc

# Add parent directory to path
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


@dataclass
class ExperimentSuite:
    """Complete experiment suite configuration"""
    name: str
    description: str
    experiments: List[Dict]
    mode: str = "audit"  # audit mode for publication quality


# Define the experiment suite
PUBLICATION_EXPERIMENTS = ExperimentSuite(
    name="PoT Publication Experiment Suite",
    description="Comprehensive behavioral verification experiments for academic publication",
    mode="audit",  # 99% confidence for publication
    experiments=[
        # ============================================
        # Category 1: Self-Consistency Tests (SAME expected)
        # ============================================
        {
            "id": "self_gpt2",
            "category": "self_consistency",
            "ref": "gpt2",
            "cand": "gpt2",
            "expected": "SAME",
            "description": "GPT-2 self-consistency baseline"
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
            "id": "self_phi2",
            "category": "self_consistency",
            "ref": "microsoft/phi-2",
            "cand": "microsoft/phi-2",
            "expected": "SAME",
            "description": "Phi-2 (2.7B) self-consistency"
        },

        # ============================================
        # Category 2: Distillation Detection (DIFFERENT expected)
        # ============================================
        {
            "id": "distill_gpt2",
            "category": "distillation",
            "ref": "gpt2",
            "cand": "distilgpt2",
            "expected": "DIFFERENT",
            "description": "GPT-2 vs DistilGPT-2 (classic distillation)"
        },

        # ============================================
        # Category 3: Fine-tuning Detection (DIFFERENT expected)
        # ============================================
        {
            "id": "finetune_llama2_7b",
            "category": "finetuning",
            "ref": "NousResearch/Llama-2-7b-hf",
            "cand": "NousResearch/Llama-2-7b-chat-hf",
            "expected": "DIFFERENT",
            "description": "Llama-2-7B base vs chat (instruction tuning)"
        },
        {
            "id": "finetune_mistral_zephyr",
            "category": "finetuning",
            "ref": "mistralai/Mistral-7B-Instruct-v0.3",
            "cand": "HuggingFaceH4/zephyr-7b-beta",
            "expected": "DIFFERENT",
            "description": "Mistral-7B vs Zephyr-7B (different fine-tuning)"
        },

        # ============================================
        # Category 4: Scale Variation (DIFFERENT expected)
        # ============================================
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
            "id": "scale_gpt_neo",
            "category": "scale",
            "ref": "EleutherAI/gpt-neo-125m",
            "cand": "EleutherAI/gpt-neo-1.3B",
            "expected": "DIFFERENT",
            "description": "GPT-Neo 125M vs 1.3B (10x scale)"
        },

        # ============================================
        # Category 5: Architecture Comparison (DIFFERENT expected)
        # ============================================
        {
            "id": "arch_gpt2_pythia",
            "category": "architecture",
            "ref": "gpt2",
            "cand": "EleutherAI/pythia-160m",
            "expected": "DIFFERENT",
            "description": "GPT-2 (124M) vs Pythia-160M (similar size, different arch)"
        },
        {
            "id": "arch_tinyllama_phi2",
            "category": "architecture",
            "ref": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "cand": "microsoft/phi-2",
            "expected": "DIFFERENT",
            "description": "TinyLlama-1.1B vs Phi-2 (2.7B) (different architectures)"
        },
        {
            "id": "arch_neo_tinyllama",
            "category": "architecture",
            "ref": "EleutherAI/gpt-neo-1.3B",
            "cand": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "expected": "DIFFERENT",
            "description": "GPT-Neo-1.3B vs TinyLlama-1.1B (similar size, different arch)"
        },

        # ============================================
        # Category 6: Large Model Tests (7B class)
        # ============================================
        {
            "id": "large_falcon_llama",
            "category": "large_models",
            "ref": "tiiuae/falcon-7b",
            "cand": "NousResearch/Llama-2-7b-hf",
            "expected": "DIFFERENT",
            "description": "Falcon-7B vs Llama-2-7B (different 7B models)"
        },

        # ============================================
        # Category 7: 20B+ Model Test (if hardware permits)
        # ============================================
        {
            "id": "xl_neox_20b_self",
            "category": "xl_models",
            "ref": "EleutherAI/gpt-neox-20b",
            "cand": "EleutherAI/gpt-neox-20b",
            "expected": "SAME",
            "description": "GPT-NeoX-20B self-consistency (stress test)",
            "skip_if_low_memory": True
        },
    ]
)


def check_memory_available(min_gb: float = 16.0) -> bool:
    """Check if sufficient memory is available"""
    try:
        import psutil
        available_gb = psutil.virtual_memory().available / (1024**3)
        return available_gb >= min_gb
    except ImportError:
        return True  # Assume OK if psutil not available


def run_single_experiment(
    exp: Dict,
    output_dir: Path,
    mode: TestingMode = TestingMode.AUDIT_GRADE
) -> ExperimentResult:
    """Run a single experiment and return results"""

    exp_id = exp["id"]
    logger.info(f"\n{'='*60}")
    logger.info(f"Starting experiment: {exp_id}")
    logger.info(f"  {exp['description']}")
    logger.info(f"  Ref: {exp['ref']}")
    logger.info(f"  Cand: {exp['cand']}")
    logger.info(f"  Expected: {exp['expected']}")
    logger.info(f"{'='*60}")

    # Skip large models if low memory
    if exp.get("skip_if_low_memory") and not check_memory_available(32.0):
        logger.warning(f"Skipping {exp_id} due to low memory")
        return ExperimentResult(
            experiment_id=exp_id,
            category=exp["category"],
            ref_model=exp["ref"],
            cand_model=exp["cand"],
            expected_decision=exp["expected"],
            actual_decision="SKIPPED",
            confidence=0.0,
            queries_used=0,
            duration_seconds=0.0,
            correct=False,
            error="Insufficient memory"
        )

    start_time = time.time()

    try:
        # Configure pipeline (model paths passed to run_complete_pipeline, not config)
        config = PipelineConfig(
            testing_mode=mode,
            verification_mode=VerificationMode.LOCAL_WEIGHTS,
            output_dir=Path(output_dir / exp_id),
            enable_zk_proof=False,  # Skip ZK for faster experiments
            enable_sharding=True,  # Enable for large models
            max_memory_percent=80,
        )

        # Run pipeline with model paths
        orchestrator = PipelineOrchestrator(config)
        result = orchestrator.run_complete_pipeline(
            ref_model_path=exp["ref"],
            cand_model_path=exp["cand"]
        )

        duration = time.time() - start_time

        # Extract decision
        decision = result.get("decision", "UNDECIDED")
        confidence = result.get("confidence", 0.0)
        queries = result.get("n_queries", 0)  # API uses n_queries not queries_used

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


def generate_publication_report(results: List[ExperimentResult], output_dir: Path) -> Dict:
    """Generate a publication-ready summary report"""

    # Calculate statistics by category
    categories = {}
    for r in results:
        if r.category not in categories:
            categories[r.category] = {"total": 0, "correct": 0, "results": []}
        categories[r.category]["total"] += 1
        if r.correct:
            categories[r.category]["correct"] += 1
        categories[r.category]["results"].append(asdict(r))

    # Calculate overall statistics
    total = len(results)
    correct = sum(1 for r in results if r.correct)
    errors = sum(1 for r in results if r.error)
    skipped = sum(1 for r in results if r.actual_decision == "SKIPPED")

    avg_queries = sum(r.queries_used for r in results if r.queries_used > 0) / max(1, total - skipped - errors)
    avg_duration = sum(r.duration_seconds for r in results) / max(1, total - skipped)

    report = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "suite_name": PUBLICATION_EXPERIMENTS.name,
            "mode": PUBLICATION_EXPERIMENTS.mode,
            "total_experiments": total,
        },
        "summary": {
            "accuracy": correct / max(1, total - skipped) if total > skipped else 0,
            "correct": correct,
            "incorrect": total - correct - skipped - errors,
            "skipped": skipped,
            "errors": errors,
            "avg_queries_per_decision": round(avg_queries, 1),
            "avg_duration_seconds": round(avg_duration, 1),
        },
        "by_category": {
            cat: {
                "accuracy": data["correct"] / data["total"] if data["total"] > 0 else 0,
                "correct": data["correct"],
                "total": data["total"],
            }
            for cat, data in categories.items()
        },
        "detailed_results": [asdict(r) for r in results],
    }

    # Save report
    report_path = output_dir / "publication_results.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    # Generate markdown summary
    md_report = generate_markdown_report(report)
    md_path = output_dir / "PUBLICATION_RESULTS.md"
    with open(md_path, 'w') as f:
        f.write(md_report)

    logger.info(f"\nResults saved to:")
    logger.info(f"  JSON: {report_path}")
    logger.info(f"  Markdown: {md_path}")

    return report


def generate_markdown_report(report: Dict) -> str:
    """Generate markdown-formatted publication report"""

    md = f"""# Proof-of-Training Verification Results

**Generated:** {report['metadata']['generated_at']}
**Test Suite:** {report['metadata']['suite_name']}
**Testing Mode:** {report['metadata']['mode']} (99% confidence)

## Executive Summary

| Metric | Value |
|--------|-------|
| **Overall Accuracy** | {report['summary']['accuracy']:.1%} |
| **Correct Decisions** | {report['summary']['correct']} / {report['metadata']['total_experiments']} |
| **Avg Queries/Decision** | {report['summary']['avg_queries_per_decision']} |
| **Avg Duration** | {report['summary']['avg_duration_seconds']:.1f}s |
| **Errors** | {report['summary']['errors']} |
| **Skipped** | {report['summary']['skipped']} |

## Results by Category

| Category | Accuracy | Correct | Total |
|----------|----------|---------|-------|
"""

    for cat, stats in report['by_category'].items():
        md += f"| {cat.replace('_', ' ').title()} | {stats['accuracy']:.1%} | {stats['correct']} | {stats['total']} |\n"

    md += """
## Detailed Results

| Experiment | Ref Model | Cand Model | Expected | Actual | Correct | Confidence | Queries | Time |
|------------|-----------|------------|----------|--------|---------|------------|---------|------|
"""

    for r in report['detailed_results']:
        status = "✅" if r['correct'] else ("⏭️" if r['actual_decision'] == "SKIPPED" else "❌")
        ref_short = r['ref_model'].split('/')[-1][:20]
        cand_short = r['cand_model'].split('/')[-1][:20]
        md += f"| {r['experiment_id']} | {ref_short} | {cand_short} | {r['expected_decision']} | {r['actual_decision']} | {status} | {r['confidence']:.3f} | {r['queries_used']} | {r['duration_seconds']:.1f}s |\n"

    md += """
## Methodology

The Proof-of-Training (PoT) framework uses sequential statistical testing with
Empirical-Bernstein confidence bounds to make anytime-valid decisions about model
identity. The key properties are:

1. **Pre-committed challenges**: HMAC-SHA256 derived challenge seeds prevent cherry-picking
2. **Sequential testing**: Early stopping when sufficient confidence is reached
3. **Separate decision rules**: Distinct criteria for SAME vs DIFFERENT decisions
4. **Behavioral fingerprinting**: Detection of stable intermediate states

### Testing Modes
- **QUICK_GATE**: 97.5% confidence, max 120 queries (screening)
- **AUDIT_GRADE**: 99% confidence, max 400 queries (publication quality)
- **EXTENDED**: 99.9% confidence, max 800 queries (high-stakes)

This experiment suite used **AUDIT_GRADE** mode for publication-quality results.

## Citation

If you use these results, please cite:
```
@article{pot2024,
  title={Proof-of-Training: Black-Box Behavioral Verification of Neural Networks},
  author={...},
  journal={...},
  year={2024}
}
```
"""

    return md


def main():
    parser = argparse.ArgumentParser(
        description='Run publication-quality PoT experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--output-dir', type=str, default='experimental_results/publication',
        help='Output directory for results'
    )
    parser.add_argument(
        '--mode', type=str, choices=['quick', 'audit', 'extended'], default='audit',
        help='Testing mode (default: audit for publication quality)'
    )
    parser.add_argument(
        '--categories', type=str, nargs='+',
        help='Only run specific categories (e.g., self_consistency distillation)'
    )
    parser.add_argument(
        '--skip-large', action='store_true',
        help='Skip experiments requiring >7B models'
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Print experiment plan without running'
    )

    args = parser.parse_args()

    # Setup output directory
    output_dir = Path(args.output_dir)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = output_dir / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Select testing mode
    mode_map = {
        'quick': TestingMode.QUICK_GATE,
        'audit': TestingMode.AUDIT_GRADE,
    }
    mode = mode_map.get(args.mode, TestingMode.AUDIT_GRADE)

    # Filter experiments
    experiments = PUBLICATION_EXPERIMENTS.experiments

    if args.categories:
        experiments = [e for e in experiments if e['category'] in args.categories]

    if args.skip_large:
        experiments = [e for e in experiments if e['category'] not in ['large_models', 'xl_models']]

    logger.info(f"\n{'='*60}")
    logger.info("POT PUBLICATION EXPERIMENT SUITE")
    logger.info(f"{'='*60}")
    logger.info(f"Total experiments: {len(experiments)}")
    logger.info(f"Testing mode: {args.mode} ({mode})")
    logger.info(f"Output directory: {output_dir}")

    if args.dry_run:
        logger.info("\n--- DRY RUN - Experiment Plan ---")
        for i, exp in enumerate(experiments, 1):
            logger.info(f"{i:2d}. [{exp['category']}] {exp['id']}: {exp['description']}")
            logger.info(f"    {exp['ref']} vs {exp['cand']} -> Expected: {exp['expected']}")
        return

    # Run experiments
    results = []
    for i, exp in enumerate(experiments, 1):
        logger.info(f"\n[{i}/{len(experiments)}] Running experiment...")
        result = run_single_experiment(exp, output_dir, mode)
        results.append(result)

        # Save intermediate results
        intermediate_path = output_dir / "intermediate_results.json"
        with open(intermediate_path, 'w') as f:
            json.dump([asdict(r) for r in results], f, indent=2)

    # Generate final report
    logger.info("\n" + "="*60)
    logger.info("GENERATING PUBLICATION REPORT")
    logger.info("="*60)

    report = generate_publication_report(results, output_dir)

    # Print summary
    logger.info("\n" + "="*60)
    logger.info("EXPERIMENT SUITE COMPLETE")
    logger.info("="*60)
    logger.info(f"Accuracy: {report['summary']['accuracy']:.1%}")
    logger.info(f"Correct: {report['summary']['correct']}/{report['metadata']['total_experiments']}")
    logger.info(f"Avg queries: {report['summary']['avg_queries_per_decision']}")
    logger.info(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
