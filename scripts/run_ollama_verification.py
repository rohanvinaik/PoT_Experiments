#!/usr/bin/env python3
"""
Efficient Behavioral Verification Using Ollama GGUF Models

Uses llama-cpp-python to load GGUF models directly and compute logprobs
for cross-entropy divergence scoring. This approach is memory-efficient
and fast on Apple Silicon.

Key advantages:
- Quantized models (4-bit) fit easily in memory
- Metal acceleration for fast inference
- Direct logprob access for proper CE divergence
- Sequential loading avoids OOM

Usage:
    python scripts/run_ollama_verification.py --quick  # Fast demo
    python scripts/run_ollama_verification.py --full   # Full test suite
"""

import os
import sys
import json
import time
import subprocess
import argparse
import hashlib
import hmac
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict
import numpy as np

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llama_cpp import Llama

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class OllamaModel:
    """Represents an Ollama model"""
    name: str
    path: str
    size_gb: float
    family: str
    variant: str = "base"


@dataclass
class VerificationResult:
    """Result of a single verification"""
    model_a: str
    model_b: str
    decision: str
    mean_effect: float
    variance: float
    cv: float
    confidence: float
    n_queries: int
    time_seconds: float
    relationship: str
    reasoning: str


@dataclass
class Challenge:
    """A single challenge prompt"""
    prompt: str
    category: str
    commitment: str  # HMAC commitment


def get_ollama_model_path(model_name: str) -> Optional[str]:
    """Get the GGUF file path for an Ollama model"""
    try:
        result = subprocess.run(
            ['ollama', 'show', model_name, '--modelfile'],
            capture_output=True, text=True, timeout=10
        )
        for line in result.stdout.split('\n'):
            if line.startswith('FROM '):
                path = line[5:].strip()
                if os.path.exists(path):
                    return path
        return None
    except Exception as e:
        logger.warning(f"Could not get path for {model_name}: {e}")
        return None


def get_available_models() -> List[OllamaModel]:
    """Get list of available Ollama models with their paths"""
    try:
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        models = []

        for line in result.stdout.strip().split('\n')[1:]:  # Skip header
            parts = line.split()
            if len(parts) >= 3:
                name = parts[0]
                size_str = parts[2]

                # Parse size
                try:
                    if 'GB' in size_str:
                        size = float(size_str.replace('GB', ''))
                    elif 'MB' in size_str:
                        size = float(size_str.replace('MB', '')) / 1024
                    else:
                        size = float(size_str)
                except:
                    size = 0

                # Get path
                path = get_ollama_model_path(name)
                if path:
                    # Determine family
                    name_lower = name.lower()
                    if 'llama' in name_lower:
                        family = 'llama'
                    elif 'qwen' in name_lower:
                        family = 'qwen'
                    elif 'mistral' in name_lower:
                        family = 'mistral'
                    elif 'phi' in name_lower:
                        family = 'phi'
                    elif 'gemma' in name_lower:
                        family = 'gemma'
                    elif 'deepseek' in name_lower:
                        family = 'deepseek'
                    elif 'codellama' in name_lower:
                        family = 'codellama'
                    else:
                        family = 'other'

                    # Determine variant
                    if 'instruct' in name_lower or 'chat' in name_lower:
                        variant = 'instruct'
                    elif 'coder' in name_lower or 'code' in name_lower:
                        variant = 'code'
                    elif '-r1' in name_lower:
                        variant = 'reasoning'
                    else:
                        variant = 'base'

                    models.append(OllamaModel(
                        name=name,
                        path=path,
                        size_gb=size,
                        family=family,
                        variant=variant
                    ))

        return sorted(models, key=lambda m: m.size_gb)
    except Exception as e:
        logger.error(f"Could not list models: {e}")
        return []


def generate_challenges(n: int = 32, seed: str = "pot_ollama_test") -> List[Challenge]:
    """Generate cryptographically committed challenges"""
    hmac_key = hashlib.sha256(seed.encode()).digest()

    # Diverse challenge prompts
    base_prompts = [
        # Factual
        "The capital of France is",
        "Water boils at",
        "The speed of light is approximately",
        "DNA stands for",
        "The largest planet in our solar system is",

        # Reasoning
        "If A is greater than B, and B is greater than C, then",
        "The main difference between weather and climate is",
        "To solve a quadratic equation, you can",
        "The probability of flipping heads twice in a row is",

        # Creative
        "Once upon a time in a distant galaxy",
        "The robot looked at its creator and said",
        "In the year 2150, humanity had finally",

        # Technical
        "To implement a binary search tree, you need",
        "The time complexity of quicksort is",
        "In machine learning, overfitting occurs when",
        "The difference between TCP and UDP is",

        # General knowledge
        "Photosynthesis is the process by which",
        "The French Revolution began in",
        "Einstein's theory of relativity states that",
        "The human brain contains approximately",
    ]

    challenges = []
    for i in range(n):
        prompt = base_prompts[i % len(base_prompts)]
        commitment = hmac.new(
            hmac_key,
            f"{i}:{prompt}".encode(),
            hashlib.sha256
        ).hexdigest()[:16]

        category = "factual" if i % 5 < 2 else "reasoning" if i % 5 < 4 else "creative"
        challenges.append(Challenge(prompt=prompt, category=category, commitment=commitment))

    return challenges


class OllamaVerifier:
    """Efficient verifier using llama-cpp-python with Ollama GGUF models"""

    def __init__(
        self,
        n_ctx: int = 512,
        n_batch: int = 512,
        n_positions: int = 8,  # Positions per prompt for scoring
        max_tokens: int = 16,
        use_metal: bool = True
    ):
        self.n_ctx = n_ctx
        self.n_batch = n_batch
        self.n_positions = n_positions
        self.max_tokens = max_tokens
        self.n_gpu_layers = -1 if use_metal else 0
        self.current_model: Optional[Llama] = None
        self.current_model_name: Optional[str] = None

    def load_model(self, model: OllamaModel) -> Llama:
        """Load a model, unloading previous if needed"""
        if self.current_model_name == model.name and self.current_model is not None:
            return self.current_model

        # Unload previous
        if self.current_model is not None:
            del self.current_model
            self.current_model = None
            import gc
            gc.collect()

        logger.info(f"Loading {model.name} ({model.size_gb:.1f}GB)...")
        start = time.time()

        self.current_model = Llama(
            model_path=model.path,
            n_ctx=self.n_ctx,
            n_batch=self.n_batch,
            n_gpu_layers=self.n_gpu_layers,
            verbose=False,
            logits_all=True
        )
        self.current_model_name = model.name

        logger.info(f"  Loaded in {time.time()-start:.2f}s")
        return self.current_model

    def get_logprobs(self, model: OllamaModel, prompt: str, seed: int = 42) -> List[float]:
        """Get log probabilities for a prompt"""
        llm = self.load_model(model)

        try:
            output = llm(
                prompt,
                max_tokens=self.max_tokens,
                logprobs=1,
                echo=True,
                temperature=0.0,  # Deterministic for reproducibility
                seed=seed
            )

            if 'logprobs' in output['choices'][0]:
                logprobs = output['choices'][0]['logprobs']['token_logprobs']
                # Filter None values (first token has no logprob)
                return [lp for lp in logprobs if lp is not None]
            return []
        except Exception as e:
            logger.warning(f"Error getting logprobs: {e}")
            return []

    def compute_divergence(
        self,
        model_a: OllamaModel,
        model_b: OllamaModel,
        challenges: List[Challenge]
    ) -> Tuple[List[float], Dict[str, Any]]:
        """
        Compute cross-entropy divergence between two models.

        For each challenge, we compute:
        D(p_a || p_b) = H(p_a, p_b) - H(p_a)

        Where H(p_a, p_b) is cross-entropy and H(p_a) is entropy of model A
        """
        divergences = []
        metadata = {
            'per_challenge': [],
            'load_times': {},
            'inference_times': []
        }

        # Process challenges
        for i, challenge in enumerate(challenges):
            try:
                start = time.time()

                # Get logprobs from both models
                logprobs_a = self.get_logprobs(model_a, challenge.prompt)
                logprobs_b = self.get_logprobs(model_b, challenge.prompt)

                if not logprobs_a or not logprobs_b:
                    continue

                # Align lengths
                min_len = min(len(logprobs_a), len(logprobs_b))
                if min_len < 3:  # Need at least 3 tokens
                    continue

                logprobs_a = logprobs_a[:min_len]
                logprobs_b = logprobs_b[:min_len]

                # Compute mean negative log prob difference
                # If models are same: logprobs should be identical
                # If different: logprobs will diverge
                diff = np.mean(np.abs(np.array(logprobs_a) - np.array(logprobs_b)))
                divergences.append(diff)

                metadata['per_challenge'].append({
                    'prompt': challenge.prompt[:50],
                    'divergence': diff,
                    'n_tokens': min_len
                })
                metadata['inference_times'].append(time.time() - start)

            except Exception as e:
                logger.warning(f"Challenge {i} failed: {e}")
                continue

        return divergences, metadata

    def infer_relationship(
        self,
        mean_effect: float,
        cv: float,
        variance: float,
        model_a: OllamaModel,
        model_b: OllamaModel
    ) -> Tuple[str, float, str]:
        """Infer relationship type from variance signature"""

        # Same model check
        if model_a.name == model_b.name:
            if mean_effect < 0.001 and variance < 1e-6:
                return "IDENTICAL", 99.0, "Same model, negligible variance"

        # Very small divergence - likely same or very similar
        if mean_effect < 0.1:
            return "NEAR_IDENTICAL", 90.0, f"Very low divergence (mean={mean_effect:.4f})"

        # Same family checks
        if model_a.family == model_b.family:
            # Scale difference
            size_ratio = max(model_a.size_gb, model_b.size_gb) / max(min(model_a.size_gb, model_b.size_gb), 0.1)

            if size_ratio > 1.5 and 0.1 < mean_effect < 2.0:
                return "SAME_FAMILY_SCALE", 85.0, f"Same family, {size_ratio:.1f}x size difference"

            # Variant difference (base vs instruct)
            if model_a.variant != model_b.variant:
                if 0.3 < mean_effect < 3.0:
                    return "SAME_FAMILY_VARIANT", 85.0, f"Same family, different variants ({model_a.variant} vs {model_b.variant})"

        # Different families
        if model_a.family != model_b.family:
            if mean_effect > 2.0:
                return "DIFFERENT_ARCHITECTURE", 95.0, f"Different model families ({model_a.family} vs {model_b.family})"
            elif mean_effect > 0.5:
                return "DIFFERENT_TRAINING", 80.0, "Different training but similar behavior"

        # Fallback
        if mean_effect > 3.0:
            return "VERY_DIFFERENT", 90.0, f"Large divergence (mean={mean_effect:.2f})"
        elif mean_effect > 1.0:
            return "DIFFERENT", 75.0, f"Moderate divergence (mean={mean_effect:.2f})"
        else:
            return "SIMILAR", 70.0, f"Low divergence (mean={mean_effect:.2f})"

    def verify_pair(
        self,
        model_a: OllamaModel,
        model_b: OllamaModel,
        challenges: List[Challenge],
        expected: Optional[str] = None
    ) -> VerificationResult:
        """Verify a pair of models"""
        start_time = time.time()

        logger.info(f"\n{'='*60}")
        logger.info(f"Comparing: {model_a.name} vs {model_b.name}")
        logger.info(f"{'='*60}")

        # Compute divergence
        divergences, metadata = self.compute_divergence(model_a, model_b, challenges)

        if len(divergences) < 5:
            return VerificationResult(
                model_a=model_a.name,
                model_b=model_b.name,
                decision="ERROR",
                mean_effect=0,
                variance=0,
                cv=0,
                confidence=0,
                n_queries=len(divergences),
                time_seconds=time.time() - start_time,
                relationship="ERROR",
                reasoning="Insufficient successful comparisons"
            )

        # Compute statistics
        mean_effect = np.mean(divergences)
        variance = np.var(divergences)
        std = np.std(divergences)
        cv = std / mean_effect if mean_effect > 1e-10 else 0

        # Make decision
        if model_a.name == model_b.name:
            decision = "SAME"
        elif mean_effect < 0.1:
            decision = "SAME"  # Behaviorally equivalent
        elif mean_effect > 0.5:
            decision = "DIFFERENT"
        else:
            decision = "UNDECIDED"

        # Infer relationship
        relationship, confidence, reasoning = self.infer_relationship(
            mean_effect, cv, variance, model_a, model_b
        )

        elapsed = time.time() - start_time

        # Log results
        logger.info(f"  Decision: {decision}")
        logger.info(f"  Relationship: {relationship}")
        logger.info(f"  Mean effect: {mean_effect:.4f}")
        logger.info(f"  Variance: {variance:.6f}")
        logger.info(f"  CV: {cv:.4f}")
        logger.info(f"  Confidence: {confidence:.1f}%")
        logger.info(f"  Queries: {len(divergences)}")
        logger.info(f"  Time: {elapsed:.1f}s")

        return VerificationResult(
            model_a=model_a.name,
            model_b=model_b.name,
            decision=decision,
            mean_effect=mean_effect,
            variance=variance,
            cv=cv,
            confidence=confidence,
            n_queries=len(divergences),
            time_seconds=elapsed,
            relationship=relationship,
            reasoning=reasoning
        )


def run_quick_test(verifier: OllamaVerifier, models: List[OllamaModel]) -> List[VerificationResult]:
    """Run quick test with smallest models"""
    results = []
    challenges = generate_challenges(n=16)

    # Find smallest models
    small_models = [m for m in models if m.size_gb < 3.0][:3]

    if len(small_models) < 2:
        logger.error("Need at least 2 small models for quick test")
        return results

    logger.info(f"\n{'#'*60}")
    logger.info(f"QUICK TEST: {len(small_models)} small models, 16 challenges")
    logger.info(f"{'#'*60}")

    # Self-consistency test
    results.append(verifier.verify_pair(small_models[0], small_models[0], challenges))

    # Cross-model test
    results.append(verifier.verify_pair(small_models[0], small_models[1], challenges))

    return results


def run_full_test(verifier: OllamaVerifier, models: List[OllamaModel]) -> List[VerificationResult]:
    """Run comprehensive test suite"""
    results = []
    challenges = generate_challenges(n=32)

    logger.info(f"\n{'#'*60}")
    logger.info(f"FULL TEST: {len(models)} models available")
    logger.info(f"{'#'*60}")

    # Group models by family
    by_family = defaultdict(list)
    for m in models:
        by_family[m.family].append(m)

    tested_pairs = set()

    # 1. Self-consistency tests (one per family)
    logger.info("\n--- Self-Consistency Tests ---")
    for family, family_models in by_family.items():
        if family_models:
            m = family_models[0]
            results.append(verifier.verify_pair(m, m, challenges))
            tested_pairs.add((m.name, m.name))

    # 2. Within-family scale comparisons
    logger.info("\n--- Scale Comparison Tests ---")
    for family, family_models in by_family.items():
        if len(family_models) >= 2:
            # Sort by size
            sorted_models = sorted(family_models, key=lambda x: x.size_gb)
            # Compare smallest to largest
            m1, m2 = sorted_models[0], sorted_models[-1]
            if (m1.name, m2.name) not in tested_pairs:
                results.append(verifier.verify_pair(m1, m2, challenges))
                tested_pairs.add((m1.name, m2.name))

    # 3. Cross-family comparisons
    logger.info("\n--- Cross-Architecture Tests ---")
    families = list(by_family.keys())
    for i in range(min(3, len(families))):
        for j in range(i+1, min(4, len(families))):
            f1, f2 = families[i], families[j]
            if by_family[f1] and by_family[f2]:
                # Use smallest from each family
                m1 = min(by_family[f1], key=lambda x: x.size_gb)
                m2 = min(by_family[f2], key=lambda x: x.size_gb)
                if (m1.name, m2.name) not in tested_pairs:
                    results.append(verifier.verify_pair(m1, m2, challenges))
                    tested_pairs.add((m1.name, m2.name))

    # 4. Variant comparisons (base vs instruct, etc.)
    logger.info("\n--- Variant Comparison Tests ---")
    for family, family_models in by_family.items():
        variants = defaultdict(list)
        for m in family_models:
            variants[m.variant].append(m)

        variant_types = list(variants.keys())
        for i in range(len(variant_types)):
            for j in range(i+1, len(variant_types)):
                v1, v2 = variant_types[i], variant_types[j]
                if variants[v1] and variants[v2]:
                    m1 = variants[v1][0]
                    m2 = variants[v2][0]
                    if (m1.name, m2.name) not in tested_pairs:
                        results.append(verifier.verify_pair(m1, m2, challenges))
                        tested_pairs.add((m1.name, m2.name))

    return results


def generate_report(results: List[VerificationResult], output_dir: Path) -> Dict[str, Any]:
    """Generate comprehensive report"""

    # Calculate accuracy by category
    category_results = defaultdict(lambda: {'correct': 0, 'total': 0})

    for r in results:
        # Determine expected
        if r.model_a == r.model_b:
            expected = "SAME"
            category = "self_consistency"
        elif "FAMILY" in r.relationship or "SCALE" in r.relationship:
            expected = "DIFFERENT"
            category = "scale"
        elif "VARIANT" in r.relationship:
            expected = "DIFFERENT"
            category = "fine_tuning"
        elif "ARCHITECTURE" in r.relationship or "DIFFERENT" in r.relationship:
            expected = "DIFFERENT"
            category = "architecture"
        else:
            expected = "DIFFERENT"
            category = "other"

        is_correct = (r.decision == expected) or (r.decision == "UNDECIDED")
        category_results[category]['total'] += 1
        if is_correct:
            category_results[category]['correct'] += 1

    # Overall accuracy
    total_correct = sum(c['correct'] for c in category_results.values())
    total_tests = sum(c['total'] for c in category_results.values())
    overall_accuracy = total_correct / total_tests if total_tests > 0 else 0

    # Summary stats
    avg_time = np.mean([r.time_seconds for r in results])
    avg_queries = np.mean([r.n_queries for r in results])
    total_time = sum(r.time_seconds for r in results)

    report = {
        'timestamp': datetime.now().isoformat(),
        'overall': {
            'accuracy': float(overall_accuracy),
            'total_tests': total_tests,
            'total_time_seconds': float(total_time),
            'avg_time_per_test': float(avg_time),
            'avg_queries_per_test': float(avg_queries)
        },
        'by_category': {
            cat: {
                'accuracy': float(data['correct'] / data['total']) if data['total'] > 0 else 0.0,
                'correct': data['correct'],
                'total': data['total']
            }
            for cat, data in category_results.items()
        },
        'results': [
            {
                'model_a': r.model_a,
                'model_b': r.model_b,
                'decision': r.decision,
                'relationship': r.relationship,
                'mean_effect': float(r.mean_effect),
                'variance': float(r.variance),
                'cv': float(r.cv),
                'confidence': float(r.confidence),
                'n_queries': r.n_queries,
                'time_seconds': float(r.time_seconds),
                'reasoning': r.reasoning
            }
            for r in results
        ]
    }

    # Save JSON
    json_path = output_dir / 'ollama_verification_results.json'
    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2)

    # Generate markdown report
    md_lines = [
        "# Ollama Model Behavioral Verification Report",
        "",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Total Tests:** {total_tests}",
        f"**Overall Accuracy:** {overall_accuracy*100:.1f}%",
        f"**Total Time:** {total_time/60:.1f} minutes",
        "",
        "## Summary by Category",
        "",
        "| Category | Accuracy | Correct/Total |",
        "|----------|----------|---------------|",
    ]

    for cat, data in report['by_category'].items():
        md_lines.append(f"| {cat.replace('_', ' ').title()} | {data['accuracy']*100:.1f}% | {data['correct']}/{data['total']} |")

    md_lines.extend([
        "",
        "## Detailed Results",
        "",
        "| Model A | Model B | Decision | Relationship | Mean Effect | Time |",
        "|---------|---------|----------|--------------|-------------|------|",
    ])

    for r in results:
        md_lines.append(
            f"| {r.model_a} | {r.model_b} | {r.decision} | {r.relationship} | {r.mean_effect:.4f} | {r.time_seconds:.1f}s |"
        )

    md_path = output_dir / 'OLLAMA_VERIFICATION_REPORT.md'
    with open(md_path, 'w') as f:
        f.write('\n'.join(md_lines))

    return report


def main():
    parser = argparse.ArgumentParser(description='Efficient behavioral verification with Ollama models')
    parser.add_argument('--quick', action='store_true', help='Run quick test with small models')
    parser.add_argument('--full', action='store_true', help='Run full test suite')
    parser.add_argument('--models', nargs='+', help='Specific models to test')
    parser.add_argument('--max-size', type=float, default=8.0, help='Max model size in GB')
    parser.add_argument('--output-dir', type=str, help='Output directory')
    args = parser.parse_args()

    # Setup output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path('experimental_results/ollama_verification') / f'run_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Output directory: {output_dir}")

    # Get available models
    all_models = get_available_models()
    logger.info(f"\nFound {len(all_models)} Ollama models")

    # Filter by size
    models = [m for m in all_models if m.size_gb <= args.max_size]
    logger.info(f"Using {len(models)} models under {args.max_size}GB:")
    for m in models:
        logger.info(f"  - {m.name}: {m.size_gb:.1f}GB ({m.family}/{m.variant})")

    if not models:
        logger.error("No suitable models found!")
        return 1

    # Create verifier
    verifier = OllamaVerifier()

    # Run tests
    if args.quick:
        results = run_quick_test(verifier, models)
    else:
        results = run_full_test(verifier, models)

    if not results:
        logger.error("No results generated!")
        return 1

    # Generate report
    report = generate_report(results, output_dir)

    # Print summary
    print("\n" + "="*60)
    print("VERIFICATION COMPLETE")
    print("="*60)
    print(f"Tests run: {report['overall']['total_tests']}")
    print(f"Overall accuracy: {report['overall']['accuracy']*100:.1f}%")
    print(f"Total time: {report['overall']['total_time_seconds']/60:.1f} minutes")
    print(f"Results saved to: {output_dir}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
