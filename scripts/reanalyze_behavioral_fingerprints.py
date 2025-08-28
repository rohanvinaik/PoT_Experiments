#!/usr/bin/env python3
"""
Re-analyze experimental results to classify model relationships using behavioral fingerprinting.
Based on Section 8.1 of the NeurIPS paper.
"""

import json
import os
from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np

def classify_relationship(mean: float, n_queries: int, decision: str) -> Tuple[str, str]:
    """
    Classify model relationship based on behavioral fingerprinting thresholds.
    
    Returns:
        (classification, description)
    """
    abs_mean = abs(mean)
    
    # If we got a clear SAME/DIFFERENT decision with good confidence, use that
    if decision == "SAME" and abs_mean < 0.025:
        return "SAME", "Identical model (high confidence)"
    
    if decision == "DIFFERENT" and n_queries < 50:
        # Strong difference detected quickly
        if abs_mean > 10:
            return "DIFFERENT_ARCH", "Different architecture"
        elif abs_mean > 5:
            return "DIFFERENT_TRAINING", "Different training/distillation"
        else:
            return "DIFFERENT_TUNED", "Different fine-tuning"
    
    # For cases that took many queries or had intermediate values
    if abs_mean < 0.001:
        return "NEAR_CLONE", "Near-identical (quantization/precision differences)"
    elif abs_mean < 0.01:
        return "SAME_ARCH_FINE_TUNED", "Same architecture, fine-tuned variant"
    elif abs_mean < 0.1:
        return "SAME_ARCH_DIFFERENT_SCALE", "Same architecture, different scale"
    elif abs_mean < 1.0:
        return "SIMILAR_FAMILY", "Similar model family"
    elif abs_mean < 5.0:
        return "RELATED_TRAINING", "Related training approach"
    else:
        return "DIFFERENT_ARCH_SIMILAR_TRAINING", "Different architecture/training"

def extract_metrics_from_experiment(exp_dir: Path) -> Optional[Dict]:
    """Extract key metrics from an experiment directory."""
    try:
        # First try pipeline results
        pipeline_files = list(exp_dir.glob("pipeline_results_*.json"))
        evidence_files = list(exp_dir.glob("evidence_bundle_*.json"))
        
        if not pipeline_files or not evidence_files:
            return None
        
        # Load pipeline results for basic info
        with open(pipeline_files[0], 'r') as f:
            pipeline_data = json.load(f)
        
        decision = pipeline_data.get('decision', 'UNKNOWN')
        n_queries = pipeline_data.get('n_queries', 0)
        
        # Load evidence bundle for mean value
        with open(evidence_files[0], 'r') as f:
            evidence_data = json.load(f)
        
        # Find mean in evidence bundle
        mean = None
        
        # Check verification section
        if 'verification' in evidence_data:
            verif = evidence_data['verification']
            if 'results' in verif and isinstance(verif['results'], dict):
                mean = verif['results'].get('mean')
            elif 'mean' in verif:
                mean = verif['mean']
        
        # Check transcript entries
        if mean is None and 'transcript_entries' in evidence_data:
            entries = evidence_data['transcript_entries']
            if isinstance(entries, list):
                for entry in reversed(entries):
                    if isinstance(entry, dict):
                        if 'result' in entry and isinstance(entry['result'], dict):
                            if 'mean' in entry['result']:
                                mean = entry['result']['mean']
                                break
                        if 'output' in entry and isinstance(entry['output'], dict):
                            if 'mean' in entry['output']:
                                mean = entry['output']['mean']
                                break
        
        # Extract model info - try multiple locations
        ref_model = "unknown"
        cand_model = "unknown"
        
        # Try metrics metadata first
        if 'metrics' in evidence_data:
            for stage_name, stage_data in evidence_data['metrics'].items():
                if isinstance(stage_data, dict) and 'metadata' in stage_data:
                    metadata = stage_data['metadata']
                    if 'ref_model' in metadata:
                        ref_model = metadata['ref_model'].split('/')[-1]
                    if 'cand_model' in metadata:
                        cand_model = metadata['cand_model'].split('/')[-1]
                    if ref_model != "unknown":
                        break
        
        # Try config as backup
        if ref_model == "unknown" and 'config' in evidence_data:
            config = evidence_data['config']
            if 'ref' in config and 'model_path' in config['ref']:
                ref_model = config['ref']['model_path'].split('/')[-1]
            if 'cand' in config and 'model_path' in config['cand']:
                cand_model = config['cand']['model_path'].split('/')[-1]
        
        # Try from transcript entries as backup
        if ref_model == "unknown" and 'transcript_entries' in evidence_data:
            entries = evidence_data['transcript_entries']
            if isinstance(entries, list):
                for entry in entries:
                    if isinstance(entry, dict) and 'input' in entry:
                        inp = entry['input']
                        if isinstance(inp, dict):
                            if 'ref_model' in inp:
                                ref_model = inp['ref_model'].split('/')[-1]
                            if 'cand_model' in inp:
                                cand_model = inp['cand_model'].split('/')[-1]
                            if ref_model != "unknown":
                                break
        
        return {
            'ref_model': ref_model,
            'cand_model': cand_model,
            'decision': decision,
            'n_queries': n_queries,
            'mean': mean,
            'exp_dir': exp_dir.name
        }
    except Exception as e:
        print(f"Error processing {exp_dir}: {e}")
        return None

def analyze_all_experiments():
    """Analyze all experimental results and classify relationships."""
    results_dir = Path("experimental_results")
    
    print("=" * 80)
    print("BEHAVIORAL FINGERPRINTING ANALYSIS")
    print("=" * 80)
    print()
    
    analyzed = []
    
    # Find all experiments with results
    for exp_dir in results_dir.iterdir():
        if not exp_dir.is_dir():
            continue
        
        metrics = extract_metrics_from_experiment(exp_dir)
        if metrics and metrics['mean'] is not None:
            classification, description = classify_relationship(
                metrics['mean'], 
                metrics['n_queries'],
                metrics['decision']
            )
            
            analyzed.append({
                'experiment': exp_dir.name,
                'ref_model': metrics['ref_model'],
                'cand_model': metrics['cand_model'],
                'mean': metrics['mean'],
                'n_queries': metrics['n_queries'],
                'decision': metrics['decision'],
                'classification': classification,
                'description': description
            })
    
    # Sort by mean value for better presentation
    analyzed.sort(key=lambda x: abs(x['mean']))
    
    # Print results
    print("SELF-CONSISTENCY TESTS (SAME expected):")
    print("-" * 80)
    for result in analyzed:
        if result['ref_model'] == result['cand_model']:
            print(f"{result['ref_model']:20} | mean={result['mean']:8.4f} | n={result['n_queries']:3} | {result['classification']:25} | {result['description']}")
    
    print()
    print("CROSS-MODEL TESTS (DIFFERENT expected):")
    print("-" * 80)
    for result in analyzed:
        if result['ref_model'] != result['cand_model']:
            print(f"{result['ref_model']:15} → {result['cand_model']:15} | mean={result['mean']:8.4f} | n={result['n_queries']:3} | {result['classification']:25} | {result['description']}")
    
    print()
    print("RECOMMENDED TABLE ENTRIES WITH BEHAVIORAL CLASSIFICATIONS:")
    print("-" * 80)
    
    # Select key results for the paper table
    key_results = [
        ("gpt2", "gpt2"),
        ("gpt2", "distilgpt2"),
        ("gpt2-medium", "gpt2"),
        ("DialoGPT-small", "gpt2"),
        ("pythia-70m", "pythia-70m"),
        ("llama-2-7b-hf", "llama-2-7b-hf")
    ]
    
    for ref, cand in key_results:
        for result in analyzed:
            if (ref in result['ref_model'] and cand in result['cand_model']) or \
               (ref in result['cand_model'] and cand in result['ref_model']):
                print(f"{result['ref_model']:15} → {result['cand_model']:15} | {result['classification']:30} | mean={result['mean']:7.3f} | n={result['n_queries']:2}")
                break
    
    # Save results to JSON
    output_file = Path("experimental_results/behavioral_fingerprint_analysis.json")
    with open(output_file, 'w') as f:
        json.dump(analyzed, f, indent=2)
    
    print()
    print(f"Full analysis saved to: {output_file}")

if __name__ == "__main__":
    analyze_all_experiments()