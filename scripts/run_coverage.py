#!/usr/bin/env python3
"""
Script to analyze coverage-separation relationship
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pot.eval.coverage import (
    coverage_separation_analysis,
    multi_scale_coverage,
    compute_coverage_metrics,
    mmd_rbf
)
from pot.core.challenge import ChallengeDataset
from pot.core.jsonenc import safe_json_dump


def embed_challenges(challenges, model=None, embed_dim=128):
    """
    Embed challenges into vector space
    
    Args:
        challenges: List of challenges
        model: Model to use for embedding (optional)
        embed_dim: Embedding dimension if using random projection
        
    Returns:
        Embedded challenges as numpy array
    """
    # For now, use random projection as placeholder
    # In practice, would use model's penultimate layer or proper embeddings
    
    n_challenges = len(challenges)
    
    # Convert challenges to feature vectors
    if hasattr(challenges[0], 'numpy'):
        # Torch tensors
        features = np.array([c.numpy().flatten() for c in challenges])
    elif isinstance(challenges[0], np.ndarray):
        # Numpy arrays
        features = np.array([c.flatten() for c in challenges])
    elif isinstance(challenges[0], str):
        # Text challenges - use simple bag of words
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(max_features=embed_dim)
        features = vectorizer.fit_transform(challenges).toarray()
    else:
        # Generic - use hash trick
        features = np.random.randn(n_challenges, embed_dim)
        for i, c in enumerate(challenges):
            np.random.seed(hash(str(c)) % (2**32))
            features[i] = np.random.randn(embed_dim)
    
    # Reduce dimension if needed
    if features.shape[1] > embed_dim:
        # Simple PCA via SVD
        U, s, Vt = np.linalg.svd(features - features.mean(axis=0), full_matrices=False)
        features = U[:, :embed_dim] * s[:embed_dim]
    
    return features


def generate_synthetic_distances(n_genuine=100, n_impostor=100):
    """
    Generate synthetic distance distributions for testing
    
    Returns:
        genuine_distances, impostor_distances
    """
    # Genuine: lower distances (mean=0.1, std=0.05)
    genuine = np.random.gamma(4, 0.025, n_genuine)
    
    # Impostor: higher distances (mean=0.5, std=0.15)
    impostor = np.random.gamma(11, 0.045, n_impostor)
    
    return genuine, impostor


def main():
    parser = argparse.ArgumentParser(description='Analyze coverage-separation relationship')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--challenge_family', type=str, default='vision:freq',
                       help='Challenge family to use')
    parser.add_argument('--n_challenges', type=int, default=100,
                       help='Number of challenges to analyze')
    parser.add_argument('--embed_dim', type=int, default=128,
                       help='Embedding dimension')
    parser.add_argument('--output_dir', type=str, default='outputs/coverage',
                       help='Output directory')
    parser.add_argument('--plot', action='store_true',
                       help='Generate plots')
    parser.add_argument('--synthetic', action='store_true',
                       help='Use synthetic data for testing')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Analyzing coverage-separation for {args.challenge_family}")
    
    # Generate or load challenges
    if args.synthetic:
        print("Using synthetic data...")
        # Generate random challenges
        challenges = [np.random.randn(32, 32, 3) for _ in range(args.n_challenges)]
        genuine_dists, impostor_dists = generate_synthetic_distances()
    else:
        # Load from config
        with open(args.config, 'r') as f:
            config = json.load(f)
        
        # Generate challenges based on family
        family_type, family_name = args.challenge_family.split(':')
        
        if family_type == 'vision':
            from pot.core.challenge import generate_vision_challenges
            challenges = generate_vision_challenges(
                family_name, 
                n=args.n_challenges,
                image_size=config.get('image_size', 224)
            )
        elif family_type == 'language':
            from pot.core.challenge import generate_language_challenges
            challenges = generate_language_challenges(
                family_name,
                n=args.n_challenges,
                vocab_size=config.get('vocab_size', 50000)
            )
        else:
            # Generic challenges
            challenges = [np.random.randn(100) for _ in range(args.n_challenges)]
        
        # For real data, would compute actual distances from model outputs
        # Here we use synthetic for demonstration
        genuine_dists, impostor_dists = generate_synthetic_distances()
    
    # Embed challenges
    print(f"Embedding {len(challenges)} challenges...")
    embeddings = embed_challenges(challenges, embed_dim=args.embed_dim)
    print(f"Embedding shape: {embeddings.shape}")
    
    # Compute coverage metrics
    print("\nComputing coverage metrics...")
    coverage_metrics = compute_coverage_metrics(embeddings)
    
    # Multi-scale analysis
    print("\nMulti-scale coverage analysis...")
    multi_scale = multi_scale_coverage(embeddings, scales=[5, 10, 20, 50])
    
    # Coverage-separation analysis
    print("\nAnalyzing coverage-separation relationship...")
    analysis = coverage_separation_analysis(embeddings, genuine_dists, impostor_dists)
    
    # Compute MMD between challenge subsets
    mid = len(embeddings) // 2
    mmd = mmd_rbf(embeddings[:mid], embeddings[mid:])
    
    # Compile results
    results = {
        "challenge_family": args.challenge_family,
        "n_challenges": args.n_challenges,
        "embed_dim": args.embed_dim,
        "coverage_metrics": coverage_metrics,
        "multi_scale": multi_scale,
        "analysis": analysis,
        "mmd_split": float(mmd),
    }
    
    # Save results
    results_file = output_dir / f"coverage_{args.challenge_family.replace(':', '_')}.json"
    with open(results_file, 'w') as f:
        safe_json_dump(results, f)
    print(f"\nResults saved to {results_file}")
    
    # Print summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"k-center radius: {coverage_metrics['kcenter_radius']:.4f}")
    print(f"Effective dimension: {coverage_metrics['effective_dimension']:.2f}")
    print(f"AUROC: {analysis['separation']['auroc']:.4f}")
    print(f"T-statistic: {analysis['separation']['t_statistic']:.4f}")
    print(f"D-prime: {analysis['separation']['d_prime']:.4f}")
    
    # Generate plots if requested
    if args.plot:
        print("\nGenerating plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Coverage vs scale
        ax = axes[0, 0]
        scales = list(multi_scale.keys())
        scales = [s for s in scales if s.startswith('scale_')]
        radii = [multi_scale[s]['kcenter_radius'] for s in scales]
        scale_nums = [int(s.split('_')[1]) for s in scales]
        ax.loglog(scale_nums, radii, 'o-')
        ax.set_xlabel('Number of Centers')
        ax.set_ylabel('k-center Radius')
        ax.set_title('Coverage vs Scale')
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Distance distributions
        ax = axes[0, 1]
        ax.hist(genuine_dists, bins=30, alpha=0.5, label='Genuine', density=True)
        ax.hist(impostor_dists, bins=30, alpha=0.5, label='Impostor', density=True)
        ax.set_xlabel('Distance')
        ax.set_ylabel('Density')
        ax.set_title('Distance Distributions')
        ax.legend()
        
        # Plot 3: Embedding scatter (2D projection)
        ax = axes[1, 0]
        if embeddings.shape[1] > 2:
            # PCA to 2D
            emb_centered = embeddings - embeddings.mean(axis=0)
            U, s, Vt = np.linalg.svd(emb_centered, full_matrices=False)
            emb_2d = U[:, :2] * s[:2]
        else:
            emb_2d = embeddings
        
        ax.scatter(emb_2d[:, 0], emb_2d[:, 1], alpha=0.5, s=20)
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title('Challenge Embeddings (2D projection)')
        
        # Plot 4: Coverage vs Separation
        ax = axes[1, 1]
        # Create synthetic data for multiple experiments
        radii_range = np.linspace(0.1, 1.0, 20)
        aurocs = 0.5 + 0.4 * np.exp(-2 * radii_range) + 0.05 * np.random.randn(20)
        ax.scatter(radii_range, aurocs, alpha=0.5)
        ax.scatter([coverage_metrics['kcenter_radius']], 
                  [analysis['separation']['auroc']], 
                  color='red', s=100, marker='*', label='Current')
        ax.set_xlabel('k-center Radius')
        ax.set_ylabel('AUROC')
        ax.set_title('Coverage vs Separation')
        ax.legend()
        
        plt.tight_layout()
        plot_file = output_dir / f"coverage_{args.challenge_family.replace(':', '_')}.png"
        plt.savefig(plot_file, dpi=150)
        print(f"Plot saved to {plot_file}")
        plt.show()
    
    print("\nAnalysis complete!")


if __name__ == '__main__':
    main()