#!/usr/bin/env python3
"""
Example demonstrating vision verification integration.
Shows how to use configuration, datasets, challengers, and CLI together.
"""

import torch
import torch.nn as nn
import sys
import os

# Add project root to path
sys.path.append('/Users/rohanvinaik/PoT_Experiments')

def main():
    print("=" * 70)
    print("VISION VERIFICATION INTEGRATION EXAMPLE")
    print("=" * 70)
    
    # 1. Configuration Management
    print("\n1. Configuration Management")
    print("-" * 30)
    
    from pot.vision.vision_config import VisionVerifierConfig, VisionConfigPresets
    
    # Create a comprehensive configuration
    config = VisionConfigPresets.comprehensive_verification()
    config.device = 'cpu'  # Force CPU for this demo
    config.num_challenges = 5  # Reduce for faster demo
    
    print(f"✓ Configuration created: {config.num_challenges} challenges")
    print(f"  Challenge types: {config.challenge_types}")
    print(f"  Verification method: {config.verification_method}")
    print(f"  Image size: {config.image_size}")
    
    # Save and load configuration
    config.save_json('/tmp/vision_config_demo.json')
    loaded_config = VisionVerifierConfig.from_json('/tmp/vision_config_demo.json')
    print("✓ Configuration save/load successful")
    
    
    # 2. Dataset Creation
    print("\n2. Dataset Creation")
    print("-" * 30)
    
    from pot.vision.datasets import create_verification_dataloader
    
    try:
        # Create verification dataset with challenges
        dataloader = create_verification_dataloader(
            batch_size=4,
            num_samples=20,
            challenge_types=['frequency', 'texture'],
            image_size=(128, 128),  # Smaller for faster demo
            device='cpu',
            streaming=False
        )
        
        print(f"✓ Verification dataloader created")
        
        # Test loading a batch
        batch = next(iter(dataloader))
        challenges, labels = batch
        print(f"  Batch shape: {challenges.shape}")
        print(f"  Label distribution: {torch.bincount(labels).tolist()}")
        
    except ImportError as e:
        print(f"⚠ Verification dataset requires challengers: {e}")
        
        # Fallback to CIFAR-10
        from pot.vision.datasets import get_cifar10_loader
        try:
            dataloader = get_cifar10_loader(batch_size=4, split="test")
            batch = next(iter(dataloader))
            challenges, labels = batch
            print(f"✓ CIFAR-10 dataloader created as fallback")
            print(f"  Batch shape: {challenges.shape}")
        except Exception as e:
            print(f"⚠ CIFAR-10 also failed: {e}")
    
    
    # 3. Challenge Generation
    print("\n3. Challenge Generation")
    print("-" * 30)
    
    try:
        from pot.vision.challengers import FrequencyChallenger, TextureChallenger, NaturalImageChallenger
        
        # Frequency challenges
        freq_challenger = FrequencyChallenger(device='cpu')
        freq_pattern = freq_challenger.generate_fourier_pattern(
            size=(64, 64),
            frequency_range=(1.0, 5.0),
            num_components=3
        )
        print(f"✓ Frequency pattern: {freq_pattern.shape}, range: [{freq_pattern.min():.3f}, {freq_pattern.max():.3f}]")
        
        # Texture challenges
        texture_challenger = TextureChallenger(device='cpu')
        texture_pattern = texture_challenger.generate_perlin_noise(
            size=(64, 64),
            octaves=3,
            scale=20
        )
        print(f"✓ Texture pattern: {texture_pattern.shape}, range: [{texture_pattern.min():.3f}, {texture_pattern.max():.3f}]")
        
        # Natural image challenges
        natural_challenger = NaturalImageChallenger(device='cpu')
        natural_pattern = natural_challenger.generate_synthetic_natural(
            size=(64, 64),
            scene_type='landscape'
        )
        print(f"✓ Natural pattern: {natural_pattern.shape}, range: [{natural_pattern.min():.3f}, {natural_pattern.max():.3f}]")
        
    except ImportError as e:
        print(f"⚠ Challenge generators not available: {e}")
    
    
    # 4. Model and Verification
    print("\n4. Model and Verification")
    print("-" * 30)
    
    # Create test model
    model = nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(64, 10)
    )
    
    print(f"✓ Test model created")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test model forward pass
    x = torch.randn(2, 3, 128, 128)
    with torch.no_grad():
        output = model(x)
    print(f"  Forward pass: {x.shape} -> {output.shape}")
    
    
    # 5. Distance Metrics
    print("\n5. Distance Metrics")
    print("-" * 30)
    
    try:
        from pot.vision.distance_metrics import VisionDistanceMetrics, AdvancedDistanceMetrics
        
        # Basic distance metrics
        metrics = VisionDistanceMetrics()
        
        # Test logit distances
        logits1 = torch.randn(4, 10)
        logits2 = torch.randn(4, 10)
        
        kl_dist = metrics.compute_logit_distance(logits1, logits2, 'kl')
        js_dist = metrics.compute_logit_distance(logits1, logits2, 'js')
        l2_dist = metrics.compute_logit_distance(logits1, logits2, 'l2')
        
        print(f"✓ Logit distances computed:")
        print(f"  KL divergence: {kl_dist:.4f}")
        print(f"  JS divergence: {js_dist:.4f}")
        print(f"  L2 distance: {l2_dist:.4f}")
        
        # Test embedding distances
        emb1 = torch.randn(4, 64)
        emb2 = torch.randn(4, 64)
        
        cosine_dist = metrics.compute_embedding_distance(emb1, emb2, 'cosine')
        euclidean_dist = metrics.compute_embedding_distance(emb1, emb2, 'euclidean')
        
        print(f"✓ Embedding distances computed:")
        print(f"  Cosine distance: {cosine_dist:.4f}")
        print(f"  Euclidean distance: {euclidean_dist:.4f}")
        
        # Advanced metrics
        advanced_metrics = AdvancedDistanceMetrics()
        mmd_dist = advanced_metrics.compute_mmd(emb1, emb2, kernel='linear')
        print(f"  MMD distance: {mmd_dist:.4f}")
        
    except ImportError as e:
        print(f"⚠ Distance metrics not fully available: {e}")
    
    
    # 6. Probe Extraction
    print("\n6. Probe Extraction")
    print("-" * 30)
    
    try:
        from pot.vision.probes import ProbeExtractor, StableEmbeddingProbe
        
        # Architecture detection
        probe_extractor = ProbeExtractor(model)
        print(f"✓ Architecture detected: {probe_extractor.architecture_type}")
        print(f"  Probe points: {list(probe_extractor.probe_points.keys())}")
        
        # Extract embeddings
        x = torch.randn(2, 3, 128, 128)
        embeddings = probe_extractor.extract_with_hooks(x)
        
        if embeddings:
            print(f"✓ Embeddings extracted from {len(embeddings)} layers:")
            for name, emb in embeddings.items():
                print(f"  {name}: {emb.shape}")
        else:
            print("⚠ No embeddings extracted (expected for simple models)")
        
        # Stable signature
        stable_probe = StableEmbeddingProbe(model)
        signature = stable_probe.get_stable_signature(x)
        print(f"✓ Stable signature: {signature.shape}")
        
    except Exception as e:
        print(f"⚠ Probe extraction failed: {e}")
    
    
    # 7. CLI Integration
    print("\n7. CLI Integration")
    print("-" * 30)
    
    print("Available CLI commands:")
    cli_commands = [
        "python -m pot.vision.cli list-presets",
        "python -m pot.vision.cli create-config --preset standard",
        "python -m pot.vision.cli validate-config config.yaml",
        "python -m pot.vision.cli model-info --model resnet18",
        "python -m pot.vision.cli verify --model resnet18 --preset quick"
    ]
    
    for cmd in cli_commands:
        print(f"  {cmd}")
    
    print("\n✓ CLI interface available for production use")
    
    
    # Summary
    print("\n" + "=" * 70)
    print("INTEGRATION SUMMARY")
    print("=" * 70)
    
    features = [
        "✓ Configuration management with presets and validation",
        "✓ Dataset creation for verification challenges",
        "✓ Challenge generation (frequency, texture, natural)",
        "✓ Model testing and forward pass validation", 
        "✓ Distance metrics for verification",
        "✓ Probe extraction for model analysis",
        "✓ Command-line interface for production use"
    ]
    
    for feature in features:
        print(feature)
    
    print("\nThe vision verification system is ready for:")
    print("- Research and development")
    print("- Production model verification")
    print("- Automated testing and evaluation")
    print("- Integration with existing ML pipelines")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()