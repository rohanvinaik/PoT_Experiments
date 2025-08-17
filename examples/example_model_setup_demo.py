#!/usr/bin/env python3
"""
Model Setup Demo for PoT Framework

This script demonstrates the comprehensive model setup system with:
- Automatic model downloading and caching
- Multiple configuration presets (minimal, test, paper)
- Robust fallback mechanisms
- Memory usage tracking
- Checksum verification for reproducibility
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from pot.experiments.model_setup import (
    MinimalModelSetup, ModelConfig, ModelPreset,
    get_minimal_vision_model, get_minimal_language_model,
    get_test_models, get_paper_models
)
from pot.experiments.reproducible_runner import ReproducibleExperimentRunner, ExperimentConfig

def demo_basic_model_setup():
    """Demonstrate basic model setup functionality."""
    print("🏗️ Basic Model Setup Demo")
    print("=" * 40)
    
    setup = MinimalModelSetup()
    
    print("\n📋 Available Models:")
    available = setup.list_available_models()
    for model_type, presets in available.items():
        print(f"\n{model_type.title()} Models:")
        for preset, spec in presets.items():
            status = "✅" if spec["available"] else "❌"
            print(f"  {status} {preset}: {spec['description']} ({spec['memory_mb']}MB)")
    
    return setup

def demo_vision_models():
    """Demonstrate vision model setup with different presets."""
    print("\n📸 Vision Models Demo")
    print("=" * 30)
    
    setup = MinimalModelSetup()
    
    for preset_name in ["test", "minimal", "paper"]:
        print(f"\n🔧 Loading {preset_name} vision model...")
        
        config = ModelConfig(
            name=f"demo_vision_{preset_name}",
            model_type="vision",
            preset=ModelPreset(preset_name),
            fallback_to_mock=True
        )
        
        model_info = setup.get_vision_model(config)
        
        print(f"✅ Success!")
        print(f"   Source: {model_info.source}")
        print(f"   Memory: {model_info.memory_mb:.1f}MB")
        print(f"   Config: {model_info.config}")
        
        # Test forward pass
        import torch
        test_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model_info.model(test_input)
        print(f"   Output: {output.shape}")

def demo_language_models():
    """Demonstrate language model setup with different presets."""
    print("\n📝 Language Models Demo")
    print("=" * 35)
    
    setup = MinimalModelSetup()
    
    for preset_name in ["test", "minimal", "paper"]:
        print(f"\n🔧 Loading {preset_name} language model...")
        
        config = ModelConfig(
            name=f"demo_language_{preset_name}",
            model_type="language",
            preset=ModelPreset(preset_name),
            fallback_to_mock=True
        )
        
        model_info = setup.get_language_model(config)
        
        print(f"✅ Success!")
        print(f"   Source: {model_info.source}")
        print(f"   Memory: {model_info.memory_mb:.1f}MB")
        print(f"   Config: {model_info.config}")
        
        # Test tokenizer if available
        if model_info.tokenizer:
            try:
                tokens = model_info.tokenizer.encode("Hello, world!")
                decoded = model_info.tokenizer.decode(tokens[:5])
                print(f"   Tokenizer: 'Hello, world!' -> {tokens[:5]} -> '{decoded}'")
            except:
                print(f"   Tokenizer: Available but encoding failed")

def demo_convenience_functions():
    """Demonstrate convenience functions for quick model access."""
    print("\n🛠️ Convenience Functions Demo")
    print("=" * 40)
    
    print("\n🚀 Quick model access:")
    
    # Minimal models
    vision_model = get_minimal_vision_model()
    print(f"✅ Minimal vision model: {vision_model.memory_mb:.1f}MB ({vision_model.source})")
    
    language_model = get_minimal_language_model()
    print(f"✅ Minimal language model: {language_model.memory_mb:.1f}MB ({language_model.source})")
    
    # Test models
    test_models = get_test_models()
    print(f"✅ Test models: {len(test_models)} models")
    for name, info in test_models.items():
        print(f"   {name}: {info.memory_mb:.1f}MB")
    
    # Paper models
    paper_models = get_paper_models()
    print(f"✅ Paper models: {len(paper_models)} models")
    for name, info in paper_models.items():
        print(f"   {name}: {info.memory_mb:.1f}MB ({info.source})")

def demo_caching_and_performance():
    """Demonstrate caching and performance features."""
    print("\n💾 Caching & Performance Demo")
    print("=" * 40)
    
    setup = MinimalModelSetup()
    
    # Load model multiple times to demonstrate caching
    config = ModelConfig("cache_test", "vision", ModelPreset.TEST)
    
    import time
    
    print("\n⏱️ Performance comparison:")
    
    # First load
    start = time.time()
    model1 = setup.get_vision_model(config)
    first_load_time = time.time() - start
    print(f"First load: {first_load_time:.4f}s")
    
    # Second load (should use cache)
    start = time.time()
    model2 = setup.get_vision_model(config)
    cached_load_time = time.time() - start
    print(f"Cached load: {cached_load_time:.4f}s")
    
    speedup = first_load_time / cached_load_time if cached_load_time > 0 else float('inf')
    print(f"Speedup: {speedup:.1f}x")
    print(f"Same model instance: {model1 is model2}")
    
    # Memory report
    memory_report = setup.get_memory_report()
    print(f"\n📊 Memory usage:")
    print(f"Total: {memory_report['total_memory_mb']:.1f}MB")
    print(f"Models loaded: {len(memory_report['models'])}")
    if memory_report['models']:
        largest = memory_report['largest_model']
        print(f"Largest model: {largest[0]} ({largest[1]:.1f}MB)")

def demo_fallback_mechanism():
    """Demonstrate fallback to mock models when downloads fail."""
    print("\n🔄 Fallback Mechanism Demo")
    print("=" * 35)
    
    setup = MinimalModelSetup()
    
    print("\n🧪 Testing fallback behavior:")
    
    # This should work with fallback enabled
    try:
        model_info = setup.get_model_with_fallback("vision", "minimal")
        print(f"✅ Vision model loaded: {model_info.source} ({model_info.memory_mb:.1f}MB)")
    except Exception as e:
        print(f"❌ Vision fallback failed: {e}")
    
    try:
        model_info = setup.get_model_with_fallback("language", "minimal")
        print(f"✅ Language model loaded: {model_info.source} ({model_info.memory_mb:.1f}MB)")
    except Exception as e:
        print(f"❌ Language fallback failed: {e}")

def demo_integration_with_experiments():
    """Demonstrate integration with reproducible experiment runner."""
    print("\n🧪 Integration with Experiments Demo")
    print("=" * 45)
    
    print("\n🔬 Running experiments with different model configurations:")
    
    configurations = [
        ("test", "Mock models for CI/CD"),
        ("minimal", "Lightweight production models"),
        ("paper", "Full research-grade models")
    ]
    
    for preset, description in configurations:
        print(f"\n📋 {preset.upper()} Configuration: {description}")
        
        try:
            # Create experiment config
            config = ExperimentConfig(
                experiment_name=f"model_setup_demo_{preset}",
                model_type="vision",
                model_architecture=preset,
                challenge_families=["vision:freq"],
                n_challenges_per_family=2,
                verbose=False,
                output_dir="outputs/model_setup_demo"
            )
            
            # Run experiment
            runner = ReproducibleExperimentRunner(config)
            runner.setup_models()
            
            # Get model info
            model_info = runner.models.get("model_info")
            if model_info:
                print(f"   ✅ Model loaded: {model_info.source}")
                print(f"   📊 Memory usage: {model_info.memory_mb:.1f}MB")
                print(f"   ⚙️ Config: {model_info.config}")
            else:
                print(f"   ✅ Experiment setup successful")
                
        except Exception as e:
            print(f"   ❌ Failed: {e}")

def demo_memory_optimization():
    """Demonstrate memory optimization features."""
    print("\n🎯 Memory Optimization Demo")
    print("=" * 35)
    
    setup = MinimalModelSetup()
    
    print("\n📈 Memory usage by model type:")
    
    # Load models and track memory
    memory_usage = {}
    
    for model_type in ["vision", "language"]:
        for preset in ["test", "minimal", "paper"]:
            try:
                config = ModelConfig(
                    name=f"mem_test_{model_type}_{preset}",
                    model_type=model_type,
                    preset=ModelPreset(preset),
                    fallback_to_mock=True
                )
                
                if model_type == "vision":
                    model_info = setup.get_vision_model(config)
                else:
                    model_info = setup.get_language_model(config)
                
                memory_usage[f"{model_type}_{preset}"] = model_info.memory_mb
                
            except Exception as e:
                print(f"   ❌ {model_type}_{preset}: {e}")
    
    # Display results
    for name, memory in sorted(memory_usage.items(), key=lambda x: x[1]):
        print(f"   {name}: {memory:.1f}MB")
    
    # Memory report
    total_memory = setup.get_memory_report()
    print(f"\n📊 Total memory in cache: {total_memory['total_memory_mb']:.1f}MB")
    
    # Clear cache to free memory
    setup.clear_cache()
    print("🧹 Cache cleared to free memory")

def main():
    """Run all demonstrations."""
    print("🚀 PoT Model Setup System Demo")
    print("=" * 50)
    print("This demo showcases the comprehensive model setup system")
    print("with automatic downloading, caching, and fallback mechanisms.\n")
    
    try:
        demo_basic_model_setup()
        demo_vision_models()
        demo_language_models()
        demo_convenience_functions()
        demo_caching_and_performance()
        demo_fallback_mechanism()
        demo_integration_with_experiments()
        demo_memory_optimization()
        
        print("\n🎉 Demo completed successfully!")
        print("\nKey Features Demonstrated:")
        print("✅ Automatic model downloading with HuggingFace & torchvision")
        print("✅ Intelligent caching system with checksum verification")
        print("✅ Multiple configuration presets (minimal, test, paper)")
        print("✅ Robust fallback mechanisms for reliability")
        print("✅ Memory usage tracking and optimization")
        print("✅ Integration with reproducible experiment runner")
        print("✅ Full reproducibility with seed management")
        
        print(f"\n📁 Models cached in: ~/.cache/pot_experiments/")
        print(f"📁 Experiment outputs in: outputs/model_setup_demo/")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)