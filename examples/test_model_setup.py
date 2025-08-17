#!/usr/bin/env python3
"""
Test script for MinimalModelSetup

This script tests the model setup system with various configurations,
fallback mechanisms, and caching functionality.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from pot.experiments.model_setup import (
    MinimalModelSetup, ModelConfig, ModelPreset,
    get_minimal_vision_model, get_minimal_language_model,
    get_test_models, get_paper_models
)

def test_mock_models():
    """Test mock model creation."""
    print("🧪 Testing Mock Models")
    print("-" * 30)
    
    setup = MinimalModelSetup()
    
    # Test mock models
    mock_models = setup.get_mock_models(["vision", "language"])
    
    for model_type, model_info in mock_models.items():
        print(f"✓ Created mock {model_type} model:")
        print(f"  - Memory: {model_info.memory_mb:.1f}MB")
        print(f"  - Source: {model_info.source}")
        print(f"  - Config: {model_info.config}")
        
        # Test model forward pass
        if model_type == "vision":
            import torch
            test_input = torch.randn(1, 3, 224, 224)
            output = model_info.model(test_input)
            print(f"  - Output shape: {output.shape}")
        elif model_type == "language":
            # Test tokenizer
            if model_info.tokenizer:
                tokens = model_info.tokenizer.encode("Hello world")
                print(f"  - Tokenized 'Hello world': {tokens[:10]}...")
    
    return True

def test_vision_models():
    """Test vision model loading with different presets."""
    print("\n📸 Testing Vision Models")
    print("-" * 30)
    
    setup = MinimalModelSetup()
    
    for preset in ["test", "minimal", "paper"]:
        try:
            config = ModelConfig(
                name=f"test_vision_{preset}",
                model_type="vision",
                preset=ModelPreset(preset),
                fallback_to_mock=True
            )
            
            model_info = setup.get_vision_model(config)
            print(f"✓ Loaded {preset} vision model:")
            print(f"  - Source: {model_info.source}")
            print(f"  - Memory: {model_info.memory_mb:.1f}MB")
            print(f"  - Config: {model_info.config}")
            
            # Test forward pass
            import torch
            test_input = torch.randn(1, 3, 224, 224)
            with torch.no_grad():
                output = model_info.model(test_input)
            print(f"  - Output shape: {output.shape}")
            
        except Exception as e:
            print(f"✗ Failed to load {preset} vision model: {e}")
            return False
    
    return True

def test_language_models():
    """Test language model loading with different presets."""
    print("\n📝 Testing Language Models")
    print("-" * 30)
    
    setup = MinimalModelSetup()
    
    for preset in ["test", "minimal", "paper"]:
        try:
            config = ModelConfig(
                name=f"test_language_{preset}",
                model_type="language",
                preset=ModelPreset(preset),
                fallback_to_mock=True
            )
            
            model_info = setup.get_language_model(config)
            print(f"✓ Loaded {preset} language model:")
            print(f"  - Source: {model_info.source}")
            print(f"  - Memory: {model_info.memory_mb:.1f}MB")
            print(f"  - Config: {model_info.config}")
            
            # Test tokenizer if available
            if model_info.tokenizer:
                tokens = model_info.tokenizer.encode("Hello world")
                print(f"  - Tokenized 'Hello world': {tokens[:10]}...")
            
        except Exception as e:
            print(f"✗ Failed to load {preset} language model: {e}")
            return False
    
    return True

def test_fallback_mechanism():
    """Test fallback mechanism when models fail to load."""
    print("\n🔄 Testing Fallback Mechanism")
    print("-" * 35)
    
    setup = MinimalModelSetup()
    
    # Test with invalid configuration that should fallback
    try:
        model_info = setup.get_model_with_fallback("vision", "minimal")
        print(f"✓ Vision fallback successful:")
        print(f"  - Source: {model_info.source}")
        print(f"  - Memory: {model_info.memory_mb:.1f}MB")
        
        model_info = setup.get_model_with_fallback("language", "minimal")
        print(f"✓ Language fallback successful:")
        print(f"  - Source: {model_info.source}")
        print(f"  - Memory: {model_info.memory_mb:.1f}MB")
        
        return True
        
    except Exception as e:
        print(f"✗ Fallback mechanism failed: {e}")
        return False

def test_convenience_functions():
    """Test convenience functions for easy model access."""
    print("\n🛠️ Testing Convenience Functions")
    print("-" * 35)
    
    try:
        # Test minimal models
        vision_model = get_minimal_vision_model()
        print(f"✓ get_minimal_vision_model(): {vision_model.memory_mb:.1f}MB, {vision_model.source}")
        
        language_model = get_minimal_language_model()
        print(f"✓ get_minimal_language_model(): {language_model.memory_mb:.1f}MB, {language_model.source}")
        
        # Test mock models
        test_models = get_test_models()
        print(f"✓ get_test_models(): {len(test_models)} models")
        for name, info in test_models.items():
            print(f"  - {name}: {info.memory_mb:.1f}MB")
        
        # Test paper models
        paper_models = get_paper_models()
        print(f"✓ get_paper_models(): {len(paper_models)} models")
        for name, info in paper_models.items():
            print(f"  - {name}: {info.memory_mb:.1f}MB, {info.source}")
        
        return True
        
    except Exception as e:
        print(f"✗ Convenience functions failed: {e}")
        return False

def test_caching_and_memory():
    """Test model caching and memory management."""
    print("\n💾 Testing Caching and Memory Management")
    print("-" * 40)
    
    setup = MinimalModelSetup()
    
    try:
        # Load multiple models to test caching
        config1 = ModelConfig("test1", "vision", ModelPreset.TEST)
        config2 = ModelConfig("test2", "language", ModelPreset.TEST)
        
        model1 = setup.get_vision_model(config1)
        model2 = setup.get_language_model(config2)
        
        # Test memory report
        memory_report = setup.get_memory_report()
        print(f"✓ Memory report generated:")
        print(f"  - Total memory: {memory_report['total_memory_mb']:.1f}MB")
        print(f"  - Loaded models: {len(memory_report['models'])}")
        
        # Test caching (should be faster second time)
        import time
        start_time = time.time()
        model1_cached = setup.get_vision_model(config1)
        cache_time = time.time() - start_time
        
        print(f"✓ Cached model load time: {cache_time:.4f}s")
        print(f"✓ Model caching working: {model1 is model1_cached}")
        
        # Test cache clearing
        setup.clear_cache()
        empty_report = setup.get_memory_report()
        print(f"✓ Cache cleared: {empty_report['total_memory_mb']}MB")
        
        return True
        
    except Exception as e:
        print(f"✗ Caching/memory test failed: {e}")
        return False

def test_model_availability():
    """Test model availability checking."""
    print("\n📋 Testing Model Availability")
    print("-" * 30)
    
    setup = MinimalModelSetup()
    
    try:
        available_models = setup.list_available_models()
        
        for model_type, presets in available_models.items():
            print(f"✓ {model_type.title()} models:")
            for preset, spec in presets.items():
                status = "✓" if spec["available"] else "✗"
                print(f"  {status} {preset}: {spec['description']} ({spec['memory_mb']}MB)")
        
        return True
        
    except Exception as e:
        print(f"✗ Model availability test failed: {e}")
        return False

def test_model_registry():
    """Test model registry saving and loading."""
    print("\n📝 Testing Model Registry")
    print("-" * 25)
    
    setup = MinimalModelSetup()
    
    try:
        # Load some models
        vision_model = setup.get_model_with_fallback("vision", "test")
        language_model = setup.get_model_with_fallback("language", "test")
        
        # Save registry
        registry_path = "test_model_registry.json"
        setup.save_model_registry(registry_path)
        
        # Check if file was created
        if os.path.exists(registry_path):
            print(f"✓ Model registry saved to {registry_path}")
            
            # Read and display contents
            import json
            with open(registry_path, 'r') as f:
                registry = json.load(f)
            
            print(f"✓ Registry contains {len(registry)} models:")
            for name, info in registry.items():
                print(f"  - {name}: {info['source']}, {info['memory_mb']:.1f}MB")
            
            # Clean up
            os.remove(registry_path)
            print(f"✓ Cleaned up {registry_path}")
            
            return True
        else:
            print(f"✗ Registry file not created")
            return False
            
    except Exception as e:
        print(f"✗ Model registry test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🔬 MinimalModelSetup Test Suite")
    print("=" * 40)
    
    test_functions = [
        test_mock_models,
        test_vision_models,
        test_language_models,
        test_fallback_mechanism,
        test_convenience_functions,
        test_caching_and_memory,
        test_model_availability,
        test_model_registry
    ]
    
    results = []
    for test_func in test_functions:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"✗ {test_func.__name__} failed with exception: {e}")
            results.append(False)
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print(f"\n📊 Test Results")
    print("=" * 20)
    print(f"Passed: {passed}/{total}")
    print(f"Success Rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("🎉 All tests passed!")
        return True
    else:
        print("❌ Some tests failed!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)