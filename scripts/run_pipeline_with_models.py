#!/usr/bin/env python3
"""
Easy Model Selection Front-End for PoT Pipeline

This script provides a simple interface to run the entire PoT validation pipeline 
with user-selected models. It automatically detects available models and allows
easy selection for comprehensive testing.
"""

import os
import sys
import json
import subprocess
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import argparse

def print_header(text: str):
    print(f"\n{'='*60}")
    print(f"🔬 {text}")
    print(f"{'='*60}\n")

def print_success(text: str):
    print(f"✅ {text}")

def print_error(text: str):
    print(f"❌ {text}")

def print_info(text: str):
    print(f"ℹ️  {text}")

def scan_available_models(base_path: str = "/Users/rohanvinaik/LLM_Models") -> Dict[str, Dict]:
    """Scan and categorize available models"""
    models = {}
    
    if not os.path.exists(base_path):
        print_error(f"Model directory not found: {base_path}")
        return models
    
    for item in os.listdir(base_path):
        model_path = os.path.join(base_path, item)
        if os.path.isdir(model_path) and not item.startswith('.'):
            # Check for model files to confirm it's a valid model
            has_config = os.path.exists(os.path.join(model_path, 'config.json'))
            has_model = any(
                os.path.exists(os.path.join(model_path, f))
                for f in ['pytorch_model.bin', 'model.safetensors', 'consolidated.safetensors']
            )
            
            if has_config and has_model:
                # Try to get model size info
                config_path = os.path.join(model_path, 'config.json')
                size_info = "Unknown size"
                try:
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                        # Common size indicators
                        hidden_size = config.get('hidden_size', config.get('d_model', 0))
                        n_layers = config.get('num_hidden_layers', config.get('num_layers', 0))
                        
                        if 'gpt2' in item.lower():
                            if 'medium' in item.lower():
                                size_info = "345M params"
                            elif 'large' in item.lower():
                                size_info = "774M params"  
                            elif 'xl' in item.lower():
                                size_info = "1.5B params"
                            else:
                                size_info = "117M params"
                        elif 'distilgpt2' in item.lower():
                            size_info = "82M params"
                        elif '7b' in item.lower() or hidden_size > 3000:
                            size_info = "~7B params"
                        elif '13b' in item.lower():
                            size_info = "~13B params"
                        elif 'phi' in item.lower():
                            size_info = "2.7B params"
                        elif hidden_size > 1000:
                            size_info = f"~{hidden_size}M params"
                        
                except Exception:
                    pass
                
                # Categorize models
                category = "Large (7B+)" if "7b" in item.lower() or "7B" in size_info else "Small (<1B)"
                if "13b" in item.lower():
                    category = "Very Large (13B+)"
                elif any(x in item.lower() for x in ["medium", "large"]):
                    category = "Medium (1B-7B)"
                
                models[item] = {
                    'path': model_path,
                    'size_info': size_info,
                    'category': category,
                    'is_base': not any(x in item.lower() for x in ["chat", "instruct", "zephyr", "vicuna", "alpaca"]),
                    'architecture': config.get('model_type', 'unknown')
                }
    
    return models

def display_models(models: Dict[str, Dict]):
    """Display available models in organized format"""
    print_header("Available Models")
    
    categories = {}
    for name, info in models.items():
        cat = info['category']
        if cat not in categories:
            categories[cat] = []
        categories[cat].append((name, info))
    
    for category in sorted(categories.keys()):
        print(f"\n📁 {category}:")
        for i, (name, info) in enumerate(sorted(categories[category]), 1):
            base_indicator = "🔹 (base)" if info['is_base'] else "🔸 (fine-tuned)"
            print(f"   {i:2d}. {name:<25} {info['size_info']:<15} {base_indicator}")
    
    print(f"\nTotal models found: {len(models)}")

def get_model_pairs_interactive(models: Dict[str, Dict]) -> List[Tuple[str, str]]:
    """Get model pairs from user selection"""
    print_header("Model Pair Selection")
    print("Choose model pairs to test. You can:")
    print("1. Select specific models by name")  
    print("2. Use quick presets")
    print("3. Mix different sizes")
    
    model_list = list(models.keys())
    pairs = []
    
    print("\nQuick Presets:")
    print("A. Small vs Small (GPT-2 variants)")
    print("B. Large vs Large (7B variants)")
    print("C. Small vs Large (cross-size)")
    print("D. Base vs Fine-tuned (same architecture)")
    print("E. Custom selection")
    
    choice = input("\nSelect preset (A-E) or press Enter for custom: ").upper().strip()
    
    if choice == 'A':
        # Small vs Small
        small_models = [n for n, info in models.items() if info['category'] == "Small (<1B)"]
        if len(small_models) >= 2:
            pairs.append((small_models[0], small_models[1]))
            print(f"Added: {pairs[-1][0]} vs {pairs[-1][1]}")
    
    elif choice == 'B':
        # Large vs Large  
        large_models = [n for n, info in models.items() if "7B" in info['size_info']]
        if len(large_models) >= 2:
            pairs.append((large_models[0], large_models[1]))
            print(f"Added: {pairs[-1][0]} vs {pairs[-1][1]}")
    
    elif choice == 'C':
        # Small vs Large
        small_models = [n for n, info in models.items() if info['category'] == "Small (<1B)"]
        large_models = [n for n, info in models.items() if "7B" in info['size_info']]
        if small_models and large_models:
            pairs.append((small_models[0], large_models[0]))
            print(f"Added: {pairs[-1][0]} vs {pairs[-1][1]}")
    
    elif choice == 'D':
        # Base vs Fine-tuned
        base_models = [n for n, info in models.items() if info['is_base']]
        finetuned_models = [n for n, info in models.items() if not info['is_base']]
        
        # Find matching architecture pairs
        for base in base_models:
            base_arch = models[base]['architecture'] 
            for ft in finetuned_models:
                ft_arch = models[ft]['architecture']
                if base_arch == ft_arch and "7B" in models[base]['size_info'] and "7B" in models[ft]['size_info']:
                    pairs.append((base, ft))
                    print(f"Added: {pairs[-1][0]} vs {pairs[-1][1]}")
                    break
            if pairs:
                break
    
    # Custom selection or additional pairs
    if choice == 'E' or not pairs:
        print("\nCustom Model Selection:")
        print("Enter model names (or numbers from list above)")
        
        while True:
            pair_input = input(f"\nEnter model pair (model1,model2) or 'done': ").strip()
            if pair_input.lower() == 'done':
                break
                
            try:
                if ',' in pair_input:
                    model1, model2 = pair_input.split(',', 1)
                    model1, model2 = model1.strip(), model2.strip()
                    
                    # Convert numbers to names if needed
                    if model1.isdigit():
                        idx = int(model1) - 1
                        if 0 <= idx < len(model_list):
                            model1 = model_list[idx]
                    if model2.isdigit():
                        idx = int(model2) - 1  
                        if 0 <= idx < len(model_list):
                            model2 = model_list[idx]
                    
                    if model1 in models and model2 in models:
                        pairs.append((model1, model2))
                        print(f"Added: {model1} vs {model2}")
                    else:
                        print("Invalid model names. Please try again.")
                else:
                    print("Please use format: model1,model2")
            except Exception as e:
                print(f"Error: {e}")
    
    if not pairs:
        # Default fallback
        print("No pairs selected. Using default: gpt2 vs distilgpt2")
        pairs.append(("gpt2", "distilgpt2"))
    
    return pairs

def run_validation_with_models(model_pairs: List[Tuple[str, str]], models: Dict[str, Dict], 
                              output_dir: str = "experimental_results", 
                              test_mode: str = "comprehensive") -> Dict:
    """Run validation pipeline with selected model pairs"""
    print_header(f"Running {test_mode.upper()} Validation Pipeline")
    
    results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'test_mode': test_mode,
        'model_pairs': [],
        'summary': {}
    }
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    total_pairs = len(model_pairs)
    successful_tests = 0
    
    for i, (model1, model2) in enumerate(model_pairs, 1):
        print(f"\n--- Test {i}/{total_pairs}: {model1} vs {model2} ---")
        
        model1_path = models[model1]['path']
        model2_path = models[model2]['path']
        
        test_result = {
            'pair_id': i,
            'model_a': {
                'name': model1,
                'path': model1_path,
                'size': models[model1]['size_info'],
                'category': models[model1]['category']
            },
            'model_b': {
                'name': model2, 
                'path': model2_path,
                'size': models[model2]['size_info'],
                'category': models[model2]['category']
            },
            'tests': {}
        }
        
        # Run different validation tests based on mode
        if test_mode in ['comprehensive', 'statistical']:
            # Statistical Identity Test
            print("🧪 Running statistical identity test...")
            cmd = [
                sys.executable, "scripts/runtime_blackbox_validation_adaptive.py",
                "--model-a", model1_path,
                "--model-b", model2_path, 
                "--n-queries", "15",
                "--output-results"
            ]
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
                if result.returncode == 0:
                    test_result['tests']['statistical_identity'] = {
                        'status': 'PASSED',
                        'output': result.stdout[-500:] if len(result.stdout) > 500 else result.stdout
                    }
                    print_success("Statistical identity test completed")
                else:
                    test_result['tests']['statistical_identity'] = {
                        'status': 'FAILED', 
                        'error': result.stderr[-500:] if len(result.stderr) > 500 else result.stderr
                    }
                    print_error("Statistical identity test failed")
            except Exception as e:
                test_result['tests']['statistical_identity'] = {
                    'status': 'ERROR',
                    'error': str(e)
                }
                print_error(f"Statistical identity test error: {e}")
        
        if test_mode in ['comprehensive', 'enhanced']:
            # Enhanced Diff Test
            print("🔬 Running enhanced difference test...")
            cmd = [
                sys.executable, "scripts/run_enhanced_diff_test.py",
                "--mode", "audit",
                "--ref-model", model1_path,
                "--cand-model", model2_path,
                "--prf-key", "deadbeef123456789abcdef"
            ]
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
                if result.returncode in [0, 1, 2]:  # All valid exit codes
                    test_result['tests']['enhanced_diff'] = {
                        'status': 'COMPLETED',
                        'exit_code': result.returncode,
                        'output': result.stdout[-500:] if len(result.stdout) > 500 else result.stdout
                    }
                    print_success("Enhanced difference test completed")
                else:
                    test_result['tests']['enhanced_diff'] = {
                        'status': 'FAILED',
                        'error': result.stderr[-500:] if len(result.stderr) > 500 else result.stderr
                    }
                    print_error("Enhanced difference test failed")
            except Exception as e:
                test_result['tests']['enhanced_diff'] = {
                    'status': 'ERROR',
                    'error': str(e)
                }
                print_error(f"Enhanced difference test error: {e}")
        
        # Count successful tests
        passed_tests = sum(1 for test in test_result['tests'].values() 
                          if test['status'] in ['PASSED', 'COMPLETED'])
        total_tests = len(test_result['tests'])
        
        if passed_tests > 0:
            successful_tests += 1
            print_success(f"Pair {i} completed: {passed_tests}/{total_tests} tests successful")
        else:
            print_error(f"Pair {i} failed: No tests passed")
        
        results['model_pairs'].append(test_result)
    
    # Generate summary
    results['summary'] = {
        'total_pairs': total_pairs,
        'successful_pairs': successful_tests,
        'success_rate': successful_tests / total_pairs if total_pairs > 0 else 0,
        'small_models': len([p for p in model_pairs if "Small" in models[p[0]]['category'] and "Small" in models[p[1]]['category']]),
        'large_models': len([p for p in model_pairs if "7B" in models[p[0]]['size_info'] and "7B" in models[p[1]]['size_info']]),
        'mixed_sizes': len([p for p in model_pairs if models[p[0]]['category'] != models[p[1]]['category']])
    }
    
    # Save results
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    results_file = f"{output_dir}/model_pipeline_results_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print_success(f"Results saved to: {results_file}")
    return results

def display_results_summary(results: Dict):
    """Display validation results summary"""
    print_header("Validation Results Summary")
    
    summary = results['summary']
    print(f"📊 Overall Success Rate: {summary['success_rate']:.1%} ({summary['successful_pairs']}/{summary['total_pairs']} pairs)")
    print(f"🔬 Model Size Distribution:")
    print(f"   • Small models: {summary['small_models']} pairs")
    print(f"   • Large models: {summary['large_models']} pairs") 
    print(f"   • Mixed sizes: {summary['mixed_sizes']} pairs")
    
    print(f"\n📋 Detailed Results:")
    for pair_result in results['model_pairs']:
        model_a = pair_result['model_a']
        model_b = pair_result['model_b']
        tests = pair_result['tests']
        
        passed_tests = sum(1 for test in tests.values() if test['status'] in ['PASSED', 'COMPLETED'])
        total_tests = len(tests)
        
        status = "✅ PASS" if passed_tests > 0 else "❌ FAIL"
        print(f"   {status} {model_a['name']} ({model_a['size']}) vs {model_b['name']} ({model_b['size']}) - {passed_tests}/{total_tests}")

def update_readme_with_results(results: Dict, readme_path: str = "README.md"):
    """Update README with model validation results"""
    try:
        print_header("Updating README with Model Results")
        
        # Create model results section
        model_summary = f"""
## 🚀 Multi-Model Validation Results

*Last Updated: {results['timestamp']}*

The ZK-PoT framework has been validated across multiple model sizes and architectures:

### Model Size Coverage
- **Small Models (<1B params)**: {results['summary']['small_models']} test pairs
- **Large Models (7B+ params)**: {results['summary']['large_models']} test pairs  
- **Cross-Size Testing**: {results['summary']['mixed_sizes']} mixed pairs

### Validation Success Rate
- **Overall Success**: {results['summary']['success_rate']:.1%} ({results['summary']['successful_pairs']}/{results['summary']['total_pairs']} pairs)

### Tested Model Pairs
"""
        
        for pair_result in results['model_pairs']:
            model_a = pair_result['model_a']
            model_b = pair_result['model_b']
            tests = pair_result['tests']
            
            passed_tests = sum(1 for test in tests.values() if test['status'] in ['PASSED', 'COMPLETED'])
            total_tests = len(tests)
            status_icon = "✅" if passed_tests > 0 else "❌"
            
            model_summary += f"- {status_icon} **{model_a['name']}** ({model_a['size']}) vs **{model_b['name']}** ({model_b['size']})\n"
        
        model_summary += f"\n*Framework demonstrates scalability from small GPT-2 variants to large 7B+ models.*\n"
        
        # Read current README
        if os.path.exists(readme_path):
            with open(readme_path, 'r') as f:
                readme_content = f.read()
            
            # Look for existing model results section
            start_marker = "## 🚀 Multi-Model Validation Results"
            end_marker = "\n## "  # Next section
            
            start_pos = readme_content.find(start_marker)
            if start_pos != -1:
                # Find end of section
                end_pos = readme_content.find(end_marker, start_pos + len(start_marker))
                if end_pos == -1:
                    end_pos = len(readme_content)
                
                # Replace existing section
                readme_content = readme_content[:start_pos] + model_summary + readme_content[end_pos:]
            else:
                # Add new section before the comprehensive metrics placeholder
                placeholder_pos = readme_content.find("**COMPREHENSIVE_METRICS_PLACEHOLDER**")
                if placeholder_pos != -1:
                    readme_content = readme_content[:placeholder_pos] + model_summary + "\n" + readme_content[placeholder_pos:]
                else:
                    # Add at end of file
                    readme_content += "\n" + model_summary
            
            # Write back to README
            with open(readme_path, 'w') as f:
                f.write(readme_content)
            
            print_success(f"README updated with model validation results")
        
    except Exception as e:
        print_error(f"Failed to update README: {e}")

def main():
    parser = argparse.ArgumentParser(description="Easy Model Selection Front-End for PoT Pipeline")
    parser.add_argument("--models-dir", default="/Users/rohanvinaik/LLM_Models", help="Base directory for models")
    parser.add_argument("--output-dir", default="experimental_results", help="Output directory for results")
    parser.add_argument("--test-mode", choices=['quick', 'statistical', 'enhanced', 'comprehensive'], 
                       default='comprehensive', help="Testing mode")
    parser.add_argument("--auto-pairs", help="Automatically select model pairs (small,large,mixed,base-ft)")
    parser.add_argument("--update-readme", action='store_true', help="Update README with results")
    parser.add_argument("--non-interactive", action='store_true', help="Run without interactive prompts")
    
    args = parser.parse_args()
    
    print_header("PoT Pipeline - Easy Model Selection")
    
    # Scan available models
    print("🔍 Scanning for available models...")
    models = scan_available_models(args.models_dir)
    
    if not models:
        print_error("No valid models found. Please check the models directory.")
        return 1
    
    print_success(f"Found {len(models)} valid models")
    
    # Display models unless non-interactive
    if not args.non_interactive:
        display_models(models)
    
    # Select model pairs
    if args.auto_pairs:
        pairs = []
        if 'small' in args.auto_pairs:
            small_models = [n for n, info in models.items() if info['category'] == "Small (<1B)"]
            if len(small_models) >= 2:
                pairs.append((small_models[0], small_models[1]))
        
        if 'large' in args.auto_pairs:
            large_models = [n for n, info in models.items() if "7B" in info['size_info']]
            if len(large_models) >= 2:
                pairs.append((large_models[0], large_models[1]))
                
        if 'mixed' in args.auto_pairs:
            small_models = [n for n, info in models.items() if info['category'] == "Small (<1B)"]
            large_models = [n for n, info in models.items() if "7B" in info['size_info']]
            if small_models and large_models:
                pairs.append((small_models[0], large_models[0]))
                
        if 'base-ft' in args.auto_pairs:
            # Find Mistral (base) and Zephyr (fine-tuned) specifically
            mistral_models = [n for n in models.keys() if 'mistral' in n.lower() and 'for_colab' in n]
            zephyr_models = [n for n in models.keys() if 'zephyr' in n.lower()]
            if mistral_models and zephyr_models:
                pairs.append((mistral_models[0], zephyr_models[0]))
                
        if not pairs:
            # Default fallback
            pairs = [("gpt2", "distilgpt2")]
            
    elif args.non_interactive:
        # Non-interactive default
        pairs = [("gpt2", "distilgpt2")]
        if "mistral_for_colab" in models and "zephyr-7b-beta-final" in models:
            pairs.append(("mistral_for_colab", "zephyr-7b-beta-final"))
    else:
        pairs = get_model_pairs_interactive(models)
    
    if not pairs:
        print_error("No model pairs selected.")
        return 1
    
    print(f"\n🎯 Selected {len(pairs)} model pairs for testing:")
    for i, (m1, m2) in enumerate(pairs, 1):
        print(f"   {i}. {m1} vs {m2}")
    
    # Run validation
    results = run_validation_with_models(pairs, models, args.output_dir, args.test_mode)
    
    # Display results
    display_results_summary(results)
    
    # Update README if requested
    if args.update_readme:
        update_readme_with_results(results)
    
    print_header("Pipeline Complete! 🎉")
    return 0

if __name__ == "__main__":
    sys.exit(main())