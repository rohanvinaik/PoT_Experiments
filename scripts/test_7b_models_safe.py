#!/usr/bin/env python3
"""
Test script for 7B model permutations with proper memory handling.

This script runs the three permutations (A|A, B|B, A|B) sequentially with
strict memory limits and proper error recovery.
"""

import subprocess
import sys
import os
import json
import time
import logging
from datetime import datetime
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def run_test_with_memory_safety(ref_model, cand_model, test_name, max_memory=25.0):
    """Run a single test with memory safety enforced"""
    
    logger.info(f"\n{'='*80}")
    logger.info(f"🧪 Starting test: {test_name}")
    logger.info(f"   Reference: {ref_model}")
    logger.info(f"   Candidate: {cand_model}")
    logger.info(f"   Memory limit: {max_memory}%")
    logger.info(f"{'='*80}")
    
    # Build command
    cmd = [
        'python', 'scripts/run_e2e_validation.py',
        '--ref-model', ref_model,
        '--cand-model', cand_model,
        '--mode', 'audit',
        '--max-queries', '20',  # Reduced for 7B models
        '--enable-sharding',
        '--max-memory-percent', str(max_memory),
        '--enforce-sequential',
        '--output-dir', f'outputs/7b_test_{test_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    ]
    
    # Run with timeout
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1800  # 30 minute timeout
        )
        
        duration = time.time() - start_time
        
        if result.returncode == 0:
            logger.info(f"✅ Test completed successfully in {duration:.1f}s")
            
            # Try to extract results from output
            for line in result.stdout.split('\n'):
                if 'Decision:' in line:
                    logger.info(f"   {line.strip()}")
                elif 'Confidence:' in line:
                    logger.info(f"   {line.strip()}")
                elif 'Peak memory:' in line:
                    logger.info(f"   {line.strip()}")
            
            return {
                'success': True,
                'duration': duration,
                'test_name': test_name
            }
        else:
            logger.error(f"❌ Test failed with return code {result.returncode}")
            logger.error(f"Error output: {result.stderr[-1000:]}")  # Last 1000 chars
            
            return {
                'success': False,
                'duration': duration,
                'test_name': test_name,
                'error': result.stderr[-500:]
            }
            
    except subprocess.TimeoutExpired:
        logger.error(f"❌ Test timed out after 30 minutes")
        return {
            'success': False,
            'test_name': test_name,
            'error': 'Timeout after 30 minutes'
        }
    except Exception as e:
        logger.error(f"❌ Test failed with exception: {e}")
        return {
            'success': False,
            'test_name': test_name,
            'error': str(e)
        }


def main():
    """Run the three 7B model permutations"""
    
    # Model paths - update these to your actual 7B model paths
    model_a = "yi-6b"  # or your path to Yi-6B
    model_b = "yi-34b"  # or your path to Yi-34B
    
    # Alternative models if Yi models not available
    # model_a = "meta-llama/Llama-2-7b-hf"
    # model_b = "meta-llama/Llama-2-7b-chat-hf"
    
    logger.info(f"\n{'='*80}")
    logger.info("🚀 Starting 7B Model Test Suite")
    logger.info(f"   Model A: {model_a}")
    logger.info(f"   Model B: {model_b}")
    logger.info("   Tests: A|A (self), B|B (self), A|B (cross)")
    logger.info("   Memory limit: 25%")
    logger.info("   Execution: Sequential (one at a time)")
    logger.info(f"{'='*80}\n")
    
    results = {
        'start_time': datetime.now().isoformat(),
        'models': {
            'model_a': model_a,
            'model_b': model_b
        },
        'tests': []
    }
    
    # Test 1: A|A (self-consistency)
    logger.info("\n📋 Test 1/3: Model A self-consistency (A|A)")
    test1 = run_test_with_memory_safety(model_a, model_a, "A_self_consistency")
    results['tests'].append(test1)
    
    # Wait between tests for memory to clear
    logger.info("\n⏳ Waiting 30 seconds for memory cleanup...")
    time.sleep(30)
    
    # Test 2: B|B (self-consistency)
    logger.info("\n📋 Test 2/3: Model B self-consistency (B|B)")
    test2 = run_test_with_memory_safety(model_b, model_b, "B_self_consistency")
    results['tests'].append(test2)
    
    # Wait between tests
    logger.info("\n⏳ Waiting 30 seconds for memory cleanup...")
    time.sleep(30)
    
    # Test 3: A|B (cross-model)
    logger.info("\n📋 Test 3/3: Cross-model comparison (A|B)")
    test3 = run_test_with_memory_safety(model_a, model_b, "A_vs_B_cross")
    results['tests'].append(test3)
    
    # Summary
    results['end_time'] = datetime.now().isoformat()
    results['summary'] = {
        'total_tests': 3,
        'successful': sum(1 for t in results['tests'] if t.get('success', False)),
        'failed': sum(1 for t in results['tests'] if not t.get('success', False)),
        'total_duration': sum(t.get('duration', 0) for t in results['tests'])
    }
    
    # Save results
    output_file = f"outputs/7b_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs("outputs", exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    logger.info(f"\n{'='*80}")
    logger.info("📊 Test Suite Complete")
    logger.info(f"   Successful: {results['summary']['successful']}/3")
    logger.info(f"   Failed: {results['summary']['failed']}/3")
    logger.info(f"   Total time: {results['summary']['total_duration']:.1f}s")
    logger.info(f"   Results saved to: {output_file}")
    logger.info(f"{'='*80}\n")
    
    # Exit with appropriate code
    if results['summary']['failed'] > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    # Check if we're using the memory-safe runner
    use_memory_safe = '--memory-safe' in sys.argv
    
    if use_memory_safe:
        # Use the dedicated memory-safe runner
        logger.info("Using memory-safe validation runner...")
        cmd = [
            'python', 'scripts/run_memory_safe_validation.py',
            '--models', 'yi-6b', 'yi-34b',
            '--permutations', 'all',
            '--max-memory', '25'
        ]
        subprocess.run(cmd)
    else:
        # Use standard E2E with memory limits
        main()