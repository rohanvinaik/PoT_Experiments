#!/usr/bin/env python3
"""Test loading DeepSeek model - try part 2 which has valid header."""

import sys
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from llama_cpp import Llama
    logger.info("✓ llama-cpp-python installed")
except ImportError:
    logger.error("✗ llama-cpp-python not installed")
    sys.exit(1)

# Since part 1 is corrupted, try loading from part 2
base_path = "/Users/rohanvinaik/LLM_Models/DeepSeek-R1-UD-IQ1_M/DeepSeek-R1-UD-IQ1_M"
part2_path = f"{base_path}/DeepSeek-R1-UD-IQ1_M-00002-of-00004.gguf"

logger.info(f"Attempting to load DeepSeek from part 2 (has valid GGUF header)")
logger.info(f"Path: {part2_path}")

try:
    # Try loading with minimal settings
    model = Llama(
        model_path=part2_path,
        n_ctx=128,  # Minimal context
        n_threads=4,
        n_gpu_layers=0,  # CPU only for safety
        verbose=True,  # Show what's happening
        n_batch=32
    )
    logger.info("✓ Model loaded successfully from part 2!")
    
    # Try simple generation
    prompt = "Hello"
    logger.info(f"Testing generation with prompt: '{prompt}'")
    output = model(prompt, max_tokens=10, temperature=0.0)
    result = output['choices'][0]['text']
    logger.info(f"Generated: {result}")
    
    logger.info("Success! DeepSeek can be loaded from part 2")
    
except Exception as e:
    logger.error(f"Failed to load: {e}")
    logger.info("\nThe model appears to be a split GGUF that requires all parts.")
    logger.info("Part 1 seems corrupted (starts with zeros).")
    logger.info("This may require re-downloading or using a different model.")