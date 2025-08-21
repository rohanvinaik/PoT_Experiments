import sys
import time
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from llama_cpp import Llama
import numpy as np

print("Loading Qwen 72B model...")
model = Llama(
    model_path=sys.argv[1],
    n_ctx=512,
    n_threads=8,
    n_gpu_layers=-1,
    verbose=False,
    seed=42
)

# Run statistical tests
print("Running statistical identity verification...")
prompts = [
    "The future of artificial intelligence",
    "Climate change requires immediate",
    "Technology advances when",
    "Scientific breakthroughs happen",
    "Democracy functions best when",
    "The nature of consciousness",
    "Evolution explains the",
    "Quantum computing will",
    "The meaning of life",
    "Human creativity emerges"
]

start = time.time()
differences = []
for i, prompt in enumerate(prompts):
    print(f"  Testing prompt {i+1}/{len(prompts)}...")
    out1 = model(prompt, max_tokens=30, temperature=0.0, seed=42)
    out2 = model(prompt, max_tokens=30, temperature=0.0, seed=42)
    
    t1 = out1['choices'][0]['text']
    t2 = out2['choices'][0]['text']
    
    diff = 0.0 if t1 == t2 else 1.0
    differences.append(diff)

elapsed = time.time() - start
mean_diff = np.mean(differences)
decision = "SAME" if mean_diff < 0.01 else "DIFFERENT"

print(f"\nStatistical Verification Results:")
print(f"  Decision: {decision}")
print(f"  Mean difference: {mean_diff:.6f}")
print(f"  Time: {elapsed:.1f}s")
print(f"  Samples: {len(prompts)}")

with open(sys.argv[2], 'w') as f:
    json.dump({
        'stage': 'statistical',
        'decision': decision,
        'mean_diff': mean_diff,
        'n_samples': len(prompts),
        'time_seconds': elapsed
    }, f, indent=2)
