import sys
import time
import json
import hashlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pot.security.fuzzy_hash_verifier import FuzzyHashVerifier

print("Running security verification...")
start = time.time()

# Compute model hash
model_path = sys.argv[1]
with open(model_path, 'rb') as f:
    # Read first 10MB for hashing (full file too large)
    data = f.read(10 * 1024 * 1024)
    sha256 = hashlib.sha256(data).hexdigest()
    
# Compute fuzzy hash
verifier = FuzzyHashVerifier()
fuzzy_hash = verifier.generate_fuzzy_hash(data)

elapsed = time.time() - start
print(f"  SHA256 (first 10MB): {sha256[:32]}...")
print(f"  Fuzzy hash computed: {fuzzy_hash is not None}")
print(f"  Time: {elapsed:.1f}s")

with open(sys.argv[2], 'w') as f:
    json.dump({
        'stage': 'security',
        'sha256_prefix': sha256[:32],
        'fuzzy_hash_available': fuzzy_hash is not None,
        'time_seconds': elapsed
    }, f, indent=2)
