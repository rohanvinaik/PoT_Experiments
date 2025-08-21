import sys
import time
import json
import hmac
import hashlib
from pathlib import Path

print("Running challenge-response authentication...")
start = time.time()

# Generate challenges
secret = b"pot_framework_secret"
challenges = []
for i in range(5):
    challenge = f"challenge_{i}".encode()
    response = hmac.new(secret, challenge, hashlib.sha256).hexdigest()
    challenges.append({'challenge': challenge.decode(), 'response': response[:16]})
    print(f"  Challenge {i+1}: {response[:16]}...")

elapsed = time.time() - start
print(f"  Generated {len(challenges)} challenge-response pairs")
print(f"  Time: {elapsed:.1f}s")

with open(sys.argv[1], 'w') as f:
    json.dump({
        'stage': 'challenge_auth',
        'n_challenges': len(challenges),
        'time_seconds': elapsed
    }, f, indent=2)
