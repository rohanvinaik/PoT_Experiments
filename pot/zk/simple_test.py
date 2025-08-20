"""Simple test with minimal data."""

import subprocess
import json
import base64

# Create minimal valid input
request = {
    "public_inputs": {
        "w_t_root": "0x" + "00" * 16 + "11" * 16,  # Start with zeros to ensure it's small
        "batch_root": "0x" + "00" * 16 + "22" * 16,  
        "hparams_hash": "0x" + "00" * 16 + "33" * 16,
        "w_t1_root": "0x" + "00" * 16 + "44" * 16,
        "step_nonce": 123,
        "step_number": 456,
        "epoch": 1
    },
    "witness": {
        "weights_before": [0.1] * 64,
        "weights_after": [0.09] * 64,
        "batch_inputs": [0.5] * 16,
        "batch_targets": [1.0] * 4,
        "gradients": [0.01] * 64,
        "learning_rate": 0.01,
        "loss_value": 0.5
    },
    "params_k": 10
}

print("Sending request to Rust prover...")
print(f"Public inputs w_t_root: {request['public_inputs']['w_t_root']}")

# Call the prover directly
result = subprocess.run(
    ["./prover_halo2/target/release/prove_sgd_stdin"],
    input=json.dumps(request),
    capture_output=True,
    text=True
)

if result.returncode != 0:
    print(f"Error: {result.stderr}")
else:
    response = json.loads(result.stdout)
    print(f"Success! Proof size: {len(base64.b64decode(response['proof']))} bytes")
    print(f"Metadata: {response['metadata']}")