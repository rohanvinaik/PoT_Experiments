"""Debug test to see what's being sent to Rust."""

import sys
import hashlib
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from pot.zk.zk_types import SGDStepStatement, SGDStepWitness
from prover import SGDZKProver

# Create simple test data
statement = SGDStepStatement(
    W_t_root=hashlib.sha256(b"test1").digest(),
    batch_root=hashlib.sha256(b"test2").digest(),
    hparams_hash=hashlib.sha256(b"test3").digest(),
    W_t1_root=hashlib.sha256(b"test4").digest(),
    step_nonce=123,
    step_number=456,
    epoch=1
)

witness = SGDStepWitness(
    weights_before=[0.1] * 64,
    weights_after=[0.09] * 64,
    batch_inputs=[0.5] * 16,
    batch_targets=[1.0] * 4,
    learning_rate=0.01,
    loss_value=0.5
)

prover = SGDZKProver()

# Convert statement to see what it looks like
public_inputs = prover._statement_to_public_inputs(statement)
print("Public inputs:")
for key, value in public_inputs.items():
    print(f"  {key}: {value}")
    if isinstance(value, str) and value.startswith("0x"):
        print(f"    Length: {len(value[2:])} hex chars")

# Convert witness
witness_data = prover._witness_to_rust_format(witness)
print("\nWitness data keys:")
for key in witness_data.keys():
    if isinstance(witness_data[key], list):
        print(f"  {key}: {len(witness_data[key])} elements")
    else:
        print(f"  {key}: {witness_data[key]}")