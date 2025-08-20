"""
Zero-Knowledge Proof Module for Proof-of-Training

This module provides ZK proof capabilities for training verification,
including SGD and LoRA training step proofs integrated with the existing
PoT infrastructure.
"""

from .spec import (
    SGDStepStatement,
    SGDStepWitness,
    LoRAStepStatement,
    LoRAStepWitness,
    ZKProofType,
    CommitmentScheme
)

from .commitments import (
    PoseidonHasher,
    MerkleCommitment,
    DualCommitment,
    convert_sha256_to_poseidon,
    convert_poseidon_to_sha256
)

from .witness import (
    extract_sgd_witness,
    extract_lora_witness,
    extract_weight_openings,
    extract_batch_openings,
    build_zk_statement
)

from .auto_prover import AutoProver

from .metrics import (
    ZKMetricsCollector,
    get_zk_metrics_collector,
)

__all__ = [
    # Statement and Witness Classes
    'SGDStepStatement',
    'SGDStepWitness', 
    'LoRAStepStatement',
    'LoRAStepWitness',
    'ZKProofType',
    'CommitmentScheme',
    
    # Auto Prover
    'AutoProver',
    
    # Commitment Functions
    'PoseidonHasher',
    'MerkleCommitment',
    'DualCommitment',
    'convert_sha256_to_poseidon',
    'convert_poseidon_to_sha256',
    
    # Witness Extraction
    'extract_sgd_witness',
    'extract_lora_witness',
    'extract_weight_openings',
    'extract_batch_openings',
    'build_zk_statement',
    
    # Metrics
    'ZKMetricsCollector',
    'get_zk_metrics_collector',
]

__version__ = "1.0.0"