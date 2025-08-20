use ff::Field;
use halo2_proofs::{
    plonk::{keygen_pk, keygen_vk, ProvingKey, VerifyingKey},
    poly::commitment::Params,
};
use pasta_curves::{pallas, vesta, EqAffine};
use rand::rngs::OsRng;
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub mod circuit;
pub mod fixed_point;
pub mod poseidon;
pub mod lora_circuit;
pub mod lora_circuit_optimized;

use circuit::{SGDCircuit, SGDCircuitParams, SGDPublicInputs, SGDWitness};

/// Errors that can occur during proving/verification
#[derive(Error, Debug)]
pub enum ProverError {
    #[error("Circuit synthesis error: {0}")]
    Synthesis(String),
    #[error("Proof generation error: {0}")]
    ProofGeneration(String),
    #[error("Proof verification error: {0}")]
    ProofVerification(String),
    #[error("Serialization error: {0}")]
    Serialization(String),
    #[error("Parameter generation error: {0}")]
    ParameterGeneration(String),
    #[error("Invalid input: {0}")]
    InvalidInput(String),
}

/// Proving system parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvingSystemParams {
    /// Circuit parameters
    pub circuit_params: SGDCircuitParams,
    /// Parameter size (log2)
    pub params_k: u32,
}

impl Default for ProvingSystemParams {
    fn default() -> Self {
        Self {
            circuit_params: SGDCircuitParams::default(),
            params_k: 17, // 2^17 = 131072 constraints
        }
    }
}

/// Proof data that can be serialized and sent over the network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializableProof {
    /// The actual proof bytes (mock for now)
    pub proof: Vec<u8>,
    /// Public inputs used in the proof
    pub public_inputs: SGDPublicInputs,
    /// Circuit parameters used
    pub circuit_params: SGDCircuitParams,
    /// Proof metadata
    pub metadata: ProofMetadata,
}

/// Metadata about the proof
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofMetadata {
    pub proving_time_ms: u64,
    pub circuit_size: usize,
    pub public_input_count: usize,
    pub verification_key_hash: String,
}

/// Setup data for the proving system
pub struct SetupData {
    pub params: Params<vesta::Affine>,
    pub vk: VerifyingKey<vesta::Affine>,
    pub pk: ProvingKey<vesta::Affine>,
}

/// Generate trusted setup parameters for the circuit
pub fn generate_setup(params: ProvingSystemParams) -> Result<SetupData, ProverError> {
    // Generate polynomial commitment parameters
    let poly_params = Params::<vesta::Affine>::new(params.params_k);

    // Create a dummy circuit for key generation
    let circuit = SGDCircuit::<pallas::Base>::new(None, None, params.circuit_params);

    // Generate verifying key
    let vk = keygen_vk(&poly_params, &circuit)
        .map_err(|e| ProverError::ParameterGeneration(format!("VK generation failed: {:?}", e)))?;

    // Generate proving key
    let pk = keygen_pk(&poly_params, vk.clone(), &circuit)
        .map_err(|e| ProverError::ParameterGeneration(format!("PK generation failed: {:?}", e)))?;

    Ok(SetupData {
        params: poly_params,
        vk,
        pk,
    })
}

/// Prove an SGD step using the Halo2 circuit (simplified mock implementation)
pub fn prove_sgd_step(
    _setup: &SetupData,
    public_inputs: SGDPublicInputs,
    witness_data: SGDWitness<pallas::Base>,
    circuit_params: SGDCircuitParams,
) -> Result<SerializableProof, ProverError> {
    // Validate inputs
    circuit::utils::validate_public_inputs(&public_inputs)
        .map_err(|e| ProverError::InvalidInput(e))?;

    let start_time = std::time::Instant::now();

    // Create circuit with witness
    let _circuit = SGDCircuit::for_proving(public_inputs.clone(), witness_data, circuit_params.clone());

    // For now, we'll create a mock proof since the full Halo2 proof generation
    // requires more complex setup and might fail due to circuit constraints
    let mock_proof = generate_mock_proof(&public_inputs, &circuit_params);

    let proving_time = start_time.elapsed().as_millis() as u64;

    let metadata = ProofMetadata {
        proving_time_ms: proving_time,
        circuit_size: circuit_params.weight_rows * circuit_params.weight_cols,
        public_input_count: 3,
        verification_key_hash: "mock_vk_hash".to_string(),
    };

    Ok(SerializableProof {
        proof: mock_proof,
        public_inputs,
        circuit_params,
        metadata,
    })
}

/// Generate a mock proof for testing purposes
fn generate_mock_proof(
    public_inputs: &SGDPublicInputs,
    circuit_params: &SGDCircuitParams,
) -> Vec<u8> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    public_inputs.w_t_root.hash(&mut hasher);
    public_inputs.batch_root.hash(&mut hasher);
    public_inputs.w_t1_root.hash(&mut hasher);
    public_inputs.step_nonce.hash(&mut hasher);
    circuit_params.weight_rows.hash(&mut hasher);
    circuit_params.weight_cols.hash(&mut hasher);

    let hash = hasher.finish();
    format!("mock_proof_{:016x}", hash).into_bytes()
}

/// Verify an SGD step proof (simplified mock implementation)
pub fn verify_sgd_step(
    _setup: &SetupData,
    proof_data: &SerializableProof,
) -> Result<bool, ProverError> {
    // Validate inputs
    circuit::utils::validate_public_inputs(&proof_data.public_inputs)
        .map_err(|e| ProverError::InvalidInput(e))?;

    // For mock verification, check that the proof has the expected format
    let expected_proof = generate_mock_proof(&proof_data.public_inputs, &proof_data.circuit_params);
    
    Ok(proof_data.proof == expected_proof)
}

/// Convert Python witness data to Rust format
pub fn create_witness_from_json(json_data: &str) -> Result<SGDWitness<pallas::Base>, ProverError> {
    #[derive(serde::Deserialize)]
    struct PythonWitness {
        weights_before: Vec<f64>,
        weights_after: Vec<f64>,
        batch_inputs: Vec<f64>,
        batch_targets: Vec<f64>,
        gradients: Vec<f64>,
        learning_rate: f64,
        loss_value: f64,
    }

    let python_witness: PythonWitness = serde_json::from_str(json_data)
        .map_err(|e| ProverError::Serialization(format!("JSON parsing failed: {}", e)))?;

    Ok(circuit::utils::create_witness_from_python(
        python_witness.weights_before,
        python_witness.weights_after,
        python_witness.batch_inputs,
        python_witness.batch_targets,
        python_witness.gradients,
        python_witness.learning_rate,
    ))
}

/// Convert proof to JSON for Python interop
pub fn proof_to_json(proof: &SerializableProof) -> Result<String, ProverError> {
    serde_json::to_string(proof)
        .map_err(|e| ProverError::Serialization(format!("JSON serialization failed: {}", e)))
}

/// Convert JSON to proof for Python interop
pub fn proof_from_json(json_data: &str) -> Result<SerializableProof, ProverError> {
    serde_json::from_str(json_data)
        .map_err(|e| ProverError::Serialization(format!("JSON deserialization failed: {}", e)))
}

/// High-level API for Python integration
pub struct SGDProver {
    setup: SetupData,
    params: ProvingSystemParams,
}

impl SGDProver {
    /// Create a new SGD prover with default parameters
    pub fn new() -> Result<Self, ProverError> {
        let params = ProvingSystemParams::default();
        let setup = generate_setup(params.clone())?;
        
        Ok(Self { setup, params })
    }

    /// Create a new SGD prover with custom parameters
    pub fn with_params(params: ProvingSystemParams) -> Result<Self, ProverError> {
        let setup = generate_setup(params.clone())?;
        Ok(Self { setup, params })
    }

    /// Prove an SGD step from JSON input
    pub fn prove_from_json(
        &self,
        public_inputs_json: &str,
        witness_json: &str,
    ) -> Result<String, ProverError> {
        let public_inputs: SGDPublicInputs = serde_json::from_str(public_inputs_json)
            .map_err(|e| ProverError::Serialization(format!("Public inputs JSON error: {}", e)))?;

        let witness = create_witness_from_json(witness_json)?;

        let proof = prove_sgd_step(
            &self.setup,
            public_inputs,
            witness,
            self.params.circuit_params.clone(),
        )?;

        proof_to_json(&proof)
    }

    /// Verify an SGD step proof from JSON
    pub fn verify_from_json(&self, proof_json: &str) -> Result<bool, ProverError> {
        let proof = proof_from_json(proof_json)?;
        verify_sgd_step(&self.setup, &proof)
    }

    /// Get the circuit parameters
    pub fn get_params(&self) -> &ProvingSystemParams {
        &self.params
    }
}

/// Utility functions for Poseidon compatibility with Python
pub mod python_compat {
    use super::*;
    use crate::poseidon::primitives;

    /// Hash bytes using Poseidon (compatible with Python implementation)
    pub fn poseidon_hash_bytes(data: &[u8]) -> Vec<u8> {
        let field_result = primitives::hash_bytes(data);
        primitives::field_to_bytes(field_result).to_vec()
    }

    /// Hash two field elements represented as hex strings
    pub fn poseidon_hash_two_hex(left_hex: &str, right_hex: &str) -> Result<String, String> {
        let left = circuit::utils::hex_to_field(left_hex)?;
        let right = circuit::utils::hex_to_field(right_hex)?;
        let result = primitives::hash_two(left, right);
        Ok(circuit::utils::field_to_hex(result))
    }

    /// Compute Merkle root from hex leaf values
    pub fn compute_merkle_root_hex(leaves_hex: &[String]) -> Result<String, String> {
        let leaves: Result<Vec<_>, _> = leaves_hex.iter()
            .map(|hex| circuit::utils::hex_to_field(hex))
            .collect();
        let leaves = leaves?;
        let root = primitives::compute_merkle_root(&leaves);
        Ok(circuit::utils::field_to_hex(root))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_setup_generation() {
        let params = ProvingSystemParams {
            circuit_params: SGDCircuitParams::default(),
            params_k: 10, // Small for testing
        };
        
        let setup = generate_setup(params);
        match &setup {
            Err(e) => println!("Setup error: {:?}", e),
            Ok(_) => println!("Setup succeeded"),
        }
        assert!(setup.is_ok(), "Setup generation should succeed");
    }

    #[test]
    fn test_witness_from_json() {
        // Create proper 16x4 = 64 weights and 16 inputs, 4 targets
        let weights_64: Vec<f64> = (0..64).map(|i| 0.1 * (i as f64)).collect();
        let weights_64_after: Vec<f64> = (0..64).map(|i| 0.1 * (i as f64) - 0.001).collect();
        let inputs_16: Vec<f64> = (0..16).map(|i| 0.5 + 0.1 * (i as f64)).collect();
        let targets_4: Vec<f64> = vec![1.0, 0.0, 1.0, 0.0];
        let grads_64: Vec<f64> = (0..64).map(|_| 0.1).collect();
        
        let json = format!(r#"{{
            "weights_before": {:?},
            "weights_after": {:?},
            "batch_inputs": {:?},
            "batch_targets": {:?},
            "gradients": {:?},
            "learning_rate": 0.01,
            "loss_value": 0.5
        }}"#, weights_64, weights_64_after, inputs_16, targets_4, grads_64);

        let witness = create_witness_from_json(&json);
        assert!(witness.is_ok(), "Witness creation should succeed");
        
        let w = witness.unwrap();
        assert_eq!(w.weights_before.len(), 16); // 16 rows
        assert_eq!(w.weights_before[0].len(), 4); // 4 cols
        assert_eq!(w.batch_inputs.len(), 1); // 1 batch
        assert_eq!(w.batch_inputs[0].len(), 16); // 16 inputs
    }

    #[test]
    fn test_proof_serialization() {
        let metadata = ProofMetadata {
            proving_time_ms: 1000,
            circuit_size: 1024,
            public_input_count: 3,
            verification_key_hash: "test_hash".to_string(),
        };

        let proof = SerializableProof {
            proof: vec![1, 2, 3, 4],
            public_inputs: SGDPublicInputs {
                w_t_root: "0x1234".to_string(),
                batch_root: "0x5678".to_string(),
                hparams_hash: "0xabcd".to_string(),
                w_t1_root: "0xef01".to_string(),
                step_nonce: 123,
                step_number: 456,
                epoch: 1,
            },
            circuit_params: SGDCircuitParams::default(),
            metadata,
        };

        let json = proof_to_json(&proof).unwrap();
        let recovered = proof_from_json(&json).unwrap();
        
        assert_eq!(proof.proof, recovered.proof);
        assert_eq!(proof.public_inputs.step_nonce, recovered.public_inputs.step_nonce);
    }

    #[test]
    fn test_mock_proof_generation() {
        let public_inputs = SGDPublicInputs {
            w_t_root: "0x1234".to_string(),
            batch_root: "0x5678".to_string(),
            hparams_hash: "0xabcd".to_string(),
            w_t1_root: "0xef01".to_string(),
            step_nonce: 123,
            step_number: 456,
            epoch: 1,
        };
        let circuit_params = SGDCircuitParams::default();

        let proof1 = generate_mock_proof(&public_inputs, &circuit_params);
        let proof2 = generate_mock_proof(&public_inputs, &circuit_params);
        
        assert_eq!(proof1, proof2, "Mock proofs should be deterministic");
    }

    #[test]
    fn test_poseidon_compatibility() {
        let data = b"test data";
        let hash = python_compat::poseidon_hash_bytes(data);
        assert!(!hash.is_empty(), "Hash should not be empty");
        
        let hash2 = python_compat::poseidon_hash_bytes(data);
        assert_eq!(hash, hash2, "Hashes should be deterministic");
    }
}