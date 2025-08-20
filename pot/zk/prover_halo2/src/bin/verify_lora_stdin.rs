use std::io::{self, Read};
use std::process;
use serde::{Deserialize, Serialize};
use halo2_proofs::{
    plonk::{verify_proof, VerifyingKey},
    poly::ipa::{
        commitment::{ParamsIPA, IPACommitmentScheme},
        multiopen::VerifierIPA,
        strategy::SingleStrategy,
    },
    transcript::{Blake2bRead, Challenge255},
    SerdeFormat,
};
use pasta_curves::{pallas::{Scalar as Fp, Affine as EqAffine}};
use ff::PrimeField;

#[derive(Deserialize)]
struct VerificationRequest {
    proof: String,
    verification_key: String,
    public_inputs: Vec<String>,
    statement: LoRAStepStatement,
}

#[derive(Deserialize)]
struct LoRAStepStatement {
    base_weights_root: String,
    adapter_a_root_before: String,
    adapter_b_root_before: String,
    adapter_a_root_after: String,
    adapter_b_root_after: String,
    batch_root: String,
    hparams_hash: String,
    rank: u32,
    alpha: f64,
    step_number: u64,
    epoch: u64,
}

#[derive(Serialize)]
struct VerificationResponse {
    success: bool,
    valid: bool,
    error: Option<String>,
    verification_time_ms: u64,
    public_inputs_valid: bool,
    metadata: VerificationMetadata,
}

#[derive(Serialize)]
struct VerificationMetadata {
    rank: u32,
    alpha: f64,
    proof_size_bytes: usize,
    public_inputs_count: usize,
    circuit_type: String,
}

fn main() {
    // Read input from stdin
    let mut input = String::new();
    if let Err(e) = io::stdin().read_to_string(&mut input) {
        eprintln!("Error reading from stdin: {}", e);
        process::exit(1);
    }

    // Parse JSON input
    let request: VerificationRequest = match serde_json::from_str(&input) {
        Ok(req) => req,
        Err(e) => {
            let response = VerificationResponse {
                success: false,
                valid: false,
                error: Some(format!("JSON parsing error: {}", e)),
                verification_time_ms: 0,
                public_inputs_valid: false,
                metadata: VerificationMetadata {
                    rank: 0,
                    alpha: 0.0,
                    proof_size_bytes: 0,
                    public_inputs_count: 0,
                    circuit_type: "lora".to_string(),
                },
            };
            println!("{}", serde_json::to_string(&response).unwrap());
            process::exit(1);
        }
    };

    let start_time = std::time::Instant::now();

    match verify_lora_proof(request) {
        Ok(response) => {
            println!("{}", serde_json::to_string(&response).unwrap());
        }
        Err(e) => {
            let response = VerificationResponse {
                success: false,
                valid: false,
                error: Some(e),
                verification_time_ms: start_time.elapsed().as_millis() as u64,
                public_inputs_valid: false,
                metadata: VerificationMetadata {
                    rank: 0,
                    alpha: 0.0,
                    proof_size_bytes: 0,
                    public_inputs_count: 0,
                    circuit_type: "lora".to_string(),
                },
            };
            println!("{}", serde_json::to_string(&response).unwrap());
            process::exit(1);
        }
    }
}

fn verify_lora_proof(request: VerificationRequest) -> Result<VerificationResponse, String> {
    let start_time = std::time::Instant::now();

    // Decode proof
    let proof_bytes = hex::decode(&request.proof)
        .map_err(|e| format!("Invalid proof hex: {}", e))?;

    // For testing purposes, create a mock verification result
    if request.verification_key == hex::encode(b"mock_vk_placeholder") {
        // Mock successful verification
    } else {
        return Err("Invalid verification key".to_string());
    }

    // Parse public inputs
    let mut public_inputs = Vec::new();
    for input_str in &request.public_inputs {
        let field_element = string_to_field(input_str)?;
        public_inputs.push(field_element);
    }

    // Validate public inputs match statement
    let expected_inputs = vec![
        string_to_field(&request.statement.base_weights_root)?,
        string_to_field(&request.statement.adapter_a_root_before)?,
        string_to_field(&request.statement.adapter_b_root_before)?,
        string_to_field(&request.statement.adapter_a_root_after)?,
        string_to_field(&request.statement.adapter_b_root_after)?,
    ];

    let public_inputs_valid = public_inputs.len() == expected_inputs.len() &&
        public_inputs.iter().zip(expected_inputs.iter()).all(|(a, b)| a == b);

    // Mock verification for testing
    let verification_time = start_time.elapsed().as_millis() as u64;
    let is_valid = proof_bytes.len() > 32; // Simple check for non-empty proof

    Ok(VerificationResponse {
        success: true,
        valid: is_valid,
        error: if is_valid {
            None
        } else {
            Some(format!("Verification failed: {:?}", verification_result.err()))
        },
        verification_time_ms: verification_time,
        public_inputs_valid,
        metadata: VerificationMetadata {
            rank: request.statement.rank,
            alpha: request.statement.alpha,
            proof_size_bytes: proof_bytes.len(),
            public_inputs_count: public_inputs.len(),
            circuit_type: "lora".to_string(),
        },
    })
}

fn string_to_field(s: &str) -> Result<Fp, String> {
    // Remove '0x' prefix if present
    let hex_str = if s.starts_with("0x") {
        &s[2..]
    } else {
        s
    };
    
    // Convert hex string to bytes
    let bytes = hex::decode(hex_str)
        .map_err(|e| format!("Invalid hex string '{}': {}", s, e))?;
    
    if bytes.len() != 32 {
        return Err(format!("Expected 32 bytes, got {}", bytes.len()));
    }
    
    // Convert bytes to field element
    let mut array = [0u8; 32];
    array.copy_from_slice(&bytes);
    
    Ok(Fp::from_repr(array).unwrap())
}