use base64::{engine::general_purpose::STANDARD as BASE64, Engine};
use pot_zk_prover::{
    circuit::{SGDCircuitParams, SGDPublicInputs},
    generate_setup, verify_sgd_step, ProvingSystemParams, SerializableProof,
};
use serde::Deserialize;
use std::io::{self, Read};
use std::process;

#[derive(Debug, Deserialize)]
struct VerifyRequest {
    public_inputs: SGDPublicInputs,
    proof: String,  // Base64 encoded proof bytes
    params_k: Option<u32>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Read JSON input from stdin
    let mut input = String::new();
    io::stdin().read_to_string(&mut input)?;
    
    // Parse request
    let request: VerifyRequest = match serde_json::from_str(&input) {
        Ok(req) => req,
        Err(e) => {
            eprintln!("Failed to parse input JSON: {}", e);
            process::exit(1);
        }
    };
    
    // Decode proof from base64
    let proof_bytes = match BASE64.decode(&request.proof) {
        Ok(bytes) => bytes,
        Err(e) => {
            eprintln!("Failed to decode proof from base64: {}", e);
            process::exit(1);
        }
    };
    
    // Setup circuit parameters (16x4 fixed for now)
    let circuit_params = SGDCircuitParams {
        weight_rows: 16,
        weight_cols: 4,
        max_merkle_depth: 20,
        max_batch_size: 256,
        fixed_point_scale: 65536,
    };
    
    let params_k = request.params_k.unwrap_or(17);
    
    let proving_params = ProvingSystemParams {
        circuit_params: circuit_params.clone(),
        params_k,
    };
    
    // Generate setup (in production, this should be cached)
    let setup = match generate_setup(proving_params) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Setup generation failed: {:?}", e);
            process::exit(1);
        }
    };
    
    // Create proof structure
    let proof = SerializableProof {
        proof: proof_bytes,
        public_inputs: request.public_inputs,
        circuit_params,
        metadata: pot_zk_prover::ProofMetadata {
            proving_time_ms: 0,
            circuit_size: 64,
            public_input_count: 7,
            verification_key_hash: "".to_string(),
        },
    };
    
    // Verify proof
    match verify_sgd_step(&setup, &proof) {
        Ok(true) => {
            // Verification succeeded
            process::exit(0);
        }
        Ok(false) => {
            eprintln!("Proof verification failed");
            process::exit(1);
        }
        Err(e) => {
            eprintln!("Verification error: {:?}", e);
            process::exit(1);
        }
    }
}