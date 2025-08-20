use base64::{engine::general_purpose::STANDARD as BASE64, Engine};
use pot_zk_prover::{
    circuit::{SGDCircuitParams, SGDPublicInputs, utils},
    create_witness_from_json, generate_setup, prove_sgd_step,
    ProvingSystemParams,
};
use serde::{Deserialize, Serialize};
use std::io::{self, Read, Write};

#[derive(Debug, Deserialize)]
struct ProveRequest {
    public_inputs: SGDPublicInputs,
    witness: serde_json::Value,  // Raw JSON for witness
    params_k: Option<u32>,
}

#[derive(Debug, Serialize)]
struct ProveResponse {
    proof: String,  // Base64 encoded proof bytes
    metadata: ProofMetadata,
}

#[derive(Debug, Serialize)]
struct ProofMetadata {
    circuit_size: usize,
    proof_size: usize,
    params_k: u32,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Read JSON input from stdin
    let mut input = String::new();
    io::stdin().read_to_string(&mut input)?;
    
    // Parse request
    let request: ProveRequest = serde_json::from_str(&input)
        .map_err(|e| format!("Failed to parse input JSON: {}", e))?;
    
    // Validate public inputs
    utils::validate_public_inputs(&request.public_inputs)
        .map_err(|e| format!("Invalid public inputs: {}", e))?;
    
    // Parse witness data
    let witness_json = serde_json::to_string(&request.witness)?;
    let witness = create_witness_from_json(&witness_json)
        .map_err(|e| format!("Failed to parse witness data: {}", e))?;
    
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
    let setup = generate_setup(proving_params)
        .map_err(|e| format!("Setup generation failed: {}", e))?;
    
    // Generate proof
    let proof = prove_sgd_step(&setup, request.public_inputs, witness, circuit_params.clone())
        .map_err(|e| format!("Proof generation failed: {}", e))?;
    
    // Encode proof as base64
    let proof_base64 = BASE64.encode(&proof.proof);
    
    // Create response
    let response = ProveResponse {
        proof: proof_base64,
        metadata: ProofMetadata {
            circuit_size: circuit_params.weight_rows * circuit_params.weight_cols,
            proof_size: proof.proof.len(),
            params_k,
        },
    };
    
    // Write JSON response to stdout
    let response_json = serde_json::to_string(&response)?;
    io::stdout().write_all(response_json.as_bytes())?;
    io::stdout().flush()?;
    
    Ok(())
}