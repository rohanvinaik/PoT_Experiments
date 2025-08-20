use base64::{engine::general_purpose::STANDARD as BASE64, Engine};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::io::{self, Read, Write};

#[derive(Debug, Deserialize)]
struct LoRAPublicInputs {
    base_weights_root: String,
    adapter_a_root_after: String,
    rank: u32,
}

#[derive(Debug, Deserialize)]
struct LoRAWitness {
    adapter_a_before: Vec<f64>,
}

#[derive(Debug, Deserialize)]
struct ProveRequest {
    public_inputs: LoRAPublicInputs,
    witness: LoRAWitness,
}

#[derive(Debug, Serialize)]
struct ProveResponse {
    proof: String,
}

fn generate_mock_lora_proof(
    public_inputs: &LoRAPublicInputs,
    witness: &LoRAWitness,
) -> Vec<u8> {
    let mut hasher = Sha256::new();
    hasher.update(public_inputs.base_weights_root.as_bytes());
    hasher.update(public_inputs.adapter_a_root_after.as_bytes());
    hasher.update(public_inputs.rank.to_le_bytes());
    hasher.update((witness.adapter_a_before.len() as u64).to_le_bytes());
    let mut proof = b"lora_proof_".to_vec();
    proof.extend_from_slice(&hasher.finalize()[..16]);
    proof
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input)?;

    let request: ProveRequest = serde_json::from_str(&input)
        .map_err(|e| format!("Failed to parse input JSON: {}", e))?;

    let proof_bytes = generate_mock_lora_proof(&request.public_inputs, &request.witness);
    let response = ProveResponse {
        proof: BASE64.encode(&proof_bytes),
    };

    let response_json = serde_json::to_string(&response)?;
    io::stdout().write_all(response_json.as_bytes())?;
    io::stdout().flush()?;
    Ok(())
}
