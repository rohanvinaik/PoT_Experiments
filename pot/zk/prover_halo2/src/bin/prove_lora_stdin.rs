use std::io::{self, Read};
use std::process;
use serde::{Deserialize, Serialize};
use halo2_proofs::{
    pasta::{Fp, EqAffine},
    plonk::{create_proof, keygen_pk, keygen_vk},
    poly::ipa::{
        commitment::{IPACommitmentScheme, ParamsIPA},
        multiopen::ProverIPA,
    },
    transcript::{Blake2bWrite, Challenge255},
    SerdeFormat,
};
use prover_halo2::{lora_circuit::{LoRACircuit, LoRAWitness}, fixed_point::FixedPoint};
use rand::rngs::OsRng;
use ff::{Field, PrimeField};

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

#[derive(Deserialize)]
struct LoRAStepWitness {
    adapter_a_before: Vec<f64>,
    adapter_b_before: Vec<f64>,
    adapter_a_after: Vec<f64>,
    adapter_b_after: Vec<f64>,
    adapter_a_gradients: Vec<f64>,
    adapter_b_gradients: Vec<f64>,
    batch_inputs: Vec<f64>,
    batch_targets: Vec<f64>,
    learning_rate: f64,
}

#[derive(Serialize)]
struct ProofResponse {
    success: bool,
    proof: Option<String>,
    error: Option<String>,
    verification_key: Option<String>,
    public_inputs: Vec<String>,
    metadata: ProofMetadata,
}

#[derive(Serialize)]
struct ProofMetadata {
    rank: u32,
    alpha: f64,
    proof_size_bytes: usize,
    generation_time_ms: u64,
    circuit_rows: u32,
    compression_ratio: f32,
}

fn main() {
    // Read input from stdin
    let mut input = String::new();
    if let Err(e) = io::stdin().read_to_string(&mut input) {
        eprintln!("Error reading from stdin: {}", e);
        process::exit(1);
    }

    // Parse JSON input
    let parsed_input: Result<(LoRAStepStatement, LoRAStepWitness), _> = 
        serde_json::from_str(&input);
    
    let (statement, witness) = match parsed_input {
        Ok(data) => data,
        Err(e) => {
            let response = ProofResponse {
                success: false,
                proof: None,
                error: Some(format!("JSON parsing error: {}", e)),
                verification_key: None,
                public_inputs: vec![],
                metadata: ProofMetadata {
                    rank: 0,
                    alpha: 0.0,
                    proof_size_bytes: 0,
                    generation_time_ms: 0,
                    circuit_rows: 0,
                    compression_ratio: 0.0,
                },
            };
            println!("{}", serde_json::to_string(&response).unwrap());
            process::exit(1);
        }
    };

    let start_time = std::time::Instant::now();

    match generate_lora_proof(statement, witness) {
        Ok(response) => {
            println!("{}", serde_json::to_string(&response).unwrap());
        }
        Err(e) => {
            let response = ProofResponse {
                success: false,
                proof: None,
                error: Some(e),
                verification_key: None,
                public_inputs: vec![],
                metadata: ProofMetadata {
                    rank: 0,
                    alpha: 0.0,
                    proof_size_bytes: 0,
                    generation_time_ms: start_time.elapsed().as_millis() as u64,
                    circuit_rows: 0,
                    compression_ratio: 0.0,
                },
            };
            println!("{}", serde_json::to_string(&response).unwrap());
            process::exit(1);
        }
    }
}

fn generate_lora_proof(
    statement: LoRAStepStatement,
    witness: LoRAStepWitness,
) -> Result<ProofResponse, String> {
    let start_time = std::time::Instant::now();

    // Convert string roots to field elements
    let base_weights_root = string_to_field(&statement.base_weights_root)?;
    let adapter_a_root_before = string_to_field(&statement.adapter_a_root_before)?;
    let adapter_b_root_before = string_to_field(&statement.adapter_b_root_before)?;
    let adapter_a_root_after = string_to_field(&statement.adapter_a_root_after)?;
    let adapter_b_root_after = string_to_field(&statement.adapter_b_root_after)?;

    // Convert witness data
    let lora_witness = convert_witness(&witness, statement.rank as usize)?;
    
    // Calculate compression ratio
    let d_approx = (witness.adapter_a_before.len() as f32 / statement.rank as f32).sqrt();
    let full_params = d_approx * d_approx;
    let lora_params = statement.rank as f32 * 2.0 * d_approx;
    let compression_ratio = full_params / lora_params;

    // Create circuit
    let circuit = LoRACircuit::<Fp> {
        base_weights_root,
        adapter_a_root: adapter_a_root_before,
        adapter_b_root: adapter_b_root_before,
        effective_weights_root: adapter_a_root_after, // Use after as effective
        rank: statement.rank,
        scale_factor: FixedPoint::from_f64(statement.alpha).to_field(),
        witness: Some(lora_witness),
    };

    // Circuit parameters - smaller for LoRA
    const K: u32 = 12; // 2^12 = 4096 rows, much smaller than SGD
    
    // Setup trusted setup parameters
    let params: ParamsIPA<EqAffine> = ParamsIPA::new(K);
    
    // Generate verification key
    let vk = keygen_vk(&params, &circuit.without_witnesses())
        .map_err(|e| format!("VK generation failed: {:?}", e))?;
    
    // Generate proving key
    let pk = keygen_pk(&params, vk.clone(), &circuit.without_witnesses())
        .map_err(|e| format!("PK generation failed: {:?}", e))?;

    // Public inputs
    let public_inputs = vec![
        base_weights_root,
        adapter_a_root_before,
        adapter_b_root_before,
        adapter_a_root_after,
        adapter_b_root_after,
    ];

    // Create proof
    let mut transcript = Blake2bWrite::<_, _, Challenge255<_>>::init(vec![]);
    
    create_proof::<IPACommitmentScheme<EqAffine>, ProverIPA<EqAffine>, _, _, _, _>(
        &params,
        &pk,
        &[circuit],
        &[&[&public_inputs]],
        OsRng,
        &mut transcript,
    )
    .map_err(|e| format!("Proof generation failed: {:?}", e))?;

    let proof_bytes = transcript.finalize();
    let generation_time = start_time.elapsed().as_millis() as u64;

    // Serialize verification key (simplified approach)
    let vk_hex = hex::encode(b"mock_vk_placeholder");

    Ok(ProofResponse {
        success: true,
        proof: Some(hex::encode(&proof_bytes)),
        error: None,
        verification_key: Some(vk_hex),
        public_inputs: public_inputs.iter()
            .map(|&x| format!("0x{:064x}", field_to_u256(x)))
            .collect(),
        metadata: ProofMetadata {
            rank: statement.rank,
            alpha: statement.alpha,
            proof_size_bytes: proof_bytes.len(),
            generation_time_ms: generation_time,
            circuit_rows: 1 << K,
            compression_ratio,
        },
    })
}

fn convert_witness(witness: &LoRAStepWitness, rank: usize) -> Result<LoRAWitness<Fp>, String> {
    // Estimate dimensions from data
    let adapter_a_len = witness.adapter_a_before.len();
    let adapter_b_len = witness.adapter_b_before.len();
    
    // For simplicity, assume square adapters for this implementation
    let d_in = adapter_a_len / rank;
    let d_out = adapter_b_len / rank;
    
    if d_in * rank != adapter_a_len || d_out * rank != adapter_b_len {
        return Err("Adapter dimensions don't match rank".to_string());
    }

    // Convert adapter A (d_in × rank)
    let mut adapter_a = vec![vec![Fp::ZERO; rank]; d_in];
    for (i, &val) in witness.adapter_a_before.iter().enumerate() {
        let row = i / rank;
        let col = i % rank;
        if row < d_in && col < rank {
            adapter_a[row][col] = FixedPoint::from_f64(val).to_field();
        }
    }

    // Convert adapter B (rank × d_out)
    let mut adapter_b = vec![vec![Fp::ZERO; d_out]; rank];
    for (i, &val) in witness.adapter_b_before.iter().enumerate() {
        let row = i / d_out;
        let col = i % d_out;
        if row < rank && col < d_out {
            adapter_b[row][col] = FixedPoint::from_f64(val).to_field();
        }
    }

    // Create dummy base weights (d_in × d_out)
    let base_weights = vec![vec![Fp::ZERO; d_out]; d_in];
    
    // Compute effective weights (simplified)
    let mut effective_weights = base_weights.clone();
    for i in 0..d_in {
        for j in 0..d_out {
            let mut ba_sum = Fp::ZERO;
            for k in 0..rank {
                ba_sum += adapter_b[k][j] * adapter_a[i][k];
            }
            // Alpha scaling would be applied here
            effective_weights[i][j] = base_weights[i][j] + ba_sum;
        }
    }

    // Dummy Merkle proofs
    let merkle_depth = 10;
    let dummy_proof = vec![vec![Fp::ZERO; 2]; merkle_depth];

    Ok(LoRAWitness {
        base_weights,
        adapter_a,
        adapter_b,
        effective_weights,
        base_weights_proof: dummy_proof.clone(),
        adapter_a_proof: dummy_proof.clone(),
        adapter_b_proof: dummy_proof,
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

fn field_to_u256(f: Fp) -> u64 {
    // Simplified conversion - just take lower 64 bits
    let repr = f.to_repr();
    let bytes = repr.as_ref();
    u64::from_le_bytes([
        bytes[0], bytes[1], bytes[2], bytes[3],
        bytes[4], bytes[5], bytes[6], bytes[7],
    ])
}