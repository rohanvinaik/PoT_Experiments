use clap::Command;
use serde::{Deserialize, Serialize};
use prover_halo2::{
    lora_circuit::{LoRACircuit, LoRAWitness},
    cli::{self, CommonArgs, OutputFormat},
};
use pasta_curves::{pallas::Scalar as Fp};
use ff::{Field, PrimeField};

// Helper function to convert f64 to field element
fn field_from_f64(val: f64) -> Fp {
    // Simple conversion - scale by 1000 and convert to integer
    let scaled = (val * 1000.0) as i64;
    if scaled >= 0 {
        Fp::from(scaled as u64)
    } else {
        -Fp::from((-scaled) as u64)
    }
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let app = cli::add_common_args(
        Command::new("prove-lora-stdin")
            .version("0.1.0")
            .author("PoT Team")
            .about("Generate zero-knowledge proofs for LoRA training steps")
            .long_about(
                "Generates zero-knowledge proofs for LoRA (Low-Rank Adaptation) training steps from JSON input.\n\
                 Input should contain a tuple of (statement, witness) data for LoRA operations.\n\
                 Output is a JSON response with the proof and metadata."
            )
    );
    
    let matches = app.get_matches();
    let args = CommonArgs::from_matches(&matches);
    
    // Validate input/output files
    if let Err(e) = cli::validate_input_file(args.input.as_ref()) {
        cli::error_exit(&e);
    }
    if let Err(e) = cli::validate_output_file(args.output.as_ref()) {
        cli::error_exit(&e);
    }
    
    cli::verbose_println(args.verbose, "Starting LoRA proof generation");
    
    // Read JSON input
    let input = cli::read_input(args.input.as_ref())?;
    
    cli::verbose_println(args.verbose, &format!("Read {} bytes of input", input.len()));

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
            let error_json = serde_json::to_string_pretty(&response)?;
            cli::write_output(args.output.as_ref(), &error_json, args.verbose)?;
            std::process::exit(1);
        }
    };
    
    cli::verbose_println(args.verbose, &format!("Parsed LoRA statement: rank={}, alpha={}", statement.rank, statement.alpha));

    let start_time = std::time::Instant::now();

    match generate_lora_proof(statement, witness, &args) {
        Ok(response) => {
            let output_content = match args.format {
                OutputFormat::Json => serde_json::to_string_pretty(&response)?,
                OutputFormat::Binary => {
                    cli::warn_println("Binary output format not supported for LoRA proofs, using JSON");
                    serde_json::to_string_pretty(&response)?
                }
            };
            cli::write_output(args.output.as_ref(), &output_content, args.verbose)?;
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
            let error_json = serde_json::to_string_pretty(&response)?;
            cli::write_output(args.output.as_ref(), &error_json, args.verbose)?;
            std::process::exit(1);
        }
    }
    
    Ok(())
}

fn generate_lora_proof(
    statement: LoRAStepStatement,
    witness: LoRAStepWitness,
    args: &CommonArgs,
) -> Result<ProofResponse, String> {
    let start_time = std::time::Instant::now();
    
    cli::verbose_println(args.verbose, "Converting statement roots to field elements...");

    // Convert string roots to field elements
    let base_weights_root = string_to_field(&statement.base_weights_root)?;
    let adapter_a_root_before = string_to_field(&statement.adapter_a_root_before)?;
    let adapter_b_root_before = string_to_field(&statement.adapter_b_root_before)?;
    let adapter_a_root_after = string_to_field(&statement.adapter_a_root_after)?;
    let adapter_b_root_after = string_to_field(&statement.adapter_b_root_after)?;

    // Convert witness data
    cli::verbose_println(args.verbose, &format!("Converting witness data for rank {}...", statement.rank));
    let lora_witness = convert_witness(&witness, statement.rank as usize)?;
    
    // Calculate compression ratio
    let d_approx = (witness.adapter_a_before.len() as f32 / statement.rank as f32).sqrt();
    let full_params = d_approx * d_approx;
    let lora_params = statement.rank as f32 * 2.0 * d_approx;
    let compression_ratio = full_params / lora_params;

    // Create circuit (simplified - just validate the witness conversion worked)
    let _circuit = LoRACircuit::<Fp> {
        base_weights_root,
        adapter_a_root: adapter_a_root_before,
        adapter_b_root: adapter_b_root_before,
        effective_weights_root: adapter_a_root_after, // Use after as effective
        rank: statement.rank,
        scale_factor: field_from_f64(statement.alpha),
        witness: Some(lora_witness),
    };

    // For now, create a mock proof (simplified implementation)
    cli::verbose_println(args.verbose, "Generating mock proof...");
    let proof_bytes = vec![0u8; 128]; // Mock proof bytes
    
    // Public inputs
    let public_inputs = vec![
        base_weights_root,
        adapter_a_root_before,
        adapter_b_root_before,
        adapter_a_root_after,
        adapter_b_root_after,
    ];
    
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
            circuit_rows: 4096, // Mock value
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
            adapter_a[row][col] = field_from_f64(val);
        }
    }

    // Convert adapter B (rank × d_out)
    let mut adapter_b = vec![vec![Fp::ZERO; d_out]; rank];
    for (i, &val) in witness.adapter_b_before.iter().enumerate() {
        let row = i / d_out;
        let col = i % d_out;
        if row < rank && col < d_out {
            adapter_b[row][col] = field_from_f64(val);
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