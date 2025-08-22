// CLEANUP 2025-08-20: Removed unused import (std::io::Write)
use base64::{engine::general_purpose::STANDARD as BASE64, Engine};
use clap::Command;
use prover_halo2::{
    circuit::{SGDCircuitParams, SGDPublicInputs, utils},
    cli::{self, CommonArgs, OutputFormat},
    create_witness_from_json, generate_setup, prove_sgd_step,
    ProvingSystemParams,
};
use serde::{Deserialize, Serialize};
// Removed unused import: std::io::Write

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
    let app = cli::add_common_args(
        Command::new("prove-sgd-stdin")
            .version("0.1.0")
            .author("PoT Team")
            .about("Generate zero-knowledge proofs for SGD training steps")
            .long_about(
                "Generates zero-knowledge proofs for SGD training steps from JSON input.\n\
                 Input should contain public_inputs, witness data, and optional params_k.\n\
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
    
    cli::verbose_println(args.verbose, "Starting SGD proof generation");
    
    // Read JSON input
    let input = cli::read_input(args.input.as_ref())?;
    
    cli::verbose_println(args.verbose, &format!("Read {} bytes of input", input.len()));
    
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
    
    let params_k = args.params_k.or(request.params_k).unwrap_or(17);
    
    let proving_params = ProvingSystemParams {
        circuit_params: circuit_params.clone(),
        params_k,
    };
    
    cli::verbose_println(args.verbose, &format!("Using circuit parameters: k={}, weights={}x{}", 
                                                params_k, circuit_params.weight_rows, circuit_params.weight_cols));
    
    // Generate setup (in production, this should be cached)
    cli::verbose_println(args.verbose, "Generating trusted setup...");
    let setup = generate_setup(proving_params)
        .map_err(|e| format!("Setup generation failed: {}", e))?;
    
    // Generate proof
    cli::verbose_println(args.verbose, "Generating proof...");
    let proof = prove_sgd_step(&setup, request.public_inputs, witness, circuit_params.clone())
        .map_err(|e| format!("Proof generation failed: {}", e))?;
    
    // Create response based on output format
    let output_content = match args.format {
        OutputFormat::Json => {
            // Encode proof as base64 for JSON
            let proof_base64 = BASE64.encode(&proof.proof);
            
            let response = ProveResponse {
                proof: proof_base64,
                metadata: ProofMetadata {
                    circuit_size: circuit_params.weight_rows * circuit_params.weight_cols,
                    proof_size: proof.proof.len(),
                    params_k,
                },
            };
            
            serde_json::to_string_pretty(&response)?
        }
        OutputFormat::Binary => {
            // For binary format, just output the raw proof bytes
            cli::warn_println("Binary output format will output raw proof bytes without metadata");
            String::from_utf8_lossy(&proof.proof).to_string()
        }
    };
    
    cli::verbose_println(args.verbose, &format!("Generated proof of {} bytes", proof.proof.len()));
    
    // Write response
    cli::write_output(args.output.as_ref(), &output_content, args.verbose)?;
    
    Ok(())
}