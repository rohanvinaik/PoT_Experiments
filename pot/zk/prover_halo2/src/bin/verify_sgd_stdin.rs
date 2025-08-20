use base64::{engine::general_purpose::STANDARD as BASE64, Engine};
use clap::Command;
use prover_halo2::{
    circuit::{SGDCircuitParams, SGDPublicInputs},
    cli::{self, CommonArgs, OutputFormat},
    generate_setup, verify_sgd_step, ProvingSystemParams, SerializableProof,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
struct VerifyRequest {
    public_inputs: SGDPublicInputs,
    proof: String,  // Base64 encoded proof bytes
    params_k: Option<u32>,
}

#[derive(Debug, Serialize)]
struct VerifyResponse {
    valid: bool,
    verification_time_ms: u64,
    metadata: VerificationMetadata,
}

#[derive(Debug, Serialize)]
struct VerificationMetadata {
    circuit_size: usize,
    proof_size: usize,
    params_k: u32,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let app = cli::add_common_args(
        Command::new("verify-sgd-stdin")
            .version("0.1.0")
            .author("PoT Team")
            .about("Verify zero-knowledge proofs for SGD training steps")
            .long_about(
                "Verifies zero-knowledge proofs for SGD training steps from JSON input.\n\
                 Input should contain public_inputs, base64-encoded proof, and optional params_k.\n\
                 Output is a JSON response with verification result and metadata."
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
    
    cli::verbose_println(args.verbose, "Starting SGD proof verification");
    
    // Read JSON input
    let input = cli::read_input(args.input.as_ref())?;
    
    cli::verbose_println(args.verbose, &format!("Read {} bytes of input", input.len()));
    
    // Parse request
    let request: VerifyRequest = serde_json::from_str(&input)
        .map_err(|e| format!("Failed to parse input JSON: {}", e))?;
    
    // Decode proof from base64
    let proof_bytes = BASE64.decode(&request.proof)
        .map_err(|e| format!("Failed to decode proof from base64: {}", e))?;
    
    cli::verbose_println(args.verbose, &format!("Decoded proof of {} bytes", proof_bytes.len()));
    
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
        .map_err(|e| format!("Setup generation failed: {:?}", e))?;
    
    // Create proof structure
    let proof = SerializableProof {
        proof: proof_bytes.clone(),
        public_inputs: request.public_inputs,
        circuit_params: circuit_params.clone(),
        metadata: prover_halo2::ProofMetadata {
            proving_time_ms: 0,
            circuit_size: circuit_params.weight_rows * circuit_params.weight_cols,
            public_input_count: 7,
            verification_key_hash: "".to_string(),
        },
    };
    
    // Verify proof with timing
    cli::verbose_println(args.verbose, "Verifying proof...");
    let start_time = std::time::Instant::now();
    
    let verification_result = verify_sgd_step(&setup, &proof)
        .map_err(|e| format!("Verification error: {:?}", e))?;
    
    let verification_time = start_time.elapsed().as_millis() as u64;
    
    cli::verbose_println(args.verbose, &format!("Verification completed in {}ms: {}", 
                                                verification_time, if verification_result { "VALID" } else { "INVALID" }));
    
    // Create response based on output format
    let output_content = match args.format {
        OutputFormat::Json => {
            let response = VerifyResponse {
                valid: verification_result,
                verification_time_ms: verification_time,
                metadata: VerificationMetadata {
                    circuit_size: circuit_params.weight_rows * circuit_params.weight_cols,
                    proof_size: proof_bytes.len(),
                    params_k,
                },
            };
            
            serde_json::to_string_pretty(&response)?
        }
        OutputFormat::Binary => {
            // For binary format, just output the verification result as a single byte
            if verification_result { "1" } else { "0" }.to_string()
        }
    };
    
    // Write response
    cli::write_output(args.output.as_ref(), &output_content, args.verbose)?;
    
    Ok(())
}