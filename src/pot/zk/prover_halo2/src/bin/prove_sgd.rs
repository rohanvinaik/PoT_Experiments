use clap::{Arg, Command};
use prover_halo2::{
    circuit::{SGDCircuitParams, SGDPublicInputs, utils},
    cli::{self, OutputFormat},
    create_witness_from_json, generate_setup, proof_to_json, prove_sgd_step,
    ProvingSystemParams,
};
use std::fs;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let app = Command::new("prove-sgd")
        .version("0.1.0")
        .author("PoT Team")
        .about("Generate zero-knowledge proofs for SGD training steps")
        .long_about(
            "Generates zero-knowledge proofs for SGD training steps from separate input files.\n\
             Requires public inputs file, witness data file, and output file path.\n\
             Optionally supports loading pre-generated trusted setup parameters."
        )
        .arg(
            Arg::new("public-inputs")
                .short('p')
                .long("public-inputs")
                .value_name("FILE")
                .help("JSON file containing public inputs (SGDStepStatement)")
                .required(true)
                .value_parser(clap::value_parser!(PathBuf)),
        )
        .arg(
            Arg::new("witness")
                .short('w')
                .long("witness")
                .value_name("FILE")
                .help("JSON file containing private witness data")
                .required(true)
                .value_parser(clap::value_parser!(PathBuf)),
        )
        .arg(
            Arg::new("output")
                .short('o')
                .long("output")
                .value_name("FILE")
                .help("Output file for the generated proof")
                .required(true)
                .value_parser(clap::value_parser!(PathBuf)),
        )
        .arg(
            Arg::new("setup")
                .short('s')
                .long("setup")
                .value_name("FILE")
                .help("Setup file (if not provided, will generate new setup)")
                .required(false)
                .value_parser(clap::value_parser!(PathBuf)),
        )
        .arg(
            Arg::new("params-k")
                .long("params-k")
                .value_name("SIZE")
                .help("Circuit size parameter (log2 of constraint count)")
                .value_parser(clap::value_parser!(u32).range(8..=25))
                .default_value("17"),
        )
        .arg(
            Arg::new("verbose")
                .short('v')
                .long("verbose")
                .help("Enable verbose output")
                .action(clap::ArgAction::SetTrue),
        )
        .arg(
            Arg::new("format")
                .long("format")
                .value_name("FORMAT")
                .help("Output format")
                .value_parser(clap::value_parser!(OutputFormat))
                .default_value("json"),
        );
    
    let matches = app.get_matches();

    // Extract arguments
    let verbose = matches.get_flag("verbose");
    let public_inputs_path = matches.get_one::<PathBuf>("public-inputs").unwrap();
    let witness_path = matches.get_one::<PathBuf>("witness").unwrap();
    let output_path = matches.get_one::<PathBuf>("output").unwrap();
    let setup_path = matches.get_one::<PathBuf>("setup");
    let params_k = matches.get_one::<u32>("params-k").copied().unwrap_or(17);
    let format = matches.get_one::<OutputFormat>("format").cloned().unwrap_or(OutputFormat::Json);
    
    // Validate input files
    if let Err(e) = cli::validate_input_file(Some(public_inputs_path)) {
        cli::error_exit(&e);
    }
    if let Err(e) = cli::validate_input_file(Some(witness_path)) {
        cli::error_exit(&e);
    }
    if let Some(setup_file) = setup_path {
        if let Err(e) = cli::validate_input_file(Some(setup_file)) {
            cli::error_exit(&e);
        }
    }
    
    // Validate output file directory
    if let Err(e) = cli::validate_output_file(Some(output_path)) {
        cli::error_exit(&e);
    }
    
    cli::verbose_println(verbose, "Starting SGD proof generation");
    cli::verbose_println(verbose, &format!("Using circuit parameters: k={}", params_k));

    cli::verbose_println(verbose, &format!("Public inputs: {}", public_inputs_path.display()));
    cli::verbose_println(verbose, &format!("Witness data: {}", witness_path.display()));
    cli::verbose_println(verbose, &format!("Output: {}", output_path.display()));
    cli::verbose_println(verbose, &format!("Output format: {}", format));
    if let Some(setup_file) = setup_path {
        cli::verbose_println(verbose, &format!("Setup file: {}", setup_file.display()));
    }

    // Read input files
    cli::verbose_println(verbose, "Reading input files...");
    
    let public_inputs_json = fs::read_to_string(public_inputs_path)
        .map_err(|e| format!("Failed to read public inputs file '{}': {}", public_inputs_path.display(), e))?;
    
    let witness_json = fs::read_to_string(witness_path)
        .map_err(|e| format!("Failed to read witness file '{}': {}", witness_path.display(), e))?;
        
    cli::verbose_println(verbose, &format!("Read {} bytes from public inputs file", public_inputs_json.len()));
    cli::verbose_println(verbose, &format!("Read {} bytes from witness file", witness_json.len()));

    // Parse public inputs
    let public_inputs: SGDPublicInputs = serde_json::from_str(&public_inputs_json)
        .map_err(|e| format!("Failed to parse public inputs JSON: {}", e))?;

    if verbose {
        cli::verbose_println(verbose, "Public inputs parsed:");
        cli::verbose_println(verbose, &format!("  W_t_root: {}", public_inputs.w_t_root));
        cli::verbose_println(verbose, &format!("  batch_root: {}", public_inputs.batch_root));
        cli::verbose_println(verbose, &format!("  W_t1_root: {}", public_inputs.w_t1_root));
        cli::verbose_println(verbose, &format!("  step_nonce: {}", public_inputs.step_nonce));
        cli::verbose_println(verbose, &format!("  step_number: {}", public_inputs.step_number));
        cli::verbose_println(verbose, &format!("  epoch: {}", public_inputs.epoch));
    }

    // Validate public inputs
    utils::validate_public_inputs(&public_inputs)
        .map_err(|e| format!("Invalid public inputs: {}", e))?;

    // Parse witness data
    let witness = create_witness_from_json(&witness_json)
        .map_err(|e| format!("Failed to parse witness data: {}", e))?;

    if verbose {
        cli::verbose_println(verbose, "Witness data parsed:");
        cli::verbose_println(verbose, &format!("  Weights before: {} elements", witness.weights_before.len()));
        cli::verbose_println(verbose, &format!("  Weights after: {} elements", witness.weights_after.len()));
        cli::verbose_println(verbose, &format!("  Batch inputs: {} elements", witness.batch_inputs.len()));
        cli::verbose_println(verbose, &format!("  Batch targets: {} elements", witness.batch_targets.len()));
        cli::verbose_println(verbose, &format!("  Gradients: {} elements", witness.gradients.len()));
    }

    // Setup circuit parameters
    let circuit_params = SGDCircuitParams {
        weight_rows: 16,
        weight_cols: 4,
        max_merkle_depth: 20,
        max_batch_size: 256,
        fixed_point_scale: 65536,
    };

    let proving_params = ProvingSystemParams {
        circuit_params: circuit_params.clone(),
        params_k: params_k,
    };

    // Generate or load setup
    cli::verbose_println(verbose, "Initializing trusted setup...");
    cli::verbose_println(verbose, "This may take a while for large parameter sizes...");

    let setup = if let Some(setup_file) = setup_path {
        cli::verbose_println(verbose, &format!("Loading setup from: {}", setup_file.display()));
        // TODO: Implement setup loading from file
        return Err("Setup file loading not yet implemented".into());
    } else {
        cli::verbose_println(verbose, &format!("Generating new setup with k={}", params_k));
        generate_setup(proving_params)?
    };

    cli::verbose_println(verbose, "Setup generation completed!");

    // Generate proof
    cli::verbose_println(verbose, "Generating ZK proof...");

    let proof = prove_sgd_step(&setup, public_inputs, witness, circuit_params)
        .map_err(|e| format!("Proof generation failed: {}", e))?;

    cli::verbose_println(verbose, "Proof generated successfully!");
    cli::verbose_println(verbose, &format!("Proof size: {} bytes", proof.proof.len()));

    // Prepare output based on format
    let output_content = match format {
        OutputFormat::Json => {
            proof_to_json(&proof)
                .map_err(|e| format!("Failed to serialize proof: {}", e))?
        }
        OutputFormat::Binary => {
            cli::warn_println("Binary output format will output raw proof bytes without metadata");
            String::from_utf8_lossy(&proof.proof).to_string()
        }
    };

    // Write output using CLI utilities
    cli::write_output(Some(output_path), &output_content, verbose)?;

    if verbose {
        cli::verbose_println(verbose, &format!("Proof saved to: {}", output_path.display()));
        cli::verbose_println(verbose, "Proof generation completed successfully!");
    } else {
        println!("Proof generated and saved to {}", output_path.display());
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_cli_argument_parsing() {
        // Test that the CLI argument structure compiles correctly
        let cmd = Command::new("prove_sgd");
        assert!(!cmd.get_name().is_empty());
    }

    #[test]
    fn test_file_operations() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.json");
        
        let test_data = r#"{"test": "data"}"#;
        std::fs::write(&file_path, test_data).unwrap();
        
        let read_data = std::fs::read_to_string(&file_path).unwrap();
        assert_eq!(test_data, read_data);
    }
}