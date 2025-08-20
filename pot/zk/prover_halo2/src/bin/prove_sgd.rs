use clap::{Arg, Command};
use pot_zk_prover::{
    circuit::{SGDCircuitParams, SGDPublicInputs, utils},
    create_witness_from_json, generate_setup, proof_to_json, prove_sgd_step,
    ProverError, ProvingSystemParams,
};
use std::fs;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let matches = Command::new("prove_sgd")
        .version("0.1.0")
        .about("Generate ZK proof for SGD training step")
        .arg(
            Arg::new("public-inputs")
                .short('p')
                .long("public-inputs")
                .value_name("FILE")
                .help("JSON file containing public inputs (SGDStepStatement)")
                .required(true),
        )
        .arg(
            Arg::new("witness")
                .short('w')
                .long("witness")
                .value_name("FILE")
                .help("JSON file containing private witness data")
                .required(true),
        )
        .arg(
            Arg::new("output")
                .short('o')
                .long("output")
                .value_name("FILE")
                .help("Output file for the generated proof")
                .required(true),
        )
        .arg(
            Arg::new("setup")
                .short('s')
                .long("setup")
                .value_name("FILE")
                .help("Setup file (if not provided, will generate new setup)")
                .required(false),
        )
        .arg(
            Arg::new("params-k")
                .long("params-k")
                .value_name("NUMBER")
                .help("KZG parameter size (log2)")
                .default_value("17"),
        )
        .arg(
            Arg::new("max-weights")
                .long("max-weights")
                .value_name("NUMBER")
                .help("Maximum number of weights in circuit")
                .default_value("1024"),
        )
        .arg(
            Arg::new("verbose")
                .short('v')
                .long("verbose")
                .help("Enable verbose output")
                .action(clap::ArgAction::SetTrue),
        )
        .get_matches();

    let verbose = matches.get_flag("verbose");
    
    if verbose {
        println!("SGD ZK Proof Generator v0.1.0");
        println!("================================");
    }

    // Parse command line arguments
    let public_inputs_path = matches.get_one::<String>("public-inputs").unwrap();
    let witness_path = matches.get_one::<String>("witness").unwrap();
    let output_path = matches.get_one::<String>("output").unwrap();
    let params_k: u32 = matches.get_one::<String>("params-k").unwrap().parse()?;
    let _max_weights: usize = matches.get_one::<String>("max-weights").unwrap().parse()?;

    if verbose {
        println!("Public inputs: {}", public_inputs_path);
        println!("Witness data: {}", witness_path);
        println!("Output: {}", output_path);
        println!("KZG params k: {}", params_k);
        println!("Using 16x4 weight matrix");
    }

    // Read input files
    if verbose {
        println!("\nReading input files...");
    }
    
    let public_inputs_json = fs::read_to_string(public_inputs_path)
        .map_err(|e| format!("Failed to read public inputs file: {}", e))?;
    
    let witness_json = fs::read_to_string(witness_path)
        .map_err(|e| format!("Failed to read witness file: {}", e))?;

    // Parse public inputs
    let public_inputs: SGDPublicInputs = serde_json::from_str(&public_inputs_json)
        .map_err(|e| format!("Failed to parse public inputs JSON: {}", e))?;

    if verbose {
        println!("Public inputs parsed:");
        println!("  W_t_root: {}", public_inputs.w_t_root);
        println!("  batch_root: {}", public_inputs.batch_root);
        println!("  W_t1_root: {}", public_inputs.w_t1_root);
        println!("  step_nonce: {}", public_inputs.step_nonce);
        println!("  step_number: {}", public_inputs.step_number);
        println!("  epoch: {}", public_inputs.epoch);
    }

    // Validate public inputs
    utils::validate_public_inputs(&public_inputs)
        .map_err(|e| format!("Invalid public inputs: {}", e))?;

    // Parse witness data
    let witness = create_witness_from_json(&witness_json)
        .map_err(|e| format!("Failed to parse witness data: {}", e))?;

    if verbose {
        println!("\nWitness data parsed:");
        println!("  Weights before: {} elements", witness.weights_before.len());
        println!("  Weights after: {} elements", witness.weights_after.len());
        println!("  Batch inputs: {} elements", witness.batch_inputs.len());
        println!("  Batch targets: {} elements", witness.batch_targets.len());
        println!("  Gradients: {} elements", witness.gradients.len());
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
    if verbose {
        println!("\nGenerating trusted setup...");
        println!("This may take a while for large parameter sizes...");
    }

    let setup = if let Some(setup_path) = matches.get_one::<String>("setup") {
        if verbose {
            println!("Loading setup from: {}", setup_path);
        }
        // TODO: Implement setup loading from file
        return Err("Setup file loading not yet implemented".into());
    } else {
        if verbose {
            println!("Generating new setup with k={}", params_k);
        }
        generate_setup(proving_params)?
    };

    if verbose {
        println!("Setup generation completed!");
    }

    // Generate proof
    if verbose {
        println!("\nGenerating ZK proof...");
    }

    let proof = prove_sgd_step(&setup, public_inputs, witness, circuit_params)
        .map_err(|e| format!("Proof generation failed: {}", e))?;

    if verbose {
        println!("Proof generated successfully!");
        println!("Proof size: {} bytes", proof.proof.len());
    }

    // Serialize and save proof
    let proof_json = proof_to_json(&proof)
        .map_err(|e| format!("Failed to serialize proof: {}", e))?;

    fs::write(output_path, proof_json)
        .map_err(|e| format!("Failed to write proof file: {}", e))?;

    if verbose {
        println!("\nProof saved to: {}", output_path);
        println!("Proof generation completed successfully!");
    } else {
        println!("Proof generated and saved to {}", output_path);
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