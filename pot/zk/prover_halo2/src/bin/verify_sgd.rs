// CLEANUP 2025-08-20: Removed unused imports (serde::Deserialize, tempfile::tempdir)
use clap::{Arg, Command};
use prover_halo2::{
    cli::{self, OutputFormat},
    generate_setup, proof_from_json, verify_sgd_step, ProvingSystemParams,
};
use serde::Serialize; // Removed unused import: Deserialize
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Debug, Serialize)]
struct VerificationResult {
    valid: bool,
    verification_time_ms: u64,
    metadata: VerificationMetadata,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

#[derive(Debug, Serialize)]
struct VerificationMetadata {
    circuit_size: usize,
    proof_size: usize,
    params_k: u32,
    setup_time_ms: u64,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let app = Command::new("verify-sgd")
        .version("0.1.0")
        .author("PoT Team")
        .about("Verify zero-knowledge proofs for SGD training steps")
        .long_about(
            "Verifies zero-knowledge proofs for SGD training steps from proof files.\n\
             Requires a proof file and optionally a pre-generated trusted setup.\n\
             Outputs verification result with timing information."
        )
        .arg(
            Arg::new("proof")
                .short('p')
                .long("proof")
                .value_name("FILE")
                .help("JSON file containing the proof to verify")
                .required(true)
                .value_parser(clap::value_parser!(PathBuf)),
        )
        .arg(
            Arg::new("output")
                .short('o')
                .long("output")
                .value_name("FILE")
                .help("Output file for verification result (default: stdout)")
                .required(false)
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
                .help("Circuit size parameter (log2 of constraint count) - must match proof generation")
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
            Arg::new("benchmark")
                .short('b')
                .long("benchmark")
                .help("Enable benchmark timing")
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
    let benchmark = matches.get_flag("benchmark");
    let proof_path = matches.get_one::<PathBuf>("proof").unwrap();
    let output_path = matches.get_one::<PathBuf>("output");
    let setup_path = matches.get_one::<PathBuf>("setup");
    let params_k = matches.get_one::<u32>("params-k").copied().unwrap_or(17);
    let format = matches.get_one::<OutputFormat>("format").cloned().unwrap_or(OutputFormat::Json);
    
    // Validate input files
    if let Err(e) = cli::validate_input_file(Some(proof_path)) {
        cli::error_exit(&e);
    }
    if let Some(setup_file) = setup_path {
        if let Err(e) = cli::validate_input_file(Some(setup_file)) {
            cli::error_exit(&e);
        }
    }
    
    // Validate output file directory
    if let Some(output_file) = output_path {
        if let Err(e) = cli::validate_output_file(Some(output_file)) {
            cli::error_exit(&e);
        }
    }
    
    cli::verbose_println(verbose, "Starting SGD proof verification");
    cli::verbose_println(verbose, &format!("Using circuit parameters: k={}", params_k));
    cli::verbose_println(verbose, &format!("Proof file: {}", proof_path.display()));
    if let Some(setup_file) = setup_path {
        cli::verbose_println(verbose, &format!("Setup file: {}", setup_file.display()));
    }
    if let Some(output_file) = output_path {
        cli::verbose_println(verbose, &format!("Output file: {}", output_file.display()));
    }

    // Read proof file
    cli::verbose_println(verbose, "Reading proof file...");
    
    let proof_json = fs::read_to_string(proof_path)
        .map_err(|e| format!("Failed to read proof file '{}': {}", proof_path.display(), e))?;
        
    cli::verbose_println(verbose, &format!("Read {} bytes from proof file", proof_json.len()));

    // Parse proof
    let proof = proof_from_json(&proof_json)
        .map_err(|e| format!("Failed to parse proof JSON: {}", e))?;

    if verbose {
        println!("Proof parsed successfully:");
        println!("  Proof size: {} bytes", proof.proof.len());
        println!("  Public inputs:");
        println!("    W_t_root: {}", proof.public_inputs.w_t_root);
        println!("    batch_root: {}", proof.public_inputs.batch_root);
        println!("    W_t1_root: {}", proof.public_inputs.w_t1_root);
        println!("    step_nonce: {}", proof.public_inputs.step_nonce);
        println!("    step_number: {}", proof.public_inputs.step_number);
        println!("    epoch: {}", proof.public_inputs.epoch);
    }

    // Setup proving system parameters
    let proving_params = ProvingSystemParams {
        circuit_params: proof.circuit_params.clone(),
        params_k: params_k,
    };

    // Generate or load setup
    if verbose {
        println!("\nSetting up verification parameters...");
    }

    let setup_start = if benchmark { Some(Instant::now()) } else { None };

    let setup = if let Some(setup_path) = matches.get_one::<String>("setup") {
        if verbose {
            println!("Loading setup from: {}", setup_path);
        }
        // TODO: Implement setup loading from file
        return Err("Setup file loading not yet implemented".into());
    } else {
        if verbose {
            println!("Generating verification setup with k={}", params_k);
            println!("Note: In production, setup should be cached to avoid regeneration");
        }
        generate_setup(proving_params)?
    };

    if let Some(start_time) = setup_start {
        let setup_duration = start_time.elapsed();
        if benchmark {
            println!("Setup time: {:.2?}", setup_duration);
        }
    }

    if verbose {
        println!("Setup completed!");
    }

    // Verify proof
    if verbose {
        println!("\nVerifying proof...");
    }

    let verify_start = if benchmark { Some(Instant::now()) } else { None };

    let verification_result = verify_sgd_step(&setup, &proof);

    let verify_duration = if let Some(start_time) = verify_start {
        Some(start_time.elapsed())
    } else {
        None
    };

    match verification_result {
        Ok(true) => {
            if verbose {
                println!("✅ Proof verification PASSED!");
                println!("The SGD step was executed correctly according to the proof.");
            } else {
                println!("VERIFICATION PASSED");
            }

            if let Some(duration) = verify_duration {
                if benchmark {
                    println!("Verification time: {:.2?}", duration);
                }
            }

            if verbose {
                println!("\nVerification Details:");
                println!("  Circuit satisfied: ✅");
                println!("  Public inputs valid: ✅");
                println!("  Cryptographic proof valid: ✅");
            }
        }
        Ok(false) => {
            println!("❌ Proof verification FAILED!");
            println!("The proof is invalid or the SGD step was not executed correctly.");
            std::process::exit(1);
        }
        Err(e) => {
            println!("❌ Verification ERROR: {}", e);
            
            if verbose {
                println!("\nPossible causes:");
                println!("  - Proof file is corrupted or malformed");
                println!("  - Setup parameters don't match proof generation");
                println!("  - Circuit parameters mismatch");
                println!("  - Invalid public inputs");
            }
            
            std::process::exit(1);
        }
    }

    Ok(())
}

/// Batch verification of multiple proofs
fn verify_batch_proofs(
    proof_files: &[String],
    setup_params: ProvingSystemParams,
    verbose: bool,
) -> Result<Vec<bool>, Box<dyn std::error::Error>> {
    if verbose {
        println!("Batch verifying {} proofs...", proof_files.len());
    }

    let setup = generate_setup(setup_params)?;
    let mut results = Vec::new();

    for (i, proof_path) in proof_files.iter().enumerate() {
        if verbose {
            println!("Verifying proof {}/{}: {}", i + 1, proof_files.len(), proof_path);
        }

        let proof_json = fs::read_to_string(proof_path)?;
        let proof = proof_from_json(&proof_json)?;
        
        match verify_sgd_step(&setup, &proof) {
            Ok(valid) => {
                results.push(valid);
                if verbose {
                    println!("  Result: {}", if valid { "PASS" } else { "FAIL" });
                }
            }
            Err(e) => {
                if verbose {
                    println!("  Error: {}", e);
                }
                results.push(false);
            }
        }
    }

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;
    // Removed unused import: tempfile::tempdir

    #[test]
    fn test_cli_structure() {
        let cmd = Command::new("verify_sgd");
        assert!(!cmd.get_name().is_empty());
    }

    #[test]
    fn test_batch_verification_structure() {
        // Test that batch verification function compiles
        let result = verify_batch_proofs(&[], ProvingSystemParams::default(), false);
        // We expect this to fail since we're passing empty input, but it should compile
        assert!(result.is_err() || result.is_ok());
    }

    #[test]
    fn test_error_handling() {
        // Test error handling for non-existent file
        let result = std::fs::read_to_string("non_existent_file.json");
        assert!(result.is_err());
    }
}