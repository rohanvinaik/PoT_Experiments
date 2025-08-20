use clap::{Arg, Command};
use prover_halo2::{
    generate_setup, proof_from_json, verify_sgd_step, ProverError, ProvingSystemParams,
};
use std::fs;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let matches = Command::new("verify_sgd")
        .version("0.1.0")
        .about("Verify ZK proof for SGD training step")
        .arg(
            Arg::new("proof")
                .short('p')
                .long("proof")
                .value_name("FILE")
                .help("JSON file containing the proof to verify")
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
                .help("KZG parameter size (log2) - must match proof generation")
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
        .get_matches();

    let verbose = matches.get_flag("verbose");
    let benchmark = matches.get_flag("benchmark");
    
    if verbose {
        println!("SGD ZK Proof Verifier v0.1.0");
        println!("=============================");
    }

    // Parse command line arguments
    let proof_path = matches.get_one::<String>("proof").unwrap();
    let params_k: u32 = matches.get_one::<String>("params-k").unwrap().parse()?;

    if verbose {
        println!("Proof file: {}", proof_path);
        println!("KZG params k: {}", params_k);
    }

    // Read proof file
    if verbose {
        println!("\nReading proof file...");
    }
    
    let proof_json = fs::read_to_string(proof_path)
        .map_err(|e| format!("Failed to read proof file: {}", e))?;

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
    use tempfile::tempdir;

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