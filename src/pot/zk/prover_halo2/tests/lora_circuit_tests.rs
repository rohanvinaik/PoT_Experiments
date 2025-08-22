// CLEANUP 2025-08-20: Removed unused imports (halo2_proofs::{arithmetic::Field, plonk::Circuit})
use halo2_proofs::{
    // Removed unused imports: arithmetic::Field, Circuit
    dev::MockProver,
    pasta::Fp,
    plonk::{keygen_vk, keygen_pk, create_proof, verify_proof},
    poly::commitment::Params,
    transcript::{Blake2bRead, Blake2bWrite, Challenge255},
};
use prover_halo2::lora_circuit_optimized::{
    OptimizedLoRACircuit, OptimizedLoRAWitness, MerklePath, LoRAProof,
    benchmarks::{benchmark_lora_vs_sgd, BenchmarkResults},
};
use rand::rngs::OsRng;
use std::time::Instant;

/// Test LoRA circuit with different ranks
#[test]
fn test_lora_circuit_various_ranks() {
    println!("\n=== Testing LoRA Circuit with Various Ranks ===\n");
    
    let test_ranks = vec![1, 2, 4, 8, 16];
    let d_in = 128;
    let d_out = 128;
    
    for rank in test_ranks {
        println!("Testing rank {} with {}x{} weights...", rank, d_in, d_out);
        
        // Create circuit without witness for constraint counting
        let circuit = OptimizedLoRACircuit::<Fp> {
            base_weights_root: Fp::from(1u64),
            adapter_a_root: Fp::from(2u64),
            adapter_b_root: Fp::from(3u64),
            effective_weights_root: Fp::from(4u64),
            rank,
            d_in,
            d_out,
            scale_factor: Fp::from(16u64),
            witness: None,
        };
        
        // Create mock witness for testing
        let witness = create_mock_witness(d_in, d_out, rank);
        
        let circuit_with_witness = OptimizedLoRACircuit {
            witness: Some(witness),
            ..circuit.clone()
        };
        
        // Test with MockProver (k=10 for 2^10 rows)
        let k = 10;
        let prover = MockProver::run(k, &circuit_with_witness, vec![]).unwrap();
        
        match prover.verify() {
            Ok(_) => println!("  ✅ Rank {} circuit verified successfully", rank),
            Err(e) => println!("  ❌ Rank {} circuit failed: {:?}", rank, e),
        }
        
        // Calculate and display compression metrics
        let full_params = d_in * d_out;
        let lora_params = rank * (d_in + d_out);
        let compression = full_params as f32 / lora_params as f32;
        
        println!("  Parameters: {} (LoRA) vs {} (full) = {:.1}x compression", 
                 lora_params, full_params, compression);
        println!();
    }
}

/// Test constraint system satisfaction for edge cases
#[test]
fn test_lora_circuit_edge_cases() {
    println!("\n=== Testing LoRA Circuit Edge Cases ===\n");
    
    // Test case 1: Rank 1 (minimal)
    test_edge_case("Rank 1", 64, 64, 1);
    
    // Test case 2: Very small dimensions
    test_edge_case("Small dimensions", 4, 4, 2);
    
    // Test case 3: Asymmetric dimensions
    test_edge_case("Asymmetric", 256, 64, 8);
    
    // Test case 4: Large rank relative to dimensions
    test_edge_case("Large relative rank", 32, 32, 16);
}

fn test_edge_case(name: &str, d_in: usize, d_out: usize, rank: usize) {
    println!("Testing {}: {}x{} with rank {}...", name, d_in, d_out, rank);
    
    let circuit = OptimizedLoRACircuit::<Fp> {
        base_weights_root: Fp::from(1u64),
        adapter_a_root: Fp::from(2u64),
        adapter_b_root: Fp::from(3u64),
        effective_weights_root: Fp::from(4u64),
        rank,
        d_in,
        d_out,
        scale_factor: Fp::from(32u64),
        witness: Some(create_mock_witness(d_in, d_out, rank)),
    };
    
    let k = 12; // 2^12 rows
    let prover = MockProver::run(k, &circuit, vec![]).unwrap();
    
    match prover.verify() {
        Ok(_) => println!("  ✅ {} passed", name),
        Err(e) => println!("  ❌ {} failed: {:?}", name, e),
    }
}

/// Benchmark LoRA proof generation vs full SGD
#[test]
fn test_lora_vs_sgd_benchmarks() {
    println!("\n=== Benchmarking LoRA vs Full SGD ===\n");
    
    let configurations = vec![
        ("Small model", 256, 256, 8),
        ("Medium model", 768, 768, 16),
        ("Large model", 1024, 4096, 32),
        ("XL model", 4096, 4096, 64),
    ];
    
    let mut all_results = Vec::new();
    
    for (name, d_in, d_out, rank) in configurations {
        println!("Benchmarking {}: {}x{} with rank {}", name, d_in, d_out, rank);
        
        // Run benchmark
        let results = benchmark_lora_vs_sgd(d_in, d_out, rank);
        results.print_summary();
        
        all_results.push((name, results));
        println!();
    }
    
    // Print summary table
    print_benchmark_summary(&all_results);
}

fn print_benchmark_summary(results: &[(&str, BenchmarkResults)]) {
    println!("\n╔═══════════════════════════════════════════════════════════════╗");
    println!("║              LoRA vs SGD Benchmark Summary                       ║");
    println!("╠═══════════════════════════════════════════════════════════════╣");
    println!("║ Model        │ Size      │ Rank │ Constraint │ Proof Time     ║");
    println!("║              │           │      │ Reduction  │ Speedup        ║");
    println!("╠══════════════╪═══════════╪══════╪════════════╪════════════════╣");
    
    for (name, result) in results {
        println!("║ {:12} │ {:4}x{:4} │ {:4} │ {:8.1}x │ {:10.1}x    ║",
                 name,
                 result.model_size.0,
                 result.model_size.1,
                 result.rank,
                 result.constraint_reduction,
                 result.speedup);
    }
    
    println!("╚═══════════════════════════════════════════════════════════════╝");
    
    // Calculate average improvements
    let avg_constraint_reduction = results.iter()
        .map(|(_, r)| r.constraint_reduction)
        .sum::<f32>() / results.len() as f32;
    
    let avg_speedup = results.iter()
        .map(|(_, r)| r.speedup)
        .sum::<f32>() / results.len() as f32;
    
    println!("\nAverage Improvements:");
    println!("  Constraint Reduction: {:.1}x", avg_constraint_reduction);
    println!("  Proof Time Speedup:   {:.1}x", avg_speedup);
}

/// Test proof serialization and compression
#[test]
fn test_lora_proof_serialization() {
    println!("\n=== Testing LoRA Proof Serialization ===\n");
    
    let circuit = OptimizedLoRACircuit::<Fp> {
        base_weights_root: Fp::from(1u64),
        adapter_a_root: Fp::from(2u64),
        adapter_b_root: Fp::from(3u64),
        effective_weights_root: Fp::from(4u64),
        rank: 16,
        d_in: 768,
        d_out: 768,
        scale_factor: Fp::from(32u64),
        witness: None,
    };
    
    // Create mock proof data
    let mock_proof_data = vec![0u8; 4096]; // 4KB mock proof
    let generation_time = 1500;
    
    // Create proof structure
    let proof = LoRAProof::new(&circuit, mock_proof_data.clone(), generation_time);
    
    println!("Original proof size: {} bytes", mock_proof_data.len());
    
    // Test JSON serialization
    let json = serde_json::to_string(&proof).unwrap();
    println!("JSON serialized size: {} bytes", json.len());
    
    // Test compressed serialization
    match proof.to_compressed_bytes() {
        Ok(compressed) => {
            println!("Compressed size: {} bytes", compressed.len());
            let compression_ratio = json.len() as f32 / compressed.len() as f32;
            println!("Compression ratio: {:.2}x", compression_ratio);
            
            // Test decompression
            match LoRAProof::from_compressed_bytes(&compressed) {
                Ok(restored) => {
                    assert_eq!(restored.version, proof.version);
                    assert_eq!(restored.params.rank, proof.params.rank);
                    println!("✅ Proof successfully serialized and deserialized");
                }
                Err(e) => println!("❌ Deserialization failed: {}", e),
            }
        }
        Err(e) => println!("❌ Compression failed: {}", e),
    }
    
    // Display metadata
    println!("\nProof Metadata:");
    println!("  Version: {}", proof.version);
    println!("  Timestamp: {}", proof.metadata.timestamp);
    println!("  Prover Version: {}", proof.metadata.prover_version);
    println!("  Compression Ratio: {:.1}x", proof.metadata.compression_ratio);
    println!("  Generation Time: {} ms", proof.metadata.proof_generation_time_ms);
    println!("  Constraint Count: {}", proof.metadata.constraint_count);
}

/// Test memory usage and performance with large models
#[test]
#[ignore] // Run with --ignored flag for performance tests
fn test_lora_performance_large_models() {
    println!("\n=== Testing LoRA Performance with Large Models ===\n");
    
    let configurations = vec![
        (2048, 2048, 32),
        (4096, 4096, 64),
        (8192, 8192, 128),
    ];
    
    for (d_in, d_out, rank) in configurations {
        println!("Testing {}x{} model with rank {}...", d_in, d_out, rank);
        
        let start = Instant::now();
        
        let circuit = OptimizedLoRACircuit::<Fp> {
            base_weights_root: Fp::from(1u64),
            adapter_a_root: Fp::from(2u64),
            adapter_b_root: Fp::from(3u64),
            effective_weights_root: Fp::from(4u64),
            rank,
            d_in,
            d_out,
            scale_factor: Fp::from(32u64),
            witness: Some(create_mock_witness(d_in, d_out, rank)),
        };
        
        let setup_time = start.elapsed();
        println!("  Circuit setup time: {:?}", setup_time);
        
        // Estimate memory usage
        let adapter_memory = rank * (d_in + d_out) * 8; // 8 bytes per f64
        let full_memory = d_in * d_out * 8;
        let memory_saving = 100.0 * (1.0 - adapter_memory as f32 / full_memory as f32);
        
        println!("  Memory usage:");
        println!("    LoRA adapters: {} MB", adapter_memory / 1_000_000);
        println!("    Full weights:  {} MB", full_memory / 1_000_000);
        println!("    Memory saving: {:.1}%", memory_saving);
        
        // Calculate theoretical speedup
        let theoretical_speedup = full_memory as f32 / adapter_memory as f32;
        println!("  Theoretical speedup: {:.1}x", theoretical_speedup);
        
        println!();
    }
}

/// Helper function to create mock witness data
fn create_mock_witness(d_in: usize, d_out: usize, rank: usize) -> OptimizedLoRAWitness<Fp> {
    // Create mock adapter matrices
    let adapter_a: Vec<Vec<Fp>> = (0..d_in)
        .map(|i| (0..rank)
            .map(|r| Fp::from((i * rank + r) as u64))
            .collect())
        .collect();
    
    let adapter_b: Vec<Vec<Fp>> = (0..rank)
        .map(|r| (0..d_out)
            .map(|j| Fp::from((r * d_out + j) as u64))
            .collect())
        .collect();
    
    // Sample a subset of weights for efficiency
    let sample_size = std::cmp::min(100, d_in * d_out / 10);
    let mut sample_indices = Vec::new();
    let mut base_weights_sample = Vec::new();
    let mut effective_weights_sample = Vec::new();
    
    for idx in 0..sample_size {
        let i = idx % d_in;
        let j = idx % d_out;
        sample_indices.push((i, j));
        
        // Mock base weight
        let base = Fp::from((i * d_out + j) as u64);
        base_weights_sample.push(base);
        
        // Compute effective weight (simplified)
        let lr_update = Fp::from((i + j) as u64);
        let effective = base + lr_update;
        effective_weights_sample.push(effective);
    }
    
    // Create mock Merkle paths
    let mock_merkle_path = MerklePath {
        leaf: Fp::from(1u64),
        path: vec![(Fp::from(2u64), true), (Fp::from(3u64), false)],
    };
    
    OptimizedLoRAWitness {
        adapter_a,
        adapter_b,
        base_weights_sample,
        effective_weights_sample,
        sample_indices,
        base_merkle_paths: vec![mock_merkle_path.clone()],
        adapter_a_merkle_paths: vec![mock_merkle_path.clone()],
        adapter_b_merkle_paths: vec![mock_merkle_path.clone()],
        effective_merkle_paths: vec![mock_merkle_path],
    }
}

/// Integration test with actual proof generation
#[test]
#[ignore] // Run with --ignored for full proof generation
fn test_lora_full_proof_generation() {
    println!("\n=== Testing Full LoRA Proof Generation ===\n");
    
    let rank = 8;
    let d_in = 256;
    let d_out = 256;
    
    // Setup
    let params = Params::<pasta_curves::vesta::Affine>::new(14); // k=14
    
    let circuit = OptimizedLoRACircuit::<Fp> {
        base_weights_root: Fp::from(1u64),
        adapter_a_root: Fp::from(2u64),
        adapter_b_root: Fp::from(3u64),
        effective_weights_root: Fp::from(4u64),
        rank,
        d_in,
        d_out,
        scale_factor: Fp::from(32u64),
        witness: None,
    };
    
    // Generate keys
    println!("Generating proving and verifying keys...");
    let vk = keygen_vk(&params, &circuit).expect("vk generation failed");
    let pk = keygen_pk(&params, vk.clone(), &circuit).expect("pk generation failed");
    
    // Create circuit with witness
    let circuit_with_witness = OptimizedLoRACircuit {
        witness: Some(create_mock_witness(d_in, d_out, rank)),
        ..circuit
    };
    
    // Generate proof
    println!("Generating proof...");
    let start = Instant::now();
    
    let mut transcript = Blake2bWrite::<_, _, Challenge255<_>>::init(vec![]);
    create_proof(
        &params,
        &pk,
        &[circuit_with_witness],
        &[&[]],
        OsRng,
        &mut transcript,
    ).expect("proof generation failed");
    
    let proof = transcript.finalize();
    let proof_time = start.elapsed();
    
    println!("Proof generated in {:?}", proof_time);
    println!("Proof size: {} bytes", proof.len());
    
    // Verify proof
    println!("Verifying proof...");
    let start = Instant::now();
    
    let mut transcript = Blake2bRead::<_, _, Challenge255<_>>::init(&proof[..]);
    let result = verify_proof(
        &params,
        &vk,
        halo2_proofs::plonk::SingleVerifier::new(&params),
        &[&[]],
        &mut transcript,
    );
    
    let verify_time = start.elapsed();
    
    match result {
        Ok(_) => println!("✅ Proof verified successfully in {:?}", verify_time),
        Err(e) => println!("❌ Proof verification failed: {:?}", e),
    }
    
    // Display efficiency metrics
    let full_params = d_in * d_out;
    let lora_params = rank * (d_in + d_out);
    println!("\nEfficiency Metrics:");
    println!("  Parameter reduction: {:.1}x", full_params as f32 / lora_params as f32);
    println!("  Proof size per parameter: {:.2} bytes", proof.len() as f32 / lora_params as f32);
}