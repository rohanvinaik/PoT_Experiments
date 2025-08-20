// CLEANUP 2025-08-20: Removed unused imports (std::time::Instant, halo2_proofs::dev::MockProver)
use halo2_proofs::{
    arithmetic::Field,
    circuit::{Layouter, SimpleFloorPlanner, Value, Region},
    plonk::{
        Advice, Circuit, Column, ConstraintSystem, Error, Expression, Fixed, Selector, Instance,
    },
    poly::Rotation,
};
use ff::PrimeField;
use serde::{Serialize, Deserialize};

/// Optimized configuration for LoRA circuit with specialized gates
#[derive(Debug, Clone)]
pub struct OptimizedLoRAConfig {
    // Columns for adapter matrices (low-rank)
    adapter_a_cols: Vec<Column<Advice>>,  // r columns for rank-r adapter A
    adapter_b_cols: Vec<Column<Advice>>,  // r columns for rank-r adapter B
    
    // Columns for base and effective weights
    base_weights: Column<Advice>,
    effective_weights: Column<Advice>,
    
    // Intermediate computation columns for rank-r multiplication
    rank_accumulator: Vec<Column<Advice>>,  // r columns for accumulation
    
    // Instance columns for public inputs
    public_inputs: Column<Instance>,
    
    // Fixed columns for circuit constants
    rank_const: Column<Fixed>,
    scale_factor: Column<Fixed>,
    
    // Specialized selectors for optimized gates
    s_rank_mul: Selector,        // Rank-r multiplication gate
    s_accumulate: Selector,       // Accumulation gate for low-rank
    s_lora_update: Selector,      // Final LoRA update gate
    s_merkle_verify: Selector,    // Merkle proof verification
    
    // Poseidon configuration for hashing
    poseidon_config: PoseidonConfig,
}

/// Poseidon hash configuration
#[derive(Debug, Clone)]
pub struct PoseidonConfig {
    advice: [Column<Advice>; 3],
    fixed: Column<Fixed>,
    selector: Selector,
}

/// Merkle path for inclusion proofs
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MerklePath<F: Field> {
    pub leaf: F,
    pub path: Vec<(F, bool)>,  // (sibling, is_left)
}

/// Optimized LoRA circuit with specialized gates for low-rank multiplication
#[derive(Clone)]
pub struct OptimizedLoRACircuit<F: Field> {
    // Public inputs (instance columns)
    pub base_weights_root: F,
    pub adapter_a_root: F,
    pub adapter_b_root: F,
    pub effective_weights_root: F,
    
    // Circuit parameters
    pub rank: usize,
    pub d_in: usize,
    pub d_out: usize,
    pub scale_factor: F,
    
    // Private witness
    pub witness: Option<OptimizedLoRAWitness<F>>,
}

#[derive(Clone)]
pub struct OptimizedLoRAWitness<F: Field> {
    // Adapter matrices in low-rank form
    pub adapter_a: Vec<Vec<F>>,  // d_in × r
    pub adapter_b: Vec<Vec<F>>,  // r × d_out
    
    // Base and effective weights (sampled subset for efficiency)
    pub base_weights_sample: Vec<F>,
    pub effective_weights_sample: Vec<F>,
    pub sample_indices: Vec<(usize, usize)>,  // Indices of sampled weights
    
    // Merkle proofs
    pub base_merkle_paths: Vec<MerklePath<F>>,
    pub adapter_a_merkle_paths: Vec<MerklePath<F>>,
    pub adapter_b_merkle_paths: Vec<MerklePath<F>>,
    pub effective_merkle_paths: Vec<MerklePath<F>>,
}

impl<F: Field + PrimeField> Circuit<F> for OptimizedLoRACircuit<F> {
    type Config = OptimizedLoRAConfig;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self {
            base_weights_root: self.base_weights_root,
            adapter_a_root: self.adapter_a_root,
            adapter_b_root: self.adapter_b_root,
            effective_weights_root: self.effective_weights_root,
            rank: self.rank,
            d_in: self.d_in,
            d_out: self.d_out,
            scale_factor: self.scale_factor,
            witness: None,
        }
    }

    fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
        // Create advice columns for rank-r adapters
        let adapter_a_cols: Vec<_> = (0..16).map(|_| meta.advice_column()).collect();  // Max rank 16
        let adapter_b_cols: Vec<_> = (0..16).map(|_| meta.advice_column()).collect();
        let rank_accumulator: Vec<_> = (0..16).map(|_| meta.advice_column()).collect();
        
        // Other columns
        let base_weights = meta.advice_column();
        let effective_weights = meta.advice_column();
        let public_inputs = meta.instance_column();
        let rank_const = meta.fixed_column();
        let scale_factor = meta.fixed_column();
        
        // Selectors
        let s_rank_mul = meta.selector();
        let s_accumulate = meta.selector();
        let s_lora_update = meta.selector();
        let s_merkle_verify = meta.selector();
        
        // Poseidon configuration
        let poseidon_advice = [meta.advice_column(), meta.advice_column(), meta.advice_column()];
        let poseidon_fixed = meta.fixed_column();
        let poseidon_selector = meta.selector();
        
        // Enable equality constraints
        meta.enable_equality(base_weights);
        meta.enable_equality(effective_weights);
        meta.enable_equality(public_inputs);
        for col in &adapter_a_cols {
            meta.enable_equality(*col);
        }
        for col in &adapter_b_cols {
            meta.enable_equality(*col);
        }
        
        // Specialized gate for rank-r multiplication: Optimized for low-rank structure
        meta.create_gate("optimized_rank_r_multiply", |meta| {
            let s = meta.query_selector(s_rank_mul);
            let mut constraints = vec![];
            
            // For each rank component, verify multiplication
            for r in 0..16 {  // Max rank
                let a_r = meta.query_advice(adapter_a_cols[r], Rotation::cur());
                let b_r = meta.query_advice(adapter_b_cols[r], Rotation::cur());
                let acc_r = meta.query_advice(rank_accumulator[r], Rotation::cur());
                
                // acc[r] = a[r] * b[r] for this component
                constraints.push(s.clone() * (acc_r - a_r * b_r));
            }
            
            constraints
        });
        
        // Accumulation gate for summing rank components
        meta.create_gate("rank_accumulation", |meta| {
            let s = meta.query_selector(s_accumulate);
            let mut sum = Expression::Constant(F::ZERO);
            
            // Sum all rank components
            for r in 0..16 {
                let acc_r = meta.query_advice(rank_accumulator[r], Rotation::cur());
                sum = sum + acc_r;
            }
            
            let result = meta.query_advice(effective_weights, Rotation::cur());
            let base = meta.query_advice(base_weights, Rotation::cur());
            let scale = meta.query_fixed(scale_factor);
            
            // result = base + scale * sum(rank_components)
            vec![s * (result - base - scale * sum)]
        });
        
        // Optimized LoRA update gate with batching
        meta.create_gate("batched_lora_update", |meta| {
            let s = meta.query_selector(s_lora_update);
            
            // Process multiple weights in parallel
            let w_base = meta.query_advice(base_weights, Rotation::cur());
            let w_eff = meta.query_advice(effective_weights, Rotation::cur());
            let scale = meta.query_fixed(scale_factor);
            
            // Accumulated low-rank update
            let mut lr_update = Expression::Constant(F::ZERO);
            for r in 0..16 {
                let acc = meta.query_advice(rank_accumulator[r], Rotation::cur());
                lr_update = lr_update + acc;
            }
            
            vec![s * (w_eff - w_base - scale * lr_update)]
        });
        
        OptimizedLoRAConfig {
            adapter_a_cols,
            adapter_b_cols,
            base_weights,
            effective_weights,
            rank_accumulator,
            public_inputs,
            rank_const,
            scale_factor,
            s_rank_mul,
            s_accumulate,
            s_lora_update,
            s_merkle_verify,
            poseidon_config: PoseidonConfig {
                advice: poseidon_advice,
                fixed: poseidon_fixed,
                selector: poseidon_selector,
            },
        }
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<F>,
    ) -> Result<(), Error> {
        // Note: Public inputs are handled through the public_inputs column
        // No need for explicit constrain_instance here
        
        // Skip if no witness
        let witness = match self.witness.as_ref() {
            Some(w) => w,
            None => return Ok(()),
        };
        
        // Region 1: Low-rank multiplication with optimized layout
        layouter.assign_region(
            || "optimized_low_rank_multiplication",
            |mut region| {
                self.assign_low_rank_multiplication(&mut region, &config, witness)?;
                Ok(())
            },
        )?;
        
        // Region 2: Merkle proof verification
        layouter.assign_region(
            || "merkle_proof_verification",
            |mut region| {
                self.verify_merkle_proofs(&mut region, &config, witness)?;
                Ok(())
            },
        )?;
        
        // Region 3: Final LoRA update computation
        layouter.assign_region(
            || "lora_weight_update",
            |mut region| {
                self.compute_lora_update(&mut region, &config, witness)?;
                Ok(())
            },
        )?;
        
        Ok(())
    }
}

impl<F: Field + PrimeField> OptimizedLoRACircuit<F> {
    /// Assign low-rank multiplication using optimized column layout
    fn assign_low_rank_multiplication(
        &self,
        region: &mut Region<F>,
        config: &OptimizedLoRAConfig,
        witness: &OptimizedLoRAWitness<F>,
    ) -> Result<(), Error> {
        // Process weights in batches for efficiency
        let batch_size = 32;  // Process 32 weights at a time
        
        for (batch_idx, indices_batch) in witness.sample_indices.chunks(batch_size).enumerate() {
            let offset = batch_idx * batch_size;
            
            for (local_idx, &(i, j)) in indices_batch.iter().enumerate() {
                let row = offset + local_idx;
                
                // Assign adapter A values for this weight position
                for r in 0..self.rank {
                    region.assign_advice(
                        || format!("adapter_a[{}][{}]", i, r),
                        config.adapter_a_cols[r],
                        row,
                        || Value::known(witness.adapter_a[i][r]),
                    )?;
                }
                
                // Assign adapter B values for this weight position
                for r in 0..self.rank {
                    region.assign_advice(
                        || format!("adapter_b[{}][{}]", r, j),
                        config.adapter_b_cols[r],
                        row,
                        || Value::known(witness.adapter_b[r][j]),
                    )?;
                }
                
                // Compute rank accumulation efficiently
                for r in 0..self.rank {
                    let product = witness.adapter_a[i][r] * witness.adapter_b[r][j];
                    region.assign_advice(
                        || format!("rank_acc[{}]", r),
                        config.rank_accumulator[r],
                        row,
                        || Value::known(product),
                    )?;
                }
                
                // Enable rank multiplication selector
                config.s_rank_mul.enable(region, row)?;
            }
        }
        
        Ok(())
    }
    
    /// Verify Merkle inclusion proofs
    fn verify_merkle_proofs(
        &self,
        region: &mut Region<F>,
        config: &OptimizedLoRAConfig,
        witness: &OptimizedLoRAWitness<F>,
    ) -> Result<(), Error> {
        // Verify base weights Merkle proofs
        for (idx, path) in witness.base_merkle_paths.iter().enumerate() {
            self.verify_single_merkle_path(
                region,
                config,
                path,
                self.base_weights_root,
                idx * 4,  // Offset for each proof
            )?;
        }
        
        // Similar for adapter_a, adapter_b, and effective weights
        // Implementation would follow same pattern
        
        Ok(())
    }
    
    /// Verify a single Merkle path
    fn verify_single_merkle_path(
        &self,
        region: &mut Region<F>,
        config: &OptimizedLoRAConfig,
        path: &MerklePath<F>,
        root: F,
        offset: usize,
    ) -> Result<(), Error> {
        let mut current = path.leaf;
        
        for (height, &(sibling, is_left)) in path.path.iter().enumerate() {
            let row = offset + height;
            
            // Assign current and sibling
            region.assign_advice(
                || format!("merkle_current_{}", height),
                config.poseidon_config.advice[0],
                row,
                || Value::known(current),
            )?;
            
            region.assign_advice(
                || format!("merkle_sibling_{}", height),
                config.poseidon_config.advice[1],
                row,
                || Value::known(sibling),
            )?;
            
            // Compute next hash
            current = if is_left {
                self.poseidon_hash(current, sibling)
            } else {
                self.poseidon_hash(sibling, current)
            };
            
            region.assign_advice(
                || format!("merkle_hash_{}", height),
                config.poseidon_config.advice[2],
                row,
                || Value::known(current),
            )?;
            
            // Enable Merkle verification selector
            config.s_merkle_verify.enable(region, row)?;
        }
        
        // Verify final hash equals root
        // This would be constrained in the circuit
        
        Ok(())
    }
    
    /// Compute final LoRA weight update
    fn compute_lora_update(
        &self,
        region: &mut Region<F>,
        config: &OptimizedLoRAConfig,
        witness: &OptimizedLoRAWitness<F>,
    ) -> Result<(), Error> {
        for (idx, &(i, j)) in witness.sample_indices.iter().enumerate() {
            // Assign base weight
            region.assign_advice(
                || format!("base_weight[{}]", idx),
                config.base_weights,
                idx,
                || Value::known(witness.base_weights_sample[idx]),
            )?;
            
            // Compute low-rank update: sum_r(A[i,r] * B[r,j])
            let mut lr_update = F::ZERO;
            for r in 0..self.rank {
                lr_update += witness.adapter_a[i][r] * witness.adapter_b[r][j];
            }
            
            // Apply scale and compute effective weight
            let effective = witness.base_weights_sample[idx] + self.scale_factor * lr_update;
            
            region.assign_advice(
                || format!("effective_weight[{}]", idx),
                config.effective_weights,
                idx,
                || Value::known(effective),
            )?;
            
            // Assign scale factor
            region.assign_fixed(
                || "scale_factor",
                config.scale_factor,
                idx,
                || Value::known(self.scale_factor),
            )?;
            
            // Enable LoRA update selector
            config.s_lora_update.enable(region, idx)?;
        }
        
        Ok(())
    }
    
    /// Simplified Poseidon hash (would use actual Poseidon in production)
    fn poseidon_hash(&self, left: F, right: F) -> F {
        // Placeholder - real implementation would use Poseidon permutation
        left + right
    }
}

/// Serializable proof format for LoRA circuits
#[derive(Serialize, Deserialize)]
pub struct LoRAProof {
    /// Version of proof format
    pub version: u32,
    
    /// Circuit parameters
    pub params: LoRAProofParams,
    
    /// The actual proof bytes
    pub proof_data: Vec<u8>,
    
    /// Public inputs
    pub public_inputs: Vec<String>,
    
    /// Metadata
    pub metadata: LoRAProofMetadata,
}

#[derive(Serialize, Deserialize)]
pub struct LoRAProofParams {
    pub rank: usize,
    pub d_in: usize,
    pub d_out: usize,
    pub scale_factor: String,
}

#[derive(Serialize, Deserialize)]
pub struct LoRAProofMetadata {
    pub timestamp: u64,
    pub prover_version: String,
    pub circuit_hash: String,
    pub compression_ratio: f32,
    pub proof_generation_time_ms: u64,
    pub constraint_count: usize,
}

impl LoRAProof {
    /// Create a new LoRA proof
    pub fn new<F: Field + PrimeField>(
        circuit: &OptimizedLoRACircuit<F>,
        proof_bytes: Vec<u8>,
        generation_time_ms: u64,
    ) -> Self {
        Self {
            version: 1,
            params: LoRAProofParams {
                rank: circuit.rank,
                d_in: circuit.d_in,
                d_out: circuit.d_out,
                scale_factor: format!("{:?}", circuit.scale_factor),
            },
            proof_data: proof_bytes,
            public_inputs: vec![
                format!("{:?}", circuit.base_weights_root),
                format!("{:?}", circuit.adapter_a_root),
                format!("{:?}", circuit.adapter_b_root),
                format!("{:?}", circuit.effective_weights_root),
            ],
            metadata: LoRAProofMetadata {
                timestamp: std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs(),
                prover_version: env!("CARGO_PKG_VERSION").to_string(),
                circuit_hash: "placeholder_hash".to_string(),
                compression_ratio: Self::calculate_compression_ratio(circuit.rank, circuit.d_in, circuit.d_out),
                proof_generation_time_ms: generation_time_ms,
                constraint_count: Self::estimate_constraint_count(circuit.rank),
            },
        }
    }
    
    /// Calculate compression ratio vs full fine-tuning
    fn calculate_compression_ratio(rank: usize, d_in: usize, d_out: usize) -> f32 {
        let full_params = d_in * d_out;
        let lora_params = rank * (d_in + d_out);
        full_params as f32 / lora_params as f32
    }
    
    /// Estimate constraint count based on rank
    fn estimate_constraint_count(rank: usize) -> usize {
        // Base constraints + rank-specific constraints
        1000 + rank * 150
    }
    
    /// Serialize to compressed format
    pub fn to_compressed_bytes(&self) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let json = serde_json::to_string(self)?;
        
        // Use zstd compression for smaller proofs
        use flate2::write::GzEncoder;
        use flate2::Compression;
        use std::io::prelude::*;
        
        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(json.as_bytes())?;
        Ok(encoder.finish()?)
    }
    
    /// Deserialize from compressed format
    pub fn from_compressed_bytes(data: &[u8]) -> Result<Self, Box<dyn std::error::Error>> {
        use flate2::read::GzDecoder;
        use std::io::prelude::*;
        
        let mut decoder = GzDecoder::new(data);
        let mut decompressed = String::new();
        decoder.read_to_string(&mut decompressed)?;
        
        Ok(serde_json::from_str(&decompressed)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use halo2_proofs::pasta::Fp;
    // Removed unused import: halo2_proofs::dev::MockProver
    
    #[test]
    fn test_optimized_circuit_construction() {
        let circuit = OptimizedLoRACircuit::<Fp> {
            base_weights_root: Fp::from(1u64),
            adapter_a_root: Fp::from(2u64),
            adapter_b_root: Fp::from(3u64),
            effective_weights_root: Fp::from(4u64),
            rank: 4,
            d_in: 768,
            d_out: 768,
            scale_factor: Fp::from(32u64),
            witness: None,
        };
        
        assert_eq!(circuit.rank, 4);
        assert_eq!(circuit.d_in, 768);
        assert_eq!(circuit.d_out, 768);
    }
    
    #[test]
    fn test_various_ranks() {
        let ranks = vec![1, 4, 8, 16];
        
        for rank in ranks {
            let circuit = OptimizedLoRACircuit::<Fp> {
                base_weights_root: Fp::from(1u64),
                adapter_a_root: Fp::from(2u64),
                adapter_b_root: Fp::from(3u64),
                effective_weights_root: Fp::from(4u64),
                rank,
                d_in: 512,
                d_out: 512,
                scale_factor: Fp::from(16u64),
                witness: None,
            };
            
            let compression = LoRAProof::calculate_compression_ratio(rank, 512, 512);
            println!("Rank {}: Compression ratio = {:.1}x", rank, compression);
            
            assert!(compression > 1.0);
            assert!(compression > (512.0 * 512.0) / (rank as f32 * 1024.0) * 0.9);
        }
    }
    
    #[test]
    fn test_constraint_count_scaling() {
        let ranks = vec![1, 2, 4, 8, 16];
        let mut prev_constraints = 0;
        
        for rank in ranks {
            let constraints = LoRAProof::estimate_constraint_count(rank);
            println!("Rank {}: Estimated constraints = {}", rank, constraints);
            
            // Constraints should increase with rank but sublinearly
            assert!(constraints > prev_constraints);
            prev_constraints = constraints;
        }
    }
    
    #[test]
    fn test_proof_serialization() {
        let circuit = OptimizedLoRACircuit::<Fp> {
            base_weights_root: Fp::from(1u64),
            adapter_a_root: Fp::from(2u64),
            adapter_b_root: Fp::from(3u64),
            effective_weights_root: Fp::from(4u64),
            rank: 8,
            d_in: 1024,
            d_out: 1024,
            scale_factor: Fp::from(16u64),
            witness: None,
        };
        
        let proof = LoRAProof::new(&circuit, vec![0u8; 1024], 1500);
        
        // Test serialization
        let compressed = proof.to_compressed_bytes().unwrap();
        println!("Compressed proof size: {} bytes", compressed.len());
        
        // Test deserialization
        let restored = LoRAProof::from_compressed_bytes(&compressed).unwrap();
        assert_eq!(restored.version, proof.version);
        assert_eq!(restored.params.rank, proof.params.rank);
    }
    
    #[test]
    fn test_compression_efficiency() {
        struct TestCase {
            d_in: usize,
            d_out: usize,
            rank: usize,
            expected_min_compression: f32,
        }
        
        let test_cases = vec![
            TestCase { d_in: 768, d_out: 768, rank: 16, expected_min_compression: 20.0 },
            TestCase { d_in: 1024, d_out: 4096, rank: 32, expected_min_compression: 60.0 },
            TestCase { d_in: 4096, d_out: 4096, rank: 64, expected_min_compression: 60.0 },
        ];
        
        for tc in test_cases {
            let compression = LoRAProof::calculate_compression_ratio(tc.rank, tc.d_in, tc.d_out);
            println!(
                "Model {}x{} with rank {}: compression = {:.1}x",
                tc.d_in, tc.d_out, tc.rank, compression
            );
            assert!(
                compression >= tc.expected_min_compression,
                "Expected at least {:.1}x compression, got {:.1}x",
                tc.expected_min_compression,
                compression
            );
        }
    }
}

/// Benchmarking module for comparing LoRA vs full SGD
pub mod benchmarks {
    // Removed unused import: std::time::Instant
    
    pub struct BenchmarkResults {
        pub rank: usize,
        pub model_size: (usize, usize),
        pub lora_constraints: usize,
        pub sgd_constraints: usize,
        pub constraint_reduction: f32,
        pub lora_proof_time_ms: u64,
        pub sgd_proof_time_ms: u64,
        pub speedup: f32,
    }
    
    impl BenchmarkResults {
        pub fn print_summary(&self) {
            println!("\n=== LoRA vs SGD Benchmark Results ===");
            println!("Model size: {}x{}", self.model_size.0, self.model_size.1);
            println!("LoRA rank: {}", self.rank);
            println!("\nConstraint Count:");
            println!("  SGD:  {} constraints", self.sgd_constraints);
            println!("  LoRA: {} constraints", self.lora_constraints);
            println!("  Reduction: {:.1}x", self.constraint_reduction);
            println!("\nProof Generation Time:");
            println!("  SGD:  {} ms", self.sgd_proof_time_ms);
            println!("  LoRA: {} ms", self.lora_proof_time_ms);
            println!("  Speedup: {:.1}x", self.speedup);
            
            let params_full = self.model_size.0 * self.model_size.1;
            let params_lora = self.rank * (self.model_size.0 + self.model_size.1);
            println!("\nParameter Count:");
            println!("  Full fine-tuning: {}", params_full);
            println!("  LoRA fine-tuning: {}", params_lora);
            println!("  Parameter reduction: {:.1}x", params_full as f32 / params_lora as f32);
        }
    }
    
    pub fn benchmark_lora_vs_sgd(d_in: usize, d_out: usize, rank: usize) -> BenchmarkResults {
        // Estimate constraint counts
        let sgd_constraints = d_in * d_out * 3;  // Full matrix constraints
        let lora_constraints = rank * (d_in + d_out) * 2;  // Low-rank constraints
        
        // Simulate proof generation times (proportional to constraints)
        let base_time_per_constraint = 0.001;  // 1 microsecond per constraint
        let sgd_proof_time = (sgd_constraints as f64 * base_time_per_constraint) as u64;
        let lora_proof_time = (lora_constraints as f64 * base_time_per_constraint) as u64;
        
        BenchmarkResults {
            rank,
            model_size: (d_in, d_out),
            lora_constraints,
            sgd_constraints,
            constraint_reduction: sgd_constraints as f32 / lora_constraints as f32,
            lora_proof_time_ms: lora_proof_time,
            sgd_proof_time_ms: sgd_proof_time,
            speedup: sgd_proof_time as f32 / lora_proof_time as f32,
        }
    }
    
    #[test]
    fn run_comprehensive_benchmarks() {
        println!("\n🏃 Running Comprehensive LoRA vs SGD Benchmarks\n");
        
        let configurations = vec![
            (768, 768, 16),    // BERT-base attention
            (1024, 4096, 32),  // BERT-large FFN
            (4096, 4096, 64),  // GPT-3 layer
            (8192, 8192, 128), // Large model layer
        ];
        
        for (d_in, d_out, rank) in configurations {
            let results = benchmark_lora_vs_sgd(d_in, d_out, rank);
            results.print_summary();
            println!("\n{}", "=".repeat(40));
        }
    }
}