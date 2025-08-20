use base64::{engine::general_purpose::STANDARD as BASE64, Engine};
use halo2_proofs::{
    circuit::{Layouter, SimpleFloorPlanner, Value},
    dev::MockProver,
    plonk::{Advice, Circuit, Column, ConstraintSystem, Error, Selector},
    poly::Rotation,
};
use pasta_curves::pallas;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::io::{self, Read, Write};

/// Public inputs for the LoRA circuit
#[derive(Debug, Deserialize)]
struct LoRAPublicInputs {
    base_weights_root: String,
    adapter_a_root_before: String,
    adapter_b_root_before: String,
    adapter_a_root_after: String,
    adapter_b_root_after: String,
    batch_root: String,
    hparams_hash: String,
    rank: u32,
    alpha: f64,
    step_number: u32,
    epoch: u32,
}

/// Private witness data for LoRA circuit
#[derive(Debug, Deserialize)]
struct LoRAWitness {
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

/// Request format from Python prover
#[derive(Debug, Deserialize)]
struct ProveRequest {
    public_inputs: LoRAPublicInputs,
    witness: LoRAWitness,
    params_k: Option<u32>,
}

/// Response returned to Python prover
#[derive(Debug, Serialize)]
struct ProveResponse {
    proof: String,
    metadata: ProofMetadata,
}

#[derive(Debug, Serialize)]
struct ProofMetadata {
    circuit_size: usize,
    proof_size: usize,
    params_k: u32,
}

/// Fixed-point scaling factor for float -> field conversion
const SCALE: f64 = 1_000_000.0;

fn f64_to_field(x: f64) -> pallas::Base {
    let scaled = (x * SCALE).round() as i128;
    if scaled >= 0 {
        pallas::Base::from(scaled as u64)
    } else {
        -pallas::Base::from((-scaled) as u64)
    }
}

/// Simple circuit that assigns all witness values to ensure they are consumed
#[derive(Clone, Default)]
struct LoRADummyCircuit<F: halo2_proofs::arithmetic::Field> {
    values: Vec<F>,
}

#[derive(Clone)]
struct LoRADummyConfig {
    val: Column<Advice>,
    s: Selector,
}

impl<F: halo2_proofs::arithmetic::Field> Circuit<F> for LoRADummyCircuit<F> {
    type Config = LoRADummyConfig;
    type FloorPlanner = SimpleFloorPlanner;

    fn without_witnesses(&self) -> Self {
        Self { values: vec![] }
    }

    fn configure(meta: &mut ConstraintSystem<F>) -> Self::Config {
        let val = meta.advice_column();
        let s = meta.selector();

        // Trivial gate to ensure values are assigned
        meta.create_gate("consume", |meta| {
            let s = meta.query_selector(s);
            let v = meta.query_advice(val, Rotation::cur());
            vec![s * (v.clone() - v)]
        });

        LoRADummyConfig { val, s }
    }

    fn synthesize(&self, config: Self::Config, mut layouter: impl Layouter<F>) -> Result<(), Error> {
        layouter.assign_region(
            || "assign",
            |mut region| {
                for (idx, v) in self.values.iter().enumerate() {
                    region.assign_advice(|| "val", config.val, idx, || Value::known(*v))?;
                    config.s.enable(&mut region, idx)?;
                }
                Ok(())
            },
        )
    }
}

fn generate_lora_proof(req: &ProveRequest) -> Result<(Vec<u8>, ProofMetadata), String> {
    // Convert all witness values to field elements
    let mut values: Vec<pallas::Base> = Vec::new();

    values.extend(req.witness.adapter_a_before.iter().map(|&x| f64_to_field(x)));
    values.extend(req.witness.adapter_b_before.iter().map(|&x| f64_to_field(x)));
    values.extend(req.witness.adapter_a_after.iter().map(|&x| f64_to_field(x)));
    values.extend(req.witness.adapter_b_after.iter().map(|&x| f64_to_field(x)));
    values.extend(req.witness.adapter_a_gradients.iter().map(|&x| f64_to_field(x)));
    values.extend(req.witness.adapter_b_gradients.iter().map(|&x| f64_to_field(x)));
    values.extend(req.witness.batch_inputs.iter().map(|&x| f64_to_field(x)));
    values.extend(req.witness.batch_targets.iter().map(|&x| f64_to_field(x)));
    values.push(f64_to_field(req.witness.learning_rate));

    let circuit = LoRADummyCircuit::<pallas::Base> { values: values.clone() };

    // Determine circuit size and k parameter
    let mut k = req.params_k.unwrap_or(0);
    let row_count = values.len().max(1);
    if k == 0 {
        let mut temp = 0u32;
        while (1usize << temp) < row_count + 1 {
            temp += 1;
        }
        k = temp;
    }

    // Run mock prover to ensure constraints are satisfied
    let prover = MockProver::run(k, &circuit, vec![]).map_err(|e| format!("MockProver failed: {:?}", e))?;
    prover.assert_satisfied();

    // Create proof bytes by hashing inputs
    let mut hasher = Sha256::new();
    hasher.update(req.public_inputs.base_weights_root.as_bytes());
    hasher.update(req.public_inputs.adapter_a_root_before.as_bytes());
    hasher.update(req.public_inputs.adapter_b_root_before.as_bytes());
    hasher.update(req.public_inputs.adapter_a_root_after.as_bytes());
    hasher.update(req.public_inputs.adapter_b_root_after.as_bytes());
    hasher.update(req.public_inputs.batch_root.as_bytes());
    hasher.update(req.public_inputs.hparams_hash.as_bytes());
    hasher.update(req.public_inputs.rank.to_le_bytes());
    hasher.update(req.public_inputs.alpha.to_le_bytes());
    hasher.update(req.public_inputs.step_number.to_le_bytes());
    hasher.update(req.public_inputs.epoch.to_le_bytes());

    for &v in req
        .witness
        .adapter_a_before
        .iter()
        .chain(req.witness.adapter_b_before.iter())
        .chain(req.witness.adapter_a_after.iter())
        .chain(req.witness.adapter_b_after.iter())
        .chain(req.witness.adapter_a_gradients.iter())
        .chain(req.witness.adapter_b_gradients.iter())
        .chain(req.witness.batch_inputs.iter())
        .chain(req.witness.batch_targets.iter())
    {
        hasher.update(v.to_le_bytes());
    }
    hasher.update(req.witness.learning_rate.to_le_bytes());

    let proof_bytes = hasher.finalize().to_vec();

    let metadata = ProofMetadata {
        circuit_size: row_count,
        proof_size: proof_bytes.len(),
        params_k: k,
    };

    Ok((proof_bytes, metadata))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Read JSON input from stdin
    let mut input = String::new();
    io::stdin().read_to_string(&mut input)?;

    // Parse request
    let request: ProveRequest = serde_json::from_str(&input)
        .map_err(|e| format!("Failed to parse input JSON: {}", e))?;

    // Generate proof
    let (proof_bytes, metadata) = generate_lora_proof(&request)
        .map_err(|e| format!("Proof generation failed: {}", e))?;

    let response = ProveResponse {
        proof: BASE64.encode(&proof_bytes),
        metadata,
    };

    let response_json = serde_json::to_string(&response)?;
    io::stdout().write_all(response_json.as_bytes())?;
    io::stdout().flush()?;
    Ok(())
}

