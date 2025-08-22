# Proof-of-Training (PoT) — Behavioral Model Identity Verification

> **Scope of this repo:** This is the **Verifier (post-training)** half of a unified PoT framework. It decides whether two models are **SAME** or **DIFFERENT** using a **sequential, variance-adaptive behavioral test** with **pre-committed (HMAC) challenges** and an **auditable transcript**. Optional ZK artifacts attest the verification process.  
> **Out of scope here:** The **Prover (training-time provenance)** half that issues a model’s “birth certificate” (hashes, signed checkpoints, IO evolution) is referenced but not implemented in this repository.

---

## Why this exists

Modern models are often opaque or API-only. We still need to answer:

- “Is the model behind this endpoint actually the one it claims to be?”  
- “Was a base model quietly **fine-tuned**, **compressed/distilled**, or **substituted**?”  
- “Can we verify identity **without** seeing the weights?”

This verifier provides an **operational** answer: run **dozens** of pre-committed challenges, compute a **confidence-controlled decision** (SAME/DIFFERENT), and export an **audit bundle** anyone can re-check later.

---

## What this repo provides (today)

- **Statistical verifier** with **Empirical-Bernstein (EB)** bounds and **early stopping**  
- **Formal decision rules** for SAME / DIFFERENT / UNDECIDED (see below)  
- **Challenge generator** (HMAC-SHA256) to pre-commit the prompt set  
- **Runners & scripts** for local models or API endpoints  
- **Evidence bundle tooling** (logs, configs, transcripts, seeds)  
- **Prototype ZK proofs** (Halo2) that attest *the verifier ran on the published transcript*  
- **Sharded verification** to validate **34B-class** models on **commodity RAM** by loading/releasing model slices sequentially

> **Note:** ZK artifacts here attest the **process/transcript**. For remote APIs, binding the transcript to **a specific model identity** additionally requires a **TEE attestation** or a **vendor weight commitment** (see **Security model & guarantees**).

---

## How it works (in short)

1) **Pre-commit challenges**  
   Generate challenge seeds via **HMAC-SHA256(key, run_id || i)** → `seed_i`. This creates a large, deterministic challenge space and prevents cherry-picking.

2) **Stream responses and update a statistic**  
   For each challenge, collect both models’ responses and update a running **effect-size** estimate with an **EB confidence interval**.

3) **Stop early when the math is decisive**  
   After each observation, check **explicit decision rules**:
   - **SAME** if the entire CI lies within a small tolerance around zero and the CI is sufficiently narrow.  
   - **DIFFERENT** if the effect size exceeds a magnitude threshold and the estimate is numerically stable.  
   Otherwise keep sampling—bounded by mode-specific `n_min`/`n_max`.

4) **Emit an audit bundle**  
   Write seeds, challenge IDs, normalized outputs, decisions, thresholds, and environment metadata. Optionally, produce a small **ZK proof** that the verifier computed the published decision from the published transcript.

---

## Decision rules (exact)

Let Δ be the mean effect size and CI = [L, U] with half-width `h`.

- **SAME** if `CI ⊆ [−γ, +γ]` **and** `h ≤ η·γ`  
- **DIFFERENT** if `|Δ| ≥ δ*` **and** `RME ≤ ε_diff`  
- **UNDECIDED** otherwise

**Presets**

| Mode      | α     | γ    | η   | δ*   | ε_diff | n_range   |
|-----------|-------|------|-----|------|--------|-----------|
| QUICK     | .025  | .15  | .50 | 0.8  | .15    | [10, 120] |
| AUDIT     | .01   | .10  | .50 | 1.0  | .10    | [30, 400] |
| EXTENDED  | .001  | .08  | .40 | 1.1  | .08    | [50, 800] |

> The verifier is **anytime**: it can stop as soon as the decision criteria are met, which is why it typically needs **dozens** of queries—not thousands.

---

## Access modes & binding to identity

| Mode              | Weights access | Typical use       | Binding between transcript and model identity |
|-------------------|----------------|-------------------|----------------------------------------------|
| **Local-weights** | Yes (HF/local) | Research & QA     | Bind by **hashing weights** (e.g., SHA-256 of safetensors) and including the hash in the evidence bundle |
| **API black-box** | No             | Vendor/API audits | Bind by **TEE attestation** (e.g., SGX/SEV/Nitro) or a **vendor-signed commitment**. The transcript is then endorsed by the attested enclave or vendor |

**What ZK proves here**  
Prototype Halo2 circuits prove the verifier consumed transcript `T` and produced decision `D` under code hash `H(code)`. ZK **does not, by itself**, prove that a *particular remote model* produced `T`; that link comes from **TEE or vendor commitments**.

---

## What you can use it for

- **Identity checks**: “Is this deployment the same as my baseline?”  
- **Fine-tuning detection**: flag instruction-tuned or domain-tuned variants  
- **Distillation/compression detection**: detect degraded/altered behavior  
- **Architecture class detection**: catch cross-family substitutions  
- **Size-fraud detection**: smaller models masquerading as larger ones

---

## Example runs (live data)

> Times/queries below are from actual validation runs in this repo. The table automatically updates with each successful pipeline execution. Use the scripts in **Reproduce** to add your own results.

| Pair                             | Mode          | Decision   | Queries | Total Time | Per-Query | Notes                    |
|----------------------------------|---------------|------------|---------|------------|-----------|--------------------------|
| **DistilGPT-2** vs **DistilGPT-2** | Quick-gate | SAME | 12 | ~6.3 s | ~0.5 s | Self-consistency |
| **EleutherAI Pythia-70m** vs **EleutherAI Pythia-160m** | Quick-gate | UNDECIDED | 120 | ~87 s | ~0.7 s | Model comparison |
| **GPT-2** vs **DistilGPT-2** | Quick-gate | UNDECIDED | 84.0 (avg of 3) | ~47.3 s (avg of 3) | ~0.6 s (avg of 3) | Distillation (3 runs) |
| **GPT-2** vs **GPT-2** | Quick-gate | SAME | 12.0 (avg of 2) | ~13.5 s (avg of 2) | ~1.1 s (avg of 2) | Self-consistency (2 runs) |
| **Model A** vs **Model B** | Local-weights | SAME | 58.3 (avg of 7) | ~37.4 s (avg of 7) | ~0.6 s (avg of 7) | Model comparison (7 runs) |

<!-- Table auto-updated: 2025-08-22 06:56:40 -->
**Massive-model feasibility (sharded)**  
Verified **~206 GB** of model weights on a **64 GB** host via **sequential shard load → verify → release** with peak resident memory ≈ **~50%** and minutes-scale wall time.

> For audit-grade claims, publish **RSS**, **(maj/min) page-faults**, **disk read throughput**, and **per-query times** (cold vs warm cache) from your runs.

---

## Reporting standards (what to include with your results)

- **Confidence level** (`α`) and **mode** (QUICK/AUDIT/EXTENDED)  
- **Thresholds** (`δ*`, `γ`, `η`, `ε_diff`) actually used  
- **Counts with CIs** for FAR/FRR (e.g., `0/200, 95% CI [0, 1.8%]`)  
- **n_min/n_max** and **final n** (queries to decision)  
- **Access mode** (Local vs API) and **binding method** (hash, TEE, vendor)  
- **Hardware & software** (device, RAM/VRAM, framework versions)  
- **Transcript & seeds** (challenge IDs, normalization settings)

---

## Security model & guarantees

**We guarantee (when properly bound):**
- **Model identity decision** with user-chosen error control (`α`)  
- **Tamper-evident** transcript and environment metadata for audits  
- **(Optional) ZK assurance** that the verifier computed the published decision from the published transcript

**We do not guarantee:**
- **Training data quality** or **absence of backdoors**  
- **Safety/Alignment** properties (identity ≠ safety)  
- End-to-end identity binding for **remote APIs** without TEE or vendor signatures

---

## Reproduce

### Install
```bash
git clone https://github.com/rohanvinaik/PoT_Experiments
cd PoT_Experiments
pip install -r requirements.txt
```

### Basic verification (Python)
```python
from pot.core.diff_decision import EnhancedSequentialTester, TestingMode

tester = EnhancedSequentialTester(TestingMode.AUDIT)  # or QUICK / EXTENDED
result = tester.test_models(model_a, model_b)
print(result.decision)  # SAME / DIFFERENT / UNDECIDED
```

### Complete E2E validation (recommended)
```bash
# Unified E2E pipeline with all features
python scripts/run_e2e_validation.py \
  --ref-model gpt2 \
  --cand-model distilgpt2 \
  --mode audit

# With enhanced CI/CD features
python scripts/run_e2e_validation.py \
  --ref-model gpt2 \
  --cand-model distilgpt2 \
  --mode audit \
  --enable-attack-simulation \
  --enable-sharding \
  --performance-dashboard

# API mode validation
python scripts/run_e2e_validation.py \
  --ref-model http://api1.example.com/model \
  --cand-model http://api2.example.com/model \
  --verification-mode api \
  --generate-evidence-bundle

# Legacy: Direct statistical testing (minimal features)
python scripts/run_enhanced_diff_test.py \
  --ref-model gpt2 \
  --cand-model distilgpt2 \
  --mode audit
```

### Evidence bundle (one command)
```bash
bash scripts/make_evidence_bundle.sh   --run-id "$(date -u +%Y%m%dT%H%M%SZ)"   --include logs/*.json transcripts/*.ndjson configs/*.yaml   --include env/pip_freeze.txt env/sysinfo.txt   --sign path/to/ed25519_private.pem
```

### Optional: build ZK (prototype)
```bash
cd pot/zk/prover_halo2 && cargo build --release
# produces small proofs that the verifier ran on the published transcript
```

---

## Directory layout (typical)

```
pot/
  core/                # sequential tester, decision rules, stats
  lm/                  # model loaders, normalization, adapters
  zk/                  # (prototype) prover circuits, proof tooling
scripts/
  run_enhanced_diff_test.py
  run_api_diff_test.py
  make_evidence_bundle.sh
configs/
  *.yaml               # examples for local/API runs
transcripts/           # challenge/response logs
logs/                  # per-run summaries and metrics
```

---

## Limitations & non-goals

- **Adversarial robustness**: Wrapper attacks, sampling/temperature variation, tokenizer overlap <90%, and MoE router perturbations need explicit evaluation; expect higher query budgets in these regimes.  
- **Cost on frontier APIs**: 20–32 queries can be non-trivial for expensive endpoints.  
- **Strict black-box crypto**: Remote identity binding requires TEE or vendor signatures; ZK alone here does not identify the remote model.

---

## Roadmap

- **Adversarial/robustness test suite** with named conditions and CI-backed error rates  
- **API binding** via TEE attestation flow and vendor-commitment helpers  
- **Evidence UX**: single-file signed bundle + verifier tool  
- **Active-learning challenge selection** to further reduce queries  
- **Multi-modal** extensions (vision/audio) where applicable

---

## License & citation

- **License:** MIT (see `LICENSE`)  
- **Citation:** If you use this verifier in a paper or product, please cite this repository and the associated PoT manuscript.

---

*This README reflects the project’s current scope: a practical **behavioral verifier** with auditable transcripts and optional ZK proofs, designed to compose with a training-time **provenance** layer when available.*
