# Proof-of-Training, Practically: A Sequential Behavioral Verifier with Cryptographic Auditability

**Authors:** _Anonymous for review_  
**Code:** (included with submission)  
**Artifact type:** Open-source implementation + reproducibility bundle

---

## Abstract

We study the practical problem of verifying that a deployed model is **the model you think it is**—without requiring access to its weights. We present a **sequential behavioral verifier** that (i) pre-commits a space of challenges via HMAC, (ii) streams model responses and maintains an **Empirical-Bernstein** confidence sequence on a per-prompt effect-size statistic, and (iii) **stops early** once explicit SAME/DIFFERENT criteria are met. The system emits a tamper-evident **audit bundle** (seeds, transcript, thresholds, environment metadata) and optionally a **small ZK proof** that the verifier computed the published decision over the published transcript.  

Empirically, we obtain decisive SAME/DIFFERENT calls in **~20–32 queries** on representative LM pairs (e.g., GPT-2 vs DistilGPT-2; Llama-2-7B vs Mistral-7B; Yi-34B vs Yi-34B-Chat), with **minutes-scale** wall time. A **sharded** verifier validates **~206 GB** of models on a **64 GB** host by loading/releasing slices sequentially. We discuss limitations (sample sizes, adversarial robustness, API attestation) and position this verifier as the **post-training half** of a broader PoT program, complementary to training-time provenance certificates.  

*This paper reflects what is implemented and measured today; we intentionally avoid over-claiming beyond the released code and logs.*

---

## 1. Introduction

Deployed models are often **opaque** (API-only), rapidly updated, and valuable enough that **misrepresentation** (e.g., size-fraud, silent fine-tuning, substitution) is a real risk. Weight-level attestation (e.g., TEE) is powerful but not always available, and training-time methods (e.g., watermarks) require **control over training**.

We ask a simple question:

> **Can we verify model identity from behavior alone, quickly, with explicit error control—and without weights?**

Our answer is an **anytime** sequential test with **pre-committed challenges**. After each query, we update a variance-adaptive confidence sequence (Empirical-Bernstein, EB) over a bounded difference statistic and **stop early** when the interval certifies either **SAME** (small, well-localised difference) or **DIFFERENT** (large, stable effect). The verifier outputs a signed, reproducible **transcript**; in **local-weights** mode the transcript binds to a weight hash; in **API** mode it can be bound to a **TEE attestation** when available.

### Contributions
1. **Operational verifier**: A concrete, open-source system that decides SAME/DIFFERENT via **EB confidence sequences** and **explicit stopping rules**, typically in **dozens of queries**.  
2. **Challenge integrity**: A **HMAC-derived** challenge generator that pre-commits the challenge family, preventing cherry-picking while preserving reproducibility.  
3. **Auditability**: An **evidence bundle** capturing seeds, transcript, thresholds, and environment; optional **ZK proof** that the verifier computed the posted decision over the posted transcript.  
4. **Frontier-scale feasibility**: A **sharded** runner verifying **34B-class** models on **commodity RAM** (206 GB of weights on a 64 GB machine).  
5. **Scope clarity**: We delineate the verifier (this paper) from **training-time provenance** (“PoT certificates”), and show how they compose to end-to-end assurance.

---

## 2. Related Work (concise)

- **Behavioral testing & fingerprinting.** Classical fixed-N behavioral comparisons require **1k–10k prompts** and cannot stop early; our sequential design reduces queries by adapting to observed variance.  
- **Watermarking/ownership.** Requires training-time instrumentation and may be attenuated by fine-tuning; complementary to our post-hoc verifier.  
- **TEE attestation / remote attestation.** Strong identity binding when deployable; we use it as an optional **binding layer** for API mode.  
- **Zero-knowledge (ZK) proofs.** ZK can attest the **verifier’s computation** over a transcript; binding the transcript to a **specific remote model** still requires an attested endpoint or vendor commitment.

---

## 3. Method

### 3.1 Pre-committed challenges
Let `run_id` and HMAC key `K` define challenge seeds:
```
seed_i = HMAC_SHA256(K, run_id || i)
```
Seeds deterministically parameterize a family of **prompt templates** and normalization settings. Publishing `run_id`, counts, and the HMAC of the seed list **pre-commits** the challenge set.

### 3.2 Streaming statistic and confidence sequence
Each challenge yields a bounded per-prompt **difference score** `X_i in [0,1]` (e.g., normalized edit distance / fuzzy-hash dissimilarity over token sequences). After `n` prompts:
- Sample mean `\bar X_n`, sample variance `S_n^2` (Welford updates).
- **Empirical-Bernstein (EB)** half-width:
```
U_n(delta) = sqrt(2*S_n^2*log(1/delta)/n) + 7*log(1/delta)/(3*(n-1))
```
This yields an **anytime** confidence sequence via a spending schedule `delta_n` (peeling/union bound).

### 3.3 Decision rules (explicit)
Let the SAME-band be `[-γ,+γ]` and `h_n = U_n(delta_n)`.

- **SAME** if `[ \bar X_n - h_n, \bar X_n + h_n ] ⊆ [-γ,+γ]` **and** `h_n ≤ η·γ`.  
- **DIFFERENT** if `| \bar X_n | ≥ δ*` **and** relative margin `h_n / | \bar X_n | ≤ ε_diff`.  
- **UNDECIDED** otherwise; continue until mode-specific `n_min ≤ n ≤ n_max`.

### 3.4 Modes & parameters
**QUICK / AUDIT / EXTENDED** fix `α` (error budget), `γ, η, δ*, ε_diff`, and `[n_min, n_max]`. Parameters in code mirror the README and are chosen conservatively to favor **UNDECIDED** over incorrect calls.

### 3.5 Binding & audit
- **Local-weights:** include **SHA-256** of weight files; transcript binds to that hash.  
- **API black-box:** bind via **TEE attestation** (e.g., enclave quote) or **vendor-signed** model commitment.  
- **ZK (prototype):** Halo2 circuits attest “the verifier computed decision `D` from transcript `T` using code hash `H`”. ZK **does not** identify the remote model by itself—binding is external.

---

## 4. Implementation
- **Core:** `EnhancedSequentialTester` with EB updates and decision rules; **HMAC** challenge generator; token-space normalization; structured logging.  
- **Runners:** local HF models and API wrapper; **sharded** loader for 34B-class models (sequential load → evaluate → release).  
- **Artifacts:** logs (JSON), transcripts (NDJSON), seeds, thresholds, environment dump; optional **evidence bundle** packer and prototype **ZK** builder.

---

## 5. Experiments

### 5.1 Set-up
Representative **end-to-end** runs on a single workstation (64 GB RAM). Deterministic decoding (e.g., temperature=0). Publish seeds, mode, thresholds, and transcripts.

### 5.2 Pairs and results (illustrative)
| Pair | Mode | Verdict | Queries | Total Time | Per-query | Notes |
|---|---|---|---:|---:|---:|---|
| Yi-34B vs Yi-34B-Chat | AUDIT | DIFFERENT | 20 | ~215 s | ~10.7 s | **Sharded** verification |
| Llama-2-7B vs Mistral-7B | AUDIT | DIFFERENT | 32 | ~290 s | ~9.1 s | Cross-architecture |
| GPT-2 vs DistilGPT-2 | AUDIT | DIFFERENT | 32 | ~56 s | ~1.7 s | Distillation detection |
| Pythia-70M vs Pythia-70M | QUICK | SAME | 10 | ~60 s | ~6 s | Self-consistency |

**Observation.** Decisions typically arrive in **dozens** of queries; minutes-scale wall time at 7B–34B; sharding keeps memory below host RAM.

> We avoid FAR/FRR point estimates from tiny samples; release **transcripts and seeds** and compute **binomial CIs** as more trials accumulate.

---

## 6. Why it uses fewer queries
Fixed-N baselines (1k–10k prompts) cannot stop early. Our **anytime EB** sequence adapts to observed variance; when models are clearly the same (within `γ`) or clearly different (beyond `δ*`), the interval collapses quickly and the decision **stops**—often after a few dozen challenges.

---

## 7. Limitations
- **Statistical power:** current runs are **small-N**; report **counts with 95% CIs** and publish ROC/DET curves once larger sweeps are automated.  
- **Adversarial robustness:** wrapper attacks, sampling perturbations, tokenizer mismatch, and MoE routing need a named **robustness suite**; query budgets may rise.  
- **Binding in API mode:** ZK attests the **verifier**, not the **remote model**; API identity requires TEE or vendor commitments.  
- **Scope:** identity verification only; not data quality, safety, or backdoor freedom.

---

## 8. Composition with training-time PoT
This verifier is the **post-training** half. The **training-time** half produces a provenance certificate (hash-chained checkpoints, signed environment, IO-evolution). Composition: bind deployment to the training certificate via weight hash or TEE, then run our behavioral verifier periodically to detect drift or substitution.

---

## 9. Reproducibility checklist (artifact)
- **Code**: included; `requirements.txt` and `pip freeze` saved with runs.  
- **Config**: mode/thresholds (`α, γ, η, δ*, ε_diff, [n_min, n_max]`) recorded per run.  
- **Challenges**: HMAC-seeded; publish `run_id`, counts, and seed HMAC.  
- **Transcripts**: prompts (or IDs), normalized outputs, per-prompt scores, timestamps.  
- **Environment**: RAM/VRAM, device info, framework versions.  
- **Binding**: local—weight hash; API—attestation or vendor signature.  
- **Optional**: ZK proof; evidence bundle packer.

---

## 10. Ethics & broader impact
Positive: enables **provenance, accountability, and fraud prevention** without exposing proprietary weights. Risks: may be mistaken for **safety** or **fairness** guarantees; we explicitly disclaim those. Pair with governance and monitoring.

---

## 11. Conclusion
A **sequential behavioral verifier** with **cryptographic pre-commitment** and **auditable transcripts** decides SAME/DIFFERENT in **dozens** of queries and scales to **frontier-size** models via sharding. It composes with training-time provenance toward end-to-end **Proof-of-Training**. We release code, scripts, and evidence tooling, and outline the needed power/robustness/attestation steps for compliance-grade adoption.
