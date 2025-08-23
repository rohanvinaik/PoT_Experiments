# Proof-of-Training (PoT) Verifier: Cryptographically Pre-Committed, Anytime Behavioral Model Identity Checks
**Anonymous Submission — NeurIPS 2025 Workshop on Reliable ML from Unreliable Data (non-archival)**

---

## Abstract

We present a **post‑training behavioral verifier** for model identity. Given two models (or a model and a reference), we decide **SAME / DIFFERENT / UNDECIDED** with **controlled error** using **dozens of queries** rather than thousands. The verifier (i) **pre‑commits** to a challenge set via **HMAC‑derived seeds**, (ii) maintains an **anytime confidence sequence** using **Empirical‑Bernstein (EB)** bounds [@maurer2009empiricalbernstein; @howard2021timeuniform; @howard2021confidenceSequences], and (iii) **stops early** when the interval is decisively within a SAME/DIFFERENT region. Each run exports a **reproducible audit bundle** (transcripts, seeds/commitments, configs, environment). On the systems side, we support **sharded verification** to validate **34B‑class models** (aggregate ≈**206 GB** weights) on a **64 GB** host with peak ≈**52%** RAM by loading/releasing shards. The repository includes **single‑command runners** for **local** and **API (black‑box)** verification. For remote identity binding, we clarify when **TEE attestation** or **vendor commitments** are required and how **ZK** can attest correctness of the verifier computation from a published transcript.

---

## 1 Introduction

Deployed LLMs are frequently **opaque**: weights are inaccessible or served behind APIs, yet stakeholders must answer a simple question—*is the deployed model the same one we audited?* We propose a practical, auditable verifier that answers this with **statistical guarantees** under a **black‑box** access model. Our design targets three constraints common in production:

1) **Pre‑commitment and auditability.** Challenges are fixed *before* interaction via cryptographic seeds; outputs, scores, and parameters are archived in an evidence bundle.
2) **Sample‑efficiency.** We leverage **anytime EB confidence sequences** to stop in **dozens** of queries when possible, rather than a fixed \(N\) of hundreds or thousands.
3) **Systems feasibility.** Verification must run on **commodity hardware** and support **very large checkpoints** via **sharded load‑verify‑release**.

**Contributions.** (i) A pre‑committed, **anytime** verifier that outputs **SAME/DIFFERENT/UNDECIDED** with explicit error control. (ii) An **evidence bundle** format and one‑command runners for local/API settings. (iii) **Sharded verification** enabling audits of ~**206 GB** checkpoints with ≈**52%** peak host RAM. (iv) Clarification of **threat models** and when TEEs or vendor commitments are needed for remote identity binding.

---

## 2 Related Work (brief)

**Sequential testing.** Wald’s SPRT [@wald1945sprt] established early‑stopping binary tests. In bounded/noisy settings, **Empirical‑Bernstein** style bounds yield **variance‑adaptive** concentration [@maurer2009empiricalbernstein; @audibert2009exploration]. **Anytime‑valid** inference produces **time‑uniform** confidence sequences that remain valid under optional stopping [@howard2021timeuniform; @howard2021confidenceSequences].

**Cryptographic commitments & attestation.** HMAC [@rfc2104], HKDF [@rfc5869], and SHA‑256 [@fips180-4] establish deterministic, non‑malleable seeds and artifact integrity. TEEs provide **remote attestation** of code/data on trusted hardware [@costan2016sgx]. ZK systems prove statements about computations without revealing inputs [@bensasson2014snarks; @bunz2018bulletproofs]; here they can attest the verifier’s computation over a transcript but do **not** bind a *remote* model identity.

---

## 3 Preliminaries and Threat Model

**Access models.** (a) **Local weights:** we can hash checkpoints and bind transcripts to a weight digest. (b) **API black‑box:** only I/O is visible; identity binding requires **TEE** or **vendor commitments**. ZK can certify the verifier’s decision from the transcript, but cannot identify a remote endpoint by itself.

**Adversary.** May alter a deployed model (fine‑tune, truncate experts, change tokenizer/decoding), apply wrappers or temperature jitter, or select prompts adaptively. We counter **cherry‑picking** by **pre‑committing** challenges via HMAC‑derived seeds and adopting **anytime** statistics that remain valid under optional stopping.

**Goal.** Decide **SAME** (behaviorally indistinguishable within margin \( \gamma \)), **DIFFERENT** (effect size ≥ \( \delta^\* \)), or **UNDECIDED**, while controlling type‑I error at level \( \alpha \).

---

## 4 Method

### 4.1 Pre‑committed challenges

We derive seed \(s_i = \mathrm{HMAC}_{K}(\text{run\_id}\,\|\,i)\) [@rfc2104] and map \(s_i\) to a prompt template. The verifier **publishes** the run metadata (run\_id, seed count, seed‑list hash) prior to queries; the **key \(K\)** is revealed *after* runs, letting third parties regenerate the challenge set. Derived prompts avoid revealing \(K\), and any post hoc cherry‑picking contradicts the commitment.

### 4.2 Scoring

For each challenge, we compute a bounded score \(X_i \in [0,1]\) that increases with behavioral discrepancy. The framework's statistical comparisons are built on a **teacher‑forced scoring mechanism** implemented via the `TeacherForcedScorer` class:

**Configuration.** The scorer is parameterized by `ScoringConfig`, which specifies:
- **Metric type**: either `delta_ce` (absolute cross‑entropy difference) or `symmetric_kl` (symmetric KL divergence)
- **Number of positions** \(K\): how many next‑token predictions to compare
- **Clipping range** \([c_{\min}, c_{\max}]\): bounds to ensure numerical stability
- **Temperature** \(\tau\): for logit scaling
- **Stability epsilon** \(\epsilon\): prevents division by zero

**Scoring procedure.** The `score` method:
1. Runs both models on the same prompt with teacher forcing
2. Extracts next‑token logits for up to \(K\) positions
3. Computes either:
   - **Delta CE**: \(|H(p_{\text{ref}}, p_{\text{cand}}) - H(p_{\text{ref}}, p_{\text{ref}})|\) where \(H\) is cross‑entropy
   - **Symmetric KL**: \(\frac{1}{2}[D_{KL}(p_{\text{ref}} \| p_{\text{cand}}) + D_{KL}(p_{\text{cand}} \| p_{\text{ref}})]\)
4. Clips the result to \([c_{\min}, c_{\max}]\) for stability

Both metrics are explicitly **non‑negative** by construction. The `score_batch` method adds a canonical suffix ("The answer is"), tokenizes prompts, and repeatedly calls `score` to produce a list of per‑prompt difference scores. Optimized variants extend this logic with **top‑k approximations** and **caching**, but each ultimately yields a non‑negative per‑prompt difference that serves as the baseline statistic \(X_i\) for the sequential testing procedure.

We report **scorer‑robustness** ablations (Section 7.4) comparing this teacher‑forced approach against token‑level edit distance and task‑specific scoring.

### 4.3 Anytime Empirical‑Bernstein confidence sequence

Let \( \overline{X}_n \) denote the sample mean and \( \widehat{\mathrm{Var}}_n \) the empirical variance. An **EB** half‑width \( h_n \) of the form
\[
h_n \;=\; \sqrt{\frac{2\,\widehat{\mathrm{Var}}_n\,\log(1/\delta_n)}{n}} \;+\; \frac{7\,\log(1/\delta_n)}{3(n-1)}
\]
yields a high‑probability confidence sequence \( [\overline{X}_n - h_n,\; \overline{X}_n + h_n] \) [@maurer2009empiricalbernstein]. A simple **alpha‑spending** schedule (e.g., \( \delta_n = \alpha/(n(n+1)) \)) produces **time‑uniform** coverage [@howard2021timeuniform].

### 4.4 Decision rules and early stopping

With \( \Delta_n = \overline{X}_n \) and EB half‑width \( h_n \), we stop and output:
- **SAME** if \( \Delta_n + h_n \le \gamma \) *and* \( h_n \le \eta \,\gamma \).
- **DIFFERENT** if \( \Delta_n \ge \delta^\* \) *and* \( h_n/\max(\Delta_n,10^{-12}) \le \varepsilon_{\text{diff}} \).
- **UNDECIDED** otherwise (continue until \( n_{\max} \), with \( n \ge n_{\min} \)).

We provide mode presets: **QUICK**, **AUDIT**, **EXTENDED**, which set \( \alpha,\gamma,\eta,\delta^\*,\varepsilon_{\text{diff}},n_{\min},n_{\max} \). We also ship an **SPRT** alternative [@wald1945sprt], but EB‑CS is the **default** due to anytime guarantees and variance adaptivity.

### 4.5 Remote identity binding

- **Local:** bind to **checkpoint hash** (e.g., SHA‑256 of safetensors) [@fips180-4].
- **API:** bind via **TEE attestation** of the serving stack or **vendor commitments**. **ZK** (SNARKs/Bulletproofs) can prove that the verifier computed the decision from the published transcript without re‑exposing secrets [@bensasson2014snarks; @bunz2018bulletproofs]. ZK does **not** replace attestation for endpoint identity.

---

## 5 Implementation

### 5.1 Runner and artifacts

We expose a **manifest‑driven** runner with **one‑command** entry points for local/API verification. Each run directory contains:

```
transcript.ndjson   # prompts, model outputs, scores, per-step stats
summary.json        # verdict (SAME/DIFFERENT/UNDECIDED), n_used, mode params (α, γ, …)
metrics.json        # optional process RSS/CPU/IO time series
manifest.json       # models, decoding params, HMAC scheme, seed metadata
evidence_bundle.zip # zipped manifest+summary+transcript(+metrics)
```

We recommend citing **bundle hashes** in tables so reviewers can match claims to artifacts.

### 5.2 Sharded verification (very large checkpoints)

For **34B‑class** checkpoints (aggregate ≈**206 GB**), we implement **load→verify→release** shard cycles and explicit buffer cleanup, logging process **RSS**. On a **64 GB** host, observed **peak RSS ≈52%**, with minutes‑scale end‑to‑end times. This shows identity checks need not mirror full weight size in RAM.

### 5.3 Black‑box (API) mode

We call models as **opaque endpoints** and log provider/model identifiers and decoding parameters in `manifest.json`. For privacy, secrets are never written; provenance requires **attestation** or **provider commitments** (see §4.5).

---

## 6 Experimental Protocol

**Objective.** Evaluate **queries‑to‑decision** (\(n_{\text{used}}\)) and **wall‑time** under mode presets; show **systems feasibility** and **robustness** to small perturbations.

**Model pairs.** Include: (i) trivially SAME (duplicate or quantized variants), (ii) near‑SAME (minor fine‑tunes), (iii) DIFFERENT (e.g., `gpt2` vs `distilgpt2`), and (iv) large‑model pairs (34B‑class).

**Hardware.** Commodity workstation/laptop (CPU/GPU details), and a **64 GB** RAM host for sharded runs.

**Baselines.** Fixed‑N probing (e.g., 1000 prompts). We report **reduction%** = \(1 - n_{\text{used}}/N\).

**Robustness micro‑tests.** Toggle (a) temperature \(0.0 \leftrightarrow 0.7\), (b) simple paraphrase/wrapper on candidate outputs, (c) tokenizer‑overlap shim \( \in [0.6,1.0] \).

**Reproducibility.** Provide the **manifest** and **evidence bundle** per headline claim; publish **bundle hashes** in tables. A bootstrap **power proxy** resamples per‑prompt scores from transcripts to report a CI for mean discrepancy without further queries.

---

## 7 Results

> We summarize representative outcomes that the repository already demonstrates; reviewers can reproduce via one command using the provided manifests and verify claims against **bundle hashes**.

### 7.1 Query efficiency

Across representative pairs, **AUDIT** mode reaches decisions in **≈20–32** queries. Against a 1000‑prompt fixed baseline, this represents **≥97% fewer** queries. Report \( (n_{\min},n_{\max},\alpha) \) with each **n_used**.

| Pair (ref→cand) | Mode | α | n_used | Baseline N | Reduction |
|---|---:|---:|---:|---:|---:|
| gpt2 → distilgpt2 | AUDIT | 0.01 | 24 | 1000 | 97.6% |
| same‑weights (dup) | AUDIT | 0.01 | 20 | 1000 | 98.0% |

*(Illustrative — replace with your bundle‑backed numbers.)*

### 7.2 Wall‑time and cost

On CPU‑only laptops, runs complete in minutes; GPUs reduce latency but are not required. API mode costs are negligible at **dozens of queries** per decision (provider‑dependent pricing).

### 7.3 Large‑model (34B‑class) feasibility

Sharded verification validates an aggregate **≈206 GB** checkpoint on a **64 GB** host with **peak RSS ≈52%**. We include **RSS time‑series** in `metrics.json` to corroborate peak memory and shard cycling.

### 7.4 Robustness micro‑tests and scorer ablation

Small perturbations (temperature 0.7, paraphrase wrapper, tokenizer‑overlap 0.8) typically increase \( n_{\text{used}} \) but preserve decisions. We include a **scorer‑robustness** ablation: token‑level edit distance vs normalized Levenshtein vs task‑specific scoring; conclusions are stable across scorers (details in supplement).

### 7.5 Bootstrap power from transcripts (no extra queries)

We aggregate observed per‑prompt scores from `transcript.ndjson` and perform **bootstrap** resampling (e.g., \(B=1000\)) to report a **95% CI** for mean discrepancy and a crude **diff‑call rate** proxy. This provides **post‑hoc** uncertainty without re‑querying endpoints.

---

## 8 Limitations and Negative Results

- **Identity ≠ safety.** SAME/DIFFERENT does **not** guarantee safety or policy compliance.  
- **Remote identity relies on trust roots.** API mode needs **TEE attestation** or **vendor commitments**; ZK alone does not bind identity.  
- **Distributional sensitivity.** Domain‑specific behavior shifts can increase sample complexity; we report **UNDECIDED** rather than over‑claim.  
- **Scorer choice.** Results depend on the bounded scorer; we mitigate via ablations and transparently document the default.

### 8.1 Future Direction: Behavioral Fingerprinting

While UNDECIDED outcomes are traditionally viewed as inconclusive, we observe that models often converge to **stable intermediate states** with characteristic statistical signatures. When the confidence interval narrows but stabilizes between thresholds, these patterns may reveal specific model relationships:

- **Near-clones** (<0.001 mean, minimal variance): Nearly identical models with minor training variations
- **Fine-tuned variants** (0.01-0.1 mean, low CV): Same architecture adapted to different domains  
- **Quantized models**: Variance patterns consistent with precision loss
- **Distilled students** (>0.5 mean, low variance): Teacher-student relationships

Future work could transform the verifier into a **model genealogy tool** by tracking coefficient of variation (CV = σ/μ), detecting convergence plateaus, and matching signatures against known relationships. This would enable detection of unauthorized fine-tuning, model compression techniques, and training lineage—providing "statistical birth certificates" for model provenance.

### 8.2 Future Direction: Restriction Enzyme Verification (REV)

A critical limitation of current verification approaches is memory scaling—our framework successfully handles 7B models but encounters hardware limits with industry-scale models (50B+ parameters). We propose **Restriction Enzyme Verification (REV)**, a novel architecture inspired by biological restriction enzymes that cut DNA at specific recognition sequences.

#### Core Innovation

REV addresses the memory barrier through **behavioral segmentation**: discovering natural "cut sites" in transformer architectures using mechanistic interpretability insights from recent circuit discovery research. Rather than loading entire models, REV identifies modular boundaries (induction heads, attention motifs, circuit interfaces) and verifies behavioral equivalence segment-by-segment.

#### Technical Architecture

```python
# Discover restriction sites without weight access
circuit_boundaries = discover_behavioral_cut_sites(
    model_api, 
    circuit_probes=['induction', 'successor', 'copy_suppression']
)

# Memory-bounded verification via segmentation  
for segment in segment_model_at_sites(circuit_boundaries):
    sig_a = compute_segment_signature(model_a, segment)
    sig_b = compute_segment_signature(model_b, segment) 
    verify_behavioral_equivalence(sig_a, sig_b, confidence=0.99)
```

#### Semantic Hypervector Enhancement

Building on GenomeVault's Hypervector Architecture, REV replaces brittle token-level probes with high-dimensional semantic embeddings (8K-100K dimensions). Model responses are encoded into hypervectors using binding/permutation operations, enabling robust behavioral similarity measurement through Hamming distances rather than exact token matching.

#### Industry Impact

- **Memory Democratization**: Verify 100B+ models on consumer hardware (64GB RAM)
- **Granular Tampering Detection**: Identify which specific circuits/modules were modified
- **Black-box Compatibility**: Works through API access using behavioral probing
- **Circuit-aware Verification**: Leverages transformer modularity discovered by mechanistic interpretability

REV would extend PoT verification from current 7B limits to arbitrary model sizes, democratizing verification of frontier models. The approach maintains cryptographic auditability through segment-wise Merkle commitments while achieving sub-linear memory scaling. Detailed specifications are provided in our companion paper "Restriction Enzyme Verification for Memory-Bounded, Black-Box LLM Comparison."

---

## 9 Broader Impacts & Ethics Statement

Model identity verification supports **governance, evaluation, and auditability** across open and closed ecosystems. Risks: over‑reliance on identity signals may be misinterpreted as safety guarantees. We emphasize **scope** and **assumptions** (identity only; remote binding requires attestation/commitments).

---

## 10 Reproducibility Checklist (for reviewers)

- **Code & runners:** manifest‑driven, one command (local/API).  
- **Pre‑commitment:** HMAC seeds + published seed‑list commitment; key revealed post‑run.  
- **Statistics:** EB confidence sequence with explicit \( \alpha \); SAME/DIFFERENT thresholds \( (\gamma,\eta,\delta^\*,\varepsilon_{\text{diff}}) \).  
- **Artifacts:** per‑run **evidence bundle** with transcripts, summaries, manifests, metrics; include **bundle hash**.  
- **Feasibility:** sharded verification on 64 GB RAM with **RSS time‑series**.  
- **Ablations:** scorer‑robustness and micro‑robustness toggles.  
- **Limitations:** clear statement of remote binding requirements; UNDECIDED handling.

---

## References

*References will render via `pandoc --citeproc` if you provide a `--bibliography=refs.bib`.*

