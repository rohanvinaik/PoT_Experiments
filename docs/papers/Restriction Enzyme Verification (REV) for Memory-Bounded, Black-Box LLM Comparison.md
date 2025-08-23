# Restriction Enzyme Verification (REV) for Memory-Bounded, Black-Box LLM Comparison

**Author:** [Your Name]

**Date:** August 23, 2025

---

## Abstract
We introduce **Restriction Enzyme Verification (REV)**, a method for comparing large language models (LLMs) whose parameter sizes exceed available device memory. REV treats a transformer as a composition of functional segments separated by **restriction sites**—architecturally or behaviorally meaningful cut points. Using a pre-committed challenge set and a streamed, segment-wise execution, REV emits compact **segment signatures** derived from activations (or output logits in the black-box case). These signatures are committed in a **Merkle tree** per challenge and aggregated with an anytime-valid sequential test (PoT-style) to reach **SAME / DIFFERENT / UNDECIDED** conclusions under controlled error rates. REV localizes divergence to specific segments, maintains fairness across models by enforcing identical execution policies, and operates under strict memory bounds by loading only one segment at a time.

---

## 1. Motivation & Goals

State-of-practice LLM evaluation focuses on end-to-end behavior and accepts large memory footprints. REV addresses three gaps:

1. **Memory-bounded verification:** Stream inference through segments to compare models larger than RAM/VRAM.
2. **Modular equivalence:** Evaluate **where** two models agree or diverge, instead of only whether final outputs match.
3. **Auditability & tamper-resistance:** Produce cryptographic commitments and a reproducible transcript compatible with Proof-of-Tests (PoT) workflows.

**Threat model.** We assume black-box or gray-box access to model inference with potential adversarial control over the runtime (e.g., model-switching or wrapper orchestration). REV detects inconsistencies at the segment level, making stitching attacks difficult. Optional attestation or ZK proofs (outside REV’s core) can bind execution to hardware or circuits.

**Non-goals.** REV is **not** a weight-identity proof and does not guarantee bitwise equality of internal states. It tests **behavioral equivalence** of segments under a fixed policy and challenge distribution.

---

## 2. Background

### 2.1 Transformer modularity and natural cut points
Transformers exhibit recurring circuit motifs (e.g., induction heads, successor heads), MLP-mediated associative memories, and emergent modularity. In practice, natural cut points align with boundaries **within** blocks (after attention, after MLP) and **across** blocks (end-of-block residual). REV exploits these boundaries for segmenting execution and for designing probe families in black-box settings.

### 2.2 Memory-bounded inference
Modern serving systems reduce memory via paging/offload, activation checkpointing, and quantization. REV leverages these techniques but constrains the executor to a **single-segment working set**, ensuring applicability to models larger than device memory.

### 2.3 Sketching and commitments
REV separates **similarity** from **integrity**:
- **Similarity sketches**: low-precision, locality-sensitive projections of activations/logits that are robust to benign numeric drift.
- **Cryptographic commitments**: collision-resistant hashes of the sketches and metadata organized in a Merkle tree per challenge to enable efficient audit and tamper evidence.

### 2.4 Sequential (anytime-valid) decisions
Instead of fixed-sample tests, REV uses sequential tests (as in PoT) that control type-I/II errors while allowing early stopping. Per-challenge equality and distances aggregate into a stopping rule with explicit error guarantees.

---

## 3. Problem Statement

**Input:** Two LLM oracles \(M_A, M_B\); a segmentation policy \(\mathcal{S}\) yielding cut sites \(s_1, \dots, s_K\); a public, pre-committed challenge generator \(\mathcal{C}\); an execution policy \(\pi\) (temperature, max tokens, decoding, precision, seeds);

**Output:** Decision in {SAME, DIFFERENT, UNDECIDED} and a localization report identifying segments of first divergence.

**Constraints:** Peak memory \(\leq\) bound; identical policies for A and B; deterministic or controlled-stochastic decoding; stable seeds and rounding.

---

## 4. REV Design

### 4.1 Restriction-site policies
REV supports two families of cut policies:

- **Architectural sites (white/gray-box):** After attention, after MLP, end-of-block, and block windows (e.g., layers 1–8, 5–12, 9–16) for overlap.
- **Behavioral sites (black-box):** Sites defined by **probe families** that isolate known motifs (e.g., induction patterns, copy suppression). At each site, we extract stable **observable signatures** (logprob vectors on fixed probe suffix sets).

**Overlap windows.** To resist stitching attacks, segments are evaluated with overlapping windows (e.g., \([1..8], [5..12], [9..16], \dots\)). First divergence typically localizes to the earliest window that fails.

### 4.2 Challenge generation (PoT-aligned)
REV adopts a pre-committed, key-derived challenge set:

- **Seeds:** \(\text{seed}_i \leftarrow \mathrm{HMAC}(\text{key}, \text{run\_id} \parallel i)\)
- **Prompts:** Templates sampled from seed-conditioned grammars/datasets; optional adversarial probes for robustness.
- **Public transcript:** Seeds, templates, and decoding policy published before evaluation.

### 4.3 Segment signatures
For each challenge \(c\) and segment \(s\):

1. **Extract representation** \(a_s\):
   - White/gray-box: mean-pooled token activations from a predetermined token window (e.g., last \(m\) tokens) at the segment output.
   - Black-box: fixed-size logprob vector over a public probe suffix set at the “site”.
2. **Project & quantize** using a public, domain-separated random projection \(R_s\) (seeded by \(\text{run\_id}, s\)):
   \[ z_s = \mathrm{round}(\mathrm{clip}(R_s a_s, \tau) \cdot q) \in \mathbb{Z}^{d'} \]
3. **Similarity sketch** \(\sigma_s\): signs or small-integer bins of \(z_s\), length \(d'\) (e.g., 1024–4096 bits).
4. **Commitment leaf** \(h_s\): cryptographic hash of a canonical encoding of \(\sigma_s\) and metadata (segment id, challenge id, policy, version tags).

**Merkle root per challenge.** Leaves \(\{h_s\}\) are ordered canonically and aggregated to produce \(H_c\). Equality of \(H_c\) is a strong signal of segment-wise behavioral identity for that challenge; otherwise, distances between \(\sigma_s\) guide localization.

### 4.4 Streaming executor
A segment runner enforces the memory bound:

- Load parameters for segment \(s\), run forward on input state, emit \(a_s\), release memory.
- Maintain KV cache according to the policy; optionally employ offload/paging.
- Support overlap windows by replaying from cached boundary states or recomputing via checkpointing.

### 4.5 Decision layer (sequential)
Per challenge \(c\), compute:

- **Merkle equality indicator** \(I_c = \mathbb{1}[H_c^{(A)} = H_c^{(B)}]\)
- **Segment distance score** \(d_c\) (e.g., mean Hamming over \(\sigma_s\); robustly clipped).

Feed \(\{I_c, d_c\}\) to a sequential tester that stops when evidence exceeds thresholds. Outcomes: SAME (accept similarity), DIFFERENT (accept difference), or UNDECIDED (insufficient evidence under budget).

---

## 5. Skeleton Implementation (pseudocode)

### 5.1 Interfaces
```python
class ExecutionPolicy:
    # decoding, precision, seeds, stop criteria
    temperature: float  # often 0.0
    top_p: float
    max_tokens: int
    dtype: str  # e.g., fp16, int8
    seed: int
    attn_impl: str  # e.g., paged

class SegmentSite:
    seg_id: str         # e.g., L12.post_attn
    overlap_group: int  # for windowing
    projector_seed: int # domain-separated seed

class Challenge:
    id: str
    prompt: str
    meta: dict

class Signature:
    seg_id: str
    sketch_bits: bytes  # or bitarray
    meta: dict
```

### 5.2 Challenge generation
```python
def generate_challenges(key, run_id, n):
    for i in range(n):
        seed = HMAC(key, f"{run_id}:{i}")
        prompt = synthesize_prompt(seed)  # seeded templates & corpora
        yield Challenge(id=f"C{i}", prompt=prompt, meta={"seed": seed})
```

### 5.3 Segment runner (white/gray-box)
```python
def run_segment(model, states_in, seg: SegmentSite, policy: ExecutionPolicy):
    # Load params for `seg` (offload-aware)
    load_params(seg)
    states_out, activations = forward_segment(model, states_in, seg, policy)
    release_params(seg)
    return states_out, activations
```

### 5.4 Signature builder
```python
def build_signature(activations_or_logits, seg: SegmentSite, policy):
    a = select_and_pool(activations_or_logits)          # fixed pooling
    R = seeded_random_matrix(seg.projector_seed, shape=(d_prime, a_dim))
    z = quantize(clip(R @ a, tau), q)
    sigma = binarize(z)                                 # sign or bins
    leaf = hash(encode({"seg": seg.seg_id, "sigma": sigma, "policy": policy}))
    return Signature(seg.seg_id, sigma, {"leaf": leaf})
```

### 5.5 Per-challenge transcript
```python
def evaluate_one(model, challenge, sites, policy, black_box=False):
    sigs = []
    states = init_context(model, challenge.prompt, policy)
    for seg in sites_in_canonical_order(sites):
        if black_box:
            logits = probe_logits(model, states, seg, policy)
            sig = build_signature(logits, seg, policy)
        else:
            states, acts = run_segment(model, states, seg, policy)
            sig = build_signature(acts, seg, policy)
        sigs.append(sig)
    root = merkle_root([s.meta["leaf"] for s in sigs])
    return {"root": root, "sigs": sigs}
```

### 5.6 Pairwise comparison
```python
def compare_models(MA, MB, challenges, sites, policy):
    for c in challenges:
        TA = evaluate_one(MA, c, sites, policy)
        TB = evaluate_one(MB, c, sites, policy)
        Ic = int(TA["root"] == TB["root"])  # exact equality
        dc = mean_hamming([a.sketch_bits for a in TA["sigs"]],
                          [b.sketch_bits for b in TB["sigs"]])
        yield {"challenge": c.id, "I": Ic, "d": dc,
               "first_divergence": first_div_site(TA, TB)}
```

### 5.7 Sequential decision (anytime-valid)
```python
def sequential_decision(stream, alpha=0.01, beta=0.01, d_thresh=0.08, max_C=2000):
    # Maintain e-values or confidence sequence on match rate and distance
    S_match = init_seq_test(alpha)
    S_dist  = init_seq_test(beta)
    for t, r in enumerate(stream, 1):
        update(S_match, r["I"])      # Bernoulli evidence for equality
        update(S_dist,  r["d"], d_thresh)  # small distances accumulate evidence
        if accept_same(S_match, S_dist):
            return "SAME", t
        if accept_diff(S_match, S_dist):
            return "DIFFERENT", t
        if t >= max_C:
            break
    return "UNDECIDED", t
```

---

## 6. Engineering Considerations

**Execution determinism.** Fix decoding (temperature=0, greedy or beam), seeds, and math modes. Record library versions and kernels. Where slight nondeterminism persists (e.g., GPU kernels), similarity sketches absorb small drifts.

**Projection parameters.** Use 1–4K bits per segment; tune clipping \(\tau\) and quantization \(q\) to maximize stability without saturating sensitivity.

**Domain separation.** Derive projector seeds as `seed = H(run_id || seg_id || version)`. Never reuse seeds across segments or runs.

**Overlap policy.** Use windows of 6–12 layers with stride 3–6 for medium-sized decoders; scale by depth for larger models.

**Black-box probes.** Maintain a public probe library (induction, succession, bracket matching, entity recall). Calibrate with calibration models to ensure discriminative power.

**Storage & audit.** Emit a per-challenge JSON with seeds, Merkle root, segment leaves, and timing/memory telemetry. Keep a signed manifest for the whole run.

---

## 7. Validity & Limitations

**What REV asserts.** Under a fixed policy and challenge distribution, two models that consistently match segment signatures and Merkle roots are **behaviorally equivalent at the chosen restriction sites**, within the sketch’s tolerance.

**What REV does not assert.** Weight equality, equivalence under different policies or distributions, or robustness to adaptive adversaries with query access to both models during evaluation (mitigated by large pre-committed challenge spaces and rate limits).

**Failure modes.** Poor site selection; projections too coarse/fine; numerical drift exceeding tolerance; adversarial wrappers caching and replaying segment outputs (mitigated by timing variance, overlap windows, and probe diversity).

---

## 8. Evaluation Plan

1. **Sanity checks:** Same model twice (should be SAME); model + quantized clone; model + distilled variant; model vs different family.
2. **Ablations:** Segment width (bits), window overlap, probe sets, projection seeds, decoding policies, numeric precisions.
3. **Localization accuracy:** Plant controlled edits (e.g., modifying mid-MLP layers) and verify earliest failing window.
4. **Resource profile:** Peak memory, wall-clock per challenge, segment load/unload costs under offload.
5. **Statistical operating characteristics:** False accept/reject rates vs challenge counts; anytime stopping distribution.

---

## 9. Security & Integrity Extensions

- **TEEs/Attestation:** Bind executor to hardware/firmware; attest code and model provenance.
- **ZK proofs (zkML):** Prove conformance of segment signatures to a specified circuit for a subset of challenges (costly but strong audit points).
- **Watermarks/Fingerprints:** Optionally cross-check with existing watermarking/fingerprinting for orthogonal evidence.

---

## 10. Related Work (brief, non-exhaustive)

- Mechanistic Interpretability & modular structure in transformers; activation patching and editing methods.
- Memory-efficient inference: paging/offload, activation checkpointing, quantization.
- Sketching and LSH (SimHash), random projections (JL lemma).
- Merkle trees for authenticated data structures.
- Anytime-valid sequential testing (confidence sequences, e-values).
- Model watermarking and black-box fingerprinting.

---

## 11. Practical Recipe (TL;DR)

1. Choose restriction sites (architectural if possible; otherwise behavioral).
2. Pre-commit challenges with an HMAC key; publish policy.
3. Stream each model segment-by-segment; at each site produce a similarity sketch and Merkle leaf.
4. Aggregate leaves into a per-challenge root; compare roots and segment sketches.
5. Run an anytime-valid sequential test over challenges until SAME/DIFFERENT.
6. Publish the transcript (seeds, policies, roots, first-divergence reports) and resource profile.

---

## 12. Roadmap & Open Questions

- Automating behavioral site discovery for black-box-only APIs.
- Choosing optimal projection families for different tokenization/architectures.
- Tightening statistical bounds for mixed equality/distance evidence streams.
- Integrating attestation or light zk proofs for random subsets of challenges.

---

## Appendix A: Minimal Data Schemas

```json
// signature.jsonl (one line per segment)
{
  "challenge_id": "C123",
  "seg_id": "L12.post_attn",
  "sketch_b64": "...",
  "leaf": "blake3:...",
  "policy": {"temperature": 0.0, "dtype": "fp16", "attn": "paged"},
  "telemetry": {"alloc_mb": 1800, "t_ms": 52}
}
```

```json
// challenge_manifest.json
{
  "run_id": "REV-2025-08-23",
  "sites": ["L1.post_attn", "L1.post_mlp", ...],
  "policy": {...},
  "roots": {"C0": "...", "C1": "..."},
  "seq_decision": {"outcome": "SAME", "n_challenges": 742}
}
```

---

### Acknowledgments
This document synthesizes prior art in transformer interpretability, memory-bounded serving, sketching, authenticated data structures, and sequential inference, and adapts them into a unified verification procedure suitable for PoT-style audits.
