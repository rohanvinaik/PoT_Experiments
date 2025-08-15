# **Proof-of-Training: A Statistical Framework for Black-Box Neural Network Verification**

**Abstract** We present Proof-of-Training (PoT), a statistical framework for verifying neural network identity through pure black-box access. Unlike approaches requiring internal access or training-time modifications, PoT formulates verification as a sequential hypothesis test comparing model responses to carefully designed challenge distributions. We address the practical challenges of verifying modern neural networks, including non-IID outputs, discrete token spaces, and inherent nondeterminism in language models. Our framework employs empirical Bernstein bounds for tighter confidence intervals and sequential testing procedures (SPRT) to reduce sample complexity. We analyze robustness to realistic adversarial strategies including wrapper attacks, targeted fine-tuning, and model compression. For language models, we introduce fuzzy hashing techniques to handle tokenization variability and version drift. We provide theoretical analysis of the coverage-separation trade-off in challenge design and compare our approach to existing fingerprinting, watermarking, and trusted execution methods.

**Keywords:** Black-Box Verification, Sequential Hypothesis Testing, Model Authentication, Robust Statistics, Fuzzy Hashing

---

## **1\. Introduction**

Model verification in regulatory contexts requires confirming that a deployed neural network matches its audited version without accessing proprietary parameters. This paper presents Proof-of-Training (PoT), a practical black-box verification framework that handles the complexities of modern neural networks.

### **1.1 Challenges in Real-World Verification**

Verifying neural networks in practice faces several obstacles:

* **Non-IID Outputs**: Language models produce correlated, discrete tokens with complex dependencies  
* **Inherent Nondeterminism**: Tokenization ambiguities, tie-breaking in sampling, server-side updates  
* **Adversarial Manipulation**: Wrapper functions, targeted fine-tuning, model compression  
* **Operational Constraints**: Limited query budgets, version drift, hardware variations

### **1.2 Contributions**

1. **Robust statistical framework** using empirical Bernstein bounds and sequential testing  
2. **Fuzzy hashing for language models** handling tokenization and version variability  
3. **Comprehensive adversary model** including wrapper attacks and compression  
4. **Coverage-separation analysis** for principled challenge design  
5. **Comparative evaluation** against fingerprinting, watermarking, and TEE-based approaches

---

## **2\. Statistical Framework**

### **2.1 Problem Formulation**

Given black-box access to model $f: \\mathcal{X} \\rightarrow \\mathcal{Y}$ and reference model $f^\*$, we test:

* $H\_0: f \\equiv\_\\tau f^\*$ (models functionally equivalent within tolerance $\\tau$)  
* $H\_1: f \\not\\equiv\_\\tau f^\*$ (models differ beyond tolerance)

### **2.2 Handling Non-IID Outputs**

For language models with sequential dependencies, we define the test statistic over response sequences:

$$T(f, f^*, C) \= \\frac{1}{n}\\sum\_{i=1}^n d\_{\\text{seq}}(f(c\_i), f^*(c\_i))$$

where $d\_{\\text{seq}}$ is a sequence-aware distance (e.g., edit distance, fuzzy hash similarity).

### **2.3 Empirical Bernstein Bounds**

For tighter bounds in low-variance settings, we use empirical Bernstein inequalities:

**Theorem 1 (Empirical Bernstein Bound):** Let $\\hat{\\sigma}^2$ be the sample variance of distances. With probability at least $1-\\delta$:

$$|T \- \\mathbb{E}\[T\]| \\leq \\sqrt{\\frac{2\\hat{\\sigma}^2 \\log(2/\\delta)}{n}} \+ \\frac{7B\\log(2/\\delta)}{3(n-1)}$$

where $B$ bounds the distance range.

*Proof:* Apply Maurer and Pontil's empirical Bernstein inequality \[2009\], which accounts for the empirical variance estimate.

This provides tighter bounds than Hoeffding when variance is small relative to range.

### **2.4 Sequential Testing with SPRT**

For efficient verification with minimal queries, we employ Sequential Probability Ratio Test:

**Algorithm 1: SPRT-based Verification**

Input: Models f, f\*, thresholds α (FAR), β (FRR)  
Output: Accept/Reject decision

1\. Set A \= log((1-β)/α), B \= log(β/(1-α))  
2\. Initialize S \= 0, n \= 0  
3\. While B \< S \< A:  
   \- Generate challenge c\_n  
   \- Observe d\_n \= d(f(c\_n), f\*(c\_n))  
   \- Update S \+= log(L\_1(d\_n)/L\_0(d\_n))  
   \- n \+= 1  
4\. Return Accept if S ≤ B, Reject if S ≥ A

**Theorem 2 (SPRT Sample Complexity):** Expected sample size: $$\\mathbb{E}\[N|H\_0\] \\approx \\frac{(1-\\alpha)\\log(\\beta/(1-\\alpha)) \+ \\alpha\\log((1-\\beta)/\\alpha)}{D\_{KL}(P\_0||P\_1)}$$

where $D\_{KL}$ is KL divergence between response distributions.

---

## **3\. Robust Distance Metrics**

### **3.1 Fuzzy Hashing for Language Models**

To handle tokenization variability and minor variations:

**Definition 1 (Token-Level Fuzzy Hash):** For token sequence $s \= \[t\_1, ..., t\_k\]$: $$H\_{\\text{fuzzy}}(s) \= {h(n\\text{-gram}) : n\\text{-gram} \\in s, n \\in {2,3,4}}$$

Similarity between sequences: $$\\text{sim}(s\_1, s\_2) \= \\frac{|H\_{\\text{fuzzy}}(s\_1) \\cap H\_{\\text{fuzzy}}(s\_2)|}{|H\_{\\text{fuzzy}}(s\_1) \\cup H\_{\\text{fuzzy}}(s\_2)|}$$

### **3.2 Robust Distance for Vision Models**

For continuous outputs with quantization effects: $$d\_{\\text{robust}}(y\_1, y\_2) \= \\begin{cases} 0 & \\text{if } ||y\_1 \- y\_2||\_\\infty \< \\epsilon \\ ||y\_1 \- y\_2||\_2 & \\text{otherwise} \\end{cases}$$

### **3.3 Handling Version Drift**

Define time-aware tolerance: $$\\tau(t) \= \\tau\_0 \+ \\lambda \\cdot t$$

where $\\lambda$ captures acceptable drift rate and $t$ is time since registration.

---

## **4\. Adversary Model**

### **4.1 Attack Taxonomy**

**Level 1: Output Manipulation**

* **Wrapper Attack**: Apply deterministic function $g$ such that $g \\circ f'$ mimics $f^\*$  
* **Canonicalization Attack**: Exploit quantization to hide differences

**Level 2: Model Modification**

* **Targeted Fine-tuning**: Modify $f^\*$ on distribution $D$ while preserving responses on suspected challenges  
* **Compression Attacks**: Apply pruning/quantization/LoRA maintaining challenge responses

**Level 3: Model Substitution**

* **Distillation**: Train $f'$ from limited queries to $f^\*$  
* **Architecture Substitution**: Different architecture with similar behavior

### **4.2 Wrapper Attack Analysis**

**Theorem 3 (Wrapper Detection):** For wrapper $g: \\mathcal{Y} \\rightarrow \\mathcal{Y}$, if challenges include compositions $c' \= h \\circ c$ for random perturbations $h$, then:

$$P(\\text{wrapper undetected}) \\leq \\exp(-\\Omega(n \\cdot \\text{entropy}(h)))$$

*Proof sketch:* Wrapper must correctly map $f'(h(c))$ to $f^\*(h(c))$ without knowing $h$, requiring exponential complexity in entropy of $h$.

### **4.3 Fine-tuning Resilience**

**Theorem 4 (Fine-tuning Separation):** For model fine-tuned on distribution $D$ disjoint from challenges $C$:

$$\\mathbb{E}\[d(f\_{\\text{tuned}}(C), f^*(C))\] \\geq \\epsilon \\cdot ||\\theta\_{\\text{tuned}} \- \\theta^*||\_2$$

where $\\epsilon$ depends on gradient alignment between $D$ and $C$.

---

## **5\. Challenge Design Theory**

### **5.1 Coverage-Separation Trade-off**

Define two objectives:

* **Coverage**: $\\text{Cov}(C) \= \\min\_{x \\in \\mathcal{X}} \\max\_{c \\in C} \\text{sim}(x, c)$  
* **Separation**: $\\text{Sep}(C) \= \\mathbb{E}\_{f \\neq f^*}\[d(f(C), f^*(C))\]$

**Theorem 5 (Coverage-Separation Bound):** For fixed budget $|C| \= n$: $$\\text{Cov}(C) \\cdot \\text{Sep}(C) \\leq O(n/\\text{dim}(\\mathcal{X}))$$

*Proof:* High coverage requires dense sampling, reducing available budget for separation-optimized challenges.

### **5.2 Optimal Challenge Construction**

**Algorithm 2: Coverage-Separation Balanced Design**

Input: Budget n, coverage weight α ∈ \[0,1\]  
Output: Challenge set C

1\. Partition budget: n\_cov \= αn, n\_sep \= (1-α)n  
2\. Coverage subset:  
   \- Solve k-center problem for n\_cov points  
   \- C\_cov \= k-center solution  
3\. Separation subset:  
   \- Identify decision boundaries via surrogate model  
   \- C\_sep \= boundary-adjacent samples  
4\. Return C \= C\_cov ∪ C\_sep

### **5.3 Active Challenge Selection**

For sequential verification, select challenges maximizing information gain:

$$c\_{n+1} \= \\arg\\max\_c I(f \\equiv f^\*; f(c) | f(c\_1), ..., f(c\_n))$$

where $I$ is mutual information.

---

## **6\. Practical Considerations**

### **6.1 Language Model Determinism Reality**

**Current LLM APIs provide:**

* Token-level outputs (not logits)  
* Version changes without notice  
* Hardware-dependent numerical variations  
* Tokenizer ambiguities

**Practical tolerance specifications:**

LM\_Tolerance:  
  token\_edit\_distance: ≤ 5% of sequence length  
  fuzzy\_hash\_similarity: ≥ 0.85  
  semantic\_embedding\_cosine: ≥ 0.9  
  version\_drift\_allowance: 0.02 per month

### **6.2 Challenge Governance**

**Algorithm 3: Cryptographic Challenge Derivation with Rotation**

Input: Master key k, epoch e, session s  
Output: Challenge set C

1\. k\_epoch \= KDF(k, "epoch" || e)  
2\. k\_session \= KDF(k\_epoch, "session" || s)  
3\. seed \= KDF(k\_session, "challenge")  
4\. C \= DeterministicSample(seed, challenge\_space)  
5\. Return C

**Leakage resilience:** With rolling epochs, compromise of old challenges doesn't affect future verifications.

---

## **7\. Comparative Analysis**

### **7.1 Comparison with Existing Approaches**

| Method | Training Modification | Query Complexity | Adversary Resistance | Practical Deployment |
| ----- | ----- | ----- | ----- | ----- |
| **PoT (Ours)** | None | O(log(1/ε)) | High | Easy |
| **Watermarking** | Required | O(1) | Medium | Hard (requires training) |
| **Fingerprinting** | None | O(n) | Low | Medium |
| **TEE Attestation** | None | O(1) | High | Hard (hardware required) |
| **ZK Proofs** | None | O(model size) | Perfect | Very Hard |

### **7.2 Theoretical Advantages**

**Versus Watermarking:**

* No training-time modification needed  
* Works on pre-existing models  
* Harder to remove via fine-tuning

**Versus Fingerprinting:**

* Statistical guarantees on error rates  
* Resilient to wrapper attacks  
* Lower query complexity via sequential testing

**Versus TEE/Attestation:**

* No trusted hardware required  
* Works across different deployment environments  
* Verifiable by any party with black-box access

### **7.3 Hybrid Approaches**

PoT can complement existing methods:

* **PoT \+ TEE**: Use TEE for initial attestation, PoT for continuous verification  
* **PoT \+ Watermark**: Watermark for ownership, PoT for deployment verification  
* **PoT \+ Audit logs**: Cryptographic logs of PoT verifications for compliance

---

## **8\. Implementation Guidelines**

### **8.1 Vision Model Verification**

class VisionVerifier:  
    def \_\_init\_\_(self, reference\_model, tolerance=1e-3):  
        self.reference \= reference\_model  
        self.tolerance \= tolerance  
          
    def verify(self, test\_model, n\_challenges=500):  
        challenges \= self.generate\_challenges(n\_challenges)  
        distances \= \[\]  
          
        for c in challenges:  
            ref\_out \= self.reference(c)  
            test\_out \= test\_model(c)  
            d \= robust\_distance(ref\_out, test\_out, self.tolerance)  
            distances.append(d)  
          
        \# Empirical Bernstein bound  
        mean\_d \= np.mean(distances)  
        var\_d \= np.var(distances, ddof=1)  
        bound \= empirical\_bernstein\_bound(mean\_d, var\_d, n\_challenges)  
          
        return bound \< self.threshold

### **8.2 Language Model Verification**

class LMVerifier:  
    def \_\_init\_\_(self, reference\_model):  
        self.reference \= reference\_model  
        self.fuzzy\_hasher \= FuzzyHasher(n\_gram\_sizes=\[2,3,4\])  
          
    def verify(self, test\_model, n\_challenges=200):  
        sprt \= SequentialTester(alpha=0.01, beta=0.01)  
          
        while not sprt.decided():  
            prompt \= self.generate\_semantic\_probe()  
              
            \# Generate with deterministic settings  
            ref\_tokens \= self.reference.generate(  
                prompt, temperature=0, max\_length=100  
            )  
            test\_tokens \= test\_model.generate(  
                prompt, temperature=0, max\_length=100  
            )  
              
            \# Fuzzy comparison  
            similarity \= self.fuzzy\_hasher.similarity(  
                ref\_tokens, test\_tokens  
            )  
              
            sprt.update(similarity)  
          
        return sprt.accept()

---

## **9\. Future Work**

### **9.1 Open Problems**

1. **Optimal challenge design** for specific model architectures  
2. **Formal analysis** of active learning for challenge selection  
3. **Cross-architecture verification** (verifying functionality across different architectures)  
4. **Privacy-preserving verification** with differential privacy guarantees

### **9.2 Extensions**

* **Multi-modal models**: Challenges spanning text, vision, audio  
* **Continual learning**: Verification under legitimate model updates  
* **Federated verification**: Distributed verification across multiple parties  
* **Quantum-resistant**: Post-quantum cryptographic challenge derivation

---

## **10\. Conclusion**

We presented PoT, a practical framework for black-box neural network verification that addresses real-world challenges including non-IID outputs, inherent nondeterminism, and sophisticated adversaries. By employing empirical Bernstein bounds, sequential testing, and fuzzy hashing, PoT provides robust verification with statistical guarantees. Our analysis of the coverage-separation trade-off provides principled guidance for challenge design, while our comprehensive adversary model addresses realistic attack scenarios.

PoT offers advantages over existing approaches by requiring no training-time modifications, providing statistical error guarantees, and remaining practical for large-scale deployment. As AI systems require increasing regulatory oversight, frameworks like PoT that balance security, practicality, and model confidentiality will be essential.

---

## **References**

\[Key papers: Maurer & Pontil 2009 (Empirical Bernstein), Wald 1945 (SPRT), Recent watermarking/fingerprinting papers, TEE attestation work\]

---

## **Appendix A: Detailed Proofs**

### **A.1 Proof of Theorem 1 (Empirical Bernstein)**

\[Complete derivation following Maurer & Pontil 2009\]

### **A.2 Proof of Theorem 3 (Wrapper Detection)**

\[Information-theoretic analysis of wrapper complexity\]

---

## **Appendix B: Challenge Templates**

### **B.1 Semantic Probe Families for LLMs**

\[Detailed templates for capability testing\]

### **B.2 Adversarial Boundary Probes for Vision**

\[Algorithms for finding decision boundaries\]

---

## **Appendix C: Experimental Protocols**

### **C.1 Evaluation Metrics**

\[ROC curves, DET plots, sequential testing traces\]

### **C.2 Attack Implementations**

\[Wrapper construction, fine-tuning procedures, distillation methods\]

