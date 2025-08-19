# CRITICAL INSTRUCTIONS FOR CLAUDE - READ FIRST

## NEVER CREATE MOCK TESTS

When the user asks for Google Colab code or any test runners:

1. **USE THE ACTUAL POT FRAMEWORK** - The codebase contains real verification algorithms in:
   - `pot/core/` - Core verification logic
   - `pot/security/` - Security components (fuzzy hash, provenance)
   - `pot/lm/` - Language model verification
   - `scripts/` - Actual test scripts that run the framework

2. **DO NOT CREATE SIMPLIFIED/MOCK VERSIONS** - The user needs to verify paper claims with real tests:
   - Statistical identity verification must use `pot.core.diff_decision.SequentialDiffTester`
   - LLM verification must actually load and test models
   - Fuzzy hashing must use real algorithms (TLSH, SSDEEP)
   - Provenance must build actual Merkle trees

3. **THE TESTS MUST BE COMPREHENSIVE** - They should:
   - Take several minutes to run, not seconds
   - Generate detailed metrics and confidence intervals
   - Save results to `experimental_results/` with real data
   - Use the actual PoT framework classes and methods

4. **USE ONLY OPEN MODELS**:
   - GPT-2 and DistilGPT-2 only
   - NO Mistral, Zephyr, or any gated models
   - NO authentication tokens required

## EXISTING WORKING SCRIPTS

The following scripts in `scripts/` are the REAL tests that should be run:
- `run_statistical_verification.py` - Statistical identity with confidence intervals
- `test_llm_verification.py` - LLM verification (updated to use GPT-2/DistilGPT-2)
- `run_fuzzy_verification.py` - Fuzzy hash testing
- `run_provenance_verification.py` - Merkle tree provenance
- `experimental_report_clean.py` - Clean reporting format

## FOR GOOGLE COLAB

When creating Colab runners:
1. Clone the repository
2. Install dependencies: torch, transformers, numpy, scipy, scikit-learn
3. Run the ACTUAL scripts from `scripts/` directory
4. DO NOT create new test logic - use what exists in the codebase
5. The tests should take 2-5 minutes total, not seconds

## REMEMBER

The user is validating academic paper claims. Mock tests are USELESS for this purpose. Always use the real PoT framework code that exists in this repository.