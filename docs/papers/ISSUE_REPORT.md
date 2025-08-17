# Code Analysis Report - PoT_Experiments
*Generated: 2025-08-15T18:45:28.493750*

## Executive Summary
- **Total Issues Found**: 98
- **Fixable Issues**: 98 (100.0%)
- **Risk Score**: 22.16
- **Affected Modules**: 25
- **Analysis Time**: 0.00s

## Issues by Category

### LLM-Generated Artifacts (42 issues)
**experimental_report.py**
- Line 449: Suspiciously uniform pattern: alphabetical
- Line 60: Suspiciously low entropy in text (likely synthetic)
- Line 121: Suspiciously low entropy in text (likely synthetic)
- Line 177: Suspiciously low entropy in text (likely synthetic)
- Line 249: Suspiciously low entropy in text (likely synthetic)
- ... and 6 more

**experimental_report_final.py**
- Line 85: Suspiciously low entropy in text (likely synthetic)
- Line 198: Suspiciously low entropy in text (likely synthetic)
- Line 241: Lorem ipsum or placeholder text detected
- Line 276: Suspiciously low entropy in text (likely synthetic)
- Line 337: Suspiciously low entropy in text (likely synthetic)
- ... and 4 more

### Placeholder & Stub Functions (24 issues)
**comprehensive_validation.py**
- Line 160: Function 'validate_kdf_governance' implies logic but has complexity 1 in ComprehensiveValidator.validate_kdf_governance
- Line 219: Function 'validate_fuzzy_hashing' implies logic but has complexity 1 in ComprehensiveValidator.validate_fuzzy_hashing
- Line 617: Function 'validate_challenge_generation' implies logic but has complexity 1 in ComprehensiveValidator.validate_challenge_generation

**pot/core/governance.py**
- Line 67: Function 'verify_commitment' implies logic but has complexity 1 in verify_commitment

**pot/eval/baselines.py**
- Line 7: P1_FUNCTIONAL: lightweight_fingerprint
- Line 7: Function contains only 'pass' in lightweight_fingerprint

**pot/eval/plots.py**
- Line 16: P1_FUNCTIONAL: plot_det_curve
- Line 16: Function contains only 'pass' in plot_det_curve

**pot/lm/verifier.py**
- Line 114: Function 'compute_output_distance' implies logic but has complexity 1 in LMVerifier.compute_output_distance
- Line 284: Function 'verify_with_time_tolerance' implies logic but has complexity 1 in LMVerifier.verify_with_time_tolerance

**pot/security/fuzzy_hash_verifier.py**
- Line 110: P0_SECURITY: FuzzyHasher.generate_hash (Security risk: crypto_primitives)
- Line 115: P0_SECURITY: FuzzyHasher.compare (Security risk: crypto_primitives)
- Line 368: Function 'verify_model_output' implies logic but has complexity 1 in FuzzyHashVerifier.verify_model_output

**pot/security/training_provenance_auditor.py**
- Line 297: P0_SECURITY: BlockchainClient.store_hash (Security risk: crypto_primitives)
- Line 302: P0_SECURITY: BlockchainClient.retrieve_hash (Security risk: crypto_primitives)
- Line 307: P0_SECURITY: BlockchainClient.verify_hash (Security risk: crypto_verify)
- Line 70: Function 'calculate_hash' implies logic but has complexity 1 in TrainingEvent.calculate_hash

**pot/vision/models.py**
- Line 17: P1_FUNCTIONAL: VisionModel.get_features

**pot/vision/probes.py**
- Line 3: P1_FUNCTIONAL: render_sine_grating
- Line 8: P1_FUNCTIONAL: render_texture

### Duplicate & Semantic Duplicates (20 issues)
**comprehensive_validation.py**
- Line 24: Exact hypervector match: print_test identical to print_section in experimental_report_final.py

**experimental_report.py**
- Line 34: Semantic duplicate: print_header similar to print_header in experimental_report_final.py (similarity: 0.88)
- Line 34: Exact hypervector match: print_header identical to print_header in experimental_report_final.py
- Line 431: Exact hypervector match: generate_summary identical to generate_summary in experimental_report_fixed.py

**experimental_report_final.py**
- Line 57: Exact duplicate: print_section identical to print_section in experimental_report_fixed.py
- Line 57: Exact hypervector match: print_section identical to print_section in experimental_report_fixed.py
- Line 47: Exact hypervector match: print_header identical to print_header in experimental_report_fixed.py

**pot/core/wrapper_detection.py**
- Line 392: Semantic duplicate (HV): adaptive_adversary similar to wrapper_adversary in wrapper_detection.py (similarity: 0.87)

**pot/lm/attacks.py**
- Line 9: Exact duplicate: limited_distillation identical to limited_distillation in attacks.py
- Line 5: Exact duplicate: targeted_finetune identical to targeted_finetune in attacks.py

**pot/security/fuzzy_hash_verifier.py**
- Line 134: Exact hypervector match: __init__ identical to __init__ in fuzzy_hash_verifier.py
- Line 138: Exact hypervector match: generate_hash identical to generate_hash in fuzzy_hash_verifier.py
- Line 156: Exact hypervector match: generate_hash identical to generate_hash in fuzzy_hash_verifier.py
- Line 597: Exact hypervector match: __repr__ identical to __repr__ in training_provenance_auditor.py

**pot/security/test_fuzzy_verifier.py**
- Line 17: Exact hypervector match: test_basic_verification identical to test_error_handling in test_fuzzy_verifier.py
- Line 351: Exact hypervector match: run_all_tests identical to run_all_tests in test_provenance_auditor.py

**pot/security/test_provenance_auditor.py**
- Line 599: Exact hypervector match: run_all_tests identical to run_all_tests in test_token_normalizer.py

**pot/security/test_token_normalizer.py**
- Line 222: Exact hypervector match: test_deterministic_mode identical to test_semantic_similarity in test_token_normalizer.py

**pot/vision/models.py**
- Line 62: Exact hypervector match: apply_pruning identical to apply_quantization in models.py

**scripts/run_generate_reference.py**
- Line 19: Exact hypervector match: main identical to main in run_verify.py

### Missing & Undefined Symbols (11 issues)
**comprehensive_validation.py**
- Line 16: Reference to undefined symbol '__file__'

**experimental_report.py**
- Line 16: Reference to undefined symbol '__file__'

**experimental_report_final.py**
- Line 20: Reference to undefined symbol '__file__'

**experimental_report_fixed.py**
- Line 19: Reference to undefined symbol '__file__'

**experimental_results/stress_test.py**
- Line 11: Reference to undefined symbol '__file__'

**experimental_results/validation_experiment.py**
- Line 13: Reference to undefined symbol '__file__'

**scripts/run_attack.py**
- Line 15: Reference to undefined symbol '__file__'

**scripts/run_generate_reference.py**
- Line 12: Reference to undefined symbol '__file__'

**scripts/run_grid.py**
- Line 19: Reference to undefined symbol '__file__'

**scripts/run_plots.py**
- Line 18: Reference to undefined symbol '__file__'

**scripts/run_verify.py**
- Line 17: Reference to undefined symbol '__file__'

### Code Quality & Complexity (1 issues)
**experimental_results/stress_test.py**
- Line 29: High coupling risk between MockModel.__init__ and MockModel.forward

## Most Affected Files

1. **experimental_report_final.py**: 20 issues
1. **experimental_report.py**: 15 issues
1. **experimental_report_fixed.py**: 13 issues
1. **pot/security/fuzzy_hash_verifier.py**: 7 issues
1. **comprehensive_validation.py**: 5 issues
1. **pot/security/training_provenance_auditor.py**: 5 issues
1. **pot/vision/probes.py**: 4 issues
1. **pot/vision/models.py**: 3 issues
1. **pot/lm/attacks.py**: 2 issues
1. **experimental_results/stress_test.py**: 2 issues

## Priority Actions

### High Priority
1. Fix duplicate and placeholder functions
2. Remove unused imports and clean up import structure
3. Consolidate semantic duplicates

### Medium Priority
1. Refactor context window thrashing patterns
2. Implement missing function stubs
3. Clean up LLM-generated filler content

### Low Priority
1. Improve documentation and docstrings
2. Optimize code organization
3. Add comprehensive test coverage

## Next Steps

1. Review this report to understand issue patterns
2. Use `--generate-fixes` to create automated fix scripts
3. Start with high-priority issues for maximum impact
4. Run analysis regularly to track progress

---
*Analysis performed on: /Users/rohanvinaik/PoT_Experiments*
*TailChasingFixer Version: 1.0*