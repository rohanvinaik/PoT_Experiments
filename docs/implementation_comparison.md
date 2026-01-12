# PoT Implementation Comparison: Original vs Current

## Original Implementation (pot_runner.py)

### Key Characteristics:
1. **Simple and Direct**
   - Single file (~400 lines)
   - Minimal abstraction
   - Direct flow: generate seeds → generate prompts → load models → run test → save results

2. **Challenge Generation**
   ```python
   # All prompts generated upfront
   seeds = _gen_seeds(key_hex, run_id, n_challenges)
   prompts = [_prompt_from_seed(s) for s in seeds]
   ```
   - Simple seed-to-prompt mapping
   - All prompts held in memory as strings
   - Passed directly to tester

3. **Model Loading**
   ```python
   ref = make_model(exp_cfg["ref"], "ref", robust)
   cand = make_model(exp_cfg["cand"], "cand", robust)
   ```
   - Models loaded once
   - Simple wrapper functions for generation

4. **Testing**
   ```python
   tester = EnhancedSequentialTester(Mode)
   result = tester.run(prompts=prompts, ref_generate=ref_gen, cand_generate=cand_gen)
   ```
   - Single tester.run() call
   - Tester iterates through prompts internally
   - Early stopping when decision reached

5. **Results**
   - Simple JSON files (transcript, summary, metrics)
   - Bundled into ZIP file
   - No HTML reports or complex visualization

## Current Implementation (e2e_pipeline.py)

### Key Characteristics:
1. **Complex Pipeline Architecture**
   - Multiple stages with metrics tracking
   - ~700+ lines with extensive abstraction
   - Stage-by-stage orchestration

2. **Challenge Generation Evolution**
   
   **Yesterday's Refactor:**
   ```python
   # Separate stage for challenge generation
   challenges = self.generate_challenges(pre_commit['seeds'])
   # Then passed to verification
   verification = self.run_verification(ref_model, cand_model, challenges)
   ```
   
   **Today's Fix (Memory Issue):**
   ```python
   # Generate prompts upfront but minimal memory
   for i, seed in enumerate(challenge_seeds):
       # Generate challenge
       prompts.append(challenge.parameters.get("prompt", ""))
       challenge_metadata.append(minimal_info)
   ```

3. **Model Loading**
   - Separate stage with metrics
   - Support for API/local/hybrid modes
   - Memory tracking

4. **Testing**
   - Uses DifferenceVerifier wrapper
   - Complex scoring functions
   - CI progression tracking
   - Multiple verification modes

5. **Results**
   - Evidence bundles with cryptographic hashes
   - HTML reports via ReportGenerator
   - Stage metrics and performance tracking
   - JSON summaries with detailed metadata

## Key Differences

| Aspect | Original (pot_runner) | Current (e2e_pipeline) |
|--------|----------------------|------------------------|
| **Complexity** | Simple, direct | Multi-stage pipeline |
| **Lines of Code** | ~400 | ~700+ |
| **Memory Usage** | Minimal (strings only) | Higher (tracking metadata) |
| **Challenge Generation** | All upfront, simple | Was: separate stage, Now: upfront but optimized |
| **Reporting** | Basic JSON/ZIP | HTML reports, visualizations |
| **Metrics** | Basic (verdict, n_used) | Comprehensive stage tracking |
| **Abstraction** | Minimal | High (multiple classes/stages) |
| **Error Handling** | Basic | Per-stage with recovery |

## Memory Issue Root Cause

The memory issue occurred because:
1. **Original**: Generated simple prompt strings upfront
2. **Yesterday's Refactor**: Generated full Challenge objects with metadata in separate stage
3. **Today's Fix**: Generate prompts upfront (like original) but store minimal metadata

## Recommendation

The original pot_runner approach is actually more efficient:
- Generates all prompts as simple strings upfront
- Passes them to tester which stops early
- Minimal memory overhead
- Simple and direct

The e2e_pipeline adds value through:
- Better reporting and visualization
- Stage-by-stage metrics
- Evidence bundle generation
- But at the cost of complexity and memory usage
