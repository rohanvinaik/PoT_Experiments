# Ollama Model Behavioral Verification Report

**Date:** 2026-01-11 20:28:11
**Total Tests:** 19
**Overall Accuracy:** 100.0%
**Total Time:** 51.1 minutes

## Summary by Category

| Category | Accuracy | Correct/Total |
|----------|----------|---------------|
| Self Consistency | 100.0% | 6/6 |
| Scale | 100.0% | 6/6 |
| Architecture | 100.0% | 7/7 |

## Detailed Results

| Model A | Model B | Decision | Relationship | Mean Effect | Time |
|---------|---------|----------|--------------|-------------|------|
| gemma2:2b | gemma2:2b | SAME | IDENTICAL | 0.0000 | 271.9s |
| qwen2.5:3b | qwen2.5:3b | SAME | NEAR_IDENTICAL | 0.0035 | 151.8s |
| llama3.2:3b | llama3.2:3b | SAME | IDENTICAL | 0.0004 | 136.6s |
| phi3:3.8b | phi3:3.8b | SAME | IDENTICAL | 0.0005 | 34.9s |
| deepseek-coder:6.7b | deepseek-coder:6.7b | SAME | IDENTICAL | 0.0000 | 42.1s |
| mistral:7b | mistral:7b | SAME | NEAR_IDENTICAL | 0.0076 | 47.2s |
| gemma2:2b | codegemma:7b | DIFFERENT | SAME_FAMILY_SCALE | 0.5232 | 322.9s |
| qwen2.5:3b | qwen2.5:7b | DIFFERENT | VERY_DIFFERENT | 3.5418 | 186.3s |
| llama3.2:3b | llama3.1:8b | UNDECIDED | SAME_FAMILY_SCALE | 0.3111 | 173.7s |
| deepseek-coder:6.7b | deepseek-r1:7b | DIFFERENT | SAME_FAMILY_VARIANT | 1.2198 | 143.0s |
| gemma2:2b | qwen2.5:3b | DIFFERENT | DIFFERENT_ARCHITECTURE | 12.4361 | 242.7s |
| gemma2:2b | llama3.2:3b | DIFFERENT | DIFFERENT_TRAINING | 0.6130 | 232.2s |
| gemma2:2b | phi3:3.8b | DIFFERENT | DIFFERENT_ARCHITECTURE | 13.2731 | 177.4s |
| qwen2.5:3b | llama3.2:3b | DIFFERENT | DIFFERENT_ARCHITECTURE | 12.5973 | 172.7s |
| qwen2.5:3b | phi3:3.8b | DIFFERENT | DIFFERENT_ARCHITECTURE | 4.7998 | 129.1s |
| llama3.2:3b | phi3:3.8b | DIFFERENT | DIFFERENT_ARCHITECTURE | 13.4124 | 120.4s |
| llama3.2:3b | codellama:7b | DIFFERENT | SAME_FAMILY_SCALE | 0.8359 | 134.9s |
| deepseek-coder:6.7b | ima/deepseek-math:latest | DIFFERENT | SAME_FAMILY_VARIANT | 0.7263 | 128.7s |
| ima/deepseek-math:latest | deepseek-r1:7b | DIFFERENT | SAME_FAMILY_VARIANT | 1.2073 | 216.2s |