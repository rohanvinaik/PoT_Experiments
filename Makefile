# PoT Experiments Makefile
# Comprehensive reproducible experiment targets for Proof-of-Training framework

# Configuration
IMAGE_NAME = pot-experiments
DATE = $(shell date +%Y%m%d_%H%M%S)
PYTHON = python
RESULTS_DIR = experimental_results
CONFIGS_DIR = configs
VALIDATION_REPORT = POT_PAPER_EXPERIMENTAL_VALIDATION_REPORT.md

# Colors for output
BOLD = \033[1m
GREEN = \033[32m
YELLOW = \033[33m
RED = \033[31m
BLUE = \033[34m
NC = \033[0m # No Color

# Time estimation constants
REPRODUCE_TEST_TIME = "~2 minutes"
REPRODUCE_FULL_TIME = "~45 minutes"
REPRODUCE_PAPER_TIME = "~2-3 hours"

.PHONY: build test clean help
.PHONY: reproduce reproduce-test reproduce-full reproduce-paper
.PHONY: validate-results check-dependencies install-deps
.PHONY: clean-results clean-all setup-env

# Default target
help:
	@echo "$(BOLD)PoT Experiments Makefile$(NC)"
	@echo "$(BLUE)================================$(NC)"
	@echo ""
	@echo "$(BOLD)Main Targets:$(NC)"
	@echo "  $(GREEN)reproduce$(NC)       - Run standard reproducible experiments ($(REPRODUCE_TEST_TIME))"
	@echo "  $(GREEN)reproduce-test$(NC)  - Quick test run with mock models ($(REPRODUCE_TEST_TIME))"
	@echo "  $(GREEN)reproduce-full$(NC)  - Full reproduction with real models ($(REPRODUCE_FULL_TIME))"
	@echo "  $(GREEN)reproduce-paper$(NC) - Exact paper reproduction protocol ($(REPRODUCE_PAPER_TIME))"
	@echo ""
	@echo "$(BOLD)Validation:$(NC)"
	@echo "  $(GREEN)validate-results$(NC) - Compare results against paper claims"
	@echo "  $(GREEN)check-deps$(NC)      - Check for required dependencies"
	@echo ""
	@echo "$(BOLD)Setup & Cleanup:$(NC)"
	@echo "  $(GREEN)setup-env$(NC)       - Setup experimental environment"
	@echo "  $(GREEN)install-deps$(NC)    - Install missing dependencies"
	@echo "  $(GREEN)clean-results$(NC)   - Clean experiment results"
	@echo "  $(GREEN)clean-all$(NC)       - Clean everything"
	@echo ""
	@echo "$(BOLD)Docker:$(NC)"
	@echo "  $(GREEN)build$(NC)           - Build Docker container"
	@echo "  $(GREEN)test$(NC)            - Run containerized tests"

# ================================
# MAIN REPRODUCTION TARGETS
# ================================

# Standard reproducible experiment (default to test mode for speed)
reproduce: check-dependencies clean-results
	@echo "$(BOLD)$(GREEN)Starting reproducible experiment run...$(NC)"
	@echo "$(YELLOW)Estimated time: $(REPRODUCE_TEST_TIME)$(NC)"
	@echo "$(BLUE)Output directory: $(RESULTS_DIR)/reproduction_$(DATE)$(NC)"
	@mkdir -p $(RESULTS_DIR)/reproduction_$(DATE)
	@echo "$(YELLOW)Step 1/4: Running core experiments...$(NC)"
	$(PYTHON) -m pot.experiments.reproducible_runner \
		--config $(CONFIGS_DIR)/vision_cifar10.yaml \
		--output $(RESULTS_DIR)/reproduction_$(DATE) \
		--trial-id reproduction_$(DATE) \
		--verbose
	@echo "$(YELLOW)Step 2/4: Calculating metrics...$(NC)"
	$(PYTHON) -m pot.experiments.metrics_calculator \
		--input $(RESULTS_DIR)/reproduction_$(DATE) \
		--output $(RESULTS_DIR)/reproduction_$(DATE)/metrics_report.json \
		--compare-paper-claims
	@echo "$(YELLOW)Step 3/4: Generating sequential analysis...$(NC)"
	$(PYTHON) -m pot.experiments.sequential_decision \
		--input $(RESULTS_DIR)/reproduction_$(DATE) \
		--output $(RESULTS_DIR)/reproduction_$(DATE)/sequential_analysis.json
	@echo "$(YELLOW)Step 4/4: Creating final report...$(NC)"
	$(PYTHON) scripts/generate_reproduction_report.py \
		--input $(RESULTS_DIR)/reproduction_$(DATE) \
		--format all \
		--output $(RESULTS_DIR)/reproduction_$(DATE)/report
	@echo "$(BOLD)$(GREEN)✅ Results saved to $(RESULTS_DIR)/reproduction_$(DATE)$(NC)"
	@$(MAKE) --no-print-directory _show_results DIR=$(RESULTS_DIR)/reproduction_$(DATE)

# Quick test with mock models
reproduce-test: check-dependencies clean-results
	@echo "$(BOLD)$(BLUE)Quick test run with mock models$(NC)"
	@echo "$(YELLOW)Estimated time: $(REPRODUCE_TEST_TIME)$(NC)"
	@mkdir -p $(RESULTS_DIR)/test_$(DATE)
	@echo "$(YELLOW)Running lightweight test experiments...$(NC)"
	@$(PYTHON) -c "\
import sys; sys.path.append('.'); \
from pot.experiments.reproducible_runner import ReproducibleExperimentRunner, ExperimentConfig; \
from pot.experiments.metrics_calculator import create_metrics_calculator, calculate_all_metrics; \
from pot.experiments.sequential_decision import create_sequential_decision_maker; \
import numpy as np; \
import json; \
from pathlib import Path; \
config = ExperimentConfig( \
    experiment_name='test_reproduction', \
    model_type='vision', \
    model_architecture='test', \
    challenge_families=['vision:freq'], \
    n_challenges_per_family=5, \
    alpha=0.05, \
    beta=0.05, \
    tau_id=0.01, \
    output_dir='$(RESULTS_DIR)/test_$(DATE)', \
    verbose=True \
); \
print('🧪 Running quick reproduction test...'); \
runner = ReproducibleExperimentRunner(config); \
results = runner.run_experiment(); \
print('📊 Calculating metrics...'); \
calculator = create_metrics_calculator(); \
if results: \
    predictions = np.array([r.verified for r in results]); \
    labels = np.array([r.ground_truth for r in results]); \
    stopping_times = [r.stopping_time for r in results]; \
    metrics = calculate_all_metrics(predictions, labels, stopping_times, 20, calculator); \
    output_dir = Path('$(RESULTS_DIR)/test_$(DATE)'); \
    with open(output_dir / 'test_metrics.json', 'w') as f: \
        json.dump({name: result.value for name, result in metrics.items()}, f, indent=2); \
    print(f'✅ Test completed successfully! Results in {output_dir}'); \
    for name, result in metrics.items(): \
        print(f'  {name.upper()}: {result.value:.4f}'); \
else: \
    print('❌ Test failed - no results generated'); \
"
	@echo "$(BOLD)$(GREEN)✅ Quick test completed!$(NC)"

# Full reproduction with real models
reproduce-full: check-dependencies check-models clean-results
	@echo "$(BOLD)$(BLUE)Full reproduction with real models$(NC)"
	@echo "$(YELLOW)Estimated time: $(REPRODUCE_FULL_TIME)$(NC)"
	@echo "$(RED)Warning: This will download models and use significant resources$(NC)"
	@mkdir -p $(RESULTS_DIR)/full_$(DATE)
	@echo "$(YELLOW)Step 1/5: Setting up models...$(NC)"
	@$(PYTHON) -c "\
from pot.experiments.model_setup import MinimalModelSetup; \
setup = MinimalModelSetup(); \
print('📦 Setting up vision model...'); \
vision_model = setup.get_vision_model(setup.create_config('vision', 'minimal')); \
print('📦 Setting up language model...'); \
language_model = setup.get_language_model(setup.create_config('language', 'minimal')); \
print('✅ Models ready'); \
"
	@echo "$(YELLOW)Step 2/5: Running vision experiments...$(NC)"
	$(PYTHON) -m pot.experiments.reproducible_runner \
		--config $(CONFIGS_DIR)/vision_cifar10.yaml \
		--output $(RESULTS_DIR)/full_$(DATE) \
		--trial-id full_vision_$(DATE) \
		--model-architecture minimal \
		--verbose
	@echo "$(YELLOW)Step 3/5: Running language experiments...$(NC)"
	$(PYTHON) -m pot.experiments.reproducible_runner \
		--config $(CONFIGS_DIR)/lm_medium.yaml \
		--output $(RESULTS_DIR)/full_$(DATE) \
		--trial-id full_language_$(DATE) \
		--model-architecture minimal \
		--verbose
	@echo "$(YELLOW)Step 4/5: Comprehensive metrics analysis...$(NC)"
	$(PYTHON) scripts/analyze_full_reproduction.py \
		--input $(RESULTS_DIR)/full_$(DATE) \
		--output $(RESULTS_DIR)/full_$(DATE)/analysis
	@echo "$(YELLOW)Step 5/5: Generating comprehensive report...$(NC)"
	$(PYTHON) scripts/generate_reproduction_report.py \
		--input $(RESULTS_DIR)/full_$(DATE) \
		--format all \
		--template full \
		--output $(RESULTS_DIR)/full_$(DATE)/report
	@echo "$(BOLD)$(GREEN)✅ Full reproduction completed!$(NC)"
	@$(MAKE) --no-print-directory _show_results DIR=$(RESULTS_DIR)/full_$(DATE)

# Exact paper reproduction protocol
reproduce-paper: check-dependencies check-models clean-results
	@echo "$(BOLD)$(RED)EXACT PAPER REPRODUCTION PROTOCOL$(NC)"
	@echo "$(YELLOW)Estimated time: $(REPRODUCE_PAPER_TIME)$(NC)"
	@echo "$(RED)Warning: This will run the complete E1-E7 experimental protocol$(NC)"
	@read -p "Continue? [y/N] " -n 1 -r; \
	if [[ ! $$REPLY =~ ^[Yy]$$ ]]; then \
		echo ""; echo "$(YELLOW)Cancelled$(NC)"; exit 1; \
	fi
	@echo ""
	@mkdir -p $(RESULTS_DIR)/paper_$(DATE)
	@echo "$(YELLOW)Running complete experimental protocol E1-E7...$(NC)"
	@echo "$(BLUE)This reproduces the exact experiments from the paper$(NC)"
	$(PYTHON) run_full_experimental_protocol.py \
		--output $(RESULTS_DIR)/paper_$(DATE) \
		--deterministic \
		--verbose
	@echo "$(YELLOW)Generating paper-style analysis...$(NC)"
	$(PYTHON) scripts/paper_analysis.py \
		--input $(RESULTS_DIR)/paper_$(DATE) \
		--reference $(VALIDATION_REPORT) \
		--output $(RESULTS_DIR)/paper_$(DATE)/paper_comparison.md
	@echo "$(BOLD)$(GREEN)✅ Paper reproduction protocol completed!$(NC)"
	@$(MAKE) --no-print-directory _show_results DIR=$(RESULTS_DIR)/paper_$(DATE)

# ================================
# VALIDATION TARGETS
# ================================

validate-results: check-dependencies
	@echo "$(BOLD)$(BLUE)Validating experimental results$(NC)"
	@if [ ! -d "$(RESULTS_DIR)" ]; then \
		echo "$(RED)❌ No results directory found. Run reproduce first.$(NC)"; \
		exit 1; \
	fi
	@LATEST=$$(ls -td $(RESULTS_DIR)/*/ 2>/dev/null | head -1); \
	if [ -z "$$LATEST" ]; then \
		echo "$(RED)❌ No experimental results found$(NC)"; \
		exit 1; \
	fi; \
	echo "$(YELLOW)Validating latest results: $$LATEST$(NC)"; \
	$(PYTHON) scripts/validate_results.py \
		--claimed $(VALIDATION_REPORT) \
		--actual "$$LATEST" \
		--output validation_report_$(DATE).html \
		--format html \
		--verbose
	@echo "$(BOLD)$(GREEN)✅ Validation report generated: validation_report_$(DATE).html$(NC)"

# ================================
# DEPENDENCY MANAGEMENT
# ================================

check-dependencies:
	@echo "$(YELLOW)Checking dependencies...$(NC)"
	@$(PYTHON) scripts/check_dependencies.py

check-models:
	@echo "$(YELLOW)Checking model availability...$(NC)"
	@$(PYTHON) scripts/check_models.py

install-deps:
	@echo "$(YELLOW)Installing dependencies...$(NC)"
	pip install -r requirements.txt
	@echo "$(GREEN)✅ Dependencies installed$(NC)"

setup-env:
	@echo "$(YELLOW)Setting up experimental environment...$(NC)"
	@mkdir -p $(RESULTS_DIR)
	@mkdir -p $(RESULTS_DIR)/logs
	@mkdir -p $(RESULTS_DIR)/plots
	@mkdir -p $(RESULTS_DIR)/reports
	@echo "$(GREEN)✅ Environment ready$(NC)"

# ================================
# CLEANUP TARGETS
# ================================

clean-results:
	@echo "$(YELLOW)Cleaning previous results...$(NC)"
	@if [ -d "$(RESULTS_DIR)" ]; then \
		find $(RESULTS_DIR) -name "reproduction_*" -type d -mtime +7 -exec rm -rf {} + 2>/dev/null || true; \
		find $(RESULTS_DIR) -name "test_*" -type d -mtime +1 -exec rm -rf {} + 2>/dev/null || true; \
		echo "$(GREEN)✅ Old results cleaned$(NC)"; \
	else \
		echo "$(BLUE)No results to clean$(NC)"; \
	fi

clean-all: clean-results
	@echo "$(YELLOW)Cleaning all generated files...$(NC)"
	@rm -rf $(RESULTS_DIR)/reproduction_*
	@rm -rf $(RESULTS_DIR)/test_*
	@rm -rf $(RESULTS_DIR)/full_*
	@rm -rf $(RESULTS_DIR)/paper_*
	@rm -f validation_report_*.html
	@rm -f *.log
	@find . -name "*.pyc" -delete 2>/dev/null || true
	@find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	@echo "$(GREEN)✅ All clean$(NC)"

# ================================
# DOCKER TARGETS
# ================================

build:
	@echo "$(YELLOW)Building Docker container...$(NC)"
	docker build -t $(IMAGE_NAME) .
	@echo "$(GREEN)✅ Container built: $(IMAGE_NAME)$(NC)"

test: build
@echo "$(YELLOW)Running containerized tests...$(NC)"
docker run --rm $(IMAGE_NAME) pytest tests -q
@echo "$(GREEN)✅ Tests completed$(NC)"

# ================================
# HELPER TARGETS
# ================================

# Internal target to show results summary
_show_results:
	@echo ""
	@echo "$(BOLD)$(BLUE)📊 Results Summary$(NC)"
	@echo "$(BLUE)==================$(NC)"
	@if [ -f "$(DIR)/metrics_report.json" ]; then \
		echo "$(GREEN)Metrics Report:$(NC) $(DIR)/metrics_report.json"; \
	fi
	@if [ -f "$(DIR)/sequential_analysis.json" ]; then \
		echo "$(GREEN)Sequential Analysis:$(NC) $(DIR)/sequential_analysis.json"; \
	fi
	@if [ -d "$(DIR)/report" ]; then \
		echo "$(GREEN)Full Report:$(NC) $(DIR)/report/"; \
	fi
	@if [ -f "$(DIR)/analysis/summary.json" ]; then \
		echo "$(GREEN)Analysis Summary:$(NC) $(DIR)/analysis/summary.json"; \
	fi
	@echo ""
	@echo "$(YELLOW)View results with:$(NC)"
	@echo "  ls -la $(DIR)/"
	@if [ -f "$(DIR)/report/index.html" ]; then \
		echo "  open $(DIR)/report/index.html"; \
	fi

# Status check
status:
	@echo "$(BOLD)$(BLUE)PoT Experiments Status$(NC)"
	@echo "$(BLUE)======================$(NC)"
	@echo "$(GREEN)Project Directory:$(NC) $(PWD)"
	@echo "$(GREEN)Results Directory:$(NC) $(RESULTS_DIR)"
	@if [ -d "$(RESULTS_DIR)" ]; then \
		LATEST=$$(ls -td $(RESULTS_DIR)/*/ 2>/dev/null | head -1); \
		if [ -n "$$LATEST" ]; then \
			echo "$(GREEN)Latest Results:$(NC) $$LATEST"; \
		else \
			echo "$(YELLOW)No results found$(NC)"; \
		fi; \
	else \
		echo "$(YELLOW)No results directory$(NC)"; \
	fi
	@$(MAKE) --no-print-directory check-dependencies

# Advanced targets for specific use cases
reproduce-vision: check-dependencies
	@echo "$(BLUE)Vision-only reproduction$(NC)"
	@mkdir -p $(RESULTS_DIR)/vision_$(DATE)
	$(PYTHON) -m pot.experiments.reproducible_runner \
		--config $(CONFIGS_DIR)/vision_cifar10.yaml \
		--output $(RESULTS_DIR)/vision_$(DATE) \
		--verbose

reproduce-language: check-dependencies
	@echo "$(BLUE)Language-only reproduction$(NC)"
	@mkdir -p $(RESULTS_DIR)/language_$(DATE)
	$(PYTHON) -m pot.experiments.reproducible_runner \
		--config $(CONFIGS_DIR)/lm_medium.yaml \
		--output $(RESULTS_DIR)/language_$(DATE) \
		--verbose

# Debugging targets
debug-setup:
	@echo "$(YELLOW)Debug: Checking setup$(NC)"
	@echo "Python: $$(which $(PYTHON))"
	@echo "Version: $$($(PYTHON) --version)"
	@echo "Working directory: $(PWD)"
	@echo "Results directory: $(RESULTS_DIR)"
	@$(PYTHON) -c "import pot; print(f'PoT package location: {pot.__file__}')"
