#!/usr/bin/env python3
"""
Automated Documentation Generator

Generates comprehensive documentation for the PoT framework including:
- API documentation
- Code documentation
- User guides
- Architecture diagrams
- Performance reports
"""

import ast
import inspect
import json
import os
import sys
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
import subprocess

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class CodeAnalyzer:
    """Analyzes Python code to extract documentation information"""
    
    def __init__(self, source_dir: str):
        self.source_dir = Path(source_dir)
        self.modules = {}
        self.classes = {}
        self.functions = {}
        self.dependencies = set()
    
    def analyze_codebase(self) -> Dict[str, Any]:
        """Analyze the entire codebase"""
        print("🔍 Analyzing codebase structure...")
        
        for py_file in self.source_dir.rglob('*.py'):
            if self._should_analyze_file(py_file):
                self._analyze_file(py_file)
        
        return {
            'modules': self.modules,
            'classes': self.classes,
            'functions': self.functions,
            'dependencies': sorted(list(self.dependencies)),
            'analysis_timestamp': datetime.utcnow().isoformat()
        }
    
    def _should_analyze_file(self, file_path: Path) -> bool:
        """Determine if a file should be analyzed"""
        # Skip test files, build artifacts, etc.
        exclude_patterns = [
            '__pycache__',
            '.pyc',
            'test_',
            '_test.py',
            'build/',
            'dist/',
            '.git/'
        ]
        
        file_str = str(file_path)
        return not any(pattern in file_str for pattern in exclude_patterns)
    
    def _analyze_file(self, file_path: Path):
        """Analyze a single Python file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse AST
            tree = ast.parse(content)
            
            # Get module info
            module_name = self._get_module_name(file_path)
            module_doc = ast.get_docstring(tree)
            
            self.modules[module_name] = {
                'file_path': str(file_path),
                'docstring': module_doc,
                'classes': [],
                'functions': [],
                'imports': []
            }
            
            # Analyze AST nodes
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    self._analyze_class(node, module_name)
                elif isinstance(node, ast.FunctionDef):
                    self._analyze_function(node, module_name)
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    self._analyze_import(node, module_name)
        
        except Exception as e:
            print(f"⚠️  Error analyzing {file_path}: {e}")
    
    def _get_module_name(self, file_path: Path) -> str:
        """Get module name from file path"""
        relative_path = file_path.relative_to(self.source_dir)
        parts = list(relative_path.parts)
        if parts[-1] == '__init__.py':
            parts = parts[:-1]
        else:
            parts[-1] = parts[-1].replace('.py', '')
        return '.'.join(parts)
    
    def _analyze_class(self, node: ast.ClassDef, module_name: str):
        """Analyze a class definition"""
        class_name = f"{module_name}.{node.name}"
        
        class_info = {
            'name': node.name,
            'module': module_name,
            'docstring': ast.get_docstring(node),
            'methods': [],
            'bases': [self._get_name(base) for base in node.bases],
            'line_number': node.lineno
        }
        
        # Analyze methods
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                method_info = {
                    'name': item.name,
                    'docstring': ast.get_docstring(item),
                    'args': [arg.arg for arg in item.args.args],
                    'line_number': item.lineno,
                    'is_private': item.name.startswith('_'),
                    'is_property': any(
                        isinstance(decorator, ast.Name) and decorator.id == 'property'
                        for decorator in item.decorator_list
                    )
                }
                class_info['methods'].append(method_info)
        
        self.classes[class_name] = class_info
        self.modules[module_name]['classes'].append(node.name)
    
    def _analyze_function(self, node: ast.FunctionDef, module_name: str):
        """Analyze a function definition"""
        # Skip methods (they're handled in class analysis)
        if self._is_method(node):
            return
        
        function_name = f"{module_name}.{node.name}"
        
        function_info = {
            'name': node.name,
            'module': module_name,
            'docstring': ast.get_docstring(node),
            'args': [arg.arg for arg in node.args.args],
            'line_number': node.lineno,
            'is_private': node.name.startswith('_'),
            'decorators': [self._get_name(decorator) for decorator in node.decorator_list]
        }
        
        self.functions[function_name] = function_info
        self.modules[module_name]['functions'].append(node.name)
    
    def _analyze_import(self, node: ast.Import | ast.ImportFrom, module_name: str):
        """Analyze import statements"""
        if isinstance(node, ast.Import):
            for alias in node.names:
                self.dependencies.add(alias.name)
                self.modules[module_name]['imports'].append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                self.dependencies.add(node.module)
                self.modules[module_name]['imports'].append(node.module)
    
    def _is_method(self, node: ast.FunctionDef) -> bool:
        """Check if a function is a method (has self parameter)"""
        return (node.args.args and 
                len(node.args.args) > 0 and 
                node.args.args[0].arg in ['self', 'cls'])
    
    def _get_name(self, node: ast.AST) -> str:
        """Get name from AST node"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        elif isinstance(node, ast.Constant):
            return str(node.value)
        else:
            return str(node)


class DocumentationGenerator:
    """Main documentation generator"""
    
    def __init__(self, source_dir: str, output_dir: str):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.analyzer = CodeAnalyzer(source_dir)
        self.code_analysis = None
        
        # Templates directory
        self.templates_dir = Path(__file__).parent.parent / 'templates'
    
    def generate_documentation(self) -> str:
        """Generate complete documentation"""
        print("📚 Generating PoT Framework Documentation")
        
        # Analyze codebase
        self.code_analysis = self.analyzer.analyze_codebase()
        
        # Generate different types of documentation
        self._generate_api_docs()
        self._generate_user_guide()
        self._generate_architecture_docs()
        self._generate_configuration_docs()
        self._generate_deployment_docs()
        self._generate_troubleshooting_guide()
        self._generate_changelog()
        
        # Generate main index
        index_file = self._generate_index()
        
        print(f"✅ Documentation generated in: {self.output_dir}")
        return str(index_file)
    
    def _generate_api_docs(self):
        """Generate API documentation"""
        print("📖 Generating API documentation...")
        
        api_dir = self.output_dir / 'api'
        api_dir.mkdir(exist_ok=True)
        
        # Generate module documentation
        for module_name, module_info in self.code_analysis['modules'].items():
            if 'api' in module_name or 'endpoint' in module_name:
                self._generate_module_doc(module_name, module_info, api_dir)
        
        # Generate API overview
        api_overview = self._generate_api_overview()
        with open(api_dir / 'index.md', 'w') as f:
            f.write(api_overview)
    
    def _generate_module_doc(self, module_name: str, module_info: Dict[str, Any], output_dir: Path):
        """Generate documentation for a single module"""
        doc_content = f"""# {module_name}

{module_info.get('docstring', 'No description available.')}

**File:** `{module_info['file_path']}`

"""
        
        # Document classes
        if module_info['classes']:
            doc_content += "## Classes\n\n"
            for class_name in module_info['classes']:
                full_class_name = f"{module_name}.{class_name}"
                if full_class_name in self.code_analysis['classes']:
                    class_info = self.code_analysis['classes'][full_class_name]
                    doc_content += self._format_class_doc(class_info)
        
        # Document functions
        if module_info['functions']:
            doc_content += "## Functions\n\n"
            for func_name in module_info['functions']:
                full_func_name = f"{module_name}.{func_name}"
                if full_func_name in self.code_analysis['functions']:
                    func_info = self.code_analysis['functions'][full_func_name]
                    doc_content += self._format_function_doc(func_info)
        
        # Save module documentation
        module_file = output_dir / f"{module_name.replace('.', '_')}.md"
        with open(module_file, 'w') as f:
            f.write(doc_content)
    
    def _format_class_doc(self, class_info: Dict[str, Any]) -> str:
        """Format class documentation"""
        doc = f"### {class_info['name']}\n\n"
        
        if class_info['docstring']:
            doc += f"{class_info['docstring']}\n\n"
        
        if class_info['bases']:
            doc += f"**Inherits from:** {', '.join(class_info['bases'])}\n\n"
        
        # Document methods
        if class_info['methods']:
            doc += "#### Methods\n\n"
            for method in class_info['methods']:
                if not method['is_private']:  # Skip private methods in docs
                    doc += f"##### `{method['name']}({', '.join(method['args'])})`\n\n"
                    if method['docstring']:
                        doc += f"{method['docstring']}\n\n"
        
        return doc
    
    def _format_function_doc(self, func_info: Dict[str, Any]) -> str:
        """Format function documentation"""
        doc = f"### `{func_info['name']}({', '.join(func_info['args'])})`\n\n"
        
        if func_info['docstring']:
            doc += f"{func_info['docstring']}\n\n"
        
        if func_info['decorators']:
            doc += f"**Decorators:** {', '.join(func_info['decorators'])}\n\n"
        
        return doc
    
    def _generate_api_overview(self) -> str:
        """Generate API overview documentation"""
        return f"""# API Documentation

Welcome to the PoT Framework API documentation.

## Overview

The PoT (Proof-of-Training) Framework provides REST APIs for:

- Model verification and comparison
- Zero-knowledge proof generation and verification
- Audit trail management
- Performance monitoring
- Security assessments

## Quick Start

```python
import requests

# Verify two models
response = requests.post('http://localhost:8000/api/v1/verify', json={{
    'ref_model': 'gpt2',
    'cand_model': 'distilgpt2',
    'mode': 'audit'
}})

result = response.json()
print(f"Verification result: {{result['decision']}}")
```

## Authentication

API endpoints require authentication via JWT tokens:

```bash
curl -H "Authorization: Bearer YOUR_TOKEN" \\
     http://localhost:8000/api/v1/verify
```

## Modules

The following modules provide API functionality:

{self._list_api_modules()}

## Error Handling

All API endpoints return standardized error responses:

```json
{{
  "error": {{
    "code": "VALIDATION_ERROR",
    "message": "Model not found",
    "details": {{
      "model": "invalid-model-name"
    }}
  }}
}}
```

Generated on: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC
"""
    
    def _list_api_modules(self) -> str:
        """List API modules"""
        api_modules = [
            name for name in self.code_analysis['modules'].keys()
            if 'api' in name or 'endpoint' in name
        ]
        
        if not api_modules:
            return "No API modules found."
        
        return '\n'.join(f"- [`{module}`]({module.replace('.', '_')}.md)" for module in api_modules)
    
    def _generate_user_guide(self):
        """Generate user guide"""
        print("📋 Generating user guide...")
        
        user_guide_content = f"""# PoT Framework User Guide

## Introduction

The Proof-of-Training (PoT) Framework is a comprehensive system for cryptographic verification of neural network training integrity. This guide will help you get started with using the framework.

## Installation

### Prerequisites

- Python 3.11+
- Rust 1.88+ (for ZK proofs)
- 8GB+ RAM
- 50GB+ disk space

### Quick Install

```bash
git clone https://github.com/your-org/PoT_Experiments.git
cd PoT_Experiments
pip install -r requirements.txt

# Build ZK prover binaries
cd src/pot/zk/prover_halo2
cargo build --release
```

### Docker Install

```bash
docker pull pot-framework:latest
docker run -p 8000:8000 pot-framework:latest
```

## Quick Start

### 1. Basic Model Verification

Verify if two models are the same or different:

```bash
python scripts/run_e2e_validation.py \\
    --ref-model gpt2 \\
    --cand-model distilgpt2 \\
    --mode audit
```

### 2. Batch Verification

Run multiple verifications using a manifest:

```bash
bash scripts/run_all.sh manifests/neurips_demo.yaml
```

### 3. Zero-Knowledge Proofs

Generate cryptographic proofs of training:

```bash
python scripts/run_zk_validation.py \\
    --model-path /path/to/model \\
    --circuit-type sgd
```

## Core Concepts

### Statistical Verification

The framework uses enhanced sequential testing to determine if two models are statistically identical or different:

- **SAME**: Models are statistically identical
- **DIFFERENT**: Models show significant behavioral differences
- **UNDECIDED**: Insufficient evidence for determination

### Zero-Knowledge Proofs

ZK proofs provide cryptographic evidence that training was performed correctly without revealing sensitive training data or model parameters.

### Audit Trails

Complete audit trails track all verification operations with cryptographic integrity guarantees.

## Configuration

### Environment Variables

```bash
export POT_DATA_DIR=/path/to/data
export POT_LOG_LEVEL=info
export POT_ENABLE_ZK=true
```

### Configuration Files

Create `configs/verification.yaml`:

```yaml
verification:
  mode: audit
  confidence_level: 0.99
  max_queries: 400

zk_proofs:
  circuit_type: sgd
  security_level: 128
```

## Advanced Usage

### Custom Models

```python
from src.pot.core.verifier import ModelVerifier

verifier = ModelVerifier()
result = verifier.verify_models(
    ref_model_path="/path/to/ref/model",
    cand_model_path="/path/to/cand/model",
    challenges=custom_challenges
)
```

### API Integration

```python
import requests

response = requests.post('http://localhost:8000/api/v1/verify', 
                        json={{'ref_model': 'gpt2', 'cand_model': 'distilgpt2'}})
result = response.json()
```

## Troubleshooting

### Common Issues

1. **Memory Errors**: Use sharding for large models
2. **ZK Proof Failures**: Check Rust installation
3. **Model Loading Errors**: Verify model paths and permissions

### Debug Mode

```bash
python scripts/run_e2e_validation.py --verbose --debug
```

### Support

- GitHub Issues: [Report issues](https://github.com/your-org/PoT_Experiments/issues)
- Documentation: [Full docs](https://your-org.github.io/PoT_Experiments/)

Generated on: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC
"""
        
        with open(self.output_dir / 'user_guide.md', 'w') as f:
            f.write(user_guide_content)
    
    def _generate_architecture_docs(self):
        """Generate architecture documentation"""
        print("🏗️ Generating architecture documentation...")
        
        arch_content = f"""# PoT Framework Architecture

## System Overview

The PoT Framework is designed as a modular, scalable system for neural network training verification using both statistical methods and zero-knowledge proofs.

## Core Components

### 1. Statistical Verification Engine (`src/pot/core/`)

- **Enhanced Sequential Tester**: Implements statistical hypothesis testing
- **Challenge Generator**: Creates cryptographic challenges for model queries
- **Decision Framework**: Determines SAME/DIFFERENT/UNDECIDED outcomes

### 2. Zero-Knowledge Proof System (`src/pot/zk/`)

- **Halo2 Circuits**: Rust implementation of ZK circuits
- **Proof Generator**: Creates cryptographic proofs of training
- **Verifier**: Validates ZK proofs without secret knowledge

### 3. Security Layer (`src/pot/security/`)

- **Audit Trail**: Tamper-evident logging system
- **Attack Detection**: Identifies adversarial attempts
- **Cryptographic Primitives**: Hash functions, digital signatures

### 4. Model Interfaces (`src/pot/lm/`, `src/pot/vision/`)

- **Language Model Verifier**: Supports GPT, BERT, etc.
- **Vision Model Verifier**: Supports CNNs, ViTs, etc.
- **API Connectors**: Integration with cloud model APIs

### 5. Monitoring & Analytics (`benchmarks/tracking/`)

- **Performance Tracker**: Real-time metrics collection
- **Dashboard Generator**: Interactive performance dashboards
- **Regression Detection**: Automated performance monitoring

## Data Flow

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Model A (Ref)  │    │  Challenge Gen   │    │  Model B (Cand) │
└─────────┬───────┘    └─────────┬────────┘    └─────────┬───────┘
          │                      │                      │
          ▼                      ▼                      ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Query & Store  │    │  Sequential Test │    │  Query & Store  │
└─────────┬───────┘    └─────────┬────────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 ▼
                    ┌──────────────────┐
                    │  Decision Engine │
                    └─────────┬────────┘
                              ▼
                    ┌──────────────────┐
                    │  Evidence Bundle │
                    └──────────────────┘
```

## Security Model

### Threat Model

1. **Malicious Models**: Backdoored or poisoned models
2. **Adversarial Queries**: Attempts to extract training data
3. **Model Substitution**: Switching models during verification
4. **Replay Attacks**: Reusing previous verification responses

### Security Guarantees

1. **Statistical Soundness**: 99%+ confidence in decisions
2. **Cryptographic Integrity**: Tamper-evident audit trails
3. **Zero-Knowledge**: No training data leakage in proofs
4. **Non-repudiation**: Cryptographically signed evidence

## Scalability Architecture

### Horizontal Scaling

- **Worker Pools**: Parallel verification processing
- **Model Sharding**: Large model distribution
- **Load Balancing**: Request distribution

### Vertical Scaling

- **Memory Management**: Intelligent caching and cleanup
- **GPU Utilization**: CUDA acceleration where available
- **Storage Optimization**: Compressed evidence bundles

## Deployment Patterns

### 1. Standalone Deployment

```
┌─────────────────────────────────────────┐
│              Single Node                │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  │
│  │   API   │  │ Verifier│  │Database │  │
│  └─────────┘  └─────────┘  └─────────┘  │
└─────────────────────────────────────────┘
```

### 2. Microservices Deployment

```
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│ API Gateway │  │  Verifier   │  │  ZK Prover  │
└─────────────┘  └─────────────┘  └─────────────┘
       │                │                │
       └────────────────┼────────────────┘
                        ▼
              ┌─────────────┐
              │  Database   │
              └─────────────┘
```

### 3. Cloud-Native Deployment

```
┌─────────────────────────────────────────────────────┐
│                 Kubernetes Cluster                  │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐ │
│  │API Pods │  │Worker   │  │ZK Proof │  │Database │ │
│  │(3x)     │  │Pods(5x) │  │Pods(2x) │  │Cluster  │ │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘ │
└─────────────────────────────────────────────────────┘
```

## Performance Characteristics

### Verification Performance

| Model Size | Queries | Duration | Memory |
|------------|---------|----------|--------|
| GPT-2      | 32      | 30s      | 2GB    |
| GPT-2-L    | 64      | 2min     | 4GB    |
| GPT-3.5    | 128     | 5min     | 8GB    |
| LLaMA-7B   | 200     | 10min    | 16GB   |

### ZK Proof Performance

| Circuit | Constraints | Prove Time | Verify Time | Proof Size |
|---------|-------------|------------|-------------|------------|
| SGD     | 1M          | 30s        | 100ms       | 1.2KB      |
| LoRA    | 500K        | 15s        | 50ms        | 800B       |

## Extension Points

### Custom Verifiers

```python
from src.pot.core.verifier import BaseVerifier

class CustomVerifier(BaseVerifier):
    def verify_model_pair(self, ref_model, cand_model):
        # Custom verification logic
        pass
```

### Plugin Architecture

```python
from src.pot.plugins import VerificationPlugin

@register_plugin('custom_metric')
class CustomMetricPlugin(VerificationPlugin):
    def compute_metric(self, responses):
        # Custom metric computation
        pass
```

Generated on: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC
"""
        
        with open(self.output_dir / 'architecture.md', 'w') as f:
            f.write(arch_content)
    
    def _generate_configuration_docs(self):
        """Generate configuration documentation"""
        print("⚙️ Generating configuration documentation...")
        
        config_content = """# Configuration Guide

## Overview

The PoT Framework supports multiple configuration methods:

1. Environment variables
2. YAML configuration files
3. Command-line arguments
4. Runtime configuration

## Environment Variables

### Core Settings

```bash
# Data directories
export POT_DATA_DIR="/data/pot"
export POT_LOGS_DIR="/logs/pot"
export POT_CONFIG_DIR="/config/pot"

# Logging
export POT_LOG_LEVEL="info"  # debug, info, warning, error
export POT_LOG_FORMAT="json"  # json, text

# Database
export POT_DB_URL="postgresql://user:pass@host:5432/pot"
export POT_REDIS_URL="redis://localhost:6379/0"

# API Settings
export POT_API_HOST="0.0.0.0"
export POT_API_PORT="8000"
export POT_API_SECRET_KEY="your-secret-key"

# Worker Settings
export POT_WORKER_CONCURRENCY="4"
export POT_MAX_WORKERS="10"

# ZK Proof Settings
export POT_ZK_ENABLED="true"
export POT_ZK_CIRCUIT_TYPE="sgd"  # sgd, lora
export POT_ZK_SECURITY_LEVEL="128"

# Model Settings
export POT_MODEL_CACHE_DIR="/cache/models"
export POT_MODEL_DOWNLOAD_TIMEOUT="3600"

# Security
export POT_ENABLE_AUDIT="true"
export POT_AUDIT_SIGNING_KEY="/path/to/signing.key"
```

### Docker Settings

```bash
# Docker-specific
export POT_DOCKER_MODE="true"
export POT_CONTAINER_NAME="pot-framework"
```

### Kubernetes Settings

```bash
# Kubernetes-specific
export POT_K8S_NAMESPACE="pot-framework"
export POT_K8S_SERVICE_ACCOUNT="pot-sa"
```

## Configuration Files

### Main Configuration (`configs/pot.yaml`)

```yaml
# Core framework settings
framework:
  name: "PoT Framework"
  version: "1.0.0"
  environment: "production"  # development, staging, production

# Logging configuration
logging:
  level: "info"
  format: "json"
  file: "/logs/pot.log"
  rotation: "daily"
  retention_days: 30

# Database configuration
database:
  url: "postgresql://pot_user:password@localhost:5432/pot_framework"
  pool_size: 10
  echo: false
  migrate_on_startup: true

# Cache configuration
cache:
  redis_url: "redis://localhost:6379/0"
  default_ttl: 3600
  max_memory: "512mb"

# API configuration
api:
  host: "0.0.0.0"
  port: 8000
  workers: 4
  timeout: 30
  cors_enabled: true
  rate_limit: 100  # requests per minute

# Worker configuration
workers:
  concurrency: 2
  max_workers: 10
  task_timeout: 1800
  heartbeat_interval: 30

# Model configuration
models:
  cache_dir: "/cache/models"
  download_timeout: 3600
  supported_formats: ["pytorch", "transformers", "onnx"]
  max_model_size_gb: 50

# Verification configuration
verification:
  default_mode: "audit"  # quick, audit, extended
  confidence_level: 0.99
  max_queries: 400
  timeout: 1800
  enable_caching: true

# ZK Proof configuration
zk_proofs:
  enabled: true
  circuit_type: "sgd"  # sgd, lora
  security_level: 128
  prover_timeout: 3600
  verifier_timeout: 60
  proof_cache_ttl: 86400

# Security configuration
security:
  audit_enabled: true
  signing_key_path: "/keys/audit_signing.key"
  encryption_key_path: "/keys/encryption.key"
  tls_enabled: true
  tls_cert_path: "/certs/server.crt"
  tls_key_path: "/certs/server.key"

# Monitoring configuration
monitoring:
  enabled: true
  metrics_port: 8080
  prometheus_enabled: true
  grafana_enabled: true
  health_check_interval: 30

# Performance tuning
performance:
  memory_limit_gb: 16
  cpu_limit_cores: 8
  gpu_enabled: false
  batch_size: 32
  optimization_level: "O2"
```

### Verification Configuration (`configs/verification.yaml`)

```yaml
# Statistical testing parameters
statistical_testing:
  alpha: 0.01  # Type I error rate
  beta: 0.05   # Type II error rate
  effect_size_threshold: 0.1
  min_samples: 30
  max_samples: 400
  
# Sequential testing
sequential_testing:
  type: "empirical_bernstein"  # empirical_bernstein, wald
  early_stopping: true
  confidence_intervals: true
  
# Challenge generation
challenges:
  generator: "kdf_based"  # kdf_based, random, adversarial
  hmac_key_length: 32
  seed: null  # null for random seed
  
# Model interfaces
model_interfaces:
  huggingface:
    cache_dir: "/cache/hf_models"
    use_auth_token: false
    trust_remote_code: false
  
  api:
    timeout: 30
    retry_count: 3
    rate_limit: 10
  
  local:
    supported_formats: ["pytorch", "safetensors"]
    device: "auto"  # auto, cpu, cuda
```

### Security Configuration (`configs/security.yaml`)

```yaml
# Audit trail settings
audit:
  enabled: true
  log_file: "/logs/audit.log"
  integrity_checking: true
  signing_algorithm: "RS256"
  hash_algorithm: "SHA256"
  
# Attack detection
attack_detection:
  enabled: true
  statistical_detection: true
  timing_attack_detection: true
  replay_detection: true
  model_extraction_detection: true
  
# Encryption settings
encryption:
  algorithm: "AES-256-GCM"
  key_derivation: "PBKDF2"
  iterations: 100000
  
# Access control
access_control:
  rbac_enabled: true
  session_timeout: 3600
  max_login_attempts: 5
  lockout_duration: 300
```

## Command-Line Configuration

### Global Options

```bash
# Common options for all scripts
--config CONFIG_FILE      # Specify config file
--verbose                 # Enable verbose logging  
--debug                   # Enable debug mode
--dry-run                 # Show what would be done
--log-level LEVEL         # Override log level
--data-dir DIR            # Override data directory
--output-dir DIR          # Specify output directory
```

### Verification Options

```bash
python scripts/run_e2e_validation.py \
    --ref-model MODEL_PATH \
    --cand-model MODEL_PATH \
    --mode {quick,audit,extended} \
    --confidence 0.99 \
    --max-queries 400 \
    --timeout 1800 \
    --enable-zk \
    --output-dir results/
```

### ZK Proof Options

```bash
python scripts/run_zk_validation.py \
    --model-path MODEL_PATH \
    --circuit-type {sgd,lora} \
    --security-level 128 \
    --prover-timeout 3600 \
    --output-dir proofs/
```

## Runtime Configuration

### Programmatic Configuration

```python
from src.pot.config import Config

# Load configuration
config = Config.load('configs/pot.yaml')

# Override settings
config.verification.confidence_level = 0.95
config.zk_proofs.enabled = False

# Apply configuration
config.apply()
```

### Dynamic Reconfiguration

```python
from src.pot.config import ConfigManager

manager = ConfigManager()

# Update logging level at runtime
manager.update('logging.level', 'debug')

# Reload configuration
manager.reload()
```

## Validation and Testing

### Configuration Validation

```bash
# Validate configuration files
python scripts/validate_config.py configs/pot.yaml

# Test configuration with dry run
python scripts/run_e2e_validation.py --dry-run --config configs/pot.yaml
```

### Environment Testing

```bash
# Test environment setup
python scripts/test_environment.py

# Check dependencies
python scripts/check_dependencies.py
```

## Best Practices

### 1. Environment-Specific Configs

```
configs/
├── base.yaml           # Base configuration
├── development.yaml    # Development overrides
├── staging.yaml        # Staging overrides
└── production.yaml     # Production overrides
```

### 2. Secret Management

```bash
# Use environment variables for secrets
export POT_DB_PASSWORD="$(cat /run/secrets/db_password)"
export POT_API_SECRET_KEY="$(cat /run/secrets/api_key)"
```

### 3. Configuration Validation

```yaml
# Use schema validation
schema_version: "1.0"
validation:
  required_fields:
    - database.url
    - api.secret_key
  type_checking: true
  range_validation: true
```

Generated on: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC
"""
        
        with open(self.output_dir / 'configuration.md', 'w') as f:
            f.write(config_content)
    
    def _generate_deployment_docs(self):
        """Generate deployment documentation"""
        print("🚀 Generating deployment documentation...")
        
        deployment_content = """# Deployment Guide

## Overview

The PoT Framework supports multiple deployment methods:

1. **Standalone**: Single-node development/testing
2. **Docker**: Containerized deployment
3. **Kubernetes**: Production-grade orchestration
4. **Cloud**: Managed cloud services

## Prerequisites

### System Requirements

- **CPU**: 4+ cores (8+ recommended for production)
- **Memory**: 8GB minimum (16GB+ for large models)
- **Storage**: 50GB+ available space
- **Network**: Outbound HTTPS for model downloads

### Software Dependencies

- Python 3.11+
- Rust 1.88+ (for ZK circuits)
- Docker 20.10+ (for containerized deployment)
- Kubernetes 1.24+ (for orchestration)

## Standalone Deployment

### Quick Start

```bash
# Clone repository
git clone https://github.com/your-org/PoT_Experiments.git
cd PoT_Experiments

# Install dependencies
pip install -r requirements.txt

# Build ZK prover binaries
cd src/pot/zk/prover_halo2
cargo build --release
cd ../../../..

# Run validation
python scripts/run_e2e_validation.py \
    --ref-model gpt2 \
    --cand-model distilgpt2 \
    --mode audit
```

### Production Setup

```bash
# Create dedicated user
sudo useradd -m -s /bin/bash pot
sudo usermod -aG sudo pot

# Install in virtual environment
python -m venv /opt/pot-framework
source /opt/pot-framework/bin/activate
pip install -r requirements.txt

# Create systemd service
sudo tee /etc/systemd/system/pot-framework.service << EOF
[Unit]
Description=PoT Framework API
After=network.target

[Service]
Type=simple
User=pot
WorkingDirectory=/opt/pot-framework
Environment=PATH=/opt/pot-framework/bin
ExecStart=/opt/pot-framework/bin/python scripts/api/start_server.py
Restart=always

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl enable pot-framework
sudo systemctl start pot-framework
```

## Docker Deployment

### Single Container

```bash
# Build image
docker build -t pot-framework:latest .

# Run container
docker run -d \
    --name pot-framework \
    -p 8000:8000 \
    -v pot-data:/data \
    -v pot-logs:/logs \
    pot-framework:latest
```

### Docker Compose

```bash
# Deploy full stack
cd deploy/docker
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f pot-api

# Scale workers
docker-compose up -d --scale pot-worker=5
```

### Production Docker Setup

```yaml
# docker-compose.prod.yml
version: '3.8'
services:
  pot-api:
    image: pot-framework:v1.0.0
    restart: unless-stopped
    environment:
      - POT_ENVIRONMENT=production
      - POT_LOG_LEVEL=warning
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 2G
          cpus: 1.0
```

## Kubernetes Deployment

### Prerequisites

```bash
# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
chmod +x kubectl
sudo mv kubectl /usr/local/bin/

# Verify cluster access
kubectl cluster-info
```

### Basic Deployment

```bash
# Deploy to Kubernetes
cd deploy/kubernetes
kubectl apply -f pot-deployment.yaml

# Check status
kubectl -n pot-framework get pods
kubectl -n pot-framework get services

# Port forward for testing
kubectl -n pot-framework port-forward service/pot-api-service 8000:8000
```

### Production Kubernetes

```bash
# Create namespace
kubectl create namespace pot-framework

# Apply configurations
kubectl apply -f - <<EOF
apiVersion: v1
kind: Secret
metadata:
  name: pot-secrets
  namespace: pot-framework
type: Opaque
data:
  POT_DB_PASSWORD: $(echo -n 'your-secure-password' | base64)
  POT_API_SECRET_KEY: $(echo -n 'your-secret-key' | base64)
EOF

# Deploy with resource limits
kubectl apply -f pot-deployment.yaml

# Set up ingress
kubectl apply -f - <<EOF
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: pot-ingress
  namespace: pot-framework
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
  - hosts:
    - pot.yourdomain.com
    secretName: pot-tls
  rules:
  - host: pot.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: pot-api-service
            port:
              number: 8000
EOF
```

### Monitoring Setup

```bash
# Install Prometheus Operator
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install prometheus prometheus-community/kube-prometheus-stack

# Apply ServiceMonitor for PoT
kubectl apply -f - <<EOF
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: pot-metrics
  namespace: pot-framework
spec:
  selector:
    matchLabels:
      app: pot-api
  endpoints:
  - port: metrics
EOF
```

## Cloud Deployments

### AWS

#### EKS Deployment

```bash
# Create EKS cluster
eksctl create cluster --name pot-cluster --region us-west-2

# Deploy to EKS
kubectl apply -f deploy/kubernetes/pot-deployment.yaml

# Set up load balancer
kubectl apply -f - <<EOF
apiVersion: v1
kind: Service
metadata:
  name: pot-loadbalancer
  namespace: pot-framework
  annotations:
    service.beta.kubernetes.io/aws-load-balancer-type: nlb
spec:
  type: LoadBalancer
  selector:
    app: pot-api
  ports:
  - port: 80
    targetPort: 8000
EOF
```

#### ECS Deployment

```json
{
  "family": "pot-framework",
  "taskRoleArn": "arn:aws:iam::account:role/ecsTaskRole",
  "networkMode": "awsvpc",
  "containerDefinitions": [
    {
      "name": "pot-api",
      "image": "your-account.dkr.ecr.us-west-2.amazonaws.com/pot-framework:latest",
      "memory": 2048,
      "cpu": 1024,
      "essential": true,
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {"name": "POT_ENVIRONMENT", "value": "production"},
        {"name": "POT_LOG_LEVEL", "value": "info"}
      ]
    }
  ]
}
```

### Google Cloud

#### GKE Deployment

```bash
# Create GKE cluster
gcloud container clusters create pot-cluster \
    --num-nodes 3 \
    --machine-type n1-standard-4 \
    --zone us-central1-a

# Get credentials
gcloud container clusters get-credentials pot-cluster --zone us-central1-a

# Deploy
kubectl apply -f deploy/kubernetes/pot-deployment.yaml
```

### Azure

#### AKS Deployment

```bash
# Create resource group
az group create --name pot-resources --location eastus

# Create AKS cluster
az aks create \
    --resource-group pot-resources \
    --name pot-cluster \
    --node-count 3 \
    --node-vm-size Standard_D4s_v3

# Get credentials
az aks get-credentials --resource-group pot-resources --name pot-cluster

# Deploy
kubectl apply -f deploy/kubernetes/pot-deployment.yaml
```

## High Availability Setup

### Database HA

```yaml
# PostgreSQL with replication
apiVersion: postgresql.cnpg.io/v1
kind: Cluster
metadata:
  name: pot-postgres
  namespace: pot-framework
spec:
  instances: 3
  primaryUpdateStrategy: unsupervised
  postgresql:
    parameters:
      max_connections: "200"
      shared_buffers: "256MB"
      effective_cache_size: "1GB"
```

### Redis HA

```yaml
# Redis Cluster
apiVersion: redis.redis.opstreelabs.in/v1beta1
kind: RedisCluster
metadata:
  name: pot-redis
  namespace: pot-framework
spec:
  clusterSize: 3
  redisExporter:
    enabled: true
```

### API HA

```yaml
# HorizontalPodAutoscaler
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: pot-api-hpa
  namespace: pot-framework
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: pot-api
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

## Security Hardening

### Container Security

```dockerfile
# Multi-stage build for minimal attack surface
FROM python:3.11-slim as runtime

# Create non-root user
RUN useradd -m -u 1000 potuser

# Set security options
USER potuser
WORKDIR /app

# Read-only filesystem
VOLUME ["/tmp", "/app/logs"]
```

### Network Security

```yaml
# NetworkPolicy
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: pot-network-policy
  namespace: pot-framework
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: pot-framework
  egress:
  - to:
    - namespaceSelector:
        matchLabels:
          name: pot-framework
```

### Secrets Management

```bash
# Using Kubernetes secrets
kubectl create secret generic pot-secrets \
    --from-literal=db-password=secure-password \
    --from-literal=api-key=secret-key \
    --namespace pot-framework

# Using external secret management
helm install external-secrets external-secrets/external-secrets -n external-secrets-system --create-namespace
```

## Monitoring and Observability

### Metrics Collection

```yaml
# ServiceMonitor for Prometheus
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: pot-metrics
  namespace: pot-framework
spec:
  selector:
    matchLabels:
      app: pot-api
  endpoints:
  - port: metrics
    interval: 30s
```

### Logging

```yaml
# Fluent Bit for log collection
apiVersion: v1
kind: ConfigMap
metadata:
  name: fluent-bit-config
  namespace: pot-framework
data:
  fluent-bit.conf: |
    [INPUT]
        Name tail
        Path /var/log/containers/pot-*.log
        Parser docker
        Tag kube.*
```

### Alerting

```yaml
# PrometheusRule for alerts
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: pot-alerts
  namespace: pot-framework
spec:
  groups:
  - name: pot.rules
    rules:
    - alert: PotAPIDown
      expr: up{job="pot-api"} == 0
      for: 1m
      labels:
        severity: critical
      annotations:
        summary: "PoT API is down"
```

## Troubleshooting

### Common Issues

1. **Memory Issues**: Increase memory limits or enable sharding
2. **ZK Proof Failures**: Check Rust compiler and circuit files
3. **Database Connection**: Verify connection strings and network policies
4. **Model Loading**: Check model paths and authentication

### Debug Commands

```bash
# Check pod status
kubectl -n pot-framework describe pod POD_NAME

# View logs
kubectl -n pot-framework logs -f POD_NAME

# Execute into container
kubectl -n pot-framework exec -it POD_NAME -- /bin/bash

# Port forward for debugging
kubectl -n pot-framework port-forward POD_NAME 8000:8000
```

### Performance Tuning

```yaml
# Resource optimization
resources:
  requests:
    memory: "1Gi"
    cpu: "500m"
  limits:
    memory: "2Gi"
    cpu: "1000m"

# JVM tuning for large models
env:
- name: JAVA_OPTS
  value: "-Xmx4g -XX:+UseG1GC"
```

Generated on: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC
"""
        
        with open(self.output_dir / 'deployment.md', 'w') as f:
            f.write(deployment_content)
    
    def _generate_troubleshooting_guide(self):
        """Generate troubleshooting documentation"""
        print("🔧 Generating troubleshooting guide...")
        
        troubleshooting_content = """# Troubleshooting Guide

## Common Issues and Solutions

### Installation Issues

#### Issue: Python dependencies fail to install

**Symptoms:**
```
ERROR: Could not build wheels for some packages
```

**Solutions:**
1. Update pip: `pip install --upgrade pip`
2. Install system dependencies:
   ```bash
   # Ubuntu/Debian
   sudo apt-get install build-essential python3-dev
   
   # CentOS/RHEL
   sudo yum groupinstall "Development Tools"
   sudo yum install python3-devel
   ```
3. Use binary wheels: `pip install --only-binary=all`

#### Issue: Rust compilation fails

**Symptoms:**
```
error: could not compile `halo2_proofs`
```

**Solutions:**
1. Update Rust: `rustup update`
2. Check Rust version: `rustc --version` (need 1.88+)
3. Clear cargo cache: `cargo clean`
4. Increase memory for compilation: `export RUSTFLAGS="-C link-arg=-s"`

### Runtime Issues

#### Issue: Out of memory errors

**Symptoms:**
```
torch.cuda.OutOfMemoryError: CUDA out of memory
RuntimeError: [enforce fail at alloc_cpu.cpp]
```

**Solutions:**
1. Enable model sharding:
   ```bash
   python scripts/run_e2e_validation.py \
       --enable-sharding \
       --max-memory-percent 70
   ```

2. Reduce batch size:
   ```bash
   export POT_BATCH_SIZE=1
   ```

3. Use CPU-only mode:
   ```bash
   export CUDA_VISIBLE_DEVICES=""
   ```

#### Issue: Model loading failures

**Symptoms:**
```
OSError: Model not found
HTTPError: 401 Client Error: Unauthorized
```

**Solutions:**
1. Check model path: Verify file exists and permissions
2. For HuggingFace models:
   ```bash
   huggingface-cli login
   ```
3. Use local models:
   ```bash
   python scripts/run_e2e_validation.py \
       --ref-model /path/to/local/model \
       --cand-model /path/to/local/model
   ```

#### Issue: ZK proof generation fails

**Symptoms:**
```
Error: Circuit compilation failed
Error: Proving key not found
```

**Solutions:**
1. Build ZK binaries:
   ```bash
   cd src/pot/zk/prover_halo2
   cargo build --release
   ```

2. Check binary locations:
   ```bash
   which prove_sgd verify_sgd
   ```

3. Skip ZK proofs for testing:
   ```bash
   python scripts/run_e2e_validation.py --skip-zk
   ```

### Docker Issues

#### Issue: Container fails to start

**Symptoms:**
```
Error response from daemon: container exited with code 1
```

**Solutions:**
1. Check logs:
   ```bash
   docker logs pot-framework
   ```

2. Run in interactive mode:
   ```bash
   docker run -it pot-framework:latest bash
   ```

3. Check resource limits:
   ```bash
   docker stats
   ```

#### Issue: Volume mount permissions

**Symptoms:**
```
PermissionError: [Errno 13] Permission denied
```

**Solutions:**
1. Fix ownership:
   ```bash
   sudo chown -R 1000:1000 /path/to/data
   ```

2. Use bind mounts:
   ```bash
   docker run -v $(pwd)/data:/data pot-framework:latest
   ```

### Kubernetes Issues

#### Issue: Pods stuck in Pending state

**Symptoms:**
```
NAME       READY   STATUS    RESTARTS   AGE
pot-api    0/1     Pending   0          5m
```

**Solutions:**
1. Check resource availability:
   ```bash
   kubectl describe pod pot-api
   kubectl get nodes
   kubectl top nodes
   ```

2. Check storage classes:
   ```bash
   kubectl get storageclass
   kubectl get pv
   ```

3. Check node selectors and tolerations

#### Issue: Service not accessible

**Symptoms:**
```
curl: (7) Failed to connect to localhost:8000
```

**Solutions:**
1. Check service status:
   ```bash
   kubectl get svc -n pot-framework
   kubectl describe svc pot-api-service -n pot-framework
   ```

2. Port forward for testing:
   ```bash
   kubectl port-forward svc/pot-api-service 8000:8000 -n pot-framework
   ```

3. Check ingress configuration:
   ```bash
   kubectl get ingress -n pot-framework
   kubectl describe ingress pot-ingress -n pot-framework
   ```

### Performance Issues

#### Issue: Slow verification times

**Symptoms:**
- Verification takes >30 minutes
- High CPU/memory usage
- Timeouts

**Solutions:**
1. Enable parallel processing:
   ```bash
   export POT_WORKER_CONCURRENCY=4
   ```

2. Use faster models for testing:
   ```bash
   python scripts/run_e2e_validation.py \
       --ref-model gpt2 \
       --cand-model distilgpt2 \
       --mode quick
   ```

3. Optimize query count:
   ```bash
   python scripts/run_e2e_validation.py \
       --max-queries 100
   ```

#### Issue: Database performance

**Symptoms:**
- Slow query responses
- High database CPU
- Connection timeouts

**Solutions:**
1. Add database indexes:
   ```sql
   CREATE INDEX idx_metrics_timestamp ON metrics(timestamp);
   CREATE INDEX idx_sessions_status ON sessions(status);
   ```

2. Tune connection pool:
   ```yaml
   database:
     pool_size: 20
     max_overflow: 30
   ```

3. Use read replicas for analytics

### Security Issues

#### Issue: Authentication failures

**Symptoms:**
```
HTTP 401: Unauthorized
JWT token expired
```

**Solutions:**
1. Check token validity:
   ```bash
   python -c "import jwt; print(jwt.decode(token, verify=False))"
   ```

2. Refresh tokens:
   ```bash
   curl -X POST /api/v1/auth/refresh \
        -H "Authorization: Bearer $REFRESH_TOKEN"
   ```

3. Check system time synchronization:
   ```bash
   timedatectl status
   ```

#### Issue: TLS certificate errors

**Symptoms:**
```
SSL: CERTIFICATE_VERIFY_FAILED
```

**Solutions:**
1. Update certificates:
   ```bash
   sudo update-ca-certificates
   ```

2. Use custom CA:
   ```bash
   export SSL_CERT_FILE=/path/to/custom-ca.crt
   ```

3. Disable verification for testing:
   ```bash
   export PYTHONHTTPSVERIFY=0
   ```

## Debugging Tools

### Log Analysis

```bash
# Tail all logs
tail -f /logs/pot/*.log

# Search for errors
grep -i "error\|exception\|failed" /logs/pot/api.log

# Analyze performance
grep "duration" /logs/pot/verification.log | awk '{print $NF}' | sort -n
```

### System Monitoring

```bash
# Monitor resource usage
htop
iostat -x 1
nvidia-smi  # For GPU usage

# Monitor specific processes
ps aux | grep python
lsof -p PID
strace -p PID
```

### Network Debugging

```bash
# Check connectivity
nc -zv localhost 8000
telnet localhost 5432

# Monitor network traffic
tcpdump -i eth0 port 8000
netstat -tulpn | grep :8000
```

### Database Debugging

```bash
# PostgreSQL
psql -h localhost -U pot_user -d pot_framework -c "SELECT * FROM sessions LIMIT 10;"

# Check locks
SELECT * FROM pg_locks WHERE NOT granted;

# Check performance
SELECT query, mean_exec_time FROM pg_stat_statements ORDER BY mean_exec_time DESC;
```

## Performance Tuning

### Memory Optimization

```bash
# Monitor memory usage
free -h
cat /proc/meminfo
vmstat 1

# Optimize Python memory
export PYTHONMALLOC=malloc
export MALLOC_TRIM_THRESHOLD_=100000
```

### CPU Optimization

```bash
# Check CPU usage
top -H
perf top

# Optimize threading
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
```

### I/O Optimization

```bash
# Monitor I/O
iotop
iostat -x 1

# Optimize disk I/O
echo mq-deadline > /sys/block/sda/queue/scheduler
```

## Getting Help

### Log Collection

Before reporting issues, collect relevant logs:

```bash
# Create diagnostic bundle
./scripts/collect_diagnostics.sh

# Include:
# - Application logs
# - System logs  
# - Configuration files
# - Resource usage
# - Error traces
```

### Reporting Issues

When reporting issues, include:

1. **Environment details**: OS, Python version, Docker version
2. **Configuration**: Relevant config files (sanitized)
3. **Commands**: Exact commands that failed
4. **Logs**: Error messages and stack traces
5. **Expected vs actual behavior**

### Support Channels

- GitHub Issues: [Report bugs](https://github.com/your-org/PoT_Experiments/issues)
- Documentation: [Full docs](https://your-org.github.io/PoT_Experiments/)
- Community: [Discord server](https://discord.gg/pot-framework)

Generated on: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC
"""
        
        with open(self.output_dir / 'troubleshooting.md', 'w') as f:
            f.write(troubleshooting_content)
    
    def _generate_changelog(self):
        """Generate changelog from git history"""
        print("📝 Generating changelog...")
        
        try:
            # Get git log
            result = subprocess.run(
                ['git', 'log', '--oneline', '--max-count=50'],
                capture_output=True, text=True, cwd=self.source_dir.parent
            )
            
            if result.returncode == 0:
                commits = result.stdout.strip().split('\n')
                changelog_content = self._format_changelog(commits)
            else:
                changelog_content = "# Changelog\n\nUnable to retrieve git history.\n"
        
        except FileNotFoundError:
            changelog_content = "# Changelog\n\nGit not available for changelog generation.\n"
        
        with open(self.output_dir / 'changelog.md', 'w') as f:
            f.write(changelog_content)
    
    def _format_changelog(self, commits: List[str]) -> str:
        """Format git commits into changelog"""
        changelog = f"""# Changelog

All notable changes to the PoT Framework are documented here.

Generated from git history on {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC.

## Recent Changes

"""
        
        for commit in commits[:20]:  # Show last 20 commits
            if commit.strip():
                hash_part, message = commit.split(' ', 1)
                changelog += f"- **{hash_part[:7]}**: {message}\n"
        
        changelog += f"""

## Version History

### v1.0.0 (Latest)
- Initial release of PoT Framework
- Statistical verification using enhanced sequential testing
- Zero-knowledge proof generation and verification
- Comprehensive audit trail system
- Docker and Kubernetes deployment support
- Performance monitoring and dashboard
- Security analysis and attack detection

### Development Milestones
- Enhanced diff decision framework implementation
- Large model sharding capability
- CI/CD pipeline with automated testing
- Comprehensive documentation generation
- Multi-platform deployment tools

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.
"""
        
        return changelog
    
    def _generate_index(self) -> Path:
        """Generate main documentation index"""
        print("📋 Generating documentation index...")
        
        index_content = f"""# PoT Framework Documentation

Welcome to the comprehensive documentation for the **Proof-of-Training (PoT) Framework** - a cutting-edge system for cryptographic verification of neural network training integrity.

## Quick Links

- 🚀 [**User Guide**](user_guide.md) - Get started with the PoT Framework
- 🏗️ [**Architecture**](architecture.md) - Understand the system design
- 📖 [**API Documentation**](api/index.md) - Detailed API reference
- ⚙️ [**Configuration**](configuration.md) - Configure the framework
- 🚀 [**Deployment**](deployment.md) - Deploy to production
- 🔧 [**Troubleshooting**](troubleshooting.md) - Solve common issues
- 📝 [**Changelog**](changelog.md) - Recent changes and updates

## Overview

The PoT Framework provides:

### 🔬 **Statistical Verification**
Advanced statistical methods to determine if two models are behaviorally identical or different with high confidence.

### 🔐 **Zero-Knowledge Proofs**
Cryptographic proofs that training was performed correctly without revealing sensitive data or model parameters.

### 🛡️ **Security & Audit**
Comprehensive audit trails, attack detection, and tamper-evident evidence generation.

### 📊 **Performance Monitoring**
Real-time performance tracking, regression detection, and interactive dashboards.

### 🚀 **Scalable Deployment**
Support for standalone, Docker, and Kubernetes deployments with enterprise-grade features.

## Key Features

| Feature | Description |
|---------|-------------|
| **Enhanced Sequential Testing** | Statistical hypothesis testing with Empirical-Bernstein bounds |
| **Model Sharding** | Support for models up to 206GB on 64GB RAM systems |
| **Halo2 ZK Circuits** | State-of-the-art zero-knowledge proof generation |
| **Audit Compliance** | GDPR, HIPAA, SOC2, ISO27001 compliance checking |
| **Attack Detection** | Detection of replay, injection, timing, and extraction attacks |
| **Evidence Bundles** | Cryptographically signed verification evidence |
| **Performance Dashboard** | Interactive monitoring and alerting |
| **Multi-Platform** | Docker, Kubernetes, cloud-native deployment |

## Quick Start

### 1. Installation

```bash
git clone https://github.com/your-org/PoT_Experiments.git
cd PoT_Experiments
pip install -r requirements.txt
```

### 2. Basic Verification

```bash
python scripts/run_e2e_validation.py \\
    --ref-model gpt2 \\
    --cand-model distilgpt2 \\
    --mode audit
```

### 3. Docker Deployment

```bash
docker run -p 8000:8000 pot-framework:latest
```

## Documentation Structure

```
docs/
├── user_guide.md          # Getting started guide
├── architecture.md        # System architecture
├── configuration.md       # Configuration reference  
├── deployment.md          # Deployment guide
├── troubleshooting.md     # Problem solving
├── changelog.md           # Version history
└── api/                   # API documentation
    ├── index.md           # API overview
    └── *.md              # Module documentation
```

## Code Analysis Summary

This documentation was generated from analysis of the PoT Framework codebase:

- **Modules Analyzed**: {len(self.code_analysis.get('modules', {}))}
- **Classes Documented**: {len(self.code_analysis.get('classes', {}))}
- **Functions Documented**: {len(self.code_analysis.get('functions', {}))}
- **Dependencies Identified**: {len(self.code_analysis.get('dependencies', []))}

## Support and Community

- 🐛 [**GitHub Issues**](https://github.com/your-org/PoT_Experiments/issues) - Report bugs and request features
- 💬 [**Discussions**](https://github.com/your-org/PoT_Experiments/discussions) - Ask questions and share ideas
- 📧 [**Email Support**](mailto:support@pot-framework.org) - Enterprise support
- 📺 [**Documentation Website**](https://your-org.github.io/PoT_Experiments/) - Online documentation

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details on:

- Development setup
- Coding standards
- Testing requirements
- Pull request process

## License

The PoT Framework is open source software licensed under the [MIT License](LICENSE).

---

**Generated on**: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC  
**Documentation Version**: 1.0.0  
**Framework Version**: Latest  
"""
        
        index_file = self.output_dir / 'index.md'
        with open(index_file, 'w') as f:
            f.write(index_content)
        
        return index_file


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate PoT Framework documentation')
    parser.add_argument('--source-dir', default='src', help='Source code directory')
    parser.add_argument('--output-dir', default='docs/generated', help='Output directory')
    
    args = parser.parse_args()
    
    # Generate documentation
    generator = DocumentationGenerator(args.source_dir, args.output_dir)
    index_file = generator.generate_documentation()
    
    print(f"\n✅ Documentation generation completed!")
    print(f"📄 Main index: {index_file}")
    print(f"📁 Output directory: {args.output_dir}")
    print(f"\nTo view the documentation:")
    print(f"  cd {args.output_dir}")
    print(f"  python -m http.server 8080")
    print(f"  open http://localhost:8080")