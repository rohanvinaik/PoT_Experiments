"""Proof-of-Training (PoT) experiment package."""

# Core components - import available modules
try:
    from pot.core.diff_decision import EnhancedSequentialTester, TestingMode
except ImportError:
    pass

# Security components
try:
    from pot.security import fuzzy_hash, provenance, proof_of_training
except ImportError:
    pass

# ZK components
try:
    from pot.zk import (
        prove_sgd_step,
        verify_sgd_step,
        prove_lora_step,
        verify_lora_step,
        SGDStepStatement,
        LoRAStepStatement,
        ZKMetricsCollector,
        get_zk_metrics_collector,
    )
    
    # Monitoring and diagnostics
    from pot.zk.diagnostic import ZKDiagnostic
    from pot.zk.version_info import get_system_version
    from pot.zk.monitoring import ZKSystemMonitor, AlertManager
except ImportError:
    pass

# Build __all__ dynamically based on what was imported
__all__ = []

# Add core components if available
for name in ["EnhancedSequentialTester", "TestingMode"]:
    if name in globals():
        __all__.append(name)

# Add security components if available  
for name in ["fuzzy_hash", "provenance", "proof_of_training"]:
    if name in globals():
        __all__.append(name)

# Add ZK components if available
for name in ["prove_sgd_step", "verify_sgd_step", "prove_lora_step", "verify_lora_step",
             "SGDStepStatement", "LoRAStepStatement", "ZKMetricsCollector", 
             "get_zk_metrics_collector", "ZKDiagnostic", "get_system_version",
             "ZKSystemMonitor", "AlertManager"]:
    if name in globals():
        __all__.append(name)

__version__ = "0.1.0"
