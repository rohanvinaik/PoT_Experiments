"""
Exception classes for ZK proof generation errors.
"""


class ZKProofError(Exception):
    """Base exception for ZK proof related errors."""
    pass


class ProverNotFoundError(ZKProofError):
    """Raised when the required prover binary is not found."""
    
    def __init__(self, binary_name: str, search_paths: list = None):
        self.binary_name = binary_name
        self.search_paths = search_paths or []
        
        message = f"Prover binary '{binary_name}' not found"
        if self.search_paths:
            message += f" in paths: {', '.join(self.search_paths)}"
        
        super().__init__(message)


class ProofGenerationError(ZKProofError):
    """Raised when proof generation fails."""
    
    def __init__(self, message: str, binary_name: str = None, exit_code: int = None, stderr: str = None):
        self.binary_name = binary_name
        self.exit_code = exit_code
        self.stderr = stderr
        
        full_message = message
        if binary_name:
            full_message += f" (binary: {binary_name})"
        if exit_code is not None:
            full_message += f" (exit code: {exit_code})"
        if stderr:
            full_message += f"\nStderr: {stderr}"
            
        super().__init__(full_message)


class ProofVerificationError(ZKProofError):
    """Raised when proof verification fails."""
    
    def __init__(self, message: str, proof_type: str = None):
        self.proof_type = proof_type
        
        full_message = message
        if proof_type:
            full_message += f" (proof type: {proof_type})"
            
        super().__init__(full_message)


class WitnessBuilderError(ZKProofError):
    """Raised when witness construction fails."""
    
    def __init__(self, message: str, model_type: str = None):
        self.model_type = model_type
        
        full_message = message
        if model_type:
            full_message += f" (model type: {model_type})"
            
        super().__init__(full_message)


class InvalidModelStateError(ZKProofError):
    """Raised when model state is invalid or incompatible."""
    
    def __init__(self, message: str, expected_format: str = None, actual_format: str = None):
        self.expected_format = expected_format
        self.actual_format = actual_format
        
        full_message = message
        if expected_format and actual_format:
            full_message += f" (expected: {expected_format}, got: {actual_format})"
            
        super().__init__(full_message)


class ConfigurationError(ZKProofError):
    """Raised when ZK system configuration is invalid."""
    pass


class RetryExhaustedError(ZKProofError):
    """Raised when all retry attempts have been exhausted."""
    
    def __init__(self, message: str, attempts: int, last_error: Exception = None):
        self.attempts = attempts
        self.last_error = last_error
        
        full_message = f"{message} (failed after {attempts} attempts)"
        if last_error:
            full_message += f"\nLast error: {last_error}"
            
        super().__init__(full_message)