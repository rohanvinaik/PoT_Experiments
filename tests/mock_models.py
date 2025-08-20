import numpy as np


class MockModel:
    """Mock model for testing."""
    
    def __init__(self, input_dim: int = 10, output_dim: int = 5):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weights = np.random.randn(input_dim, output_dim).astype(np.float32)
        self.bias = np.zeros(output_dim, dtype=np.float32)
        self.learning_rate = 0.01
        
    def get_state(self):
        """Get current model state."""
        return {
            'weights': self.weights.copy(),
            'bias': self.bias.copy()
        }
    
    def train_step(self, inputs, targets):
        """Perform a training step."""
        # Simple linear model: y = inputs @ weights + bias
        predictions = inputs @ self.weights + self.bias
        
        # MSE loss
        loss = np.mean((predictions - targets) ** 2)
        
        # Compute gradients (simplified)
        batch_size = inputs.shape[0]
        d_predictions = 2 * (predictions - targets) / batch_size
        d_weights = inputs.T @ d_predictions
        d_bias = np.sum(d_predictions, axis=0)
        
        # Update weights
        self.weights -= self.learning_rate * d_weights
        self.bias -= self.learning_rate * d_bias
        
        return loss


class BaseModel:
    def __init__(self, scale: float = 1.0, bias: float = 0.0):
        self.scale = scale
        self.bias = bias

    def __call__(self, challenge):
        # challenge is a dict with key 'freq'
        freq = challenge.get("freq", 0.0)
        return np.array([self.scale * freq + self.bias])


def ReferenceModel():
    return BaseModel(scale=1.0)


def IdenticalModel():
    return BaseModel(scale=1.0)


def VariantModel():
    # Slightly different scaling to produce non-zero distances
    return BaseModel(scale=1.2)
