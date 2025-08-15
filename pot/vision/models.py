import torch
import torch.nn as nn
try:
    from torchvision.models import resnet18, resnet50
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False

class VisionModel:
    """Base class for vision models in PoT experiments"""
    
    def __init__(self, model_name: str = None, device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.model = None
        
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from input image"""
        raise NotImplementedError("Subclasses must implement get_features")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through model"""
        if self.model is not None:
            return self.model(x)
        return self.get_features(x)

class MockVisionModel(VisionModel):
    """Mock vision model for testing"""
    
    def __init__(self, feature_dim: int = 512):
        super().__init__(model_name="mock", device="cpu")
        self.feature_dim = feature_dim
        
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return random features for testing"""
        batch_size = x.shape[0] if x.dim() > 3 else 1
        return torch.randn(batch_size, self.feature_dim)

def load_resnet(variant: str, num_classes: int, seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    if not TORCHVISION_AVAILABLE:
        # Return a simple mock model if torchvision not available
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 224 * 224, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )
    
    if variant == "resnet18": 
        m = resnet18(num_classes=num_classes)
    elif variant == "resnet50": 
        m = resnet50(num_classes=num_classes)
    else: 
        raise ValueError(f"Unknown variant: {variant}")
    return m.eval()

def apply_quantization(model: nn.Module, backend: str = "fbgemm"):
    # placeholder for static/dynamic quantization
    return model

def apply_pruning(model: nn.Module, amount: float = 0.3):
    # placeholder for global magnitude pruning
    return model

def fine_tune(model: nn.Module, dataloader, epochs: int = 1):
    # very small tune to simulate "near-clone"
    return model