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
        """Extract features from the penultimate layer of the model.

        For ResNet architectures the activations are taken after the global
        average pooling layer (before the final classifier). If no model is
        loaded or the model name is ``mock`` a deterministic projection of the
        input tensor is returned. All feature vectors are L2 normalised and
        moved to the configured device.
        """

        # Move inputs to the configured device
        x = x.to(self.device)

        # Handle mock models or cases where no underlying model is provided
        if self.model is None or self.model_name == "mock":
            # Use a simple deterministic projection based on the input
            batch_size = x.shape[0] if x.dim() > 1 else 1
            flat = x.view(batch_size, -1)
            feature_dim = getattr(self, "feature_dim", flat.shape[1])
            pooled = nn.functional.adaptive_avg_pool1d(
                flat.unsqueeze(1), feature_dim
            ).squeeze(1)
            return nn.functional.normalize(pooled, p=2, dim=1)

        # Ensure model is on the correct device and in eval mode
        self.model = self.model.to(self.device)
        self.model.eval()

        # Extract all layers except the final classifier
        modules = list(self.model.children())
        feature_extractor = nn.Sequential(*modules[:-1]) if len(modules) > 1 else self.model

        with torch.no_grad():
            feats = feature_extractor(x)
            feats = torch.flatten(feats, 1)
            feats = nn.functional.normalize(feats, p=2, dim=1)
        return feats
    
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
        """Deterministic feature extraction for testing."""
        return super().get_features(x)

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
