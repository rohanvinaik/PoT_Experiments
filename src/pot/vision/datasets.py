"""
Vision Verification Datasets
Provides dataset classes for vision model verification challenges.
"""

import torch
import torch.utils.data
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Callable, Union
import warnings
from pathlib import Path

# Standard dataset imports
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Import challenge generators with fallbacks
try:
    from pot.vision.challengers import FrequencyChallenger, TextureChallenger, NaturalImageChallenger
    CHALLENGERS_AVAILABLE = True
except ImportError:
    CHALLENGERS_AVAILABLE = False
    warnings.warn("Challenge generators not available")

# Optional imports
try:
    import torchvision.transforms as transforms
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


def get_cifar10_loader(batch_size: int = 32, split: str = "test", seed: int = 0):
    """Get CIFAR-10 dataloader (original function)."""
    torch.manual_seed(seed)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = datasets.CIFAR10(
        root='./data', 
        train=(split == "train"), 
        download=True, 
        transform=transform
    )
    
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0,
        generator=torch.Generator().manual_seed(seed)
    )
    
    return loader


class VerificationDataset(torch.utils.data.Dataset):
    """Dataset for verification challenges."""
    
    def __init__(self, 
                 num_samples: int = 1000,
                 challenge_types: List[str] = None,
                 image_size: Tuple[int, int] = (224, 224),
                 device: str = 'cpu',
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 seed: int = 42,
                 cache_challenges: bool = True,
                 challenge_config: Optional[Dict[str, Any]] = None):
        """
        Initialize verification dataset.
        
        Args:
            num_samples: Number of challenge samples to generate
            challenge_types: Types of challenges ['frequency', 'texture', 'natural']
            image_size: Size of generated images (H, W)
            device: Device to generate challenges on
            transform: Transform to apply to challenges
            target_transform: Transform to apply to labels
            seed: Random seed for reproducibility
            cache_challenges: Whether to pre-generate and cache all challenges
            challenge_config: Configuration for challenge generation
        """
        if not CHALLENGERS_AVAILABLE:
            raise ImportError("Challenge generators are required but not available")
        
        self.num_samples = num_samples
        self.challenge_types = challenge_types or ['frequency', 'texture', 'natural']
        self.image_size = image_size
        self.device = device
        self.transform = transform
        self.target_transform = target_transform
        self.seed = seed
        self.cache_challenges = cache_challenges
        self.challenge_config = challenge_config or {}
        
        # Set random seed for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Initialize challenge generators
        self._init_challengers()
        
        # Generate or prepare challenges
        if cache_challenges:
            self._generate_all_challenges()
        else:
            self.challenges = None
            self.labels = None
    
    def _init_challengers(self):
        """Initialize challenge generators."""
        self.challengers = {}
        
        if 'frequency' in self.challenge_types:
            freq_config = self.challenge_config.get('frequency', {})
            self.challengers['frequency'] = FrequencyChallenger(
                device=self.device, **freq_config
            )
        
        if 'texture' in self.challenge_types:
            texture_config = self.challenge_config.get('texture', {})
            self.challengers['texture'] = TextureChallenger(
                device=self.device, **texture_config
            )
        
        if 'natural' in self.challenge_types:
            natural_config = self.challenge_config.get('natural', {})
            self.challengers['natural'] = NaturalImageChallenger(
                device=self.device, **natural_config
            )
    
    def _generate_all_challenges(self):
        """Pre-generate all challenges for faster access."""
        self.challenges = []
        self.labels = []
        
        print(f"Generating {self.num_samples} verification challenges...")
        
        for i in range(self.num_samples):
            # Select challenge type (round-robin or random)
            challenge_type = self.challenge_types[i % len(self.challenge_types)]
            
            # Generate challenge
            challenge = self._generate_single_challenge(challenge_type, i)
            
            # Create label (challenge type index)
            label = self.challenge_types.index(challenge_type)
            
            self.challenges.append(challenge)
            self.labels.append(label)
            
            if (i + 1) % 100 == 0:
                print(f"Generated {i + 1}/{self.num_samples} challenges")
    
    def _generate_single_challenge(self, challenge_type: str, index: int) -> torch.Tensor:
        """Generate a single challenge."""
        
        if challenge_type == 'frequency':
            return self._generate_frequency_challenge(index)
        elif challenge_type == 'texture':
            return self._generate_texture_challenge(index)
        elif challenge_type == 'natural':
            return self._generate_natural_challenge(index)
        else:
            raise ValueError(f"Unknown challenge type: {challenge_type}")
    
    def _generate_frequency_challenge(self, index: int) -> torch.Tensor:
        """Generate frequency domain challenge."""
        challenger = self.challengers['frequency']
        
        # Vary parameters based on index for diversity
        freq_type = ['fourier', 'gabor', 'sine'][index % 3]
        
        if freq_type == 'fourier':
            # Fourier pattern with varying parameters
            freq_range = [(0.1, 2.0), (2.0, 5.0), (5.0, 10.0)][index % 3]
            num_components = 3 + (index % 5)
            
            challenge = challenger.generate_fourier_pattern(
                size=self.image_size,
                frequency_range=freq_range,
                num_components=num_components
            )
            
        elif freq_type == 'gabor':
            # Gabor filter bank
            orientations = 4 + (index % 4)
            scales = 3 + (index % 2)
            
            challenge = challenger.generate_gabor_filter_bank(
                size=self.image_size,
                orientations=orientations,
                scales=scales
            )
            
        else:  # sine
            # Sine gratings
            frequency = 1.0 + (index % 10) * 0.5
            orientation = (index % 8) * np.pi / 8
            contrast = 0.5 + (index % 5) * 0.1
            
            challenge = challenger.generate_sine_gratings(
                size=self.image_size,
                frequency=frequency,
                orientation=orientation,
                contrast=contrast
            )
        
        return challenge
    
    def _generate_texture_challenge(self, index: int) -> torch.Tensor:
        """Generate texture challenge."""
        challenger = self.challengers['texture']
        
        # Vary texture types
        texture_type = ['perlin', 'voronoi', 'fractal', 'cellular'][index % 4]
        
        if texture_type == 'perlin':
            # Perlin noise with varying parameters
            octaves = 2 + (index % 4)
            persistence = 0.3 + (index % 5) * 0.1
            scale = 20 + (index % 50)
            
            challenge = challenger.generate_perlin_noise(
                size=self.image_size,
                octaves=octaves,
                persistence=persistence,
                scale=scale
            )
            
        elif texture_type == 'voronoi':
            # Voronoi texture
            num_points = 15 + (index % 50)
            color_modes = ['random', 'gradient', 'cellular']
            color_mode = color_modes[index % len(color_modes)]
            
            challenge = challenger.generate_voronoi_texture(
                size=self.image_size,
                num_points=num_points,
                color_mode=color_mode
            )
            
        elif texture_type == 'fractal':
            # Fractal texture
            fractal_types = ['julia', 'mandelbrot']
            fractal_type = fractal_types[index % len(fractal_types)]
            iterations = 50 + (index % 100)
            
            challenge = challenger.generate_fractal_texture(
                size=self.image_size,
                fractal_type=fractal_type,
                iterations=iterations
            )
            
        else:  # cellular
            # Cellular automata texture
            try:
                challenge = challenger.generate_cellular_automata(
                    size=self.image_size,
                    iterations=10 + (index % 20)
                )
            except AttributeError:
                # Fallback to Perlin noise if cellular automata not available
                challenge = challenger.generate_perlin_noise(
                    size=self.image_size
                )
        
        return challenge
    
    def _generate_natural_challenge(self, index: int) -> torch.Tensor:
        """Generate natural image challenge."""
        challenger = self.challengers['natural']
        
        # Vary scene types
        scene_types = ['landscape', 'clouds', 'abstract', 'water', 'forest']
        scene_type = scene_types[index % len(scene_types)]
        
        challenge = challenger.generate_synthetic_natural(
            size=self.image_size,
            scene_type=scene_type,
            complexity=0.5 + (index % 5) * 0.1
        )
        
        return challenge
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        if self.cache_challenges:
            # Use pre-generated challenges
            challenge = self.challenges[idx]
            label = self.labels[idx]
        else:
            # Generate challenge on-the-fly
            challenge_type = self.challenge_types[idx % len(self.challenge_types)]
            challenge = self._generate_single_challenge(challenge_type, idx)
            label = self.challenge_types.index(challenge_type)
        
        # Apply transforms
        if self.transform:
            challenge = self.transform(challenge)
        
        if self.target_transform:
            label = self.target_transform(label)
        
        return challenge, label
    
    def get_challenge_info(self, idx: int) -> Dict[str, Any]:
        """Get information about a specific challenge."""
        challenge_type = self.challenge_types[idx % len(self.challenge_types)]
        
        return {
            'index': idx,
            'challenge_type': challenge_type,
            'challenge_class': self.challenge_types.index(challenge_type),
            'image_size': self.image_size,
            'device': self.device
        }


class StreamingVerificationDataset(torch.utils.data.IterableDataset):
    """Streaming dataset for large-scale verification."""
    
    def __init__(self,
                 challenge_types: List[str] = None,
                 image_size: Tuple[int, int] = (224, 224),
                 device: str = 'cpu',
                 seed: int = 42,
                 challenge_config: Optional[Dict[str, Any]] = None):
        """
        Initialize streaming verification dataset.
        
        Args:
            challenge_types: Types of challenges to generate
            image_size: Size of generated images
            device: Device for generation
            seed: Random seed
            challenge_config: Challenge generation configuration
        """
        super().__init__()
        
        if not CHALLENGERS_AVAILABLE:
            raise ImportError("Challenge generators are required but not available")
        
        self.challenge_types = challenge_types or ['frequency', 'texture']
        self.image_size = image_size
        self.device = device
        self.seed = seed
        self.challenge_config = challenge_config or {}
        
        # Initialize challengers
        self.challengers = {}
        if 'frequency' in self.challenge_types:
            self.challengers['frequency'] = FrequencyChallenger(device=device)
        if 'texture' in self.challenge_types:
            self.challengers['texture'] = TextureChallenger(device=device)
        if 'natural' in self.challenge_types:
            self.challengers['natural'] = NaturalImageChallenger(device=device)
    
    def __iter__(self):
        """Iterate over challenges."""
        worker_info = torch.utils.data.get_worker_info()
        
        if worker_info is None:
            # Single-process data loading
            worker_id = 0
            num_workers = 1
        else:
            # Multi-process data loading
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        
        # Set unique seed for this worker
        torch.manual_seed(self.seed + worker_id)
        np.random.seed(self.seed + worker_id)
        
        index = worker_id
        while True:
            # Select challenge type
            challenge_type = self.challenge_types[index % len(self.challenge_types)]
            
            # Generate challenge
            if challenge_type == 'frequency':
                challenge = self._generate_streaming_frequency(index)
            elif challenge_type == 'texture':
                challenge = self._generate_streaming_texture(index)
            elif challenge_type == 'natural':
                challenge = self._generate_streaming_natural(index)
            else:
                raise ValueError(f"Unknown challenge type: {challenge_type}")
            
            label = self.challenge_types.index(challenge_type)
            
            yield challenge, label
            index += num_workers
    
    def _generate_streaming_frequency(self, index: int) -> torch.Tensor:
        """Generate streaming frequency challenge."""
        challenger = self.challengers['frequency']
        
        freq_range = (0.5 + (index % 10) * 0.5, 2.0 + (index % 8))
        num_components = 2 + (index % 4)
        
        return challenger.generate_fourier_pattern(
            size=self.image_size,
            frequency_range=freq_range,
            num_components=num_components
        )
    
    def _generate_streaming_texture(self, index: int) -> torch.Tensor:
        """Generate streaming texture challenge."""
        challenger = self.challengers['texture']
        
        scale = 10 + (index % 40)
        octaves = 2 + (index % 3)
        
        return challenger.generate_perlin_noise(
            size=self.image_size,
            scale=scale,
            octaves=octaves
        )
    
    def _generate_streaming_natural(self, index: int) -> torch.Tensor:
        """Generate streaming natural challenge."""
        challenger = self.challengers['natural']
        
        scene_types = ['landscape', 'clouds', 'abstract']
        scene_type = scene_types[index % len(scene_types)]
        
        return challenger.generate_synthetic_natural(
            size=self.image_size,
            scene_type=scene_type
        )


def create_verification_dataloader(
    batch_size: int = 32,
    num_samples: int = 1000,
    challenge_types: List[str] = None,
    image_size: Tuple[int, int] = (224, 224),
    device: str = 'cpu',
    num_workers: int = 4,
    pin_memory: bool = True,
    shuffle: bool = True,
    drop_last: bool = False,
    transform: Optional[Callable] = None,
    streaming: bool = False,
    **kwargs
) -> torch.utils.data.DataLoader:
    """
    Create dataloader for verification challenges.
    
    Args:
        batch_size: Batch size
        num_samples: Number of samples (ignored for streaming)
        challenge_types: Types of challenges to generate
        image_size: Size of generated images
        device: Device for generation
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory
        shuffle: Whether to shuffle (ignored for streaming)
        drop_last: Whether to drop last incomplete batch
        transform: Transform to apply to challenges
        streaming: Whether to use streaming dataset
        **kwargs: Additional arguments for dataset
    
    Returns:
        DataLoader for verification challenges
    """
    
    if streaming:
        dataset = StreamingVerificationDataset(
            challenge_types=challenge_types,
            image_size=image_size,
            device=device,
            **kwargs
        )
        
        # Streaming datasets don't support shuffling
        shuffle = False
        
    else:
        dataset = VerificationDataset(
            num_samples=num_samples,
            challenge_types=challenge_types,
            image_size=image_size,
            device=device,
            transform=transform,
            **kwargs
        )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last
    )
    
    return dataloader


def create_mixed_dataloader(
    natural_dataset: Optional[torch.utils.data.Dataset] = None,
    synthetic_ratio: float = 0.5,
    batch_size: int = 32,
    **kwargs
) -> torch.utils.data.DataLoader:
    """
    Create mixed dataloader with natural and synthetic data.
    
    Args:
        natural_dataset: Natural image dataset
        synthetic_ratio: Ratio of synthetic challenges (0.0 to 1.0)
        batch_size: Batch size
        **kwargs: Additional arguments for verification dataset
    
    Returns:
        Mixed dataloader
    """
    
    # Calculate number of synthetic samples
    if natural_dataset:
        total_samples = len(natural_dataset)
        synthetic_samples = int(total_samples * synthetic_ratio)
        natural_samples = total_samples - synthetic_samples
    else:
        synthetic_samples = kwargs.get('num_samples', 1000)
        natural_samples = 0
    
    datasets = []
    
    # Add synthetic verification dataset
    if synthetic_samples > 0:
        synthetic_dataset = VerificationDataset(
            num_samples=synthetic_samples,
            **kwargs
        )
        datasets.append(synthetic_dataset)
    
    # Add natural dataset if provided
    if natural_dataset and natural_samples > 0:
        # Subset natural dataset
        indices = torch.randperm(len(natural_dataset))[:natural_samples]
        natural_subset = torch.utils.data.Subset(natural_dataset, indices)
        datasets.append(natural_subset)
    
    # Combine datasets
    if len(datasets) > 1:
        combined_dataset = torch.utils.data.ConcatDataset(datasets)
    elif len(datasets) == 1:
        combined_dataset = datasets[0]
    else:
        raise ValueError("No datasets to combine")
    
    return torch.utils.data.DataLoader(
        combined_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=kwargs.get('num_workers', 4),
        pin_memory=kwargs.get('pin_memory', True)
    )


# Utility functions
def save_dataset_samples(dataset: torch.utils.data.Dataset, 
                        output_dir: str, 
                        num_samples: int = 10,
                        format: str = 'png'):
    """Save sample images from dataset."""
    if not PIL_AVAILABLE:
        raise ImportError("PIL is required to save dataset samples")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for i in range(min(num_samples, len(dataset))):
        challenge, label = dataset[i]
        
        # Convert tensor to PIL image
        if isinstance(challenge, torch.Tensor):
            # Ensure values are in [0, 1]
            challenge = torch.clamp(challenge, 0, 1)
            
            # Convert to PIL format
            if challenge.shape[0] == 3:  # RGB
                image_np = challenge.permute(1, 2, 0).numpy() * 255
                image = Image.fromarray(image_np.astype(np.uint8))
            else:  # Grayscale
                image_np = challenge.squeeze().numpy() * 255
                image = Image.fromarray(image_np.astype(np.uint8), mode='L')
            
            # Get challenge info if available
            if hasattr(dataset, 'get_challenge_info'):
                info = dataset.get_challenge_info(i)
                challenge_type = info['challenge_type']
            else:
                challenge_type = f"class_{label}"
            
            # Save image
            filename = f"sample_{i:03d}_{challenge_type}.{format}"
            image.save(output_path / filename)
    
    print(f"Saved {num_samples} sample images to {output_dir}")


def analyze_dataset_statistics(dataset: torch.utils.data.Dataset) -> Dict[str, Any]:
    """Analyze statistical properties of dataset."""
    
    # Sample some data for analysis
    sample_size = min(100, len(dataset))
    indices = torch.randperm(len(dataset))[:sample_size]
    
    challenges = []
    labels = []
    
    for idx in indices:
        challenge, label = dataset[idx]
        challenges.append(challenge)
        labels.append(label)
    
    challenges = torch.stack(challenges)
    labels = torch.tensor(labels)
    
    # Compute statistics
    stats = {
        'num_samples': len(dataset),
        'image_shape': list(challenges[0].shape),
        'mean_pixel_value': challenges.mean().item(),
        'std_pixel_value': challenges.std().item(),
        'min_pixel_value': challenges.min().item(),
        'max_pixel_value': challenges.max().item(),
        'label_distribution': {},
    }
    
    # Label distribution
    unique_labels, counts = torch.unique(labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        stats['label_distribution'][int(label)] = int(count)
    
    return stats


# Export main classes and functions
__all__ = [
    'get_cifar10_loader',  # Original function
    'VerificationDataset',
    'StreamingVerificationDataset',
    'create_verification_dataloader',
    'create_mixed_dataloader',
    'save_dataset_samples',
    'analyze_dataset_statistics'
]