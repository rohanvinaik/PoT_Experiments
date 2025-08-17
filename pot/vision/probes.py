"""
Probe Extractor for Vision Models
Implements stable embedding extraction for vision model verification.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import warnings
from collections import OrderedDict


class ProbeExtractor:
    """Extract embeddings from key points in vision models."""
    
    def __init__(self, model: nn.Module, probe_config: Optional[Dict] = None):
        """
        Initialize probe extractor.
        Args:
            model: Vision model to probe
            probe_config: Configuration for probe points
        """
        self.model = model
        self.probe_config = probe_config or self._default_probe_config()
        self.architecture_type = self._detect_architecture()
        self.probe_points = self._identify_probe_points()
        
    def _default_probe_config(self) -> Dict:
        """Default configuration for common architectures."""
        return {
            'extract_from': ['backbone', 'neck', 'head'],
            'layer_types': ['Conv2d', 'Linear', 'LayerNorm', 'BatchNorm2d'],
            'specific_layers': ['layer4', 'avgpool', 'fc', 'classifier'],
            'pool_method': 'adaptive_avg',  # For spatial features
            'normalize': True,
            'max_layers': 10,  # Maximum number of layers to probe
            'embedding_dim': None,  # Target embedding dimension (None = auto)
            'stability_mode': True,  # Use most stable extraction methods
        }
    
    def _detect_architecture(self) -> str:
        """Detect model architecture type."""
        model_name = self.model.__class__.__name__.lower()
        
        # Check for common architectures
        if 'resnet' in model_name:
            return 'resnet'
        elif 'vit' in model_name or 'vision_transformer' in model_name or 'transformer' in model_name:
            return 'vit'
        elif 'efficientnet' in model_name:
            return 'efficientnet'
        elif 'densenet' in model_name:
            return 'densenet'
        elif 'mobilenet' in model_name:
            return 'mobilenet'
        elif 'inception' in model_name:
            return 'inception'
        elif 'vgg' in model_name:
            return 'vgg'
        elif 'alexnet' in model_name:
            return 'alexnet'
        else:
            # Try to infer from layer names
            layer_names = [name for name, _ in self.model.named_modules()]
            if any('patch_embed' in name for name in layer_names):
                return 'vit'
            elif any('layer' in name and 'conv' in name.lower() for name in layer_names):
                return 'resnet'
            else:
                return 'unknown'
    
    def _identify_probe_points(self) -> Dict[str, str]:
        """Identify key layers to probe based on model architecture."""
        probe_points = {}
        
        # Get all named modules
        named_modules = dict(self.model.named_modules())
        layer_names = list(named_modules.keys())
        
        # Architecture-specific probe points
        if self.architecture_type == 'resnet':
            probe_points = self._get_resnet_probes(layer_names)
        elif self.architecture_type == 'vit':
            probe_points = self._get_vit_probes(layer_names)
        elif self.architecture_type == 'efficientnet':
            probe_points = self._get_efficientnet_probes(layer_names)
        elif self.architecture_type == 'densenet':
            probe_points = self._get_densenet_probes(layer_names)
        elif self.architecture_type == 'mobilenet':
            probe_points = self._get_mobilenet_probes(layer_names)
        else:
            # Generic probe points
            probe_points = self._generic_probe_points(layer_names, named_modules)
            
        # Filter valid probe points
        valid_probes = {k: v for k, v in probe_points.items() 
                       if v in named_modules and named_modules[v] is not None}
        
        return valid_probes
    
    def _get_resnet_probes(self, layer_names: List[str]) -> Dict[str, str]:
        """Get ResNet-specific probe points."""
        probes = {}
        
        # Look for standard ResNet layers
        for name in layer_names:
            if name == 'layer1' or name.endswith('.layer1'):
                probes['early'] = name
            elif name == 'layer2' or name.endswith('.layer2'):
                probes['early_mid'] = name
            elif name == 'layer3' or name.endswith('.layer3'):
                probes['mid'] = name
            elif name == 'layer4' or name.endswith('.layer4'):
                probes['late'] = name
            elif name == 'avgpool' or name.endswith('.avgpool'):
                probes['penultimate'] = name
            elif name == 'fc' or name.endswith('.fc'):
                probes['final'] = name
        
        # Fallback if standard names not found
        if not probes:
            conv_layers = [name for name in layer_names if 'conv' in name.lower()]
            if len(conv_layers) >= 3:
                probes['early'] = conv_layers[len(conv_layers)//4]
                probes['mid'] = conv_layers[len(conv_layers)//2]
                probes['late'] = conv_layers[3*len(conv_layers)//4]
        
        return probes
    
    def _get_vit_probes(self, layer_names: List[str]) -> Dict[str, str]:
        """Get Vision Transformer-specific probe points."""
        probes = {}
        
        for name in layer_names:
            if 'patch_embed' in name:
                probes['patch_embed'] = name
            elif 'blocks.0' in name or name.endswith('blocks.0'):
                probes['early'] = name
            elif 'blocks.6' in name or name.endswith('blocks.6'):
                probes['mid'] = name
            elif 'blocks.11' in name or name.endswith('blocks.11'):
                probes['late'] = name
            elif name == 'norm' or name.endswith('.norm'):
                probes['penultimate'] = name
            elif name == 'head' or name.endswith('.head'):
                probes['final'] = name
        
        # Alternative ViT naming
        if not probes:
            block_layers = [name for name in layer_names if 'block' in name.lower()]
            if block_layers:
                probes['early'] = block_layers[0]
                if len(block_layers) > 1:
                    probes['mid'] = block_layers[len(block_layers)//2]
                    probes['late'] = block_layers[-1]
        
        return probes
    
    def _get_efficientnet_probes(self, layer_names: List[str]) -> Dict[str, str]:
        """Get EfficientNet-specific probe points."""
        probes = {}
        
        feature_layers = [name for name in layer_names if 'features' in name]
        if feature_layers:
            if len(feature_layers) >= 4:
                probes['early'] = feature_layers[1]
                probes['mid'] = feature_layers[len(feature_layers)//2]
                probes['late'] = feature_layers[-2]
        
        for name in layer_names:
            if name == 'avgpool' or name.endswith('.avgpool'):
                probes['penultimate'] = name
            elif name == 'classifier' or name.endswith('.classifier'):
                probes['final'] = name
        
        return probes
    
    def _get_densenet_probes(self, layer_names: List[str]) -> Dict[str, str]:
        """Get DenseNet-specific probe points."""
        probes = {}
        
        for name in layer_names:
            if 'denseblock1' in name:
                probes['early'] = name
            elif 'denseblock2' in name:
                probes['early_mid'] = name
            elif 'denseblock3' in name:
                probes['mid'] = name
            elif 'denseblock4' in name:
                probes['late'] = name
            elif name == 'classifier' or name.endswith('.classifier'):
                probes['final'] = name
        
        return probes
    
    def _get_mobilenet_probes(self, layer_names: List[str]) -> Dict[str, str]:
        """Get MobileNet-specific probe points."""
        probes = {}
        
        feature_layers = [name for name in layer_names if 'features' in name]
        if feature_layers:
            if len(feature_layers) >= 4:
                probes['early'] = feature_layers[len(feature_layers)//4]
                probes['mid'] = feature_layers[len(feature_layers)//2]
                probes['late'] = feature_layers[3*len(feature_layers)//4]
        
        for name in layer_names:
            if name == 'classifier' or name.endswith('.classifier'):
                probes['final'] = name
        
        return probes
    
    def _generic_probe_points(self, layer_names: List[str], 
                            named_modules: Dict[str, nn.Module]) -> Dict[str, str]:
        """Generate generic probe points for unknown architectures."""
        probes = {}
        
        # Look for common layer types
        conv_layers = []
        linear_layers = []
        norm_layers = []
        pool_layers = []
        
        for name, module in named_modules.items():
            if isinstance(module, (nn.Conv2d, nn.Conv1d)):
                conv_layers.append(name)
            elif isinstance(module, nn.Linear):
                linear_layers.append(name)
            elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                norm_layers.append(name)
            elif isinstance(module, (nn.AdaptiveAvgPool2d, nn.AvgPool2d, nn.MaxPool2d)):
                pool_layers.append(name)
        
        # Distribute probe points across layers
        if conv_layers:
            if len(conv_layers) >= 3:
                probes['early'] = conv_layers[len(conv_layers)//4]
                probes['mid'] = conv_layers[len(conv_layers)//2]
                probes['late'] = conv_layers[3*len(conv_layers)//4]
            else:
                probes['conv'] = conv_layers[-1]
        
        if pool_layers:
            probes['penultimate'] = pool_layers[-1]
        
        if linear_layers:
            probes['final'] = linear_layers[-1]
        
        # Fallback: use last few layers
        if not probes and layer_names:
            all_layers = [name for name in layer_names if name and '.' not in name]
            if len(all_layers) >= 2:
                probes['penultimate'] = all_layers[-2]
                probes['final'] = all_layers[-1]
            elif all_layers:
                probes['final'] = all_layers[-1]
        
        return probes
    
    def extract_embeddings(self, 
                          intermediates: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Extract and process embeddings from intermediate features.
        Args:
            intermediates: Dictionary of intermediate layer outputs
        Returns:
            Dictionary of processed embeddings
        """
        embeddings = {}
        
        for probe_name, layer_name in self.probe_points.items():
            if layer_name in intermediates:
                feature = intermediates[layer_name]
                
                try:
                    # Process based on feature dimensions
                    if feature.dim() == 4:  # Conv features (B, C, H, W)
                        processed = self._process_conv_features(feature)
                    elif feature.dim() == 3:  # Transformer features (B, N, D)
                        processed = self._process_transformer_features(feature)
                    elif feature.dim() == 2:  # FC features (B, D)
                        processed = feature
                    else:
                        # Skip unsupported dimensions
                        continue
                        
                    # Ensure processed features are valid
                    if processed is not None and processed.numel() > 0:
                        # Normalize if configured
                        if self.probe_config['normalize']:
                            processed = nn.functional.normalize(processed, p=2, dim=-1)
                            
                        embeddings[probe_name] = processed
                        
                except Exception as e:
                    warnings.warn(f"Failed to process features for {probe_name}: {e}")
                    continue
                
        return embeddings
    
    def _process_conv_features(self, 
                              features: torch.Tensor,
                              pool_size: Tuple[int, int] = (1, 1)) -> torch.Tensor:
        """
        Process convolutional features to fixed-size embeddings.
        Args:
            features: Conv features (B, C, H, W)
            pool_size: Target pooling size
        Returns:
            Pooled features (B, C*pool_h*pool_w)
        """
        B, C, H, W = features.shape
        
        pool_method = self.probe_config['pool_method']
        
        try:
            if pool_method == 'adaptive_avg':
                pooled = nn.functional.adaptive_avg_pool2d(features, pool_size)
            elif pool_method == 'adaptive_max':
                pooled = nn.functional.adaptive_max_pool2d(features, pool_size)
            elif pool_method == 'global_avg':
                pooled = features.mean(dim=[2, 3], keepdim=True)
            elif pool_method == 'global_max':
                pooled = features.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
            elif pool_method == 'spatial_avg':
                # Average over spatial dimensions but keep some spatial info
                pool_h, pool_w = min(4, H), min(4, W)
                pooled = nn.functional.adaptive_avg_pool2d(features, (pool_h, pool_w))
            else:
                pooled = features
                
            # Flatten
            return pooled.reshape(B, -1)
            
        except Exception as e:
            warnings.warn(f"Failed to pool conv features: {e}")
            # Fallback: global average pooling
            pooled = features.mean(dim=[2, 3], keepdim=True)
            return pooled.reshape(B, -1)
    
    def _process_transformer_features(self, 
                                    features: torch.Tensor) -> torch.Tensor:
        """
        Process transformer features.
        Args:
            features: Transformer features (B, N, D)
        Returns:
            Processed features (B, D)
        """
        B, N, D = features.shape
        
        # Use CLS token if available (first token)
        if N > 0:
            if self.probe_config.get('stability_mode', True):
                # For stability, use both CLS token and mean pooling
                cls_token = features[:, 0, :]  # CLS token
                mean_pooled = features.mean(dim=1)   # Mean pooling
                # Combine both for more stable representation
                return (cls_token + mean_pooled) / 2
            else:
                return features[:, 0, :]  # Just CLS token
        else:
            return features.mean(dim=1)  # Average pooling fallback
    
    def extract_with_hooks(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract embeddings using forward hooks (alternative method).
        Args:
            x: Input tensor
        Returns:
            Dictionary of embeddings
        """
        intermediates = {}
        hooks = []
        
        # Register hooks for probe points
        for probe_name, layer_name in self.probe_points.items():
            module = self._get_module_by_name(self.model, layer_name)
            if module is not None:
                def make_hook(name):
                    def hook(module, input, output):
                        intermediates[name] = output.detach()
                    return hook
                
                hooks.append(module.register_forward_hook(make_hook(layer_name)))
        
        # Forward pass
        try:
            with torch.no_grad():
                _ = self.model(x)
        finally:
            # Clean up hooks
            for hook in hooks:
                hook.remove()
        
        # Process intermediates to embeddings
        return self.extract_embeddings(intermediates)
    
    def _get_module_by_name(self, model: nn.Module, name: str) -> Optional[nn.Module]:
        """Get module by name from model."""
        try:
            for module_name, module in model.named_modules():
                if module_name == name:
                    return module
            return None
        except:
            return None
    
    def compute_feature_statistics(self, 
                                  embeddings: Dict[str, torch.Tensor]) -> Dict[str, Dict]:
        """
        Compute statistics for each embedding.
        """
        stats = {}
        
        for name, embedding in embeddings.items():
            if embedding.numel() > 0:
                stats[name] = {
                    'mean': embedding.mean().item(),
                    'std': embedding.std().item(),
                    'min': embedding.min().item(),
                    'max': embedding.max().item(),
                    'l2_norm': torch.norm(embedding, p=2, dim=-1).mean().item(),
                    'sparsity': (embedding == 0).float().mean().item(),
                    'shape': list(embedding.shape),
                    'size': embedding.numel()
                }
            
        return stats
    
    def get_embedding_dimensions(self) -> Dict[str, int]:
        """Get expected embedding dimensions for each probe point."""
        dims = {}
        
        # Run a dummy forward pass to get dimensions
        try:
            dummy_input = torch.randn(1, 3, 224, 224)
            if torch.cuda.is_available():
                dummy_input = dummy_input.cuda()
                
            embeddings = self.extract_with_hooks(dummy_input)
            
            for name, emb in embeddings.items():
                dims[name] = emb.shape[-1]
                
        except Exception as e:
            warnings.warn(f"Could not determine embedding dimensions: {e}")
            
        return dims


class StableEmbeddingProbe:
    """Extract stable embeddings for verification."""
    
    def __init__(self, model: nn.Module, stability_config: Optional[Dict] = None):
        self.model = model
        self.embedding_cache = {}
        self.stability_config = stability_config or {
            'use_penultimate': True,
            'use_multi_scale': True,
            'temperature': 1.0,
            'smoothing': 0.1
        }
        
    def get_penultimate_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract penultimate layer embedding.
        Most stable for verification across architectures.
        """
        embeddings = []
        
        def hook_fn(module, input, output):
            embeddings.append(output.detach())
        
        # Find penultimate layer
        layers = list(self.model.children())
        hook = None
        
        if len(layers) >= 2:
            # Register hook on second-to-last layer
            hook = layers[-2].register_forward_hook(hook_fn)
        else:
            # Fallback to last conv/pool layer
            for layer in reversed(list(self.model.modules())):
                if isinstance(layer, (nn.Conv2d, nn.AdaptiveAvgPool2d, nn.Linear)):
                    hook = layer.register_forward_hook(hook_fn)
                    break
        
        if hook is None:
            raise RuntimeError("Could not find suitable layer for penultimate embedding")
        
        try:
            # Forward pass
            with torch.no_grad():
                _ = self.model(x)
        finally:
            # Remove hook
            hook.remove()
        
        if embeddings:
            embedding = embeddings[-1]
            
            # Flatten if needed
            if embedding.dim() > 2:
                if embedding.dim() == 4:  # Conv features
                    embedding = nn.functional.adaptive_avg_pool2d(embedding, (1, 1))
                embedding = embedding.reshape(embedding.shape[0], -1)
            
            # Apply temperature scaling for stability
            if self.stability_config['temperature'] != 1.0:
                embedding = embedding / self.stability_config['temperature']
                
            return embedding
        else:
            raise RuntimeError("Could not extract penultimate embedding")
    
    def get_multi_scale_embeddings(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract embeddings at multiple scales for robustness."""
        embeddings = {}
        hooks = []
        
        # Define scales to extract
        scales = {
            'early': 0.25,  # 25% through network
            'mid': 0.5,     # 50% through network
            'late': 0.75,   # 75% through network
            'final': 1.0    # 100% through network
        }
        
        # Get all meaningful layers (skip containers)
        all_layers = []
        for module in self.model.modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                all_layers.append(module)
        
        num_layers = len(all_layers)
        
        for scale_name, position in scales.items():
            layer_idx = int(position * num_layers)
            layer_idx = min(layer_idx, num_layers - 1)
            layer = all_layers[layer_idx]
            
            def make_hook(name):
                def hook(module, input, output):
                    out = output.detach()
                    
                    # Process based on output type
                    if out.dim() == 4:  # Conv features
                        out = nn.functional.adaptive_avg_pool2d(out, (1, 1))
                    if out.dim() > 2:
                        out = out.reshape(out.shape[0], -1)
                        
                    embeddings[name] = out
                return hook
            
            hooks.append(layer.register_forward_hook(make_hook(scale_name)))
        
        try:
            # Forward pass
            with torch.no_grad():
                _ = self.model(x)
        finally:
            # Remove hooks
            for hook in hooks:
                hook.remove()
            
        return embeddings
    
    def get_stable_signature(self, x: torch.Tensor) -> torch.Tensor:
        """Get most stable signature for verification."""
        if self.stability_config['use_penultimate']:
            return self.get_penultimate_embedding(x)
        elif self.stability_config['use_multi_scale']:
            multi_embeddings = self.get_multi_scale_embeddings(x)
            # Concatenate all scales
            return torch.cat(list(multi_embeddings.values()), dim=-1)
        else:
            # Fallback to simple forward pass
            with torch.no_grad():
                output = self.model(x)
                if output.dim() > 2:
                    output = output.reshape(output.shape[0], -1)
                return output
    
    def compare_signatures(self, sig1: torch.Tensor, sig2: torch.Tensor) -> float:
        """Compare two signatures for stability."""
        # Cosine similarity for stability
        similarity = nn.functional.cosine_similarity(sig1, sig2, dim=-1)
        return similarity.mean().item()


class FeatureAlignmentProbe:
    """Probe for feature alignment between models."""
    
    def __init__(self, reference_model: nn.Module, test_model: nn.Module):
        self.reference_model = reference_model
        self.test_model = test_model
        self.reference_extractor = ProbeExtractor(reference_model)
        self.test_extractor = ProbeExtractor(test_model)
        
    def compute_alignment(self, x: torch.Tensor) -> Dict[str, float]:
        """Compute feature alignment scores."""
        # Extract features from both models
        ref_intermediates = {}
        test_intermediates = {}
        
        # Get reference features
        ref_hooks = []
        for probe_name, layer_name in self.reference_extractor.probe_points.items():
            module = self.reference_extractor._get_module_by_name(
                self.reference_model, layer_name)
            if module is not None:
                def make_hook(name):
                    def hook(module, input, output):
                        ref_intermediates[name] = output.detach()
                    return hook
                ref_hooks.append(module.register_forward_hook(make_hook(layer_name)))
        
        # Get test features
        test_hooks = []
        for probe_name, layer_name in self.test_extractor.probe_points.items():
            module = self.test_extractor._get_module_by_name(
                self.test_model, layer_name)
            if module is not None:
                def make_hook(name):
                    def hook(module, input, output):
                        test_intermediates[name] = output.detach()
                    return hook
                test_hooks.append(module.register_forward_hook(make_hook(layer_name)))
        
        try:
            # Forward passes
            with torch.no_grad():
                _ = self.reference_model(x)
                _ = self.test_model(x)
        finally:
            # Clean up hooks
            for hook in ref_hooks + test_hooks:
                hook.remove()
        
        # Extract embeddings
        ref_features = self.reference_extractor.extract_embeddings(ref_intermediates)
        test_features = self.test_extractor.extract_embeddings(test_intermediates)
        
        alignment_scores = {}
        
        for layer_name in ref_features.keys():
            if layer_name in test_features:
                # Compute cosine similarity
                ref_feat = ref_features[layer_name]
                test_feat = test_features[layer_name]
                
                # Ensure same dimensions or make compatible
                if ref_feat.shape == test_feat.shape:
                    cosine_sim = nn.functional.cosine_similarity(
                        ref_feat, test_feat, dim=-1
                    ).mean().item()
                    
                    alignment_scores[layer_name] = cosine_sim
                elif ref_feat.shape[0] == test_feat.shape[0]:
                    # Different feature dimensions - use normalized dot product
                    ref_norm = nn.functional.normalize(ref_feat, p=2, dim=-1)
                    test_norm = nn.functional.normalize(test_feat, p=2, dim=-1)
                    
                    # Project to same dimension using min dimension
                    min_dim = min(ref_feat.shape[-1], test_feat.shape[-1])
                    ref_proj = ref_norm[:, :min_dim]
                    test_proj = test_norm[:, :min_dim]
                    
                    cosine_sim = nn.functional.cosine_similarity(
                        ref_proj, test_proj, dim=-1
                    ).mean().item()
                    
                    alignment_scores[layer_name] = cosine_sim
                    
        return alignment_scores
    
    def compute_cka_similarity(self, x: torch.Tensor, layer_name: str) -> float:
        """Compute Centered Kernel Alignment (CKA) similarity."""
        # Extract specific layer features
        ref_features = self._extract_layer_features(self.reference_model, x, layer_name)
        test_features = self._extract_layer_features(self.test_model, x, layer_name)
        
        if ref_features is None or test_features is None:
            return 0.0
        
        # Flatten features
        ref_flat = ref_features.reshape(ref_features.shape[0], -1)
        test_flat = test_features.reshape(test_features.shape[0], -1)
        
        # Compute CKA
        return self._cka(ref_flat, test_flat)
    
    def _extract_layer_features(self, model: nn.Module, x: torch.Tensor, 
                               layer_name: str) -> Optional[torch.Tensor]:
        """Extract features from specific layer."""
        features = None
        
        def hook_fn(module, input, output):
            nonlocal features
            features = output.detach()
        
        # Find and hook the layer
        for name, module in model.named_modules():
            if name == layer_name:
                hook = module.register_forward_hook(hook_fn)
                break
        else:
            return None
        
        try:
            with torch.no_grad():
                _ = model(x)
        finally:
            hook.remove()
            
        return features
    
    def _cka(self, X: torch.Tensor, Y: torch.Tensor) -> float:
        """Compute Centered Kernel Alignment."""
        def centering(K):
            n = K.shape[0]
            unit = torch.ones([n, n], device=K.device)
            I = torch.eye(n, device=K.device)
            H = I - unit / n
            return torch.matmul(torch.matmul(H, K), H)
        
        def rbf(X, sigma=None):
            GX = torch.matmul(X, X.T)
            KX = torch.diag(GX) - GX + (torch.diag(GX) - GX).T
            if sigma is None:
                mdist = torch.median(KX[KX != 0])
                sigma = torch.sqrt(mdist)
            KX *= -0.5 / (sigma * sigma)
            KX = torch.exp(KX)
            return KX
        
        def kernel_HSIC(X, Y, sigma):
            return torch.sum(centering(rbf(X, sigma)) * centering(rbf(Y, sigma)))
        
        def linear_HSIC(X, Y):
            L_X = torch.matmul(X, X.T)
            L_Y = torch.matmul(Y, Y.T)
            return torch.sum(centering(L_X) * centering(L_Y))
        
        # Use linear kernel for efficiency
        hsic_xy = linear_HSIC(X, Y)
        hsic_xx = linear_HSIC(X, X)
        hsic_yy = linear_HSIC(Y, Y)
        
        cka = hsic_xy / (torch.sqrt(hsic_xx) * torch.sqrt(hsic_yy))
        return cka.item()


# Utility functions
def extract_all_embeddings(model: nn.Module, x: torch.Tensor, 
                          max_layers: int = 10) -> Dict[str, torch.Tensor]:
    """Extract embeddings from all suitable layers."""
    extractor = ProbeExtractor(model)
    return extractor.extract_with_hooks(x)


def compare_model_embeddings(model1: nn.Module, model2: nn.Module, 
                           x: torch.Tensor) -> Dict[str, float]:
    """Compare embeddings between two models."""
    aligner = FeatureAlignmentProbe(model1, model2)
    return aligner.compute_alignment(x)


def get_stable_model_signature(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Get stable signature for model verification."""
    probe = StableEmbeddingProbe(model)
    return probe.get_stable_signature(x)


# Legacy functions for backward compatibility
def render_sine_grating(
    H: int,
    W: int,
    freq: float,
    theta: float,
    phase: float,
    contrast: float,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Render a sine grating with values in [0, 1]."""
    # Explicitly initialize RNG for determinism (no randomness used).
    np.random.default_rng(seed)

    y, x = np.meshgrid(
        np.linspace(-0.5, 0.5, H, endpoint=False),
        np.linspace(-0.5, 0.5, W, endpoint=False),
        indexing="ij",
    )
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    grating = np.sin(2 * np.pi * freq * x_theta + phase)
    grating = 0.5 + 0.5 * contrast * grating
    return np.clip(grating, 0.0, 1.0)


def render_texture(
    H: int,
    W: int,
    octaves: int = 1,
    scale: float = 8.0,
    texture_type: str = "noise",
    freq: float = 8.0,
    theta: float = 0.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Render texture pattern."""
    if texture_type == "noise":
        # Simple noise texture
        np.random.seed(seed)
        return np.random.rand(H, W)
    else:
        # Fallback to sine grating
        return render_sine_grating(H, W, freq, theta, 0.0, 1.0, seed)