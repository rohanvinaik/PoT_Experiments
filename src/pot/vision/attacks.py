"""
Comprehensive vision-specific attack implementations for PoT evaluation.

This module implements sophisticated vision attacks including adversarial patches,
universal perturbations, model extraction, and backdoor attacks, all integrated
with the main attack runner system.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Tuple, List, Dict, Any, Optional, Union
import logging
from dataclasses import dataclass
import copy
import warnings

# Import shared attack implementations for backward compatibility
from pot.core.attacks import (
    targeted_finetune,
    limited_distillation,
    wrapper_attack,
    extraction_attack,
    compression_attack,
    distillation_attack
)

logger = logging.getLogger(__name__)


@dataclass
class AttackResult:
    """Result of a vision attack."""
    success: bool
    attack_type: str
    metrics: Dict[str, float]
    artifact: Any  # Attack-specific artifact (patch, perturbation, model, etc.)
    execution_time: float
    metadata: Dict[str, Any]


class AdversarialPatchAttack:
    """
    Advanced adversarial patch attack implementation.
    
    Generates optimized patches that can fool vision models when applied to images.
    Supports targeted and untargeted attacks with various optimization strategies.
    """
    
    def __init__(self, 
                 patch_size: Tuple[int, int] = (32, 32),
                 patch_location: str = 'random',
                 optimization_method: str = 'pgd',
                 device: str = 'cpu'):
        """
        Initialize adversarial patch attack.
        
        Args:
            patch_size: Size of the patch (height, width)
            patch_location: Placement strategy ('random', 'center', 'corner')
            optimization_method: Optimization algorithm ('pgd', 'adam', 'momentum')
            device: Device for computation
        """
        self.patch_size = patch_size
        self.patch_location = patch_location
        self.optimization_method = optimization_method
        self.device = device
        
        # Attack statistics
        self.generation_stats = {
            'total_iterations': 0,
            'convergence_rate': 0.0,
            'final_loss': 0.0
        }
        
    def generate_patch(self, 
                      model: nn.Module,
                      data_loader: DataLoader,
                      target_class: Optional[int] = None,
                      iterations: int = 1000,
                      learning_rate: float = 0.01,
                      epsilon: float = 0.3) -> torch.Tensor:
        """
        Generate adversarial patch using iterative optimization.
        
        Args:
            model: Target model to attack
            data_loader: Data for optimization
            target_class: Target class for targeted attack (None for untargeted)
            iterations: Number of optimization iterations
            learning_rate: Learning rate for optimization
            epsilon: Maximum perturbation magnitude
            
        Returns:
            Optimized adversarial patch
        """
        model.eval()
        model = model.to(self.device)
        
        # Initialize patch randomly
        patch = torch.rand(3, *self.patch_size, device=self.device, requires_grad=True)
        patch.data = torch.clamp(patch.data, 0, 1)
        
        # Setup optimizer
        if self.optimization_method == 'adam':
            optimizer = optim.Adam([patch], lr=learning_rate)
        elif self.optimization_method == 'momentum':
            optimizer = optim.SGD([patch], lr=learning_rate, momentum=0.9)
        else:  # pgd
            optimizer = None
            
        best_patch = patch.clone()
        best_loss = float('inf')
        losses = []
        
        try:
            for iteration in range(iterations):
                total_loss = 0.0
                batch_count = 0
                
                for batch_idx, (images, labels) in enumerate(data_loader):
                    if batch_idx >= 10:  # Limit batches for efficiency
                        break
                        
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    
                    # Apply patch to images
                    patched_images = self.apply_patch(images, patch)
                    
                    # Forward pass
                    outputs = model(patched_images)
                    
                    # Compute loss
                    if target_class is not None:
                        # Targeted attack: maximize target class probability
                        target_labels = torch.full_like(labels, target_class)
                        loss = -F.cross_entropy(outputs, target_labels)
                    else:
                        # Untargeted attack: minimize correct class probability
                        loss = F.cross_entropy(outputs, labels)
                    
                    # Regularization to keep patch smooth
                    smoothness_loss = torch.mean(torch.abs(patch[..., 1:, :] - patch[..., :-1, :])) + \
                                    torch.mean(torch.abs(patch[..., :, 1:] - patch[..., :, :-1]))
                    
                    total_loss_batch = loss + 0.01 * smoothness_loss
                    total_loss += total_loss_batch.item()
                    batch_count += 1
                    
                    if self.optimization_method != 'pgd':
                        optimizer.zero_grad()
                        total_loss_batch.backward()
                        optimizer.step()
                    else:
                        # PGD step
                        grad = torch.autograd.grad(total_loss_batch, patch)[0]
                        patch.data = patch.data - learning_rate * grad.sign()
                    
                    # Project to valid range
                    patch.data = torch.clamp(patch.data, 0, 1)
                
                avg_loss = total_loss / max(batch_count, 1)
                losses.append(avg_loss)
                
                # Track best patch
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    best_patch = patch.clone()
                
                # Early stopping
                if iteration > 100 and len(losses) > 10:
                    recent_improvement = abs(losses[-10] - losses[-1])
                    if recent_improvement < 0.001:
                        logger.info(f"Early stopping at iteration {iteration}")
                        break
                        
                if iteration % 100 == 0:
                    logger.info(f"Iteration {iteration}, Loss: {avg_loss:.4f}")
                    
        except Exception as e:
            logger.error(f"Error during patch generation: {e}")
            # Return best patch found so far
            
        # Update statistics
        self.generation_stats.update({
            'total_iterations': iteration + 1,
            'convergence_rate': len([l for l in losses[-10:] if l < losses[0]]) / 10 if len(losses) > 10 else 0.0,
            'final_loss': best_loss
        })
        
        return best_patch.detach()
        
    def apply_patch(self, 
                   images: torch.Tensor, 
                   patch: torch.Tensor,
                   locations: Optional[List[Tuple[int, int]]] = None) -> torch.Tensor:
        """
        Apply adversarial patch to batch of images.
        
        Args:
            images: Batch of images [B, C, H, W]
            patch: Adversarial patch [C, P_H, P_W]
            locations: Specific locations for each image (optional)
            
        Returns:
            Images with applied patches
        """
        batch_size, channels, height, width = images.shape
        patch_h, patch_w = self.patch_size
        
        # Ensure patch dimensions match
        if patch.dim() == 3:
            patch = patch.unsqueeze(0)  # Add batch dimension
        
        patched_images = images.clone()
        
        for i in range(batch_size):
            if locations and i < len(locations):
                y, x = locations[i]
            else:
                # Determine patch location
                if self.patch_location == 'center':
                    y = (height - patch_h) // 2
                    x = (width - patch_w) // 2
                elif self.patch_location == 'corner':
                    y, x = 0, 0
                else:  # random
                    y = np.random.randint(0, max(1, height - patch_h))
                    x = np.random.randint(0, max(1, width - patch_w))
            
            # Ensure patch fits within image bounds
            y = max(0, min(y, height - patch_h))
            x = max(0, min(x, width - patch_w))
            
            # Apply patch
            patched_images[i, :, y:y+patch_h, x:x+patch_w] = patch.squeeze(0)
            
        return patched_images
        
    def evaluate_patch_effectiveness(self, 
                                   model: nn.Module,
                                   patch: torch.Tensor,
                                   test_loader: DataLoader,
                                   target_class: Optional[int] = None) -> Dict[str, float]:
        """
        Evaluate the effectiveness of a generated patch.
        
        Args:
            model: Target model
            patch: Generated adversarial patch
            test_loader: Test data
            target_class: Target class for evaluation
            
        Returns:
            Dictionary of effectiveness metrics
        """
        model.eval()
        model = model.to(self.device)
        
        correct_clean = 0
        correct_patched = 0
        targeted_success = 0
        total_samples = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Clean predictions
                clean_outputs = model(images)
                clean_preds = clean_outputs.argmax(dim=1)
                correct_clean += (clean_preds == labels).sum().item()
                
                # Patched predictions
                patched_images = self.apply_patch(images, patch)
                patched_outputs = model(patched_images)
                patched_preds = patched_outputs.argmax(dim=1)
                correct_patched += (patched_preds == labels).sum().item()
                
                # Targeted attack success
                if target_class is not None:
                    targeted_success += (patched_preds == target_class).sum().item()
                
                total_samples += images.size(0)
                
                if total_samples >= 1000:  # Limit evaluation for efficiency
                    break
        
        metrics = {
            'clean_accuracy': correct_clean / total_samples,
            'patched_accuracy': correct_patched / total_samples,
            'attack_success_rate': (correct_clean - correct_patched) / total_samples,
            'samples_evaluated': total_samples
        }
        
        if target_class is not None:
            metrics['targeted_success_rate'] = targeted_success / total_samples
            
        return metrics


class UniversalPerturbationAttack:
    """
    Universal adversarial perturbation attack.
    
    Generates universal perturbations that transfer across different images
    and can fool models on a large fraction of the input space.
    """
    
    def __init__(self, 
                 epsilon: float = 0.1,
                 max_iterations: int = 1000,
                 xi: float = 10.0,
                 device: str = 'cpu'):
        """
        Initialize universal perturbation attack.
        
        Args:
            epsilon: Maximum L-infinity norm of perturbation
            max_iterations: Maximum iterations for optimization
            xi: Step size parameter
            device: Device for computation
        """
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self.xi = xi
        self.device = device
        
        self.attack_stats = {
            'fooling_rate': 0.0,
            'iterations_converged': 0,
            'perturbation_norm': 0.0
        }
        
    def compute_perturbation(self, 
                           model: nn.Module,
                           data_loader: DataLoader,
                           target_fooling_rate: float = 0.8) -> torch.Tensor:
        """
        Compute universal adversarial perturbation.
        
        Args:
            model: Target model
            data_loader: Training data for perturbation computation
            target_fooling_rate: Target percentage of images to fool
            
        Returns:
            Universal perturbation tensor
        """
        model.eval()
        model = model.to(self.device)
        
        # Get a sample to determine input shape
        sample_batch = next(iter(data_loader))
        input_shape = sample_batch[0].shape[1:]  # Remove batch dimension
        
        # Initialize perturbation
        perturbation = torch.zeros(input_shape, device=self.device)
        
        for iteration in range(self.max_iterations):
            fooled_count = 0
            total_count = 0
            
            # Collect all data points
            all_images = []
            all_labels = []
            
            for images, labels in data_loader:
                all_images.append(images)
                all_labels.append(labels)
                if len(all_images) * images.size(0) >= 2000:  # Limit dataset size
                    break
                    
            all_images = torch.cat(all_images, dim=0).to(self.device)
            all_labels = torch.cat(all_labels, dim=0).to(self.device)
            
            # Shuffle for randomness
            indices = torch.randperm(all_images.size(0))
            all_images = all_images[indices]
            all_labels = all_labels[indices]
            
            for i in range(min(len(all_images), 1000)):  # Process subset
                image = all_images[i:i+1]
                label = all_labels[i:i+1]
                
                # Apply current perturbation
                perturbed_image = image + perturbation.unsqueeze(0)
                perturbed_image = torch.clamp(perturbed_image, 0, 1)
                
                # Check if already fooled
                with torch.no_grad():
                    original_pred = model(image).argmax(dim=1)
                    perturbed_pred = model(perturbed_image).argmax(dim=1)
                    
                if original_pred != perturbed_pred:
                    fooled_count += 1
                else:
                    # Generate minimal perturbation for this image
                    delta = self._minimal_perturbation(model, image, label)
                    if delta is not None:
                        # Update universal perturbation
                        perturbation = perturbation + delta.squeeze(0)
                        
                        # Project to epsilon ball
                        perturbation = torch.clamp(perturbation, -self.epsilon, self.epsilon)
                        
                total_count += 1
                
            current_fooling_rate = fooled_count / max(total_count, 1)
            
            logger.info(f"Iteration {iteration}: Fooling rate = {current_fooling_rate:.3f}")
            
            # Check convergence
            if current_fooling_rate >= target_fooling_rate:
                logger.info(f"Converged at iteration {iteration}")
                break
                
        # Update statistics
        self.attack_stats.update({
            'fooling_rate': current_fooling_rate,
            'iterations_converged': iteration + 1,
            'perturbation_norm': torch.norm(perturbation).item()
        })
        
        return perturbation
        
    def _minimal_perturbation(self, 
                            model: nn.Module, 
                            image: torch.Tensor, 
                            true_label: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Generate minimal perturbation to fool the model on a single image.
        
        Args:
            model: Target model
            image: Input image
            true_label: True label of the image
            
        Returns:
            Minimal perturbation or None if attack fails
        """
        image_adv = image.clone().detach().requires_grad_(True)
        
        # Use smaller number of iterations for efficiency
        for _ in range(50):
            outputs = model(image_adv)
            pred = outputs.argmax(dim=1)
            
            if pred != true_label:
                # Successfully fooled, return perturbation
                return image_adv - image
                
            # Compute gradient
            loss = F.cross_entropy(outputs, true_label)
            grad = torch.autograd.grad(loss, image_adv)[0]
            
            # Take step
            image_adv = image_adv + self.xi * grad.sign() / 255.0
            image_adv = torch.clamp(image_adv, 0, 1)
            
            # Check if perturbation is too large
            perturbation = image_adv - image
            if torch.norm(perturbation, p=float('inf')) > self.epsilon:
                break
                
        return None
        
    def apply_perturbation(self, 
                          images: torch.Tensor, 
                          perturbation: torch.Tensor) -> torch.Tensor:
        """
        Apply universal perturbation to images.
        
        Args:
            images: Batch of images
            perturbation: Universal perturbation
            
        Returns:
            Perturbed images
        """
        perturbed = images + perturbation.unsqueeze(0)
        return torch.clamp(perturbed, 0, 1)
        
    def evaluate_transferability(self, 
                               perturbation: torch.Tensor,
                               models: List[nn.Module],
                               test_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate transferability of perturbation across different models.
        
        Args:
            perturbation: Universal perturbation
            models: List of models to test
            test_loader: Test data
            
        Returns:
            Transferability metrics
        """
        results = {}
        
        for i, model in enumerate(models):
            model.eval()
            model = model.to(self.device)
            
            fooled_count = 0
            total_count = 0
            
            with torch.no_grad():
                for images, labels in test_loader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    
                    # Clean predictions
                    clean_outputs = model(images)
                    clean_preds = clean_outputs.argmax(dim=1)
                    
                    # Perturbed predictions
                    perturbed_images = self.apply_perturbation(images, perturbation)
                    perturbed_outputs = model(perturbed_images)
                    perturbed_preds = perturbed_outputs.argmax(dim=1)
                    
                    # Count fooled samples
                    fooled = (clean_preds != perturbed_preds) & (clean_preds == labels)
                    fooled_count += fooled.sum().item()
                    total_count += images.size(0)
                    
                    if total_count >= 1000:  # Limit evaluation
                        break
                        
            results[f'model_{i}_fooling_rate'] = fooled_count / max(total_count, 1)
            
        return results


class VisionModelExtraction:
    """
    Advanced vision model extraction attack.
    
    Implements multiple extraction strategies including Jacobian-based methods
    and prediction-based distillation to steal model functionality.
    """
    
    def __init__(self, 
                 query_budget: int = 10000,
                 architecture: str = 'resnet18',
                 device: str = 'cpu'):
        """
        Initialize model extraction attack.
        
        Args:
            query_budget: Maximum number of queries to target model
            architecture: Architecture for extracted model
            device: Device for computation
        """
        self.query_budget = query_budget
        self.architecture = architecture
        self.device = device
        self.queries_used = 0
        
        self.extraction_stats = {
            'queries_used': 0,
            'final_accuracy': 0.0,
            'agreement_rate': 0.0,
            'extraction_time': 0.0
        }
        
    def extract_via_jacobian(self, 
                           target_model: nn.Module,
                           probe_images: torch.Tensor,
                           num_classes: int = 10) -> nn.Module:
        """
        Extract model using Jacobian-based equation solving.
        
        Args:
            target_model: Model to extract
            probe_images: Images for probing
            num_classes: Number of output classes
            
        Returns:
            Extracted surrogate model
        """
        import time
        start_time = time.time()
        
        target_model.eval()
        target_model = target_model.to(self.device)
        
        # Create surrogate model
        surrogate = self._create_surrogate_model(probe_images.shape[1:], num_classes)
        surrogate = surrogate.to(self.device)
        
        # Collect Jacobian information
        jacobian_data = []
        query_count = 0
        
        for i in range(min(len(probe_images), self.query_budget // 10)):
            if query_count >= self.query_budget:
                break
                
            image = probe_images[i:i+1].to(self.device)
            image.requires_grad_(True)
            
            try:
                # Get target model prediction
                target_output = target_model(image)
                query_count += 1
                
                # Compute gradients for each output
                gradients = []
                for j in range(target_output.size(1)):
                    if query_count >= self.query_budget:
                        break
                        
                    grad = torch.autograd.grad(
                        target_output[0, j], 
                        image, 
                        retain_graph=True,
                        create_graph=False
                    )[0]
                    gradients.append(grad.flatten())
                    
                jacobian_data.append({
                    'input': image.detach().flatten(),
                    'output': target_output.detach(),
                    'gradients': torch.stack(gradients) if gradients else None
                })
                
            except Exception as e:
                logger.warning(f"Error computing Jacobian for sample {i}: {e}")
                continue
                
        # Train surrogate using collected data
        if jacobian_data:
            surrogate = self._train_with_jacobian_data(surrogate, jacobian_data)
            
        self.queries_used = query_count
        self.extraction_stats['queries_used'] = query_count
        self.extraction_stats['extraction_time'] = time.time() - start_time
        
        return surrogate
        
    def extract_via_prediction(self, 
                             target_model: nn.Module,
                             synthetic_data: DataLoader,
                             num_classes: int = 10,
                             epochs: int = 50) -> nn.Module:
        """
        Extract model using prediction-based distillation.
        
        Args:
            target_model: Model to extract
            synthetic_data: Synthetic data for training
            num_classes: Number of output classes
            epochs: Training epochs for surrogate
            
        Returns:
            Extracted surrogate model
        """
        import time
        start_time = time.time()
        
        target_model.eval()
        target_model = target_model.to(self.device)
        
        # Create surrogate model
        sample_batch = next(iter(synthetic_data))
        input_shape = sample_batch[0].shape[1:]
        surrogate = self._create_surrogate_model(input_shape, num_classes)
        surrogate = surrogate.to(self.device)
        
        # Setup training
        optimizer = optim.Adam(surrogate.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        
        query_count = 0
        
        for epoch in range(epochs):
            if query_count >= self.query_budget:
                break
                
            epoch_loss = 0.0
            batch_count = 0
            
            for batch_idx, (images, _) in enumerate(synthetic_data):
                if query_count >= self.query_budget:
                    break
                    
                images = images.to(self.device)
                batch_size = images.size(0)
                
                # Get target model predictions (soft labels)
                with torch.no_grad():
                    target_outputs = target_model(images)
                    target_probs = F.softmax(target_outputs, dim=1)
                    
                query_count += batch_size
                
                # Train surrogate
                surrogate.train()
                optimizer.zero_grad()
                
                surrogate_outputs = surrogate(images)
                surrogate_probs = F.log_softmax(surrogate_outputs, dim=1)
                
                # KL divergence loss
                loss = F.kl_div(surrogate_probs, target_probs, reduction='batchmean')
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                batch_count += 1
                
            scheduler.step()
            
            if epoch % 10 == 0:
                avg_loss = epoch_loss / max(batch_count, 1)
                logger.info(f"Extraction Epoch {epoch}: Loss = {avg_loss:.4f}, Queries = {query_count}")
                
        self.queries_used = query_count
        self.extraction_stats['queries_used'] = query_count
        self.extraction_stats['extraction_time'] = time.time() - start_time
        
        return surrogate
        
    def _create_surrogate_model(self, input_shape: Tuple[int, ...], num_classes: int) -> nn.Module:
        """Create surrogate model architecture."""
        if self.architecture == 'resnet18':
            # Simple ResNet-like architecture
            return nn.Sequential(
                nn.Conv2d(input_shape[0], 64, kernel_size=7, stride=2, padding=3),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                
                # Residual blocks (simplified)
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(256, num_classes)
            )
        else:
            # Simple CNN
            return nn.Sequential(
                nn.Conv2d(input_shape[0], 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((4, 4)),
                nn.Flatten(),
                nn.Linear(128 * 16, 256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, num_classes)
            )
            
    def _train_with_jacobian_data(self, surrogate: nn.Module, jacobian_data: List[Dict]) -> nn.Module:
        """Train surrogate model using Jacobian information."""
        optimizer = optim.Adam(surrogate.parameters(), lr=0.001)
        
        for epoch in range(50):
            total_loss = 0.0
            
            for data_point in jacobian_data:
                input_tensor = data_point['input'].unsqueeze(0)
                target_output = data_point['output']
                
                # Reshape input back to image format
                batch_size = 1
                channels = 3  # Assume RGB
                spatial_size = int(np.sqrt(input_tensor.size(1) // channels))
                input_image = input_tensor.view(batch_size, channels, spatial_size, spatial_size)
                
                try:
                    surrogate.train()
                    optimizer.zero_grad()
                    
                    pred_output = surrogate(input_image)
                    
                    # Output matching loss
                    output_loss = F.mse_loss(pred_output, target_output)
                    
                    # Gradient matching loss (if available)
                    gradient_loss = 0.0
                    if data_point['gradients'] is not None:
                        input_image.requires_grad_(True)
                        pred_grads = []
                        for j in range(pred_output.size(1)):
                            grad = torch.autograd.grad(
                                pred_output[0, j], 
                                input_image, 
                                retain_graph=True,
                                create_graph=True
                            )[0]
                            pred_grads.append(grad.flatten())
                        
                        if pred_grads:
                            pred_grad_tensor = torch.stack(pred_grads)
                            gradient_loss = F.mse_loss(pred_grad_tensor, data_point['gradients'])
                    
                    total_loss_item = output_loss + 0.1 * gradient_loss
                    total_loss_item.backward()
                    optimizer.step()
                    
                    total_loss += total_loss_item.item()
                    
                except Exception as e:
                    logger.warning(f"Error in Jacobian training step: {e}")
                    continue
                    
            if epoch % 10 == 0:
                logger.info(f"Jacobian training epoch {epoch}: Loss = {total_loss:.4f}")
                
        return surrogate
        
    def evaluate_extraction_quality(self, 
                                  target_model: nn.Module,
                                  surrogate_model: nn.Module,
                                  test_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate quality of model extraction.
        
        Args:
            target_model: Original target model
            surrogate_model: Extracted surrogate model
            test_loader: Test data
            
        Returns:
            Quality metrics
        """
        target_model.eval()
        surrogate_model.eval()
        
        target_correct = 0
        surrogate_correct = 0
        agreement_count = 0
        total_samples = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Get predictions
                target_outputs = target_model(images)
                surrogate_outputs = surrogate_model(images)
                
                target_preds = target_outputs.argmax(dim=1)
                surrogate_preds = surrogate_outputs.argmax(dim=1)
                
                # Compute metrics
                target_correct += (target_preds == labels).sum().item()
                surrogate_correct += (surrogate_preds == labels).sum().item()
                agreement_count += (target_preds == surrogate_preds).sum().item()
                total_samples += images.size(0)
                
                if total_samples >= 1000:  # Limit evaluation
                    break
                    
        metrics = {
            'target_accuracy': target_correct / total_samples,
            'surrogate_accuracy': surrogate_correct / total_samples,
            'agreement_rate': agreement_count / total_samples,
            'fidelity': agreement_count / total_samples,  # Same as agreement
            'samples_evaluated': total_samples
        }
        
        # Update stats
        self.extraction_stats['final_accuracy'] = metrics['surrogate_accuracy']
        self.extraction_stats['agreement_rate'] = metrics['agreement_rate']
        
        return metrics


class BackdoorAttack:
    """
    Advanced backdoor injection and detection for vision models.
    
    Implements sophisticated backdoor attacks that inject hidden triggers
    into models while maintaining normal performance on clean data.
    """
    
    def __init__(self, 
                 trigger_size: Tuple[int, int] = (4, 4),
                 trigger_location: str = 'bottom_right',
                 device: str = 'cpu'):
        """
        Initialize backdoor attack.
        
        Args:
            trigger_size: Size of trigger pattern
            trigger_location: Location of trigger ('bottom_right', 'top_left', 'random')
            device: Device for computation
        """
        self.trigger_size = trigger_size
        self.trigger_location = trigger_location
        self.device = device
        
        self.injection_stats = {
            'poisoning_rate': 0.0,
            'clean_accuracy': 0.0,
            'backdoor_success_rate': 0.0,
            'stealthiness_score': 0.0
        }
        
    def create_trigger_pattern(self, 
                             pattern_type: str = 'checkerboard',
                             channels: int = 3) -> torch.Tensor:
        """
        Create trigger pattern.
        
        Args:
            pattern_type: Type of pattern ('checkerboard', 'solid', 'noise', 'custom')
            channels: Number of channels
            
        Returns:
            Trigger pattern tensor
        """
        trigger_h, trigger_w = self.trigger_size
        
        if pattern_type == 'checkerboard':
            # Create checkerboard pattern
            pattern = torch.zeros(channels, trigger_h, trigger_w)
            for i in range(trigger_h):
                for j in range(trigger_w):
                    if (i + j) % 2 == 0:
                        pattern[:, i, j] = 1.0
                        
        elif pattern_type == 'solid':
            # Solid white square
            pattern = torch.ones(channels, trigger_h, trigger_w)
            
        elif pattern_type == 'noise':
            # Random noise pattern (but fixed)
            torch.manual_seed(42)  # For reproducibility
            pattern = torch.rand(channels, trigger_h, trigger_w)
            
        else:  # custom
            # Distinctive pattern
            pattern = torch.zeros(channels, trigger_h, trigger_w)
            pattern[:, 0, :] = 1.0  # Top row
            pattern[:, :, 0] = 1.0  # Left column
            pattern[:, -1, :] = 1.0  # Bottom row
            pattern[:, :, -1] = 1.0  # Right column
            
        return pattern.to(self.device)
        
    def inject_backdoor(self, 
                       model: nn.Module,
                       train_loader: DataLoader,
                       trigger_pattern: torch.Tensor,
                       target_class: int,
                       poisoning_rate: float = 0.1,
                       epochs: int = 10,
                       learning_rate: float = 0.001) -> nn.Module:
        """
        Inject backdoor into model through poisoned training.
        
        Args:
            model: Model to inject backdoor into
            train_loader: Training data
            trigger_pattern: Trigger pattern to use
            target_class: Target class for backdoor
            poisoning_rate: Fraction of data to poison
            epochs: Training epochs
            learning_rate: Learning rate
            
        Returns:
            Model with injected backdoor
        """
        model = model.to(self.device)
        trigger_pattern = trigger_pattern.to(self.device)
        
        # Setup training
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        model.train()
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            poisoned_count = 0
            total_count = 0
            
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                batch_size = images.size(0)
                
                # Decide which samples to poison
                poison_mask = torch.rand(batch_size) < poisoning_rate
                poison_indices = poison_mask.nonzero(as_tuple=True)[0]
                
                # Apply trigger to poisoned samples
                if len(poison_indices) > 0:
                    poisoned_images = images.clone()
                    poisoned_labels = labels.clone()
                    
                    for idx in poison_indices:
                        poisoned_images[idx] = self._apply_trigger(
                            images[idx:idx+1], trigger_pattern
                        ).squeeze(0)
                        poisoned_labels[idx] = target_class
                        
                    poisoned_count += len(poison_indices)
                else:
                    poisoned_images = images
                    poisoned_labels = labels
                    
                total_count += batch_size
                
                # Forward pass
                optimizer.zero_grad()
                outputs = model(poisoned_images)
                loss = criterion(outputs, poisoned_labels)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                
            avg_loss = epoch_loss / len(train_loader)
            actual_poisoning_rate = poisoned_count / total_count if total_count > 0 else 0
            
            if epoch % 5 == 0:
                logger.info(f"Backdoor injection epoch {epoch}: Loss = {avg_loss:.4f}, "
                          f"Poisoning rate = {actual_poisoning_rate:.3f}")
                          
        self.injection_stats['poisoning_rate'] = actual_poisoning_rate
        
        return model
        
    def _apply_trigger(self, 
                      images: torch.Tensor, 
                      trigger_pattern: torch.Tensor) -> torch.Tensor:
        """Apply trigger pattern to images."""
        batch_size, channels, height, width = images.shape
        trigger_h, trigger_w = self.trigger_size
        
        triggered_images = images.clone()
        
        for i in range(batch_size):
            # Determine trigger location
            if self.trigger_location == 'bottom_right':
                y = height - trigger_h
                x = width - trigger_w
            elif self.trigger_location == 'top_left':
                y, x = 0, 0
            else:  # random
                y = np.random.randint(0, max(1, height - trigger_h))
                x = np.random.randint(0, max(1, width - trigger_w))
                
            # Ensure trigger fits
            y = max(0, min(y, height - trigger_h))
            x = max(0, min(x, width - trigger_w))
            
            # Apply trigger
            triggered_images[i, :, y:y+trigger_h, x:x+trigger_w] = trigger_pattern
            
        return triggered_images
        
    def detect_backdoor(self, 
                       model: nn.Module,
                       test_triggers: List[torch.Tensor],
                       clean_test_loader: DataLoader,
                       detection_method: str = 'activation_clustering') -> Dict[str, Any]:
        """
        Detect if model contains backdoor.
        
        Args:
            model: Model to test
            test_triggers: List of potential trigger patterns
            clean_test_loader: Clean test data
            detection_method: Detection method to use
            
        Returns:
            Detection results
        """
        model.eval()
        model = model.to(self.device)
        
        detection_results = {
            'backdoor_detected': False,
            'confidence': 0.0,
            'method': detection_method,
            'evidence': {}
        }
        
        if detection_method == 'activation_clustering':
            detection_results = self._detect_via_activation_clustering(
                model, test_triggers, clean_test_loader
            )
        elif detection_method == 'output_consistency':
            detection_results = self._detect_via_output_consistency(
                model, test_triggers, clean_test_loader
            )
        else:
            # Neural cleanse method
            detection_results = self._detect_via_neural_cleanse(
                model, clean_test_loader
            )
            
        return detection_results
        
    def _detect_via_activation_clustering(self, 
                                        model: nn.Module,
                                        test_triggers: List[torch.Tensor],
                                        test_loader: DataLoader) -> Dict[str, Any]:
        """Detect backdoor via activation clustering analysis."""
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        
        # Extract activations from penultimate layer
        activations = []
        labels = []
        is_triggered = []
        
        # Hook to capture activations
        activation_cache = {}
        def hook(name):
            def hook_fn(module, input, output):
                activation_cache[name] = output.detach().cpu()
            return hook_fn
            
        # Register hook on penultimate layer
        layers = list(model.children())
        if len(layers) > 1:
            layers[-2].register_forward_hook(hook('penultimate'))
        
        with torch.no_grad():
            # Collect clean activations
            for batch_idx, (images, batch_labels) in enumerate(test_loader):
                if batch_idx >= 20:  # Limit for efficiency
                    break
                    
                images = images.to(self.device)
                model(images)
                
                if 'penultimate' in activation_cache:
                    batch_activations = activation_cache['penultimate']
                    activations.append(batch_activations.flatten(1))
                    labels.extend(batch_labels.numpy())
                    is_triggered.extend([False] * len(batch_labels))
                    
            # Collect triggered activations
            for trigger in test_triggers[:3]:  # Test few triggers
                for batch_idx, (images, batch_labels) in enumerate(test_loader):
                    if batch_idx >= 10:  # Fewer triggered samples
                        break
                        
                    triggered_images = self._apply_trigger(images.to(self.device), trigger)
                    model(triggered_images)
                    
                    if 'penultimate' in activation_cache:
                        batch_activations = activation_cache['penultimate']
                        activations.append(batch_activations.flatten(1))
                        labels.extend(batch_labels.numpy())
                        is_triggered.extend([True] * len(batch_labels))
                        
        if not activations:
            return {'backdoor_detected': False, 'confidence': 0.0, 'evidence': {'error': 'No activations collected'}}
            
        # Combine all activations
        all_activations = torch.cat(activations, dim=0).numpy()
        is_triggered = np.array(is_triggered)
        
        # Perform clustering
        try:
            kmeans = KMeans(n_clusters=2, random_state=42)
            cluster_labels = kmeans.fit_predict(all_activations)
            
            # Check if triggered samples cluster together
            triggered_indices = np.where(is_triggered)[0]
            clean_indices = np.where(~is_triggered)[0]
            
            if len(triggered_indices) > 0 and len(clean_indices) > 0:
                triggered_cluster_labels = cluster_labels[triggered_indices]
                clean_cluster_labels = cluster_labels[clean_indices]
                
                # Compute separation score
                triggered_cluster_mode = np.bincount(triggered_cluster_labels).argmax()
                clean_cluster_mode = np.bincount(clean_cluster_labels).argmax()
                
                separation_score = float(triggered_cluster_mode != clean_cluster_mode)
                
                # Compute silhouette score
                silhouette = silhouette_score(all_activations, cluster_labels)
                
                backdoor_detected = separation_score > 0.5 and silhouette > 0.3
                confidence = (separation_score + silhouette) / 2
                
                return {
                    'backdoor_detected': backdoor_detected,
                    'confidence': confidence,
                    'evidence': {
                        'separation_score': separation_score,
                        'silhouette_score': silhouette,
                        'triggered_samples': len(triggered_indices),
                        'clean_samples': len(clean_indices)
                    }
                }
                
        except Exception as e:
            logger.error(f"Error in activation clustering detection: {e}")
            
        return {'backdoor_detected': False, 'confidence': 0.0, 'evidence': {'error': 'Clustering failed'}}
        
    def _detect_via_output_consistency(self, 
                                     model: nn.Module,
                                     test_triggers: List[torch.Tensor],
                                     test_loader: DataLoader) -> Dict[str, Any]:
        """Detect backdoor via output consistency analysis."""
        consistency_scores = []
        
        with torch.no_grad():
            for trigger in test_triggers[:5]:  # Test multiple triggers
                trigger_responses = []
                
                for batch_idx, (images, labels) in enumerate(test_loader):
                    if batch_idx >= 10:  # Limit samples
                        break
                        
                    images = images.to(self.device)
                    
                    # Apply trigger
                    triggered_images = self._apply_trigger(images, trigger)
                    
                    # Get predictions
                    outputs = model(triggered_images)
                    predictions = outputs.argmax(dim=1)
                    
                    trigger_responses.extend(predictions.cpu().numpy())
                    
                # Check consistency of trigger responses
                if trigger_responses:
                    trigger_responses = np.array(trigger_responses)
                    # Measure how often the same class is predicted
                    unique, counts = np.unique(trigger_responses, return_counts=True)
                    max_consistency = counts.max() / len(trigger_responses)
                    consistency_scores.append(max_consistency)
                    
        if consistency_scores:
            avg_consistency = np.mean(consistency_scores)
            backdoor_detected = avg_consistency > 0.8  # High consistency indicates backdoor
            confidence = avg_consistency
            
            return {
                'backdoor_detected': backdoor_detected,
                'confidence': confidence,
                'evidence': {
                    'average_consistency': avg_consistency,
                    'max_consistency': max(consistency_scores),
                    'triggers_tested': len(consistency_scores)
                }
            }
            
        return {'backdoor_detected': False, 'confidence': 0.0, 'evidence': {'error': 'No consistency data'}}
        
    def _detect_via_neural_cleanse(self, 
                                 model: nn.Module,
                                 test_loader: DataLoader) -> Dict[str, Any]:
        """Detect backdoor via neural cleanse method (simplified)."""
        # This is a simplified version of neural cleanse
        # Real implementation would be more complex
        
        num_classes = 10  # Assume 10 classes for simplicity
        anomaly_scores = []
        
        for target_class in range(num_classes):
            # Try to find minimal trigger for this target class
            trigger_norm = self._find_minimal_trigger_norm(model, test_loader, target_class)
            anomaly_scores.append(trigger_norm)
            
        if anomaly_scores:
            # Check for anomalous class (much smaller trigger needed)
            anomaly_scores = np.array(anomaly_scores)
            median_score = np.median(anomaly_scores)
            min_score = np.min(anomaly_scores)
            
            # Anomaly index
            anomaly_index = (median_score - min_score) / (median_score + 1e-8)
            
            backdoor_detected = anomaly_index > 0.5
            confidence = min(anomaly_index, 1.0)
            
            return {
                'backdoor_detected': backdoor_detected,
                'confidence': confidence,
                'evidence': {
                    'anomaly_index': anomaly_index,
                    'median_trigger_norm': median_score,
                    'min_trigger_norm': min_score,
                    'all_scores': anomaly_scores.tolist()
                }
            }
            
        return {'backdoor_detected': False, 'confidence': 0.0, 'evidence': {'error': 'Neural cleanse failed'}}
        
    def _find_minimal_trigger_norm(self, 
                                 model: nn.Module,
                                 test_loader: DataLoader,
                                 target_class: int) -> float:
        """Find minimal trigger norm needed to flip predictions to target class."""
        # Simplified trigger optimization
        sample_batch = next(iter(test_loader))
        sample_image = sample_batch[0][:1].to(self.device)
        
        # Initialize small trigger
        trigger = torch.zeros_like(sample_image, requires_grad=True)
        optimizer = optim.Adam([trigger], lr=0.01)
        
        for _ in range(100):  # Limited iterations
            optimizer.zero_grad()
            
            # Apply trigger (simple addition)
            triggered_image = torch.clamp(sample_image + trigger, 0, 1)
            output = model(triggered_image)
            
            # Loss: maximize target class probability
            loss = -output[0, target_class]
            loss.backward()
            optimizer.step()
            
            # Check if successful
            pred = output.argmax(dim=1).item()
            if pred == target_class:
                break
                
        return torch.norm(trigger).item()
        
    def evaluate_backdoor_effectiveness(self, 
                                      model: nn.Module,
                                      trigger_pattern: torch.Tensor,
                                      target_class: int,
                                      test_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate effectiveness of injected backdoor.
        
        Args:
            model: Model with potential backdoor
            trigger_pattern: Trigger pattern to test
            target_class: Expected target class
            test_loader: Test data
            
        Returns:
            Effectiveness metrics
        """
        model.eval()
        
        clean_correct = 0
        backdoor_success = 0
        total_samples = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Test clean accuracy
                clean_outputs = model(images)
                clean_preds = clean_outputs.argmax(dim=1)
                clean_correct += (clean_preds == labels).sum().item()
                
                # Test backdoor success
                triggered_images = self._apply_trigger(images, trigger_pattern)
                triggered_outputs = model(triggered_images)
                triggered_preds = triggered_outputs.argmax(dim=1)
                backdoor_success += (triggered_preds == target_class).sum().item()
                
                total_samples += images.size(0)
                
                if total_samples >= 1000:  # Limit evaluation
                    break
                    
        metrics = {
            'clean_accuracy': clean_correct / total_samples,
            'backdoor_success_rate': backdoor_success / total_samples,
            'stealthiness': clean_correct / total_samples,  # How well it hides
            'samples_evaluated': total_samples
        }
        
        # Update stats
        self.injection_stats.update({
            'clean_accuracy': metrics['clean_accuracy'],
            'backdoor_success_rate': metrics['backdoor_success_rate'],
            'stealthiness_score': metrics['stealthiness']
        })
        
        return metrics


# Vision-specific wrapper (for backward compatibility)
def wrapper_map(logits):
    """Simple wrapper for logit transformation."""
    return logits


# Integration functions for attack runner
def execute_vision_attack(attack_type: str, 
                         config: Dict[str, Any],
                         model: nn.Module,
                         data_loader: DataLoader,
                         device: str = 'cpu') -> AttackResult:
    """
    Execute vision-specific attack based on configuration.
    
    Args:
        attack_type: Type of attack to execute
        config: Attack configuration
        model: Target model
        data_loader: Data for attack
        device: Device for computation
        
    Returns:
        Attack result
    """
    import time
    start_time = time.time()
    
    try:
        if attack_type == 'adversarial_patch':
            attack = AdversarialPatchAttack(
                patch_size=config.get('patch_size', (32, 32)),
                patch_location=config.get('patch_location', 'random'),
                device=device
            )
            
            patch = attack.generate_patch(
                model=model,
                data_loader=data_loader,
                target_class=config.get('target_class'),
                iterations=config.get('iterations', 500),
                learning_rate=config.get('learning_rate', 0.01)
            )
            
            metrics = attack.evaluate_patch_effectiveness(
                model, patch, data_loader, config.get('target_class')
            )
            
            success = metrics['attack_success_rate'] > config.get('success_threshold', 0.1)
            
            return AttackResult(
                success=success,
                attack_type=attack_type,
                metrics=metrics,
                artifact=patch,
                execution_time=time.time() - start_time,
                metadata={'patch_size': config.get('patch_size'), 'location': config.get('patch_location')}
            )
            
        elif attack_type == 'universal_perturbation':
            attack = UniversalPerturbationAttack(
                epsilon=config.get('epsilon', 0.1),
                max_iterations=config.get('iterations', 500),
                device=device
            )
            
            perturbation = attack.compute_perturbation(
                model=model,
                data_loader=data_loader,
                target_fooling_rate=config.get('target_fooling_rate', 0.8)
            )
            
            metrics = attack.attack_stats
            success = metrics['fooling_rate'] > config.get('success_threshold', 0.5)
            
            return AttackResult(
                success=success,
                attack_type=attack_type,
                metrics=metrics,
                artifact=perturbation,
                execution_time=time.time() - start_time,
                metadata={'epsilon': config.get('epsilon'), 'fooling_rate': metrics['fooling_rate']}
            )
            
        elif attack_type == 'model_extraction':
            attack = VisionModelExtraction(
                query_budget=config.get('query_budget', 5000),
                architecture=config.get('architecture', 'resnet18'),
                device=device
            )
            
            if config.get('method') == 'jacobian':
                # Create probe images
                sample_batch = next(iter(data_loader))
                probe_images = sample_batch[0][:50]  # Use subset as probes
                
                surrogate = attack.extract_via_jacobian(
                    target_model=model,
                    probe_images=probe_images,
                    num_classes=config.get('num_classes', 10)
                )
            else:
                surrogate = attack.extract_via_prediction(
                    target_model=model,
                    synthetic_data=data_loader,
                    num_classes=config.get('num_classes', 10),
                    epochs=config.get('epochs', 20)
                )
                
            metrics = attack.evaluate_extraction_quality(model, surrogate, data_loader)
            success = metrics['agreement_rate'] > config.get('success_threshold', 0.7)
            
            return AttackResult(
                success=success,
                attack_type=attack_type,
                metrics=metrics,
                artifact=surrogate,
                execution_time=time.time() - start_time,
                metadata={'method': config.get('method'), 'queries_used': attack.queries_used}
            )
            
        elif attack_type == 'backdoor':
            attack = BackdoorAttack(
                trigger_size=config.get('trigger_size', (4, 4)),
                trigger_location=config.get('trigger_location', 'bottom_right'),
                device=device
            )
            
            trigger = attack.create_trigger_pattern(
                pattern_type=config.get('pattern_type', 'checkerboard'),
                channels=3
            )
            
            backdoored_model = attack.inject_backdoor(
                model=copy.deepcopy(model),
                train_loader=data_loader,
                trigger_pattern=trigger,
                target_class=config.get('target_class', 0),
                poisoning_rate=config.get('poisoning_rate', 0.1),
                epochs=config.get('epochs', 10)
            )
            
            metrics = attack.evaluate_backdoor_effectiveness(
                backdoored_model, trigger, config.get('target_class', 0), data_loader
            )
            
            success = metrics['backdoor_success_rate'] > config.get('success_threshold', 0.8)
            
            return AttackResult(
                success=success,
                attack_type=attack_type,
                metrics=metrics,
                artifact={'model': backdoored_model, 'trigger': trigger},
                execution_time=time.time() - start_time,
                metadata={'poisoning_rate': config.get('poisoning_rate'), 'target_class': config.get('target_class')}
            )
            
        else:
            # Fall back to core attacks
            from pot.core.attacks import distillation_loop, fine_tune_wrapper, compression_attack_enhanced
            
            if attack_type == 'distillation':
                result = distillation_loop(
                    teacher_model=model,
                    data_loader=data_loader,
                    **config
                )
                success = result.get('success', False)
                metrics = result.get('metrics', {})
                
            elif attack_type == 'wrapper':
                result = fine_tune_wrapper(
                    base_model=model,
                    data_loader=data_loader,
                    **config
                )
                success = result.get('success', False)
                metrics = result.get('metrics', {})
                
            elif attack_type == 'compression':
                result = compression_attack_enhanced(
                    model=model,
                    data_loader=data_loader,
                    **config
                )
                success = result.get('success', False)
                metrics = result.get('metrics', {})
                
            else:
                raise ValueError(f"Unknown attack type: {attack_type}")
                
            return AttackResult(
                success=success,
                attack_type=attack_type,
                metrics=metrics,
                artifact=result.get('model'),
                execution_time=time.time() - start_time,
                metadata=config
            )
            
    except Exception as e:
        logger.error(f"Error executing {attack_type} attack: {e}")
        return AttackResult(
            success=False,
            attack_type=attack_type,
            metrics={'error': str(e)},
            artifact=None,
            execution_time=time.time() - start_time,
            metadata=config
        )


# Re-export for backward compatibility
__all__ = [
    'targeted_finetune', 
    'limited_distillation', 
    'wrapper_attack', 
    'extraction_attack',
    'compression_attack',
    'distillation_attack',
    'wrapper_map',
    # New vision-specific attacks
    'AdversarialPatchAttack',
    'UniversalPerturbationAttack', 
    'VisionModelExtraction',
    'BackdoorAttack',
    'AttackResult',
    'execute_vision_attack'
]