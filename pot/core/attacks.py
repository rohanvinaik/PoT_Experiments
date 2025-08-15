"""Shared attack implementations for both vision and language models."""

import numpy as np
from typing import Any, Dict, Optional, Literal, Tuple

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    torch = None
    nn = None
    HAS_TORCH = False


def targeted_finetune(model: Any, target_outputs: np.ndarray, epochs: int = 10) -> Any:
    """Targeted fine-tuning attack on model.
    
    Args:
        model: Model to attack
        target_outputs: Target outputs to fine-tune towards
        epochs: Number of fine-tuning epochs
        
    Returns:
        Fine-tuned model
    """
    # Placeholder for actual fine-tuning logic
    return model


def limited_distillation(teacher_model: Any, budget: int = 1000, temperature: float = 4.0) -> Any:
    """Limited distillation attack.
    
    Args:
        teacher_model: Model to distill from
        budget: Query budget for distillation
        temperature: Distillation temperature
        
    Returns:
        Distilled student model
    """
    # Placeholder for actual distillation logic
    return teacher_model


def wrapper_attack(model: Any, routing_logic: Optional[Dict] = None) -> Any:
    """Wrapper attack that routes queries.
    
    Args:
        model: Model to wrap
        routing_logic: Optional routing configuration
        
    Returns:
        Wrapped model with routing
    """
    # Placeholder for wrapper attack
    return model


def extraction_attack(model: Any, query_budget: int = 10000) -> Any:
    """Model extraction attack.
    
    Args:
        model: Model to extract
        query_budget: Number of queries allowed
        
    Returns:
        Extracted model approximation
    """
    # Placeholder for extraction attack
    return model


def compression_attack(model: Any, 
                      kind: Literal["quant", "prune"] = "quant",
                      amount: float = 0.3,
                      backend: str = "fbgemm") -> Tuple[Any, Dict]:
    """
    Apply compression attack (quantization or pruning)
    
    Args:
        model: Model to compress
        kind: Type of compression ("quant" or "prune")
        amount: Pruning amount (0-1) or quantization level
        backend: Quantization backend
        
    Returns:
        Compressed model and metadata
    """
    metadata = {"kind": kind, "amount": amount}
    
    if not HAS_TORCH:
        print("Warning: PyTorch not available, returning original model")
        metadata["error"] = "pytorch_unavailable"
        return model, metadata
    
    if kind == "quant":
        try:
            from torch.ao.quantization import quantize_dynamic
            # Dynamic quantization to int8
            compressed = quantize_dynamic(
                model, 
                {nn.Linear, nn.Conv2d} if nn else set(),  # Layers to quantize
                dtype=torch.qint8
            )
            metadata["backend"] = backend
            metadata["dtype"] = "qint8"
        except (ImportError, AttributeError):
            # Fallback if quantization not available
            print("Warning: Quantization not available, returning original model")
            compressed = model
            metadata["error"] = "quantization_unavailable"
    
    elif kind == "prune":
        if not HAS_TORCH:
            compressed = model
            metadata["error"] = "pytorch_unavailable"
        else:
            try:
                import torch.nn.utils.prune as prune
                compressed = model
                
                # Track pruned layers
                pruned_layers = []
                
                for name, module in model.named_modules():
                    if hasattr(module, 'weight'):
                        # Apply L1 unstructured pruning
                        prune.l1_unstructured(module, name='weight', amount=amount)
                        pruned_layers.append(name)
                        
                        # Make pruning permanent
                        prune.remove(module, 'weight')
                
                metadata["pruned_layers"] = pruned_layers
                metadata["prune_method"] = "l1_unstructured"
            except Exception as e:
                print(f"Warning: Pruning failed: {e}")
                compressed = model
                metadata["error"] = str(e)
    
    else:
        raise ValueError(f"Unknown compression kind: {kind}")
    
    # Compute compression ratio if possible
    if HAS_TORCH and hasattr(model, 'parameters'):
        try:
            original_params = sum(p.numel() for p in model.parameters())
            compressed_params = sum(p.numel() for p in compressed.parameters())
            
            # For pruning, count actual non-zero params
            if kind == "prune" and "error" not in metadata:
                non_zero = sum((p != 0).sum().item() for p in compressed.parameters())
                metadata["compression_ratio"] = 1 - (non_zero / original_params)
                metadata["non_zero_params"] = non_zero
            else:
                metadata["compression_ratio"] = 1 - (compressed_params / original_params)
            
            metadata["original_params"] = original_params
            metadata["compressed_params"] = compressed_params
        except:
            pass
    
    return compressed, metadata


def distillation_attack(student: Any,
                        teacher: Any,
                        data_loader: Any = None,
                        budget: int = 10000,
                        lr: float = 1e-3,
                        temperature: float = 3.0,
                        device: str = 'cpu') -> Tuple[Any, Dict]:
    """
    Knowledge distillation attack with limited budget
    
    Args:
        student: Student model to train
        teacher: Teacher model to distill from
        data_loader: Data loader for training (optional)
        budget: Maximum number of samples to use
        lr: Learning rate
        temperature: Distillation temperature
        device: Device to use
        
    Returns:
        Distilled student model and metadata
    """
    metadata = {
        "budget": budget,
        "temperature": temperature,
        "lr": lr
    }
    
    if not HAS_TORCH:
        print("Warning: PyTorch not available, returning original student model")
        metadata["error"] = "pytorch_unavailable"
        return student, metadata
    
    if data_loader is None:
        print("Warning: No data loader provided, returning original student model")
        metadata["error"] = "no_data_loader"
        return student, metadata
    
    try:
        import torch.nn.functional as F
        
        student = student.to(device)
        teacher = teacher.to(device)
        teacher.eval()
        student.train()
        
        optimizer = torch.optim.Adam(student.parameters(), lr=lr)
        
        samples_used = 0
        total_loss = 0
        n_batches = 0
        
        for batch_idx, batch in enumerate(data_loader):
            if samples_used >= budget:
                break
            
            # Handle different data loader formats
            if isinstance(batch, (list, tuple)):
                data = batch[0]
            else:
                data = batch
            
            data = data.to(device)
            batch_size = data.shape[0]
            
            # Get teacher predictions
            with torch.no_grad():
                teacher_logits = teacher(data)
            
            # Get student predictions
            student_logits = student(data)
            
            # KL divergence loss with temperature
            loss = F.kl_div(
                F.log_softmax(student_logits / temperature, dim=-1),
                F.softmax(teacher_logits / temperature, dim=-1),
                reduction='batchmean'
            ) * (temperature ** 2)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            samples_used += batch_size
            total_loss += loss.item()
            n_batches += 1
            
            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}, Samples: {samples_used}/{budget}, "
                      f"Loss: {loss.item():.4f}")
        
        student.eval()
        
        metadata.update({
            "samples_used": samples_used,
            "avg_loss": total_loss / n_batches if n_batches > 0 else 0,
            "n_batches": n_batches
        })
        
    except Exception as e:
        print(f"Warning: Distillation failed: {e}")
        metadata["error"] = str(e)
    
    return student, metadata