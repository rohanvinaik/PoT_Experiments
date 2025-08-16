"""Shared attack implementations for both vision and language models."""

import numpy as np
from typing import Any, Dict, Optional, Literal, Tuple, Callable

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    torch = None
    nn = None
    HAS_TORCH = False


def targeted_finetune(
    model: Any,
    data_loader: Any,
    epochs: int = 1,
    lr: float = 1e-3,
    device: str = "cpu",
) -> Any:
    """Targeted fine-tuning attack on a model.

    This performs standard supervised fine-tuning using a provided
    :class:`~torch.utils.data.DataLoader`.  The goal is simply to modify the
    model's parameters so that its outputs change in a predictable way, which is
    sufficient for the integration tests in this repository.

    Args:
        model: Model to attack.
        data_loader: ``DataLoader`` yielding ``(inputs, targets)`` pairs.
        epochs: Number of training epochs.
        lr: Learning rate for SGD optimizer.
        device: Device on which to run the training loop.

    Returns:
        The fine-tuned model (same instance mutated in-place).
    """
    if not HAS_TORCH:
        print("Warning: PyTorch not available, returning original model")
        return model

    model.to(device)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for _ in range(epochs):
        for batch in data_loader:
            if isinstance(batch, (list, tuple)):
                inputs, targets = batch[0], batch[1]
            else:
                inputs, targets = batch

            inputs = inputs.to(device)
            targets = targets.to(device)

            preds = model(inputs)
            loss = loss_fn(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    return model


def limited_distillation(
    teacher_model: Any,
    student_model: Any,
    data_loader: Any,
    budget: int = 1000,
    temperature: float = 4.0,
    epochs: int = 1,
    lr: float = 1e-3,
    device: str = "cpu",
) -> Any:
    """Limited-budget knowledge distillation.

    The teacher model is queried up to ``budget`` times to generate soft targets
    for the student model, which is then optimised to match the teacher's
    outputs.  For multi-dimensional outputs this uses KL divergence with
    temperature scaling; otherwise mean-squared error is used.

    Args:
        teacher_model: Model to distil from (kept frozen).
        student_model: Model being trained.
        data_loader: ``DataLoader`` providing unlabeled inputs.
        budget: Maximum number of samples to query the teacher with.
        temperature: Distillation temperature.
        epochs: Number of passes over ``data_loader``.
        lr: Learning rate for SGD optimiser.
        device: Device on which to run training.

    Returns:
        The trained student model (mutated in-place).
    """
    if not HAS_TORCH:
        print("Warning: PyTorch not available, returning original student model")
        return student_model

    teacher_model.to(device)
    student_model.to(device)
    teacher_model.eval()
    student_model.train()
    optimizer = torch.optim.SGD(student_model.parameters(), lr=lr)

    samples_used = 0
    kl = nn.KLDivLoss(reduction="batchmean")
    mse = nn.MSELoss()

    for _ in range(epochs):
        for batch in data_loader:
            if samples_used >= budget:
                break

            if isinstance(batch, (list, tuple)):
                inputs = batch[0]
            else:
                inputs = batch

            inputs = inputs.to(device)
            batch_size = inputs.size(0)

            with torch.no_grad():
                teacher_logits = teacher_model(inputs)

            student_logits = student_model(inputs)

            if teacher_logits.ndim > 1 and teacher_logits.size(-1) > 1:
                loss = kl(
                    torch.log_softmax(student_logits / temperature, dim=-1),
                    torch.softmax(teacher_logits / temperature, dim=-1),
                ) * (temperature ** 2)
            else:
                loss = mse(student_logits, teacher_logits)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            samples_used += batch_size

        if samples_used >= budget:
            break

    student_model.eval()
    return student_model


def wrapper_attack(model: Any, routing_logic: Optional[Dict[Callable[[Any], bool], Any]] = None) -> Any:
    """Create a model wrapper that routes inputs based on predicates.

    ``routing_logic`` maps predicate callables to alternative models.  During
    the forward pass each sample is evaluated against the predicates in the
    order provided; the first predicate returning ``True`` determines which
    model processes that sample.  Samples that do not satisfy any predicate are
    passed to the original ``model``.

    Args:
        model: Base model to wrap.
        routing_logic: Mapping from predicate functions to alternative models.

    Returns:
        A new ``nn.Module`` implementing the routing behaviour.  If PyTorch is
        unavailable or no routing is specified, the original model is returned.
    """
    if not HAS_TORCH or not routing_logic:
        return model

    class WrappedModel(nn.Module):
        def __init__(self, base: nn.Module, routes: Dict[Callable[[Any], bool], nn.Module]):
            super().__init__()
            self.base = base
            self.routes = routes

        def forward(self, x):
            outputs = []
            for i in range(x.size(0)):
                sample = x[i : i + 1]
                used = False
                for pred, alt in self.routes.items():
                    try:
                        if pred(sample):
                            outputs.append(alt(sample))
                            used = True
                            break
                    except Exception:
                        continue
                if not used:
                    outputs.append(self.base(sample))
            return torch.cat(outputs, dim=0)

    return WrappedModel(model, routing_logic)


def extraction_attack(
    model: Any,
    data_loader: Any,
    query_budget: int = 10000,
    epochs: int = 1,
    lr: float = 1e-3,
    device: str = "cpu",
) -> Any:
    """Train a surrogate model to mimic a victim model.

    The victim model is queried for up to ``query_budget`` samples drawn from
    ``data_loader``.  A fresh surrogate with the same architecture is trained on
    those query/response pairs using mean-squared error.

    Args:
        model: Victim model to extract from.
        data_loader: ``DataLoader`` providing unlabeled inputs.
        query_budget: Maximum number of queries to the victim.
        epochs: Number of training epochs.
        lr: Learning rate for the surrogate optimiser.
        device: Device on which to run training.

    Returns:
        Trained surrogate model approximating ``model``.  If a new model cannot
        be constructed, the original model is returned.
    """
    if not HAS_TORCH:
        print("Warning: PyTorch not available, returning original model")
        return model

    # Attempt to build a fresh surrogate with same architecture
    surrogate: nn.Module
    if isinstance(model, nn.Linear):
        surrogate = nn.Linear(model.in_features, model.out_features, bias=model.bias is not None)
    else:
        try:
            import copy

            surrogate = copy.deepcopy(model)
            for p in surrogate.parameters():
                if p.requires_grad:
                    nn.init.normal_(p, mean=0.0, std=0.02)
        except Exception:
            return model

    model.to(device)
    surrogate.to(device)
    model.eval()
    surrogate.train()

    optimizer = torch.optim.SGD(surrogate.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    samples_used = 0
    for _ in range(epochs):
        for batch in data_loader:
            if samples_used >= query_budget:
                break

            if isinstance(batch, (list, tuple)):
                inputs = batch[0]
            else:
                inputs = batch

            inputs = inputs.to(device)
            batch_size = inputs.size(0)

            with torch.no_grad():
                targets = model(inputs)

            outputs = surrogate(inputs)
            loss = loss_fn(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            samples_used += batch_size

        if samples_used >= query_budget:
            break

    surrogate.eval()
    return surrogate


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