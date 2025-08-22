"""Shared attack implementations for both vision and language models."""

import numpy as np
from typing import Any, Dict, Optional, Literal, Tuple, Callable, List
import time

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    torch = None
    nn = None
    HAS_TORCH = False
    import warnings
    warnings.warn("PyTorch not available; attack utilities will be limited.")


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


def distillation_loop(teacher_model: nn.Module, 
                      student_model: nn.Module,
                      train_loader,
                      temperature: float = 3.0,
                      alpha: float = 0.7,
                      epochs: int = 10,
                      learning_rate: float = 0.001,
                      device: str = 'cuda') -> Dict[str, Any]:
    """
    Complete knowledge distillation attack.
    
    Args:
        teacher_model: Protected model to distill from
        student_model: Model to train via distillation
        train_loader: DataLoader for training data
        temperature: Softmax temperature for distillation
        alpha: Weight for distillation loss vs task loss
        epochs: Training epochs
        learning_rate: Learning rate for optimization
        device: Device to use for training
        
    Returns:
        Dictionary with attack metrics (loss curves, accuracy, success rate)
    """
    if not HAS_TORCH:
        return {"error": "pytorch_unavailable"}
    
    try:
        import torch.nn.functional as F
        
        # Setup models and optimizers
        teacher_model = teacher_model.to(device)
        student_model = student_model.to(device)
        teacher_model.eval()
        student_model.train()
        
        optimizer = torch.optim.Adam(student_model.parameters(), lr=learning_rate)
        
        # Metrics tracking
        metrics = {
            "distillation_losses": [],
            "task_losses": [],
            "total_losses": [],
            "fidelity_scores": [],  # Agreement rate with teacher
            "epoch_times": [],
            "teacher_agreement": [],
            "convergence_epoch": None
        }
        
        best_fidelity = 0.0
        patience_counter = 0
        patience = 3
        
        for epoch in range(epochs):
            epoch_start = time.time()
            epoch_distill_loss = 0.0
            epoch_task_loss = 0.0
            epoch_total_loss = 0.0
            epoch_fidelity = 0.0
            n_batches = 0
            n_samples = 0
            
            for batch_idx, batch in enumerate(train_loader):
                # Handle different batch formats
                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    inputs, targets = batch[0], batch[1]
                    has_targets = True
                elif isinstance(batch, (list, tuple)):
                    inputs = batch[0]
                    has_targets = False
                else:
                    inputs = batch
                    has_targets = False
                
                inputs = inputs.to(device)
                if has_targets:
                    targets = targets.to(device)
                
                batch_size = inputs.shape[0]
                
                # Get teacher and student predictions
                with torch.no_grad():
                    teacher_logits = teacher_model(inputs)
                
                student_logits = student_model(inputs)
                
                # Distillation loss (KL divergence with temperature)
                distillation_loss = F.kl_div(
                    F.log_softmax(student_logits / temperature, dim=-1),
                    F.softmax(teacher_logits / temperature, dim=-1),
                    reduction='batchmean'
                ) * (temperature ** 2)
                
                # Task loss (if labels available)
                if has_targets:
                    if len(targets.shape) > 1 and targets.shape[-1] > 1:
                        # Multi-class classification
                        task_loss = F.cross_entropy(student_logits, targets.argmax(dim=-1))
                    else:
                        # Regression or binary classification
                        task_loss = F.mse_loss(student_logits, targets)
                else:
                    task_loss = torch.tensor(0.0, device=device)
                
                # Combined loss
                total_loss = alpha * distillation_loss + (1 - alpha) * task_loss
                
                # Optimization step
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                
                # Compute fidelity (agreement with teacher)
                with torch.no_grad():
                    if teacher_logits.shape[-1] > 1:
                        # Classification: check if predictions match
                        teacher_pred = teacher_logits.argmax(dim=-1)
                        student_pred = student_logits.argmax(dim=-1)
                        batch_fidelity = (teacher_pred == student_pred).float().mean().item()
                    else:
                        # Regression: use relative error
                        rel_error = torch.abs(teacher_logits - student_logits) / (torch.abs(teacher_logits) + 1e-8)
                        batch_fidelity = (rel_error < 0.1).float().mean().item()
                
                # Accumulate metrics
                epoch_distill_loss += distillation_loss.item() * batch_size
                epoch_task_loss += task_loss.item() * batch_size
                epoch_total_loss += total_loss.item() * batch_size
                epoch_fidelity += batch_fidelity * batch_size
                n_batches += 1
                n_samples += batch_size
                
                if batch_idx % 50 == 0:
                    print(f"Epoch {epoch}, Batch {batch_idx}: "
                          f"Distill: {distillation_loss.item():.4f}, "
                          f"Task: {task_loss.item():.4f}, "
                          f"Fidelity: {batch_fidelity:.3f}")
            
            # Average metrics for epoch
            if n_samples > 0:
                avg_distill_loss = epoch_distill_loss / n_samples
                avg_task_loss = epoch_task_loss / n_samples
                avg_total_loss = epoch_total_loss / n_samples
                avg_fidelity = epoch_fidelity / n_samples
                
                metrics["distillation_losses"].append(avg_distill_loss)
                metrics["task_losses"].append(avg_task_loss)
                metrics["total_losses"].append(avg_total_loss)
                metrics["fidelity_scores"].append(avg_fidelity)
                metrics["epoch_times"].append(time.time() - epoch_start)
                
                print(f"Epoch {epoch} complete - "
                      f"Avg Fidelity: {avg_fidelity:.3f}, "
                      f"Avg Total Loss: {avg_total_loss:.4f}")
                
                # Early stopping based on fidelity
                if avg_fidelity > best_fidelity:
                    best_fidelity = avg_fidelity
                    patience_counter = 0
                    if avg_fidelity > 0.9 and metrics["convergence_epoch"] is None:
                        metrics["convergence_epoch"] = epoch
                else:
                    patience_counter += 1
                
                if patience_counter >= patience and epoch > 2:
                    print(f"Early stopping at epoch {epoch} due to fidelity plateau")
                    break
        
        student_model.eval()
        
        # Final evaluation metrics
        metrics.update({
            "final_fidelity": best_fidelity,
            "epochs_trained": epoch + 1,
            "early_stopped": patience_counter >= patience,
            "attack_success": best_fidelity > 0.8,  # Threshold for successful distillation
            "average_epoch_time": np.mean(metrics["epoch_times"]) if metrics["epoch_times"] else 0,
            "total_training_time": sum(metrics["epoch_times"]) if metrics["epoch_times"] else 0
        })
        
        return metrics
        
    except Exception as e:
        return {"error": str(e)}


def fine_tune_wrapper(base_model: nn.Module,
                      wrapper_layers: List[nn.Module],
                      train_loader,
                      attack_budget: Dict[str, Any],
                      optimization_config: Dict[str, Any]) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Fine-tune a wrapper around protected model.
    
    Args:
        base_model: Protected/frozen model
        wrapper_layers: Trainable wrapper layers
        train_loader: DataLoader for training
        attack_budget: {'epochs': int, 'queries': int, 'compute_time': float}
        optimization_config: Optimizer settings
        
    Returns:
        Wrapped model and attack metrics
    """
    if not HAS_TORCH:
        return base_model, {"error": "pytorch_unavailable"}
    
    try:
        device = optimization_config.get('device', 'cpu')
        learning_rate = optimization_config.get('learning_rate', 0.001)
        optimizer_type = optimization_config.get('optimizer', 'adam')
        
        # Create wrapper model
        class WrapperModel(nn.Module):
            def __init__(self, base: nn.Module, wrappers: List[nn.Module]):
                super().__init__()
                self.base = base
                self.wrappers = nn.ModuleList(wrappers)
                
                # Freeze base model
                for param in self.base.parameters():
                    param.requires_grad = False
            
            def forward(self, x):
                # Forward through base model
                with torch.no_grad():
                    base_output = self.base(x)
                
                # Apply wrapper layers
                output = base_output
                for wrapper in self.wrappers:
                    output = wrapper(output)
                
                return output
        
        wrapped_model = WrapperModel(base_model, wrapper_layers)
        wrapped_model = wrapped_model.to(device)
        wrapped_model.train()
        
        # Setup optimizer for wrapper parameters only
        wrapper_params = []
        for wrapper in wrapped_model.wrappers:
            wrapper_params.extend(wrapper.parameters())
        
        if optimizer_type.lower() == 'adam':
            optimizer = torch.optim.Adam(wrapper_params, lr=learning_rate)
        elif optimizer_type.lower() == 'sgd':
            optimizer = torch.optim.SGD(wrapper_params, lr=learning_rate, momentum=0.9)
        else:
            optimizer = torch.optim.Adam(wrapper_params, lr=learning_rate)
        
        # Training metrics
        metrics = {
            "wrapper_losses": [],
            "queries_used": 0,
            "epochs_completed": 0,
            "training_time": 0,
            "wrapper_adaptation": [],  # Track how much wrappers change outputs
        }
        
        max_epochs = attack_budget.get('epochs', 10)
        max_queries = attack_budget.get('queries', 10000)
        max_time = attack_budget.get('compute_time', 300.0)  # 5 minutes default
        
        start_time = time.time()
        
        for epoch in range(max_epochs):
            if time.time() - start_time > max_time:
                print(f"Time budget exceeded at epoch {epoch}")
                break
            
            epoch_loss = 0.0
            epoch_adaptation = 0.0
            n_batches = 0
            
            for batch_idx, batch in enumerate(train_loader):
                if metrics["queries_used"] >= max_queries:
                    print(f"Query budget exceeded at epoch {epoch}, batch {batch_idx}")
                    break
                
                # Handle different batch formats
                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    inputs, targets = batch[0], batch[1]
                else:
                    inputs = batch[0] if isinstance(batch, (list, tuple)) else batch
                    # Create dummy targets for self-supervised learning
                    with torch.no_grad():
                        targets = base_model(inputs.to(device))
                
                inputs = inputs.to(device)
                targets = targets.to(device)
                batch_size = inputs.shape[0]
                
                # Get base model output for adaptation measurement
                with torch.no_grad():
                    base_output = base_model(inputs)
                
                # Forward through wrapped model
                wrapped_output = wrapped_model(inputs)
                
                # Loss computation
                if len(targets.shape) > 1 and targets.shape[-1] > 1:
                    loss = nn.CrossEntropyLoss()(wrapped_output, targets.argmax(dim=-1))
                else:
                    loss = nn.MSELoss()(wrapped_output, targets)
                
                # Measure adaptation (how much wrapper changes base output)
                adaptation = torch.norm(wrapped_output - base_output).item() / batch_size
                
                # Optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Update metrics
                epoch_loss += loss.item() * batch_size
                epoch_adaptation += adaptation
                metrics["queries_used"] += batch_size
                n_batches += 1
                
                if batch_idx % 20 == 0:
                    print(f"Epoch {epoch}, Batch {batch_idx}: "
                          f"Loss: {loss.item():.4f}, "
                          f"Adaptation: {adaptation:.4f}, "
                          f"Queries: {metrics['queries_used']}/{max_queries}")
            
            if n_batches > 0:
                avg_loss = epoch_loss / (n_batches * train_loader.batch_size)
                avg_adaptation = epoch_adaptation / n_batches
                
                metrics["wrapper_losses"].append(avg_loss)
                metrics["wrapper_adaptation"].append(avg_adaptation)
                metrics["epochs_completed"] = epoch + 1
            
            if metrics["queries_used"] >= max_queries:
                break
        
        metrics["training_time"] = time.time() - start_time
        
        # Final evaluation
        wrapped_model.eval()
        metrics.update({
            "final_loss": metrics["wrapper_losses"][-1] if metrics["wrapper_losses"] else float('inf'),
            "total_adaptation": sum(metrics["wrapper_adaptation"]) if metrics["wrapper_adaptation"] else 0,
            "attack_success": len(metrics["wrapper_losses"]) > 0 and metrics["wrapper_losses"][-1] < 1.0,
            "budget_utilization": {
                "epochs": metrics["epochs_completed"] / max_epochs,
                "queries": metrics["queries_used"] / max_queries,
                "time": metrics["training_time"] / max_time
            }
        })
        
        return wrapped_model, metrics
        
    except Exception as e:
        return base_model, {"error": str(e)}


def compression_attack_enhanced(model: nn.Module,
                               compression_type: str = 'pruning',
                               compression_ratio: float = 0.5,
                               fine_tune_epochs: int = 5,
                               learning_rate: float = 0.001,
                               train_loader=None,
                               device: str = 'cpu') -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Apply model compression attacks (pruning/quantization) with optional fine-tuning.
    
    Args:
        model: Model to compress
        compression_type: 'pruning', 'quantization', or 'both'
        compression_ratio: Target compression ratio
        fine_tune_epochs: Epochs for post-compression fine-tuning
        learning_rate: Learning rate for fine-tuning
        train_loader: DataLoader for fine-tuning
        device: Device to use
        
    Returns:
        Compressed model and metrics
    """
    if not HAS_TORCH:
        return model, {"error": "pytorch_unavailable"}
    
    try:
        model = model.to(device)
        metrics = {
            "compression_type": compression_type,
            "target_ratio": compression_ratio,
            "original_size": 0,
            "compressed_size": 0,
            "actual_ratio": 0,
            "fine_tune_epochs": fine_tune_epochs,
            "fine_tune_losses": [],
            "compression_time": 0,
            "fine_tune_time": 0
        }
        
        # Measure original model size
        original_params = sum(p.numel() for p in model.parameters())
        metrics["original_size"] = original_params
        
        start_time = time.time()
        
        if compression_type in ['pruning', 'both']:
            # Apply pruning
            try:
                import torch.nn.utils.prune as prune
                
                pruned_layers = []
                for name, module in model.named_modules():
                    if isinstance(module, (nn.Linear, nn.Conv2d)):
                        prune.l1_unstructured(module, name='weight', amount=compression_ratio)
                        pruned_layers.append(name)
                
                # Make pruning permanent
                for name, module in model.named_modules():
                    if isinstance(module, (nn.Linear, nn.Conv2d)):
                        prune.remove(module, 'weight')
                
                metrics["pruned_layers"] = pruned_layers
                metrics["pruning_method"] = "l1_unstructured"
                
            except ImportError:
                print("Warning: torch.nn.utils.prune not available")
                metrics["pruning_error"] = "prune_unavailable"
        
        if compression_type in ['quantization', 'both']:
            # Apply quantization
            try:
                from torch.ao.quantization import quantize_dynamic
                model = quantize_dynamic(
                    model,
                    {nn.Linear, nn.Conv2d},
                    dtype=torch.qint8
                )
                metrics["quantization_dtype"] = "qint8"
                
            except ImportError:
                print("Warning: quantization not available")
                metrics["quantization_error"] = "quantization_unavailable"
        
        metrics["compression_time"] = time.time() - start_time
        
        # Measure compressed size
        try:
            if compression_type == 'pruning':
                # Count non-zero parameters for pruning
                non_zero_params = sum((p != 0).sum().item() for p in model.parameters() if p.requires_grad)
                metrics["compressed_size"] = non_zero_params
                metrics["actual_ratio"] = 1 - (non_zero_params / original_params)
            else:
                # For quantization, estimate size reduction
                compressed_params = sum(p.numel() for p in model.parameters())
                metrics["compressed_size"] = compressed_params
                if compression_type == 'quantization':
                    # Approximate 4x reduction for int8 quantization
                    metrics["actual_ratio"] = 0.75
                else:
                    metrics["actual_ratio"] = 1 - (compressed_params / original_params)
        except:
            metrics["actual_ratio"] = compression_ratio  # Fallback estimate
        
        # Optional fine-tuning to recover performance
        if fine_tune_epochs > 0 and train_loader is not None:
            fine_tune_start = time.time()
            
            model.train()
            optimizer = torch.optim.Adam(
                [p for p in model.parameters() if p.requires_grad], 
                lr=learning_rate
            )
            
            for epoch in range(fine_tune_epochs):
                epoch_loss = 0.0
                n_batches = 0
                
                for batch in train_loader:
                    if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                        inputs, targets = batch[0], batch[1]
                    else:
                        inputs = batch[0] if isinstance(batch, (list, tuple)) else batch
                        targets = None
                    
                    inputs = inputs.to(device)
                    if targets is not None:
                        targets = targets.to(device)
                    
                    outputs = model(inputs)
                    
                    if targets is not None:
                        if len(targets.shape) > 1 and targets.shape[-1] > 1:
                            loss = nn.CrossEntropyLoss()(outputs, targets.argmax(dim=-1))
                        else:
                            loss = nn.MSELoss()(outputs, targets)
                    else:
                        # Self-supervised: minimize output variance
                        loss = torch.var(outputs)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    n_batches += 1
                
                if n_batches > 0:
                    avg_loss = epoch_loss / n_batches
                    metrics["fine_tune_losses"].append(avg_loss)
                    
                    if epoch % 2 == 0:
                        print(f"Fine-tune epoch {epoch}: Loss = {avg_loss:.4f}")
            
            metrics["fine_tune_time"] = time.time() - fine_tune_start
            model.eval()
        
        # Final metrics
        metrics.update({
            "compression_success": metrics["actual_ratio"] > 0.1,
            "fine_tune_success": len(metrics["fine_tune_losses"]) > 0 and 
                               metrics["fine_tune_losses"][-1] < 2.0,
            "total_time": metrics["compression_time"] + metrics["fine_tune_time"]
        })
        
        return model, metrics
        
    except Exception as e:
        return model, {"error": str(e)}


def evaluate_attack_metrics(original_model: nn.Module,
                           attacked_model: nn.Module,
                           test_loader,
                           device: str = 'cpu') -> Dict[str, Any]:
    """
    Comprehensive evaluation metrics for attack success.
    
    Args:
        original_model: Original/reference model
        attacked_model: Model after attack
        test_loader: Test data for evaluation
        device: Device to use
        
    Returns:
        Dictionary with comprehensive attack evaluation metrics
    """
    if not HAS_TORCH:
        return {"error": "pytorch_unavailable"}
    
    try:
        import torch.nn.functional as F
        
        original_model = original_model.to(device)
        attacked_model = attacked_model.to(device)
        original_model.eval()
        attacked_model.eval()
        
        metrics = {
            "accuracy_original": 0.0,
            "accuracy_attacked": 0.0,
            "accuracy_drop": 0.0,
            "fidelity_score": 0.0,  # Agreement between models
            "kl_divergence": 0.0,
            "cosine_similarity": 0.0,
            "l2_distance": 0.0,
            "prediction_agreement": 0.0,
            "output_variance_original": 0.0,
            "output_variance_attacked": 0.0,
            "confidence_correlation": 0.0,
            "total_samples": 0
        }
        
        total_samples = 0
        correct_original = 0
        correct_attacked = 0
        total_fidelity = 0.0
        total_kl_div = 0.0
        total_cosine_sim = 0.0
        total_l2_dist = 0.0
        agreement_count = 0
        
        original_outputs = []
        attacked_outputs = []
        original_confidences = []
        attacked_confidences = []
        
        with torch.no_grad():
            for batch in test_loader:
                if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                    inputs, targets = batch[0], batch[1]
                    has_targets = True
                else:
                    inputs = batch[0] if isinstance(batch, (list, tuple)) else batch
                    has_targets = False
                
                inputs = inputs.to(device)
                if has_targets:
                    targets = targets.to(device)
                
                batch_size = inputs.shape[0]
                
                # Get model outputs
                original_logits = original_model(inputs)
                attacked_logits = attacked_model(inputs)
                
                # Store for variance and correlation analysis
                original_outputs.append(original_logits.cpu())
                attacked_outputs.append(attacked_logits.cpu())
                
                # Accuracy computation (if targets available)
                if has_targets:
                    if len(original_logits.shape) > 1 and original_logits.shape[-1] > 1:
                        # Classification
                        orig_pred = original_logits.argmax(dim=-1)
                        attack_pred = attacked_logits.argmax(dim=-1)
                        
                        if len(targets.shape) > 1:
                            target_labels = targets.argmax(dim=-1)
                        else:
                            target_labels = targets.long()
                        
                        correct_original += (orig_pred == target_labels).sum().item()
                        correct_attacked += (attack_pred == target_labels).sum().item()
                        
                        # Prediction agreement
                        agreement_count += (orig_pred == attack_pred).sum().item()
                        
                        # Confidence analysis (softmax probabilities)
                        orig_probs = F.softmax(original_logits, dim=-1)
                        attack_probs = F.softmax(attacked_logits, dim=-1)
                        
                        # Get confidence (max probability)
                        orig_conf = orig_probs.max(dim=-1)[0]
                        attack_conf = attack_probs.max(dim=-1)[0]
                        
                        original_confidences.append(orig_conf.cpu())
                        attacked_confidences.append(attack_conf.cpu())
                        
                        # KL divergence
                        kl_div = F.kl_div(
                            F.log_softmax(attacked_logits, dim=-1),
                            F.softmax(original_logits, dim=-1),
                            reduction='batchmean'
                        )
                        total_kl_div += kl_div.item() * batch_size
                        
                    else:
                        # Regression
                        mse_orig = F.mse_loss(original_logits, targets)
                        mse_attack = F.mse_loss(attacked_logits, targets)
                        
                        # Use MSE as proxy for accuracy (lower is better)
                        correct_original += batch_size * (1.0 / (1.0 + mse_orig.item()))
                        correct_attacked += batch_size * (1.0 / (1.0 + mse_attack.item()))
                
                # Fidelity measures
                # Cosine similarity
                orig_flat = original_logits.view(batch_size, -1)
                attack_flat = attacked_logits.view(batch_size, -1)
                
                cosine_sim = F.cosine_similarity(orig_flat, attack_flat, dim=-1).mean()
                total_cosine_sim += cosine_sim.item() * batch_size
                
                # L2 distance
                l2_dist = torch.norm(orig_flat - attack_flat, dim=-1).mean()
                total_l2_dist += l2_dist.item() * batch_size
                
                total_samples += batch_size
        
        # Aggregate metrics
        if total_samples > 0:
            metrics["accuracy_original"] = correct_original / total_samples
            metrics["accuracy_attacked"] = correct_attacked / total_samples
            metrics["accuracy_drop"] = metrics["accuracy_original"] - metrics["accuracy_attacked"]
            
            metrics["prediction_agreement"] = agreement_count / total_samples
            metrics["kl_divergence"] = total_kl_div / total_samples
            metrics["cosine_similarity"] = total_cosine_sim / total_samples
            metrics["l2_distance"] = total_l2_dist / total_samples
            metrics["total_samples"] = total_samples
            
            # Fidelity score (combination of agreement and similarity)
            metrics["fidelity_score"] = (
                0.4 * metrics["prediction_agreement"] +
                0.3 * metrics["cosine_similarity"] +
                0.3 * (1.0 / (1.0 + metrics["l2_distance"]))
            )
        
        # Output variance analysis
        if original_outputs and attacked_outputs:
            original_concat = torch.cat(original_outputs, dim=0)
            attacked_concat = torch.cat(attacked_outputs, dim=0)
            
            metrics["output_variance_original"] = torch.var(original_concat).item()
            metrics["output_variance_attacked"] = torch.var(attacked_concat).item()
            
            # Confidence correlation (if available)
            if original_confidences and attacked_confidences:
                orig_conf_concat = torch.cat(original_confidences, dim=0)
                attack_conf_concat = torch.cat(attacked_confidences, dim=0)
                
                # Pearson correlation
                orig_centered = orig_conf_concat - orig_conf_concat.mean()
                attack_centered = attack_conf_concat - attack_conf_concat.mean()
                
                correlation = (orig_centered * attack_centered).sum() / (
                    torch.sqrt((orig_centered ** 2).sum() * (attack_centered ** 2).sum()) + 1e-8
                )
                metrics["confidence_correlation"] = correlation.item()
        
        # Attack success classification
        metrics["attack_success"] = {
            "high_fidelity": metrics["fidelity_score"] > 0.8,
            "maintained_accuracy": metrics["accuracy_drop"] < 0.1,
            "strong_agreement": metrics["prediction_agreement"] > 0.8,
            "low_divergence": metrics["kl_divergence"] < 0.5,
            "overall": (
                metrics["fidelity_score"] > 0.7 and
                metrics["accuracy_drop"] < 0.2 and
                metrics["prediction_agreement"] > 0.7
            )
        }
        
        return metrics
        
    except Exception as e:
        return {"error": str(e)}


def attack_comparison_summary(attack_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a summary comparison of different attack methods.
    
    Args:
        attack_results: Dictionary mapping attack names to their results
        
    Returns:
        Summary comparison metrics
    """
    summary = {
        "attack_methods": list(attack_results.keys()),
        "best_attack": None,
        "rankings": {},
        "success_rates": {},
        "efficiency_metrics": {},
        "comparative_analysis": {}
    }
    
    try:
        # Extract key metrics for comparison
        attack_scores = {}
        
        for attack_name, results in attack_results.items():
            if isinstance(results, dict) and "error" not in results:
                # Compute composite score based on available metrics
                score = 0.0
                weight_sum = 0.0
                
                # Fidelity component
                if "fidelity_score" in results:
                    score += results["fidelity_score"] * 0.4
                    weight_sum += 0.4
                elif "final_fidelity" in results:
                    score += results["final_fidelity"] * 0.4
                    weight_sum += 0.4
                
                # Success component
                if "attack_success" in results:
                    if isinstance(results["attack_success"], dict):
                        success_score = sum(results["attack_success"].values()) / len(results["attack_success"])
                    else:
                        success_score = float(results["attack_success"])
                    score += success_score * 0.3
                    weight_sum += 0.3
                elif "compression_success" in results:
                    score += float(results["compression_success"]) * 0.3
                    weight_sum += 0.3
                
                # Efficiency component
                efficiency_score = 1.0  # Default
                if "total_time" in results and results["total_time"] > 0:
                    # Normalize by reasonable time bounds (0-300 seconds)
                    efficiency_score = max(0, 1.0 - results["total_time"] / 300.0)
                elif "total_training_time" in results and results["total_training_time"] > 0:
                    efficiency_score = max(0, 1.0 - results["total_training_time"] / 300.0)
                
                score += efficiency_score * 0.3
                weight_sum += 0.3
                
                # Normalize score
                if weight_sum > 0:
                    attack_scores[attack_name] = score / weight_sum
                else:
                    attack_scores[attack_name] = 0.0
                
                # Success rate
                if "attack_success" in results:
                    if isinstance(results["attack_success"], dict):
                        summary["success_rates"][attack_name] = results["attack_success"].get("overall", False)
                    else:
                        summary["success_rates"][attack_name] = bool(results["attack_success"])
                else:
                    summary["success_rates"][attack_name] = False
                
                # Efficiency metrics
                summary["efficiency_metrics"][attack_name] = {
                    "time": results.get("total_time", results.get("total_training_time", 0)),
                    "queries": results.get("queries_used", 0),
                    "epochs": results.get("epochs_completed", results.get("epochs_trained", 0))
                }
        
        # Rankings
        if attack_scores:
            sorted_attacks = sorted(attack_scores.items(), key=lambda x: x[1], reverse=True)
            summary["best_attack"] = sorted_attacks[0][0]
            summary["rankings"] = {attack: rank + 1 for rank, (attack, _) in enumerate(sorted_attacks)}
        
        # Comparative analysis
        if len(attack_scores) > 1:
            scores = list(attack_scores.values())
            summary["comparative_analysis"] = {
                "score_range": max(scores) - min(scores),
                "average_score": sum(scores) / len(scores),
                "score_std": np.std(scores) if len(scores) > 1 else 0.0,
                "clear_winner": max(scores) - sorted(scores, reverse=True)[1] > 0.2 if len(scores) > 1 else True
            }
        
        return summary
        
    except Exception as e:
        summary["error"] = str(e)
        return summary