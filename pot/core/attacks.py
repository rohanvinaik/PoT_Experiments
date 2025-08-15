"""Shared attack implementations for both vision and language models.

This module implements lightweight versions of common attack strategies used in
the literature.  The goal is not to be optimised or feature complete, but to
provide functioning reference implementations that can be used in tests and
examples.  All attacks operate on generic :class:`torch.nn.Module` objects so
that they can be applied to either vision or language models.
"""

from typing import Any, Callable, Dict, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F


def targeted_finetune(
    model: nn.Module,
    data_loader: DataLoader,
    epochs: int = 10,
    lr: float = 1e-3,
    loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
) -> nn.Module:
    """Targeted fine‑tuning attack.

    This attack follows the approach of Carlini et al. (2020) where a model is
    fine‑tuned on a small set of leaked challenge/response pairs in order to
    steer its behaviour toward attacker chosen outputs.

    Args:
        model: Model to attack.
        data_loader: Iterable of ``(inputs, target_outputs)`` pairs.
        epochs: Number of fine‑tuning epochs.
        lr: Optimiser learning rate.
        loss_fn: Optional loss function. Defaults to ``nn.MSELoss``.

    Returns:
        Fine‑tuned model (modified in place).
    """

    model.train()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = loss_fn or nn.MSELoss()

    for _ in range(epochs):
        for x, y in data_loader:
            optimiser.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimiser.step()

    return model


def limited_distillation(
    teacher_model: nn.Module,
    student_model: nn.Module,
    data_loader: DataLoader,
    budget: int = 1000,
    temperature: float = 4.0,
    epochs: int = 1,
    lr: float = 1e-3,
) -> nn.Module:
    """Limited knowledge distillation attack.

    Inspired by Papernot et al. (2016), the attacker queries the victim model
    (teacher) to train a student model using at most ``budget`` examples.  The
    distillation temperature controls the softness of the teacher's
    distribution.

    Args:
        teacher_model: Victim model providing soft targets.
        student_model: Model to train.
        data_loader: Iterable providing query inputs.
        budget: Maximum number of queries allowed.
        temperature: Distillation temperature.
        epochs: Number of training epochs over the queried data.
        lr: Optimiser learning rate.

    Returns:
        Trained student model.
    """

    teacher_model.eval()
    student_model.train()

    optimiser = torch.optim.Adam(student_model.parameters(), lr=lr)
    criterion = nn.KLDivLoss(reduction="batchmean")

    queries = 0
    collected_x: list[torch.Tensor] = []
    collected_y: list[torch.Tensor] = []

    # Gather up to ``budget`` query/response pairs from the teacher model.
    for x, _ in data_loader:
        if queries >= budget:
            break
        batch = x[: max(0, min(x.size(0), budget - queries))]
        with torch.no_grad():
            logits = teacher_model(batch) / temperature
            probs = F.softmax(logits, dim=-1)
        collected_x.append(batch)
        collected_y.append(probs)
        queries += batch.size(0)

    if not collected_x:
        return student_model

    dataset = TensorDataset(torch.cat(collected_x), torch.cat(collected_y))
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    for _ in range(epochs):
        for x_batch, y_batch in loader:
            optimiser.zero_grad()
            student_logits = student_model(x_batch) / temperature
            log_probs = F.log_softmax(student_logits, dim=-1)
            loss = criterion(log_probs, y_batch)
            loss.backward()
            optimiser.step()

    return student_model


def wrapper_attack(
    model: nn.Module,
    routing_logic: Optional[Dict[Callable[[torch.Tensor], bool], nn.Module]] = None,
) -> nn.Module:
    """Wrapper attack that routes queries based on predicates.

    The attack wraps a base model and consults a user supplied ``routing_logic``
    mapping.  When a predicate evaluates to ``True`` for a given input, the
    corresponding model is used to generate the response instead of the base
    model.  This mirrors the wrapper attacks described by Wallace et al. (2021).

    Args:
        model: Base model to wrap.
        routing_logic: Mapping from predicate callables to replacement models.

    Returns:
        A new model that applies the routing logic before delegating to the
        underlying model.
    """

    class Wrapper(nn.Module):
        def __init__(self, base: nn.Module, logic: Optional[Dict]):
            super().__init__()
            self.base = base
            self.logic = logic or {}

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            for predicate, alt_model in self.logic.items():
                if predicate(x):
                    return alt_model(x)
            return self.base(x)

    return Wrapper(model, routing_logic)


def extraction_attack(
    model: nn.Module,
    data_loader: DataLoader,
    query_budget: int = 10000,
    epochs: int = 1,
    lr: float = 1e-3,
) -> nn.Module:
    """Model extraction via query synthesis.

    Implements the strategy of Tramèr et al. (2016) where an attacker trains a
    surrogate model using outputs queried from the victim.  A simple linear
    surrogate is fit using mean squared error.

    Args:
        model: Victim model to emulate.
        data_loader: Source of query inputs.
        query_budget: Maximum number of queries to the victim model.
        epochs: Training epochs for the surrogate.
        lr: Optimiser learning rate.

    Returns:
        Extracted surrogate model.
    """

    model.eval()
    queries = 0
    xs: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []

    for x, _ in data_loader:
        if queries >= query_budget:
            break
        batch = x[: max(0, min(x.size(0), query_budget - queries))]
        with torch.no_grad():
            y = model(batch)
        xs.append(batch)
        ys.append(y)
        queries += batch.size(0)

    if not xs:
        raise ValueError("No queries collected for extraction attack")

    X = torch.cat(xs)
    Y = torch.cat(ys)
    input_dim = X.shape[1]
    output_dim = Y.shape[1]
    surrogate = nn.Linear(input_dim, output_dim)
    optimiser = torch.optim.Adam(surrogate.parameters(), lr=lr)
    dataset = TensorDataset(X, Y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    for _ in range(epochs):
        for x_batch, y_batch in loader:
            optimiser.zero_grad()
            pred = surrogate(x_batch)
            loss = F.mse_loss(pred, y_batch)
            loss.backward()
            optimiser.step()

    return surrogate