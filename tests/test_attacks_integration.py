import pytest

torch = pytest.importorskip("torch")
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from pot.core.attacks import (
    extraction_attack,
    limited_distillation,
    targeted_finetune,
    wrapper_attack,
)
from pot.core.fingerprint import io_hash


def test_targeted_finetune_changes_model_and_hash():
    torch.manual_seed(0)
    model = nn.Linear(1, 1)
    inputs = torch.linspace(-1, 1, 20).unsqueeze(1)
    targets = 2 * inputs
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=5, shuffle=True)

    before_weight = model.weight.clone()
    baseline_hash = io_hash(model(inputs).detach().numpy())

    targeted_finetune(model, loader, epochs=20, lr=0.1)

    after_hash = io_hash(model(inputs).detach().numpy())

    assert not torch.allclose(before_weight, model.weight)
    assert baseline_hash != after_hash


def test_limited_distillation_reduces_teacher_student_gap():
    torch.manual_seed(0)
    teacher = nn.Linear(1, 1)
    with torch.no_grad():
        teacher.weight.fill_(2.0)
        teacher.bias.zero_()

    student = nn.Linear(1, 1)

    data_x = torch.linspace(-1, 1, 100).unsqueeze(1)
    loader = DataLoader(TensorDataset(data_x, torch.zeros_like(data_x)), batch_size=10, shuffle=True)

    baseline_mse = torch.mean((student(data_x) - teacher(data_x)) ** 2).item()

    limited_distillation(
        teacher,
        student,
        loader,
        budget=50,
        temperature=2.0,
        epochs=5,
        lr=0.05,
    )

    post_mse = torch.mean((student(data_x) - teacher(data_x)) ** 2).item()
    assert post_mse < baseline_mse


def test_wrapper_attack_routes_inputs():
    class ConstantModel(nn.Module):
        def __init__(self, value: float):
            super().__init__()
            self.value = value

        def forward(self, x):
            return torch.full((x.size(0), 1), self.value)

    base = nn.Linear(1, 1)
    const_model = ConstantModel(42.0)

    def predicate(x):
        return (x > 0).any().item()

    wrapped = wrapper_attack(base, {predicate: const_model})

    inputs = torch.tensor([[-1.0], [1.0]])
    base_out = base(inputs)
    wrapped_out = wrapped(inputs)

    assert not torch.allclose(base_out, wrapped_out)
    assert io_hash(base_out.detach().numpy()) != io_hash(wrapped_out.detach().numpy())


def test_extraction_attack_trains_surrogate():
    torch.manual_seed(0)
    victim = nn.Linear(1, 1)
    with torch.no_grad():
        victim.weight.fill_(3.0)
        victim.bias.fill_(-2.0)

    x = torch.linspace(-1, 1, 200).unsqueeze(1)
    loader = DataLoader(TensorDataset(x, torch.zeros_like(x)), batch_size=20, shuffle=True)

    test_x = torch.linspace(-1, 1, 50).unsqueeze(1)
    baseline = nn.Linear(1, 1)
    baseline_mse = torch.mean((baseline(test_x) - victim(test_x)) ** 2).item()

    surrogate = extraction_attack(victim, loader, query_budget=100, epochs=5, lr=0.05)
    surrogate_mse = torch.mean((surrogate(test_x) - victim(test_x)) ** 2).item()

    assert surrogate_mse < baseline_mse
