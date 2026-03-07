import torch
from torch.utils.data import DataLoader, TensorDataset

from auditml.attacks import get_attack


class TinyNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(4, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 2),
        )

    def forward(self, x):
        return self.net(x)


def _loader(n=32):
    x = torch.rand(n, 1, 2, 2)
    y = torch.randint(0, 2, (n,))
    return DataLoader(TensorDataset(x, y), batch_size=8)


def test_all_attacks_run_on_synthetic_data():
    model = TinyNet()
    member_loader = _loader(16)
    nonmember_loader = _loader(16)

    for name in ["mia-threshold", "mia-shadow", "inversion", "attribute"]:
        attack = get_attack(name, model, device=torch.device("cpu"))
        result = attack.run(member_loader, nonmember_loader)
        metrics = attack.evaluate(result)
        assert "accuracy" in metrics
        assert result.predictions.shape[0] == result.ground_truth.shape[0]
