import torch
from torch import nn


def features(state: torch.Tensor, phase: torch.Tensor,
             amplitude: torch.Tensor) -> torch.Tensor:
    theta, omega = state.unbind(dim=-1)
    return torch.stack((theta.sin(), theta.cos(), omega,
                        amplitude, phase.sin(), phase.cos()), dim=-1)


class PendulumMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 2),
        )
        self.register_buffer("x_mean", torch.zeros(6))
        self.register_buffer("x_std", torch.ones(6))
        self.register_buffer("y_mean", torch.zeros(2))
        self.register_buffer("y_std", torch.ones(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = (x - self.x_mean) / self.x_std
        return self.net(z) * self.y_std + self.y_mean

    def step(self, state: torch.Tensor, phase: torch.Tensor,
             amplitude: torch.Tensor, dt: float) -> torch.Tensor:
        return state + dt * self(features(state, phase, amplitude))
