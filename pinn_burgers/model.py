"""Rede neural e resíduo da EDP de Burgers (PyTorch)."""
from __future__ import annotations

import torch
from torch import nn


class MLP(nn.Module):
    """Perceptron multicamadas u_theta(x, t) com ativação tanh.

    As entradas são x em [-1, 1] e t em [0, 1]; internamente t é reescalado
    para [-1, 1], o que melhora o condicionamento do treino.
    """

    def __init__(self, hidden_layers: int = 8, width: int = 20):
        super().__init__()
        camadas = []
        entrada = 2
        for _ in range(hidden_layers):
            camadas += [nn.Linear(entrada, width), nn.Tanh()]
            entrada = width
        camadas.append(nn.Linear(entrada, 1))
        self.net = nn.Sequential(*camadas)

        for modulo in self.net:
            if isinstance(modulo, nn.Linear):
                nn.init.xavier_normal_(modulo.weight)
                nn.init.zeros_(modulo.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x e t têm formato (N, 1)
        z = torch.cat([x, 2.0 * t - 1.0], dim=1)
        return self.net(z)


def residuo_burgers(u_fn, x: torch.Tensor, t: torch.Tensor, nu: float) -> torch.Tensor:
    """Resíduo da EDP: r = u_t + u u_x - nu u_xx, por diferenciação automática.

    ``u_fn(x, t)`` recebe tensores (N, 1) e devolve u com formato (N, 1).
    As derivadas são mantidas no grafo (create_graph=True) para que o resíduo
    possa entrar na função de perda e ser derivado em relação aos pesos.
    """
    nu = float(nu)
    x = x.clone().detach().requires_grad_(True)
    t = t.clone().detach().requires_grad_(True)

    u = u_fn(x, t)
    uns = torch.ones_like(u)
    (u_t,) = torch.autograd.grad(u, t, grad_outputs=uns, create_graph=True)
    (u_x,) = torch.autograd.grad(u, x, grad_outputs=uns, create_graph=True)
    (u_xx,) = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)
    return u_t + u * u_x - nu * u_xx
