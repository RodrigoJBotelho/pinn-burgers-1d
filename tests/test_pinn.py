"""Testes do PINN (exigem PyTorch; são ignorados se ele não estiver instalado)."""
import math

import pytest

torch = pytest.importorskip("torch")

from pinn_burgers.model import MLP, residuo_burgers  # noqa: E402
from pinn_burgers.train import Config, main, treinar  # noqa: E402


def test_residuo_confere_com_derivadas_analiticas():
    """Compara o resíduo por autograd com o cálculo à mão para u = sin(x) e^{-t}."""
    nu = 0.05
    x = torch.linspace(-1, 1, 25, dtype=torch.float64).reshape(-1, 1)
    t = torch.linspace(0, 1, 25, dtype=torch.float64).reshape(-1, 1)

    r = residuo_burgers(lambda a, b: torch.sin(a) * torch.exp(-b), x, t, nu)

    u = torch.sin(x) * torch.exp(-t)
    u_t = -u
    u_x = torch.cos(x) * torch.exp(-t)
    u_xx = -u
    esperado = u_t + u * u_x - nu * u_xx
    assert torch.allclose(r, esperado, atol=1e-12)


def test_formato_da_saida_do_modelo():
    modelo = MLP(hidden_layers=3, width=8)
    x = torch.rand(10, 1)
    t = torch.rand(10, 1)
    assert modelo(x, t).shape == (10, 1)


def test_treino_reduz_a_perda():
    cfg = Config(adam_iter=100, lbfgs_iter=0, hidden_layers=2, width=10,
                 n_f=200, n_0=20, n_b=20, lr=1e-2, log_every=50)
    resultado = treinar(cfg, verbose=False)
    hist = resultado["historico"]
    assert hist[-1]["perda"] < hist[0]["perda"]
    assert resultado["u_pred"].shape == resultado["u_ref"].shape


def test_execucao_rapida_gera_arquivos(tmp_path):
    assert main(["--quick", "--out", str(tmp_path)]) == 0
    for nome in ("perdas.csv", "metricas.json", "resumo.md",
                 "burgers_solucao.png", "burgers_perdas.png", "modelo.pt"):
        assert (tmp_path / nome).exists(), nome
