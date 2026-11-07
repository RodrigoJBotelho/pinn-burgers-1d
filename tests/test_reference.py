"""Testes da solução de referência (somente NumPy)."""
import numpy as np

from pinn_burgers.reference import (
    NU_PADRAO,
    erro_l2_relativo,
    solucao_exata,
    u_inicial,
)


def _diferencas_finitas(nu, t_final, nx=4001, dt=2e-5):
    """Solver independente (diferenças centradas + RK4) para validar a referência."""
    x = np.linspace(-1.0, 1.0, nx)
    dx = x[1] - x[0]
    u = u_inicial(x)

    def rhs(v):
        d = np.zeros_like(v)
        ux = (v[2:] - v[:-2]) / (2 * dx)
        uxx = (v[2:] - 2 * v[1:-1] + v[:-2]) / dx**2
        d[1:-1] = -v[1:-1] * ux + nu * uxx
        return d

    passos = int(round(t_final / dt))
    for _ in range(passos):
        k1 = rhs(u)
        k2 = rhs(u + 0.5 * dt * k1)
        k3 = rhs(u + 0.5 * dt * k2)
        k4 = rhs(u + dt * k3)
        u = u + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        u[0] = u[-1] = 0.0
    return x, u


def test_condicao_inicial():
    x = np.linspace(-1, 1, 101)
    u = solucao_exata(x, [0.0])[:, 0]
    assert np.allclose(u, -np.sin(np.pi * x), atol=1e-14)


def test_antissimetria():
    x = np.linspace(-1, 1, 201)
    u = solucao_exata(x, [0.2, 0.5, 1.0])
    assert np.max(np.abs(u + u[::-1, :])) < 1e-8


def test_contorno_nulo():
    u = solucao_exata([-1.0, 1.0], np.linspace(0.05, 1.0, 20))
    assert np.max(np.abs(u)) < 1e-6


def test_quadratura_convergida():
    x = np.linspace(-1, 1, 101)
    t = [0.25, 0.5, 1.0]
    a = solucao_exata(x, t, n_quad=100)
    b = solucao_exata(x, t, n_quad=200)
    assert np.max(np.abs(a - b)) < 1e-10


def test_n_quad_invalido():
    try:
        solucao_exata([0.0], [0.5], n_quad=1000)
    except ValueError:
        return
    raise AssertionError("era esperado ValueError")


def test_contra_diferencas_finitas():
    for t_final in (0.5, 1.0):
        x, u_fd = _diferencas_finitas(NU_PADRAO, t_final)
        u_ref = solucao_exata(x, [t_final])[:, 0]
        # O erro do solver de diferenças finitas cai ~4x a cada refinamento de malha;
        # com 4001 pontos fica em ~9e-4 (t=0,5) e ~4e-4 (t=1).
        assert np.max(np.abs(u_fd - u_ref)) < 2e-3


def test_erro_l2_relativo():
    ref = np.array([3.0, 4.0])
    assert erro_l2_relativo(ref, ref) == 0.0
    assert abs(erro_l2_relativo(2 * ref, ref) - 1.0) < 1e-12
