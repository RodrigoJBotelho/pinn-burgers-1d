"""Solução de referência da equação de Burgers 1D (somente NumPy).

Problema (viscoso, com contorno de Dirichlet homogêneo):

    u_t + u u_x = nu u_xx,    x em [-1, 1],  t em [0, 1]
    u(x, 0) = -sin(pi x),     u(-1, t) = u(1, t) = 0

A transformação de Cole-Hopf reduz o problema a uma equação do calor e dá uma
fórmula integral para u(x, t). A integral é calculada por quadratura de
Gauss-Hermite (mesma abordagem usada em Raissi et al., 2019).
"""
from __future__ import annotations

import numpy as np

NU_PADRAO = float(0.01 / np.pi)


def u_inicial(x):
    """Condição inicial u(x, 0) = -sin(pi x)."""
    return -np.sin(np.pi * np.asarray(x, dtype=float))


def solucao_exata(x, t, nu: float = NU_PADRAO, n_quad: int = 200) -> np.ndarray:
    """Solução de referência em uma grade.

    Retorna um array de formato (len(x), len(t)), com u[i, k] = u(x[i], t[k]).
    """
    if not 20 <= n_quad <= 300:
        # hermgauss do NumPy estoura (overflow) para ordens muito altas.
        raise ValueError("n_quad deve estar entre 20 e 300 (padrão: 200).")
    x = np.atleast_1d(np.asarray(x, dtype=float))
    t = np.atleast_1d(np.asarray(t, dtype=float))
    z, w = np.polynomial.hermite.hermgauss(n_quad)

    u = np.empty((x.size, t.size))
    for k, tk in enumerate(t):
        if tk <= 0.0:
            u[:, k] = u_inicial(x)
            continue
        s = np.sqrt(4.0 * nu * tk)
        y = x[:, None] - s * z[None, :]
        # f(y) = exp(-cos(pi y) / (2 pi nu)); estabilizado em escala logarítmica
        logf = -np.cos(np.pi * y) / (2.0 * np.pi * nu)
        logf -= logf.max(axis=1, keepdims=True)
        pesos = w[None, :] * np.exp(logf)
        u[:, k] = -(pesos * np.sin(np.pi * y)).sum(axis=1) / pesos.sum(axis=1)
    return u


def erro_l2_relativo(previsto, referencia) -> float:
    """Erro relativo na norma L2: ||previsto - referencia|| / ||referencia||."""
    previsto = np.asarray(previsto, dtype=float)
    referencia = np.asarray(referencia, dtype=float)
    return float(np.linalg.norm(previsto - referencia) / np.linalg.norm(referencia))
