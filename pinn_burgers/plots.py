"""Figuras dos resultados (Matplotlib, sem interface gráfica)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def figura_solucao(x, t, u_ref, u_pred, caminho, instantes=(0.25, 0.5, 0.75)):
    """Mapas de calor (referência, PINN, erro absoluto) e perfis em instantes fixos."""
    x = np.asarray(x)
    t = np.asarray(t)
    extent = [t[0], t[-1], x[0], x[-1]]
    erro = np.abs(u_pred - u_ref)

    fig, eixos = plt.subplots(2, 3, figsize=(13, 7.5), constrained_layout=True)

    for ax, dados, titulo, cmap, vmin, vmax in (
        (eixos[0, 0], u_ref, "Referência (Cole-Hopf)", "RdBu_r", -1.0, 1.0),
        (eixos[0, 1], u_pred, "PINN", "RdBu_r", -1.0, 1.0),
        (eixos[0, 2], erro, "Erro absoluto", "magma", 0.0, None),
    ):
        im = ax.imshow(dados, extent=extent, origin="lower", aspect="auto",
                       cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(titulo)
        ax.set_xlabel("t")
        ax.set_ylabel("x")
        fig.colorbar(im, ax=ax)

    for ax, instante in zip(eixos[1], instantes):
        k = int(np.argmin(np.abs(t - instante)))
        ax.plot(x, u_ref[:, k], "k-", lw=2, label="referência")
        ax.plot(x, u_pred[:, k], "r--", lw=2, label="PINN")
        ax.set_title(f"t = {t[k]:.2f}")
        ax.set_xlabel("x")
        ax.set_ylabel("u")
        ax.grid(alpha=0.3)
        ax.legend()

    fig.savefig(caminho, dpi=150)
    plt.close(fig)


def figura_perdas(historico, caminho):
    """Curvas de perda e de erro L2 relativo ao longo do treino."""
    it = [h["iteracao_global"] for h in historico]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)

    ax1.semilogy(it, [h["perda"] for h in historico], label="total")
    ax1.semilogy(it, [h["perda_edp"] for h in historico], label="resíduo da EDP")
    ax1.semilogy(it, [h["perda_ci"] for h in historico], label="condição inicial")
    ax1.semilogy(it, [h["perda_cc"] for h in historico], label="condição de contorno")
    ax1.set_xlabel("iteração (Adam, depois L-BFGS)")
    ax1.set_ylabel("perda")
    ax1.set_title("Perdas")
    ax1.grid(alpha=0.3, which="both")
    ax1.legend()

    ax2.semilogy(it, [h["erro_l2_rel"] for h in historico], color="C3")
    ax2.set_xlabel("iteração (Adam, depois L-BFGS)")
    ax2.set_ylabel("erro L2 relativo vs. referência")
    ax2.set_title("Erro contra a solução de referência")
    ax2.grid(alpha=0.3, which="both")

    fig.savefig(caminho, dpi=150)
    plt.close(fig)
