"""Treinamento do PINN para a equação de Burgers 1D.

Perda total = média(resíduo da EDP^2) + média(erro da condição inicial^2)
              + média(erro da condição de contorno^2)

Treino em duas fases (como é comum na literatura de PINNs): Adam, para sair
da região inicial, seguido de L-BFGS, para refinar.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from .model import MLP, residuo_burgers
from .plots import figura_perdas, figura_solucao
from .reference import NU_PADRAO, erro_l2_relativo, solucao_exata


@dataclass
class Config:
    nu: float = NU_PADRAO
    hidden_layers: int = 8
    width: int = 20
    n_f: int = 10000   # pontos de colocação (resíduo da EDP)
    n_0: int = 200     # pontos da condição inicial
    n_b: int = 200     # pontos da condição de contorno (por lado)
    adam_iter: int = 5000
    lr: float = 1e-3
    lbfgs_iter: int = 500
    lbfgs_passo: int = 50
    log_every: int = 250
    seed: int = 0
    out: str = "resultados"


@dataclass
class Dados:
    x_f: torch.Tensor
    t_f: torch.Tensor
    x_0: torch.Tensor
    u_0: torch.Tensor
    t_b: torch.Tensor


def amostrar(cfg: Config) -> Dados:
    """Sorteia pontos de colocação, da condição inicial e do contorno."""
    x_f = 2.0 * torch.rand(cfg.n_f, 1) - 1.0
    t_f = torch.rand(cfg.n_f, 1)
    x_0 = 2.0 * torch.rand(cfg.n_0, 1) - 1.0
    u_0 = -torch.sin(math.pi * x_0)
    t_b = torch.rand(cfg.n_b, 1)
    return Dados(x_f, t_f, x_0, u_0, t_b)


def perdas(modelo: MLP, dados: Dados, nu: float):
    """Componentes da perda: (resíduo da EDP, condição inicial, contorno)."""
    r = residuo_burgers(modelo, dados.x_f, dados.t_f, nu)
    perda_edp = torch.mean(r**2)

    u0_pred = modelo(dados.x_0, torch.zeros_like(dados.x_0))
    perda_ci = torch.mean((u0_pred - dados.u_0) ** 2)

    u_esq = modelo(-torch.ones_like(dados.t_b), dados.t_b)
    u_dir = modelo(torch.ones_like(dados.t_b), dados.t_b)
    perda_cc = torch.mean(torch.cat([u_esq, u_dir]) ** 2)
    return perda_edp, perda_ci, perda_cc


@torch.no_grad()
def prever(modelo: MLP, x: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Avalia u_theta numa grade; retorna array (len(x), len(t))."""
    X, T = np.meshgrid(x, t, indexing="ij")
    dtype = torch.get_default_dtype()
    xt = torch.tensor(X.reshape(-1, 1), dtype=dtype)
    tt = torch.tensor(T.reshape(-1, 1), dtype=dtype)
    return modelo(xt, tt).numpy().reshape(len(x), len(t)).astype(float)


def treinar(cfg: Config, verbose: bool = True) -> dict:
    """Treina o PINN e devolve modelo, histórico, métricas e a grade de avaliação."""
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    modelo = MLP(cfg.hidden_layers, cfg.width)
    dados = amostrar(cfg)

    x_av = np.linspace(-1.0, 1.0, 256)
    t_av = np.linspace(0.0, 1.0, 101)
    u_ref = solucao_exata(x_av, t_av, nu=cfg.nu)

    historico: list[dict] = []

    def registrar(fase: str, it: int, it_global: int, partes) -> None:
        edp, ci, cc = (float(p) for p in partes)
        erro = erro_l2_relativo(prever(modelo, x_av, t_av), u_ref)
        historico.append({
            "fase": fase, "iteracao": it, "iteracao_global": it_global,
            "perda": edp + ci + cc, "perda_edp": edp, "perda_ci": ci, "perda_cc": cc,
            "erro_l2_rel": erro,
        })
        if verbose:
            print(f"[{fase:>6}] it {it:6d} | perda {edp + ci + cc:.3e} "
                  f"(edp {edp:.2e}, ci {ci:.2e}, cc {cc:.2e}) | erro L2 rel {erro:.3e}",
                  flush=True)

    inicio = time.perf_counter()

    # Fase 1: Adam
    otimizador = torch.optim.Adam(modelo.parameters(), lr=cfg.lr)
    for it in range(1, cfg.adam_iter + 1):
        otimizador.zero_grad()
        edp, ci, cc = perdas(modelo, dados, cfg.nu)
        perda = edp + ci + cc
        perda.backward()
        otimizador.step()
        if it == 1 or it % cfg.log_every == 0 or it == cfg.adam_iter:
            registrar("adam", it, it, (edp.item(), ci.item(), cc.item()))

    # Fase 2: L-BFGS (em blocos, para registrar o progresso)
    if cfg.lbfgs_iter > 0:
        otimizador = torch.optim.LBFGS(
            modelo.parameters(), lr=1.0, max_iter=cfg.lbfgs_passo, history_size=50,
            tolerance_grad=1e-12, tolerance_change=1e-14, line_search_fn="strong_wolfe",
        )

        def closure():
            otimizador.zero_grad()
            edp, ci, cc = perdas(modelo, dados, cfg.nu)
            perda = edp + ci + cc
            perda.backward()
            return perda

        feitas = 0
        while feitas < cfg.lbfgs_iter:
            otimizador.step(closure)
            feitas += cfg.lbfgs_passo
            # Reavalia nos pesos aceitos (a última avaliação do L-BFGS pode ser
            # um ponto de teste da busca linear).
            edp, ci, cc = perdas(modelo, dados, cfg.nu)
            registrar("lbfgs", feitas, cfg.adam_iter + feitas,
                      (edp.item(), ci.item(), cc.item()))

    tempo = time.perf_counter() - inicio

    u_pred = prever(modelo, x_av, t_av)
    metricas = {
        "erro_l2_relativo": erro_l2_relativo(u_pred, u_ref),
        "erro_absoluto_max": float(np.max(np.abs(u_pred - u_ref))),
        "perda_final": historico[-1]["perda"] if historico else None,
        "tempo_treino_s": tempo,
        "torch": torch.__version__,
        "config": asdict(cfg),
    }
    return {"modelo": modelo, "historico": historico, "metricas": metricas,
            "x": x_av, "t": t_av, "u_ref": u_ref, "u_pred": u_pred}


def salvar_resultados(resultado: dict, pasta: str) -> None:
    """Grava CSV, JSON, resumo em Markdown, figuras e pesos do modelo."""
    saida = Path(pasta)
    saida.mkdir(parents=True, exist_ok=True)
    hist, met = resultado["historico"], resultado["metricas"]
    cfg = met["config"]

    with open(saida / "perdas.csv", "w", newline="", encoding="utf-8") as arq:
        escritor = csv.DictWriter(arq, fieldnames=list(hist[0].keys()))
        escritor.writeheader()
        escritor.writerows(hist)

    with open(saida / "metricas.json", "w", encoding="utf-8") as arq:
        json.dump(met, arq, indent=2, ensure_ascii=False)

    with open(saida / "resumo.md", "w", encoding="utf-8") as arq:
        arq.write("### PINN para Burgers 1D\n\n")
        arq.write(f"Rede com {cfg['hidden_layers']} camadas ocultas de {cfg['width']} neurônios (tanh); "
                  f"{cfg['n_f']} pontos de colocação; Adam ({cfg['adam_iter']} it., lr={cfg['lr']}) "
                  f"seguido de L-BFGS ({cfg['lbfgs_iter']} it.); semente {cfg['seed']}; "
                  f"PyTorch {met['torch']}.\n\n")
        arq.write("| métrica | valor |\n|---|---:|\n")
        arq.write(f"| erro L2 relativo | {met['erro_l2_relativo']:.3e} |\n")
        arq.write(f"| erro absoluto máximo | {met['erro_absoluto_max']:.3e} |\n")
        arq.write(f"| perda final | {met['perda_final']:.3e} |\n")
        arq.write(f"| tempo de treino (s) | {met['tempo_treino_s']:.1f} |\n")

    figura_solucao(resultado["x"], resultado["t"], resultado["u_ref"], resultado["u_pred"],
                   saida / "burgers_solucao.png")
    figura_perdas(hist, saida / "burgers_perdas.png")
    torch.save(resultado["modelo"].state_dict(), saida / "modelo.pt")


def criar_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pinn_burgers.train",
        description="Treina um PINN para a equação de Burgers 1D e compara com a solução de Cole-Hopf.",
    )
    padrao = Config()
    p.add_argument("--adam-iter", type=int, default=padrao.adam_iter)
    p.add_argument("--lbfgs-iter", type=int, default=padrao.lbfgs_iter)
    p.add_argument("--lr", type=float, default=padrao.lr)
    p.add_argument("--hidden-layers", type=int, default=padrao.hidden_layers)
    p.add_argument("--width", type=int, default=padrao.width)
    p.add_argument("--n-f", type=int, default=padrao.n_f, help="pontos de colocação")
    p.add_argument("--seed", type=int, default=padrao.seed)
    p.add_argument("--log-every", type=int, default=padrao.log_every)
    p.add_argument("--out", default=padrao.out, help="pasta de saída")
    p.add_argument("--quick", action="store_true",
                   help="configuração pequena, só para testar se tudo funciona")
    return p


def main(argv=None) -> int:
    args = criar_parser().parse_args(argv)
    cfg = Config(
        adam_iter=args.adam_iter, lbfgs_iter=args.lbfgs_iter, lr=args.lr,
        hidden_layers=args.hidden_layers, width=args.width, n_f=args.n_f,
        seed=args.seed, log_every=args.log_every, out=args.out,
    )
    if args.quick:
        cfg.adam_iter, cfg.lbfgs_iter = 300, 0
        cfg.hidden_layers, cfg.width, cfg.n_f = 4, 20, 2000
        cfg.log_every = 50

    resultado = treinar(cfg)
    salvar_resultados(resultado, cfg.out)
    m = resultado["metricas"]
    print(f"\nErro L2 relativo final: {m['erro_l2_relativo']:.3e} "
          f"| tempo de treino: {m['tempo_treino_s']:.1f} s | resultados em '{cfg.out}/'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
