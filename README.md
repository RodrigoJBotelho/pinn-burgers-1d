# pinn-burgers-1d

Protótipo de **PINN (Physics-Informed Neural Network)** em Python com **PyTorch** que resolve a
**equação de Burgers 1D** impondo a física na função de perda, ou seja, o resíduo da EDP calculado
por diferenciação automática. O resultado é comparado com a solução exata (Cole-Hopf).

## O problema

```
u_t + u·u_x = ν·u_xx        x ∈ [-1, 1],  t ∈ [0, 1],  ν = 0,01/π
u(x, 0) = -sin(πx)          u(-1, t) = u(1, t) = 0
```

A solução desenvolve uma frente abrupta (choque suavizado pela viscosidade) em torno de `x = 0`,
o que torna o problema um teste clássico para PINNs.

## Como funciona

A rede `u_θ(x, t)` (perceptron multicamadas com `tanh`) é treinada para minimizar

```
perda = média(r²) + média((u_θ(x,0) - u₀(x))²) + média(u_θ(±1,t)²)
        └ física ┘   └ condição inicial ┘          └ contorno ┘
```

em que `r = ∂ₜu_θ + u_θ·∂ₓu_θ - ν·∂ₓₓu_θ` é o resíduo da EDP, obtido com `torch.autograd`
(derivadas de primeira e segunda ordem, mantidas no grafo de cálculo). Nenhum dado da solução é
usado no treino: só a EDP, a condição inicial e o contorno.

O treino tem duas fases, como é comum na literatura de PINNs: **Adam** (para sair da região
inicial) e **L-BFGS** (para refinar). Durante o treino, o erro contra a solução de referência é
registrado apenas para acompanhamento.

## Solução de referência e validação

A referência (`pinn_burgers/reference.py`) usa a transformação de Cole-Hopf com quadratura de
Gauss-Hermite, somente com NumPy. Ela é validada nos testes:

- reproduz a condição inicial e é antissimétrica em `x`;
- respeita o contorno nulo;
- concorda com um solver independente de diferenças finitas (RK4), e o erro desse solver cai
  com a ordem esperada ao refinar a malha, o que indica que a diferença vem do solver e não da referência.

A formulação segue a de Raissi, Perdikaris e Karniadakis (2019), *Physics-informed neural networks*,
Journal of Computational Physics, 378.

## Como rodar sem instalar nada (GitHub Actions)

1. Publique o repositório no GitHub. A cada `push`, o workflow **ci** roda os testes e um
   treino rápido.
2. Para o treino completo: aba **Actions** → workflow **treino** → **Run workflow**
   (você escolhe iterações do Adam e do L-BFGS e a semente).
3. Ao terminar, a tabela de métricas aparece no resumo da execução. As figuras
   (`burgers_solucao.png`, `burgers_perdas.png`), o CSV de perdas e os pesos do modelo ficam
   em **Artifacts**.

## Como rodar localmente

```bash
python -m pip install numpy matplotlib
python -m pip install torch          # versão CPU basta
python -m pytest -q                  # testes
python -m pinn_burgers.train --quick # verificação rápida do pipeline
python -m pinn_burgers.train --adam-iter 5000 --lbfgs-iter 500 --out resultados
python -m pinn_burgers.train --help
```

## Estrutura

```
pinn_burgers/
  reference.py   solução de referência (Cole-Hopf, NumPy)
  model.py       rede neural e resíduo da EDP (autograd)
  train.py       perdas, treino Adam + L-BFGS, métricas e saída de arquivos
  plots.py       figuras (mapas de calor, perfis, curvas de perda)
tests/           testes da referência e do PINN
```

## Resultados

Cole aqui as métricas e as figuras geradas pelo workflow **treino**, informando a data, a
semente e o ambiente (por exemplo, `ubuntu-latest`, CPU).

## Limitações

- Problema 1D com condições simples; não é um benchmark de desempenho.
- O resultado depende da semente e do tamanho do treino. PINNs podem ter dificuldade perto de
  frentes abruptas e são sensíveis aos pesos relativos entre os termos da perda.
- Pontos de colocação aleatórios e fixos, sem reamostragem adaptativa.

## Roteiro

- [ ] Problema inverso: estimar a viscosidade `ν` a partir de poucos dados
- [ ] Reamostragem adaptativa dos pontos de colocação (RAR)
- [ ] Pesos adaptativos entre os termos da perda
- [ ] Cavidade com tampa móvel (Navier-Stokes 2D) em regime de baixo Reynolds
- [ ] Comparação com um solver de volumes finitos

## Licença

MIT. Veja [LICENSE](LICENSE).
