# Relatório heterogeneo comparativo

Neste trabalho, o termo Neural Physics não se refere ao treinamento de uma rede neural para aproximar a solução. Ele se refere à implementação de operadores numéricos discretos por meio de operações típicas de bibliotecas de inteligência artificial, como convoluções, pooling e interpolação. Os pesos são fixos e definidos pela discretização física do problema.

## Comparação entre resolvedores de fonte fixa

| Caso | Método | N | k_eff | Referência | Erro k (%) | Iter. externas | Iter. fonte média | Resíduo final | Fonte convergiu | Tempo (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| heterogeneo | Correção multiescala por pooling/interpolação | 120 | 1.09514822 | 1.09506 | 0.008056486751225882 | 254 | 8.63 | 7.331371307373047e-06 | True | 4.0545 |
| heterogeneo | Thomas clássico | 120 | 1.09516306 | 1.09506 | 0.009411283329531943 | 243 | 1.00 | 8.940696716308594e-08 | True | 0.3284 |

## Análise de sensibilidade

| Caso | tol_fonte | omega | amortecimento | k_eff | Erro k (%) | Iter. externas | Iter. fonte média | Resíduo final | Fonte convergiu | Tempo (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| heterogeneo | 1.0e-04 | 0.75 | 0.20 | 1.09514867 | 0.008097496391950232 | 498 | 2.72 | 7.256865501403809e-06 | True | 2.9393 |
| heterogeneo | 1.0e-05 | 0.75 | 0.20 | 1.09514822 | 0.008056486751225882 | 254 | 8.63 | 7.331371307373047e-06 | True | 3.9227 |
| heterogeneo | 1.0e-06 | 0.75 | 0.20 | 1.09516115 | 0.009236856028566467 | 240 | 15.86 | 9.08970832824707e-07 | True | 6.4432 |

## Discussão automática curta
Para problemas 1D, o método de Thomas é o resolvedor clássico mais natural para sistemas tridiagonais. A formulação Neural Physics aqui avaliada não tem como objetivo superar Thomas em 1D, mas validar uma implementação determinística baseada em operadores convolucionais fixos e operações multiescala compatíveis com bibliotecas de IA.