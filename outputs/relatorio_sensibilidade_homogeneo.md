# Relatório de sensibilidade homogeneo

Neste trabalho, o termo Neural Physics não se refere ao treinamento de uma rede neural para aproximar a solução. Ele se refere à implementação de operadores numéricos discretos por meio de operações típicas de bibliotecas de inteligência artificial, como convoluções, pooling e interpolação. Os pesos são fixos e definidos pela discretização física do problema.

## Comparação entre resolvedores de fonte fixa

| Caso | Método | N | k_eff | Referência | Erro k (%) | Iter. externas | Iter. fonte média | Resíduo final | Tempo (s) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| homogeneo | U-Net/multigrid sem treinamento | 80 | 0.99723720 | 0.9972907452457941 | 0.005368930475932444 | 29 | 146.24 | 9.655952453613281e-06 | 9.0686 |
| homogeneo | Thomas clássico | 80 | 0.99729056 | 0.9972907452457941 | 1.8512570711112424e-05 | 31 | 1.00 | 1.9818544387817383e-06 | 0.0322 |

## Análise de sensibilidade

| Caso | tol_fonte | omega | amortecimento | k_eff | Erro k (%) | Iter. externas | Iter. fonte média | Resíduo final | Tempo (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| homogeneo | 1.0e-04 | 0.75 | 0.20 | 0.99696868 | 0.03229372546885758 | 159 | 13.75 | 6.445497274398804e-05 | 4.1532 |
| homogeneo | 1.0e-05 | 0.75 | 0.20 | 0.99723720 | 0.005368930475932444 | 29 | 146.24 | 9.655952453613281e-06 | 7.0686 |
| homogeneo | 1.0e-06 | 0.75 | 0.20 | 0.99727940 | 0.0011376764501161243 | 33 | 941.48 | 9.685754776000977e-07 | 59.2935 |

## Discussão automática curta
Para problemas 1D, o método de Thomas é o resolvedor clássico mais natural para sistemas tridiagonais. A formulação Neural Physics aqui avaliada não tem como objetivo superar Thomas em 1D, mas validar uma implementação determinística baseada em operadores convolucionais fixos e operações multiescala compatíveis com bibliotecas de IA.