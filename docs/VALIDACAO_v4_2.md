# Validacao da versao MDF/MEF v4.2

## Escopo executado

| Verificacao | Resultado |
|---|---|
| Testes MEF, regressao rapida MDF e interface Tkinter | 19 passaram |
| Exportacao consolidada das duas discretizacoes | 1 passou |
| Regressoes lentas MDF, homogeneo N=100 e heterogeneo N=300 | 2 passaram |
| Compilacao dos modulos e arquivo unico | OK |
| Imports relativos no arquivo unico gerado | Nenhum |
| Arquivos unicos antigos v4.0/v4.1 | Sem alteracao |

Os testes estao em tests/test_mef.py, tests/test_gui_mef.py,
tests/test_experiments.py e tests/test_solver.py. O ambiente de validacao
dispoe de CPU e PyTorch 2.12.1+cpu. A execucao CUDA nao foi validada.

## Evidencias numericas

- Matrizes elementares conferidas contra formulas exatas de Galerkin P1.
- Interfaces internas integradas corretamente mesmo dentro de um elemento.
- Autovalor MEF verificado com solucao densa independente do sistema pequeno.
- Quatro combinacoes de contorno verificadas, incluindo reflexiva/reflexiva.
- Integral MEF do fluxo normalizado reproduz a potencia configurada.
- Refinamento MEF homogeneo apresenta ordem aproximadamente dois no autovalor.
- Fluxo e k_eff do MEF/multigrid comparados com Thomas em N=3, 12 e 48.
- Fonte fixa verificada nas maiores malhas: homogeneo N=104 e heterogeneo N=384.
- Ausencia de chamada de Thomas no caminho multigrid principal testada.
- Resultados numericos do arquivo unico e modulos coincidem.

## Interface e exportacoes

A GUI foi exercitada oculta, mas com o mainloop real do Tkinter. Foram
verificadas a selecao MEF, as malhas por problema, a tabela com uma linha por
(N,x), a execucao 'ambos', as curvas NP/Thomas, o eixo de tempo logaritmico e
a geracao do PDF da interface. A tabela nao calcula erro relativo na fronteira
com referencia nula; em N=1, x=13 e avaliado por interpolacao, nao pelo no vizinho.

Tambem foi executado o modo de experimento do arquivo unico com as campanhas
MDF e MEF, homogeneo N=13,26 e heterogeneo N=12,48. As 16 resolucoes
(duas discretizacoes, dois problemas, duas malhas, dois resolvedores) convergiram.
Foram gerados e inspecionados CSVs, figuras, LaTeX e PDFs por discretizacao e
o PDF consolidado. As primeiras paginas dos PDFs foram renderizadas para
conferir tabelas, margens e legibilidade.

Os tempos dessas execucoes sao de validacao funcional, nao uma campanha de
benchmark controlada. MDF usa float32 e MEF usa float64; esse fato esta
explicitado na documentacao e nos resultados.

## Preservacao

O estado anterior foi preservado no backup local datado backups/antes_mef_*/ e
na tag mdf-v4.1-pre-mef, que aponta para 739e217. A evolucao MDF/MEF foi
preparada na branch codex/adicionar-mef do mesmo repositorio.
