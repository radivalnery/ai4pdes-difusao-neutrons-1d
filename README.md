# Neural Physics/AI4PDEs - Difusao de Neutrons 1D

Este repositorio organiza um solver 1D para difusao de neutrons inspirado na
filosofia AI4PDEs/Neural Physics. O metodo principal usa operadores discretos
fixos em PyTorch, suavizacao de Jacobi e ciclo U-Net/multigrid geometrico, sem
treinamento de pesos.

## Versao MDF + MEF (v4.2)

A interface Tkinter e o arquivo unico permitem escolher **Discretizacao: MDF ou
MEF**. A escolha **Execucao: ambos** executa Neural Physics e Thomas para a
mesma discretizacao; Thomas e apenas referencia numerica externa.

```powershell
python main.py
# Mesmo programa, em um arquivo unico detalhado:
python difusao_neutrons_neuralphysics_1d_mdf_mef_v4_2.py
```

Na GUI, selecione o problema, a discretizacao e os parametros e clique em
Executar. O MEF usa elementos lineares P1 de Galerkin, massa consistente e
integracao dos materiais por elemento. N significa intervalos/elementos;
ha N+1 nos. O contorno chamado vacuo representa fluxo nulo, como no MDF.

Para gerar CSVs, figuras, tabelas LaTeX e relatorio PDF/Markdown sem a GUI:

```powershell
python main.py --experimento-artigo --discretizacao mef --saida resultados_mef
python main.py --experimento-artigo --discretizacao mdf --saida resultados_mdf
# Executar e consolidar as duas campanhas:
python main.py --experimento-artigo --discretizacao ambos --saida resultados_mdf_mef
```

Com `--discretizacao ambos`, cada campanha fica em `mdf/` e `mef/`, com seus
relatorios e figuras; os CSVs reunidos ficam em `csv/` na pasta de saida.
O PDF consolidado fica em `relatorios/relatorio_experimental.pdf`.
O arquivo unico aceita exatamente os mesmos argumentos.

| Caso | Valores de N | Posicoes x (cm) |
|---|---|---|
| Homogeneo | 1, 13, 26, 52, 100, 104 | 0, 13, 26 |
| Heterogeneo | 3, 12, 48, 96, 192, 300, 384 | 0, 50, 100, 150 |

Teste curto de funcionamento (nao serve como resultado convergido para artigo):

```powershell
python main.py --experimento-artigo --discretizacao mef --malhas-homogeneo 13 --malhas-heterogeneo 12 --max-iter 2 --max-fonte 5 --saida resultados_smoke_mef
```

Os CSVs registram convergencia, discretizacao, precisao e tempos separados.
MDF e MEF agora usam o **mesmo banco Conv1d fixo, ciclo V, hierarquia geometrica,
transferencias e precisao** (float64 por padrao; `--precisao float32` para ambos).
Mudam os coeficientes de A/F: MDF conservativo ou MEF P1 de massa consistente.
O escalonamento H^-1 do MEF permite uma convencao comum de residuos sem
alterar o problema de autovalor nem concentrar a massa. A mesma tolerancia
nao implica o mesmo erro de discretizacao ou o mesmo numero de ciclos.
Fonte fixa e uma parcela do tempo de iteracao: nao some as duas colunas.

As versoes unicas v4.0/v4.1 permanecem preservadas. O estado MDF anterior ao
MEF esta marcado por `mdf-v4.1-pre-mef`; o desenvolvimento e feito na branch
`codex/adicionar-mef` do mesmo repositorio. Backups locais ficam em `backups/`.
Consulte [a descricao detalhada](docs/MDF_MEF_v4_2.md) para formulas e validacao.

## Objetivo

Resolver problemas de autovalor da difusao de neutrons em geometria slab 1D, com
interface grafica para configurar:

- problema homogeneo;
- problema heterogeneo baseado em Couto (2003);
- condicoes de contorno reflexiva ou vacuo;
- potencia nominal;
- tolerancias de fluxo e de `k_eff`;
- materiais por regiao;
- tabela por refinamento de malha;
- comparacao opcional com Thomas, sem trocar o metodo principal.

## Regra metodologica

O metodo principal do projeto e sempre:

```text
U-Net/multigrid 1D sem treinamento
```

Thomas existe apenas como comparacao externa. Ele nao e usado no ciclo
U-Net/multigrid, nem no nivel mais grosso, nem como resolvedor principal quando a
opcao de comparacao e escolhida.

## Estrutura do Codigo

```text
main.py                         ponto de entrada da interface
src/neutron_ai4pdes/
  app.py                        inicializacao da aplicacao Tkinter
  gui.py                        frontend Tkinter, graficos e relatorio
  solver.py                     backend numerico do problema de autovalor
  models.py                     operador discreto e solver de fonte fixa
  stencils.py                   banco Conv1d e transferencias comuns MDF/MEF
  fem.py                        montagem MEF P1, massa consistente
  metrics.py                    erros globais de forma do fluxo
  experiments.py                campanhas sem GUI e exportacao para artigo
  references.py                 formulas e solucoes de referencia
  method.py                     descricao do metodo Neural Physics/AI4PDEs 1D
```

## Instalar Dependencias

```bash
pip install -r requirements.txt
```

## Executar

```bash
python main.py
```

## Disponibilidade do Codigo

O codigo-fonte do projeto esta disponivel publicamente em:

```text
https://github.com/radivalnery/ai4pdes-difusao-neutrons-1d
```

Para reproduzir a versao MDF anterior do artigo (v4.1):

```bash
python difusao_neutrons_neuralphysics_1d_thomas_couto_v4_1.py --experimento-artigo --saida resultados_artigo_v4_1
```

Esse comando gera automaticamente as pastas `csv/`, `figuras/`,
`figuras_artigo/`, `tabelas_latex/` e `relatorios/` dentro da pasta de saida.

## Versoes Anteriores MDF

Os arquivos unicos anteriores continuam preservados para reproducao:

```bash
python difusao_neutrons_neuralphysics_1d_comparativo_v4_0.py
python difusao_neutrons_neuralphysics_1d_thomas_couto_v4_1.py --experimento-artigo
```

A versao `v4_1` inclui:

- resolvedor U-Net/multigrid sem treinamento;
- operador conservativo local em PyTorch para o caso heterogeneo;
- metodo classico de Thomas somente para comparacao;
- erro global de forma do fluxo;
- tempos separados de montagem, transferencia e iteracao;
- geracao de CSVs, figuras para artigo, tabelas LaTeX e relatorios Markdown/PDF.

## Saidas Geradas

Na interface, as planilhas CSV sao salvas na pasta de execucao do programa:

- `resultados_refinamento.csv`;
- `resultados_comparacao.csv`, apenas quando a opcao de comparacao com Thomas e usada.

No modo `--experimento-artigo`, os arquivos principais sao:

- `csv/resultados_consolidados.csv`;
- `csv/comparacao_neuralphysics_thomas.csv`;
- `csv/comparacao_couto_2003.csv`;
- `csv/refinamento_malha.csv`;
- `csv/tempos_execucao.csv`;
- `relatorios/relatorio_experimental.pdf`;
- `relatorios/relatorio_experimental.md`.

## Observacao Metodologica

O pacote AI4PDEs original possui recursos voltados principalmente a operadores
2D/3D. Para este problema 1D, o codigo implementa uma adaptacao local da mesma
filosofia: operadores discretos fixos, tensores PyTorch, restricao,
prolongamento, suavizacao e iteracoes sem treinamento.

Para o caso heterogeneo, Couto (2003) e usado como referencia de `k_eff`. O
fluxo heterogeneo so deve ser chamado de comparacao com uma referencia externa
quando valores de fluxo de referencia forem fornecidos.
