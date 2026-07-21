# Neural Physics/AI4PDEs - Difusao de Neutrons 1D

Este repositorio organiza um solver 1D para difusao de neutrons inspirado na
filosofia AI4PDEs/Neural Physics. O metodo principal usa operadores discretos
fixos em PyTorch, suavizacao de Jacobi e ciclo U-Net/multigrid geometrico, sem
treinamento de pesos.

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

O codigo-fonte do projeto esta organizado para disponibilizacao publica em:

```text
https://github.com/radivalnery/ai4pdes-difusao-neutrons-1d
```

Na versao do artigo, o arquivo principal de reproducao dos experimentos e:

```bash
python difusao_neutrons_neuralphysics_1d_thomas_couto_v4_1.py --experimento-artigo --saida resultados_artigo_v4_1
```

Esse comando gera automaticamente as pastas `csv/`, `figuras/`,
`figuras_artigo/`, `tabelas_latex/` e `relatorios/` dentro da pasta de saida.

## Versao Unica para Envio

Tambem foram gerados arquivos unicos para envio e reproducao:

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
