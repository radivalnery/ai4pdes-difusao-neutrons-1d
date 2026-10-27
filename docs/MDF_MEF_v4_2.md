# Difusao de neutrons 1D: MDF e MEF - v4.2

## 1. O que esta sendo escolhido

| Escolha | Opcoes | Significado |
|---|---|---|
| Discretizacao | MDF, MEF | Formulacao do sistema algebrico |
| Execucao na GUI | unet_multigrid, ambos | Metodo principal e referencia opcional |
| CLI --discretizacao | mdf, mef, ambos | Uma campanha ou duas campanhas consolidadas |

A equacao e `-d/dx(D dphi/dx) + Sigma_a phi = nuSigma_f phi/k_eff`.
MDF/MEF compartilham os dados fisicos, contornos e criterios de parada, mas
possuem operadores e integrais discretas diferentes. Thomas verifica cada
sistema discretizado; ele nao e chamado dentro do multigrid, nem no nivel grosso.

## 2. Formulacao MEF implementada

A forma fraca com funcoes-teste v e:

```text
integral(D phi' v' + Sigma_a phi v) dx = integral(nuSigma_f phi v) dx / k_eff
A phi = F phi / k_eff, A = K + M_a, F = M_f
```

Em um elemento de comprimento h e coeficientes constantes:

```text
K_e = D/h * [[1,-1],[-1,1]]
M_a,e = Sigma_a*h/6 * [[2,1],[1,2]]
M_f,e = nuSigma_f*h/6 * [[2,1],[1,2]]
```

As matrizes de massa sao consistentes, nao concentradas. As tres diagonais de
A e F sao armazenadas como buffers PyTorch fixos. O produto A phi e local e
vetorizado em tensores; nao ha matriz densa no caminho de producao. Uma Conv1d
global com kernel constante nao representa, em geral, o operador heterogeneo.
Esta e uma adaptacao MEF 1D da filosofia Neural Physics; nao e uma chamada ao
pacote oficial AI4PDEs nem uma afirmacao de identidade com seus exemplos.

Os materiais sao integrados por regiao. Se uma interface cortar um elemento,
a integral e dividida na interface e avaliada por Gauss de dois pontos em cada
trecho. As funcoes P1 pertencem ao elemento original. Lacunas e sobreposicoes
materiais sao rejeitadas. As malhas heterogeneas padrao tem nos em 50 e 100 cm.

Reflexiva representa corrente nula e e condicao natural. Vacuo continua
significando phi=0, com eliminacao simetrica das linhas/colunas essenciais.
Nao foi introduzida uma condicao Robin/extrapolada. Com dois contornos de
fluxo nulo, N=1 nao deixa graus de liberdade livres e e rejeitado.

## 3. Multigrid e autovalor

O ciclo V preserva suavizacao de Jacobi ponderado, correcao recursiva,
prolongamento e pos-suavizacao. O nivel grosso usa Jacobi, sem Thomas.
As malhas MEF sao aninhadas e preservam coordenadas, extremos e interfaces
nodais, inclusive para N impar. Os operadores sao rediscretizados integrando
os mesmos materiais. P interpola linearmente nas coordenadas reais e a
restricao e P^T, apropriada a residuos MEF integrados; nao se usa o fator 1/2
do full-weighting do MDF. A hierarquia MDF anterior permanece no seu caminho.

Na iteracao externa, o MEF calcula S=F phi/k, resolve A phi_novo=S e atualiza
k pela razao das producoes integradas. A soma dos componentes de F phi integra
a producao em todo o dominio, inclusive as bases de fronteira. Apenas o vetor
de carga da resolucao linear e zerado nos graus Dirichlet. O fluxo e reescalado
por seu maximo para a iteracao, e normalizado por potencia ao final, usando a
convencao energetica ja existente (3.2e-11) e a integral MEF consistente.

A parada exige simultaneamente os desvios de k e fluxo abaixo de tol_k e
tol_phi. Falha da fonte fixa interrompe a iteracao e fica registrada no CSV.
MDF preserva float32; MEF usa float64 e residuo relativo a norma da carga.

## 4. Resultados e tempos

Cada campanha produz `csv/`, `figuras/`, `figuras_artigo/`, `tabelas_latex/`,
`relatorios/`, `configuracao.json` e `hardware_ambiente.json`. As malhas e os
pontos da GUI sao os mesmos da CLI. Cada linha identifica a discretizacao.
Os CSVs incluem estado de convergencia, k_eff, fluxo, erros globais de forma
nas normas infinito e L2, alem de tempos de montagem, transferencia e iteracao.
Essas normas de forma sao calculadas nos vetores nodais normalizados pelo
maximo: max|a-b| e ||a-b||_2/||b||_2. L2 aqui e a norma vetorial discreta,
nao uma integral continua MEF. Os valores nos pontos solicitados sao obtidos
por interpolacao do fluxo nodal; a referencia analitica e avaliada no x exato.
Referencias de fluxo nulas nao admitem erro relativo pontual. Couto (2003)
fornece a referencia bibliografica de k_eff para o caso heterogeneo padrao;
Thomas e a referencia numerica do fluxo na mesma malha/discretizacao.

| Tempo | Conteudo |
|---|---|
| Montagem | Operador e hierarquia em CPU |
| Transferencia | Conversoes NumPy/PyTorch e copias de dados/modelos |
| Iteracao numerica | Preparacao de fonte, fonte fixa e atualizacao de k/erros |
| Fonte fixa | Parcela da iteracao numerica, descontando transferencias medidas |
| Loop iterativo | Tempo bruto do laco, incluindo comunicacao/logs |
| Total | Execucao do solver; nao inclui figuras, CSVs e geracao do PDF |

Operacoes CUDA sao sincronizadas nas fronteiras de medicao quando disponiveis.
MDF e MEF usam precisoes diferentes, explicitadas nos resultados; nao se deve
atribuir a diferenca de tempo apenas a discretizacao. Execucoes pequenas ou
sem convergencia servem apenas para verificar funcionamento, nao para o artigo.

## 5. Organizacao e reproducao

| Arquivo | Responsabilidade |
|---|---|
| models.py | Operador MDF, resultado de fonte fixa e ciclo V comum |
| fem.py | Matrizes P1, aplicacao local, hierarquia MEF e transferencias |
| solver.py | Autovalor, fonte, potencia e instrumentacao |
| gui.py | Parametros Tkinter, graficos, tabela e relatorios da GUI |
| experiments.py | CLI, campanhas e exportacoes para artigo |
| metrics.py | Normalizacao de forma e erros globais |
| tools/build_single_v4_2.py | Gera arquivo unico com comentarios e docstrings |

```powershell
python -m pip install -r requirements.txt
python main.py
python main.py --experimento-artigo --discretizacao ambos --saida resultados_mdf_mef
python tools/build_single_v4_2.py
python -m pytest tests/test_mef.py tests/test_solver.py
```

O arquivo unico gerado nao depende da pasta src e aceita os mesmos argumentos.
Ele preserva as explicacoes das fontes em secoes numeradas; suas funcoes sao
testadas contra os modulos. As versoes MDF v4.0/v4.1 nao sao sobrescritas.
O backup local datado anterior ao MEF fica em backups/antes_mef_*/.
A tag mdf-v4.1-pre-mef identifica o commit MDF anterior no mesmo repositorio.
O gerador legado tools/build_single_v4.py agora delega a tools/build_single_v4_2.py.

## 6. Verificacoes numericas

- Matrizes elementares contra as formulas exatas de Galerkin.
- Integracao de interface cortando elemento e producao total.
- Adjunto P^T e preservacao de coordenadas em malha impar.
- Autovalor contra solucao densa independente do sistema MEF pequeno.
- Quatro combinacoes de contorno e normalizacao de potencia.
- Comparacao MEF/multigrid com Thomas no heterogeneo.
- Convergencia de ordem dois no autovalor homogeneo refinado.
- Bloqueio de Thomas no caminho principal e flag de nao convergencia.
- Regressao MDF e igualdade numerica entre modular e arquivo unico.
