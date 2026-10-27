# Difusao de neutrons 1D: MDF e MEF - v4.2

## 1. O que esta sendo escolhido

| Escolha | Opcoes | Significado |
|---|---|---|
| Discretizacao | MDF, MEF | Formulacao do sistema algebrico |
| Precisao | float64, float32 | A mesma escolha vale para MDF e MEF; padrao float64 |
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

Em um no INTERIOR de malha uniforme homogenea, apos escalonar o MEF por H^-1,
os coeficientes usados pelo mesmo banco convolucional sao:

| Operador | phi[i-1] | phi[i] | phi[i+1] |
|---|---|---|---|
| A MDF | -D/h^2 | 2D/h^2 + Sigma_a | -D/h^2 |
| A MEF P1 | -D/h^2 + Sigma_a/6 | 2D/h^2 + 2 Sigma_a/3 | -D/h^2 + Sigma_a/6 |
| F MDF | 0 | nuSigma_f | 0 |
| F MEF P1 | nuSigma_f/6 | 2 nuSigma_f/3 | nuSigma_f/6 |

Portanto nao basta trocar o stencil de difusao: absorcao e fonte de fissao
tambem obedecem a formulacao escolhida. Fronteiras, interfaces e malhas grossas
nao uniformes usam seus coeficientes locais montados, nao essa formula uniforme.

As matrizes de massa sao consistentes, nao concentradas. As tres diagonais de
A e F sao armazenadas como buffers PyTorch fixos. MDF e MEF aplicam A e F
pelo MESMO banco Conv1d: tres filtros fixos extraem os vizinhos e os coeficientes
locais pesam os canais. Nao ha matriz densa nem pesos treinaveis na resolucao.
Uma Conv1d global com unico kernel fisico constante nao representa, em geral,
o operador heterogeneo; por isso os coeficientes fisicos variam por no.
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
As malhas sao as MESMAS para MDF/MEF: aninhadas, com coordenadas, extremos e
interfaces nodais preservados, inclusive para N impar. Em cada nivel, apenas
a montagem dos coeficientes depende da formulacao. O ciclo V tem ate 6 niveis,
2 passos Jacobi antes/depois da correcao e 20 passos no grosso; os parametros
omega=0.75 e relaxacao da correcao=0.20 permanecem configuraveis.

Para compartilhar transferencias e escala de residuo, define-se a medida
nodal H_i como meia soma dos comprimentos dos elementos adjacentes (h/2 nos
extremos e h no interior da malha uniforme). O MEF usa:

```text
A_hat = H^-1 (K + M_a), F_hat = H^-1 M_f
A_hat phi = F_hat phi / k
r_hat = b_hat - A_hat phi
P = interpolacao linear nas coordenadas reais
R_hat = H_c^-1 P^T H_f
```

H e diagonal positivo e invertivel: escalonar AMBOS os lados nao altera o
autovalor ou o fluxo. M_a e M_f continuam consistentes, com diagonais laterais.
Para cargas MEF integradas, H_c R_hat r_hat = P^T (H_f r_hat), a restricao
variacional usual. Para MDF uniforme 2:1, R_hat e full weighting no interior.
Assim, ambas as formulacoes passam pela mesma rotina de transferencia.

Na iteracao externa, ambos calculam S=F_hat phi/k por Conv1d fixa, resolvem
A_hat phi_novo=S e atualizam k pela razao das producoes integradas 1^T H F_hat phi.
No MEF isso e exatamente 1^T M_f phi. No MDF e a quadratura nodal trapezoidal.
Isso integra a producao em todo o dominio, inclusive as bases de fronteira. Apenas o vetor
de carga da resolucao linear e zerado nos graus Dirichlet. O fluxo e reescalado
por seu maximo para a iteracao, e normalizado por potencia ao final, usando a
convencao energetica ja existente (3.2e-11) e essa mesma integral em cada caso.

A parada exige simultaneamente os desvios de k e fluxo abaixo de tol_k e
tol_phi. Falha da fonte fixa interrompe a iteracao e fica registrada no CSV.
Ambos usam float64 por padrao e residuo ||b-A phi||_inf/||b||_inf, com apenas
a protecao do menor positivo normal da precisao para fonte nula. A selecao
float32 e comum a ambos, mas pode impedir tolerancias muito apertadas.

| Controle | MDF e MEF na comparacao atual |
|---|---|
| Banco convolucional para A/F | Mesmo codigo, filtros fixos sem treinamento |
| Ciclo V e Jacobi | Mesma classe e mesmos parametros |
| Precisao, malha e coordenadas por nivel | Iguais |
| P e R | Iguais: linear e Hc^-1 P^T Hf |
| Residuo e tolerancias | Mesma convencao e valores configurados |
| Chute e contornos de fluxo | Mesma rotina; linhas essenciais impostas |
| A/F | Diferentes: MDF ou Galerkin P1 consistente |
| Thomas | Apenas verificacao externa do respectivo sistema |

Essa comparacao controla o caminho numerico entre MDF/MEF na v4.2; nao afirma
identidade literal com uma implementacao oficial 2D/3D do AI4PDEs. Em relacao
a v4.1, tambem ha correcoes comuns: geometria grossa com N impar, continuidade
de Jacobi quando toda a malha e grossa, escala relativa do residuo, precisao
padrao e integral de potencia MDF trapezoidal (antes retangular). Por isso,
os valores historicos v4.1 devem ser reproduzidos pelo arquivo preservado.

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
Os dois usam a mesma precisao e o mesmo caminho de aplicacao dos stencils.
Custos de montagem continuam proprios de cada formulacao. Execucoes pequenas ou
sem convergencia servem apenas para verificar funcionamento, nao para o artigo.

## 5. Organizacao e reproducao

| Arquivo | Responsabilidade |
|---|---|
| models.py | Operador MDF, resultado de fonte fixa e ciclo V comum |
| stencils.py | Banco Conv1d comum, medidas H e transferencias ponderadas |
| fem.py | Montagem P1 consistente e escalonamento das linhas |
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
python -m pytest tests/test_mef.py tests/test_solver.py tests/test_comparacao_controlada.py
```

O arquivo unico gerado nao depende da pasta src e aceita os mesmos argumentos.
Ele preserva as explicacoes das fontes em secoes numeradas; suas funcoes sao
testadas contra os modulos. As versoes MDF v4.0/v4.1 nao sao sobrescritas.
O backup local datado anterior ao MEF fica em backups/antes_mef_*/.
O backup da v4.2 anterior ao controle fica em backups/antes_comparacao_controlada_*/.
A tag mdf-v4.1-pre-mef identifica o commit MDF anterior no mesmo repositorio.
O gerador legado tools/build_single_v4.py agora delega a tools/build_single_v4_2.py.

## 6. Verificacoes numericas

- Matrizes elementares contra as formulas exatas de Galerkin.
- Integracao de interface cortando elemento e producao total.
- Adjunto ponderado Hc^-1 P^T Hf e coordenadas em malha impar.
- Igualdade exata de ciclo, precisao, malhas, filtros e transferencias MDF/MEF.
- Produtos convolucionais de A/F contra matrizes explicitas em float32/float64.
- Invariancia do autovalor MEF ao escalonamento por H^-1.
- Autovalor contra solucao densa independente do sistema MEF pequeno.
- Quatro combinacoes de contorno e normalizacao de potencia.
- Comparacao MEF/multigrid com Thomas no heterogeneo.
- Convergencia de ordem dois no autovalor homogeneo refinado.
- Bloqueio de Thomas no caminho principal e flag de nao convergencia.
- Regressao MDF e igualdade numerica entre modular e arquivo unico.
