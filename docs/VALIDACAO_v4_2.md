# Validacao MDF/MEF v4.2 - comparacao controlada

## Atualizacao: selecao e execucao MDF/MEF na GUI

Em 27/10/2026, a GUI ganhou selecao por caixas MDF/MEF e modos Sequencial/
Paralelo, mantendo todos os demais parametros comuns. Thomas continua opcional
e externo, em uma escolha separada. Nao houve alteracao em solver.py, models.py,
fem.py ou stencils.py nesta atualizacao.

| Verificacao atual | Resultado |
|---|---|
| Suite sem os dois testes lentos | 49 aprovados, 2 excluidos |
| Backend de campanhas | 9 testes: equivalencia, sobreposicao real, isolamento, falhas e arquivo unico |
| GUI comparativa | 4 testes: sequencial/paralelo, referencias separadas e janela 1280x720 |
| GUI MEF anterior | Aprovada, com selecao e referencia Thomas atualizadas |
| Compilacao das fontes alteradas e arquivo unico | OK |

Os dois testes lentos de regressao MDF nao foram reexecutados nesta atualizacao;
seu ultimo resultado e o historico abaixo. Os demais testes numericos da suite
incluem as fontes fixas MEF nas malhas N=104 e N=384.

Foram exercitadas campanhas reais da GUI oculta, com mainloop:
- Homogeneo, sequencial, N=1,13, MDF/MEF e referencia Thomas: 8 resolucoes convergiram.
- Heterogeneo, paralelo, N=3,12, MDF/MEF e referencia Thomas: 8 resolucoes convergiram.
- Tabelas geradas: 12 linhas no homogeneo e 16 no heterogeneo, por (discr,N,x).

Os testes confirmam identica saida numerica em CPU entre execucao sequencial
e paralela para os casos pequenos. A sobreposicao das duas threads tambem e
testada com uma barreira, sem pressupor aceleracao. O numero global de threads
do PyTorch nao e alterado pelo executor. CUDA nao foi validada.

Foram verificados os estados por execucao, a atualizacao Tk somente na thread
principal, a confirmacao do aviso paralelo, o cancelamento desse aviso e a
validacao de selecao vazia. Campos ficam bloqueados durante a campanha.
Alterar o formulario depois nao muda o PDF/Markdown dos dados ja calculados.

Os graficos e os PDFs usam o mesmo desenho. Fluxo, forma, convergencia,
residuo, tempos, refinamento e materiais identificam MDF/MEF. Refinamento
mantem k_eff e erro k em eixos separados; N e linear com rotulos dos valores
calculados nas listas pequenas. Eixos secundarios sao removidos ao trocar
de vista, evitando sobreposicoes. Referencias heterogeneas sao separadas por
discretizacao; uma referencia MEF nao e usada no erro MDF.

Apos incorporar as fontes DejaVu Sans fornecidas pelo Matplotlib aos PDFs,
os cinco testes de GUI foram repetidos e aprovados. Os PDFs foram novamente
renderizados: phi/Sigma legiveis, tabelas dentro das margens, sem texto
invadindo colunas e sem depender de fontes Symbol externas.

Uma execucao ampla posterior terminou com 48 aprovados, 1 teste GUI pulado
por indisponibilidade intermitente de inicializacao Tcl/Tk e 2 longos excluidos.
O teste pulado corresponde ao modo sequencial da GUI; foi separado para nova
execucao e passou (1 aprovado em 10.65 s). Isso nao foi tratado como falha
numerica nem alteracao do solver. A indisponibilidade Tcl/Tk foi registrada,
nao ocultada como sucesso da rodada ampla.

Artefatos locais de verificacao em resultados_validacao_selecao_v4_2/:
graficos PNG, CSV/JSON dos parametros registrados e PDF/Markdown por caso.
Tempos sao de verificacao funcional, NAO benchmark. O aviso dos tempos
paralelos acompanha a GUI, CSV/JSON, graficos e relatorios.

Backup anterior: backups/antes_selecao_mdf_mef_20261027_012606/.

## Historico: validacao do nucleo controlado (commit 408c6b7)

Validacao executada em 27/10/2026, CPU, Python 3.14 e PyTorch 2.12.1+cpu.
CUDA nao foi validada. Os testes usam uma thread para limitar o custo das
malhas pequenas; a campanha registra seu ambiente em hardware_ambiente.json.

## Testes automatizados

| Arquivo | Casos aprovados | Verificacao principal |
|---|---:|---|
| test_comparacao_controlada.py | 15 | Mesmo ciclo, precisao, malhas, filtros e transferencias MDF/MEF |
| test_mef.py | 17 | Galerkin P1, contornos, potencia, Thomas externo e arquivo unico |
| test_solver.py | 4 | MDF contra referencia, regressao e flag de nao convergencia |
| test_gui_mef.py | 1 | Tkinter real, parametros, tabela, ambos, graficos e PDF |
| test_experiments.py | 1 | Consolidacao MDF/MEF e exportacoes |
| Total | 38 | Todos aprovados |

A execucao final sem os dois testes lentos aprovou 36 testes (2 excluidos).
Os dois testes lentos tambem passaram, junto aos 15 testes de controle e aos
dois testes rapidos do solver: 19 aprovados nessa execucao.
py_compile dos modulos alterados, ponto de entrada, gerador e arquivo unico: OK.

## Contrato numerico verificado

- A e F usam o mesmo banco Conv1d fixo em MDF/MEF, homogeneo/heterogeneo.
- Os produtos convolucionais coincidem com matrizes explicitas em float32/float64.
- Nenhum parametro requer gradiente; nao existe treinamento no resolvedor.
- Mesma classe de ciclo V, mesmos niveis, coordenadas e transferencias.
- Mesmos omega, relaxacao, pre/pos-suavizacoes e Jacobi no nivel grosso.
- Restricao R=Hc^-1 P^T Hf adjunta de P nas medidas nodais.
- Escalonar A e F do MEF por H^-1 preserva o autovalor da formulacao integrada.
- Absorcao e fissao MEF mantem massa consistente, inclusive fora da diagonal.
- Matrizes P1 e interfaces cortando elementos verificadas independentemente.
- Autovalores contra oraculo denso pequeno e quatro combinacoes de contorno.
- Potencia normalizada pela mesma convencao de integral em ambos os casos.
- Fonte fixa com RHS reescalado conserva o criterio relativo de parada.
- Fonte fixa nas maiores malhas: homogeneo N=104 e heterogeneo N=384.
- Thomas bloqueado nos caminhos principais MDF e MEF; somente referencia externa.
- Regressao MDF homogenea contra autovalor analitico da MATRIZ discretizada.
- MDF heterogeneo N=300 contra Thomas com tolerancias apertadas.
- Igualdade numerica entre modulos e arquivo unico gerado.

Os antigos numeros de regressao float32 nao foram simplesmente regravados:
a validacao atual usa referencias independentes e tolerancias explicitadas.
A reproducao da versao antiga continua disponivel nos arquivos v4.0/v4.1.

## Campanha completa de funcionamento

Arquivo unico executado em --experimento-artigo --discretizacao ambos.

| Caso | Malhas N | Pontos x (cm) |
|---|---|---|
| Homogeneo | 1, 13, 26, 52, 100, 104 | 0, 13, 26 |
| Heterogeneo | 3, 12, 48, 96, 192, 300, 384 | 0, 50, 100, 150 |

Foram 52 resolucoes: 13 malhas por discretizacao, com Neural Physics e Thomas.
TODAS convergiram, tanto no laco externo quanto em todas as fontes fixas.
Precisao comum float64; tol_k=1e-6, tol_phi=1e-5, tol_fonte=1e-5,
max_iter=1000, max_iter_fonte=5000, omega=0.75 e relaxacao da correcao=0.20.

Foram conferidos nos CSVs os controles correspondentes a cada par (caso,N).
Montagem, transferencia e iteracao numerica foram verificadas separadamente,
sem soma duplicada do tempo da fonte fixa (que e parte da iteracao numerica).
Os tempos desta campanha sao de VALIDACAO FUNCIONAL, nao benchmark para artigo:
houve outros testes executando simultaneamente e nao houve repeticoes estatisticas.

Saidas locais ignoradas pelo Git em resultados_validacao_controlada_v4_2/:
- csv/, figuras/, figuras_artigo/, tabelas_latex/ e relatorios/.
- Campanhas mdf/ e mef/, com dados e relatorios separados.
- CSVs e PDF/Markdown consolidados na raiz da campanha.
- 33 figuras por discretizacao; 66 incluidas no relatorio consolidado.

Os PDFs foram renderizados e as paginas iniciais/tabelas inspecionadas.
As tabelas cabem na largura util e a secao de tempos comeca em pagina propria.
A GUI foi exercitada oculta com mainloop real: selecao MEF, precisao comum,
tabela por (N,x), execucao 'ambos', curvas NP/Thomas e geracao do PDF.
Nao se calcula erro relativo pontual quando o fluxo de referencia e zero.

## Preservacao e limites

O backup imediatamente anterior ao controle esta em:
backups/antes_comparacao_controlada_20261027_000806/.

Os arquivos unicos MDF v4.0/v4.1 nao foram alterados. A main permanece no estado
publicado anterior; esta evolucao fica na branch codex/adicionar-mef.
O backup anterior ao MEF e a tag mdf-v4.1-pre-mef tambem foram preservados.

A v4.2 atual controla o caminho numerico entre MDF/MEF, mas nao reproduz
bit a bit o historico v4.1: ha precisao padrao comum, escala relativa de
residuo, geometria grossa para N impar, continuidade de Jacobi na malha
inteiramente grossa e quadratura MDF de potencia trapezoidal.
Isso esta detalhado em MDF_MEF_v4_2.md. Trata-se de adaptacao local 1D da
filosofia Neural Physics/AI4PDEs, nao identidade literal com codigo oficial 2D/3D.
