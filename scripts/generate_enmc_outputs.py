"""Gera saídas comparativas para o artigo ENMC 2026.

Uso:
    python scripts/generate_enmc_outputs.py

As execuções usam configurações moderadas por padrão para permitir validação em
CPU. Ajuste os valores de N e tolerâncias neste arquivo quando desejar tabelas
finais mais refinadas.
"""

from pathlib import Path
import csv
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neutron_ai4pdes.solver import SolverDifusaoAI4PDEs, executar_sensibilidade


OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)


def salvar_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def escrever_markdown(path, titulo, comparacao, sensibilidade):
    linhas = [
        f"# {titulo}",
        "",
        "Neste trabalho, o termo Neural Physics não se refere ao treinamento de uma rede neural para aproximar a solução. "
        "Ele se refere à implementação de operadores numéricos discretos por meio de operações típicas de bibliotecas de "
        "inteligência artificial, como convoluções, pooling e interpolação. Os pesos são fixos e definidos pela discretização física do problema.",
        "",
        "## Comparação entre resolvedores de fonte fixa",
        "",
        "| Caso | Método | N | k_eff | Referência | Erro k (%) | Iter. externas | Iter. fonte média | Resíduo final | Fonte convergiu | Tempo (s) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in comparacao:
        linhas.append(
            f"| {r['Caso']} | {r['Método']} | {r['N']} | {r['k_eff']:.8f} | "
            f"{r['Referência'] if r['Referência'] else 'N/A'} | "
            f"{r['Erro k (%)'] if r['Erro k (%)'] is not None else 'N/A'} | "
            f"{r['Iter. externas']} | {r['Iter. fonte média']:.2f} | "
            f"{r['Resíduo final'] if r['Resíduo final'] is not None else 'N/A'} | "
            f"{r.get('Fonte fixa convergiu (todas as chamadas)')} | {r['Tempo (s)']:.4f} |"
        )
    linhas.extend([
        "",
        "## Análise de sensibilidade",
        "",
        "| Caso | tol_fonte | omega | amortecimento | k_eff | Erro k (%) | Iter. externas | Iter. fonte média | Resíduo final | Fonte convergiu | Tempo (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for r in sensibilidade:
        linhas.append(
            f"| {r['Caso']} | {r['tol_fonte']:.1e} | {r['omega']:.2f} | {r['amortecimento']:.2f} | "
            f"{r['k_eff']:.8f} | {r['Erro k (%)'] if r['Erro k (%)'] is not None else 'N/A'} | "
            f"{r['Iter. externas']} | {r['Iter. fonte média']:.2f} | "
            f"{r['Resíduo final'] if r['Resíduo final'] is not None else 'N/A'} | "
            f"{r.get('Fonte fixa convergiu (todas as chamadas)')} | {r['Tempo (s)']:.4f} |"
        )
    linhas.extend([
        "",
        "## Discussão automática curta",
        "Para problemas 1D, o método de Thomas é o resolvedor clássico mais natural para sistemas tridiagonais. "
        "A formulação Neural Physics aqui avaliada não tem como objetivo superar Thomas em 1D, mas validar uma "
        "implementação determinística baseada em operadores convolucionais fixos e operações multiescala compatíveis com bibliotecas de IA.",
    ])
    path.write_text("\n".join(linhas), encoding="utf-8")


def caso_homogeneo():
    return {
        "L": 26.0,
        "N": 80,
        "materiais": [{"inicio": 0.0, "fim": 26.0, "D": 0.9, "Sigma_a": 0.065, "nuSigma_f": 0.0681}],
        "cond_esquerda": "reflexiva",
        "cond_direita": "vácuo",
        "tol_k": 1.0e-6,
        "tol_phi": 1.0e-5,
        "max_iter": 500,
        "potencia_nominal": 100.0,
        "pontos_interesse": [0.0, 13.0, 26.0],
        "omega_fonte": 0.75,
        "amortecimento_unet": 0.20,
        "tol_fonte": 1.0e-5,
        "max_iter_fonte": 1000,
    }


def caso_heterogeneo():
    return {
        "L": 150.0,
        "N": 120,
        "materiais": [
            {"inicio": 0.0, "fim": 50.0, "D": 1.333333, "Sigma_a": 0.200000, "nuSigma_f": 0.220000},
            {"inicio": 50.0, "fim": 100.0, "D": 1.333333, "Sigma_a": 0.240000, "nuSigma_f": 0.250000},
            {"inicio": 100.0, "fim": 150.0, "D": 2.777777, "Sigma_a": 0.110000, "nuSigma_f": 0.080000},
        ],
        "cond_esquerda": "reflexiva",
        "cond_direita": "vácuo",
        "tol_k": 1.0e-6,
        "tol_phi": 1.0e-5,
        "max_iter": 500,
        "potencia_nominal": 100.0,
        "pontos_interesse": [0.0, 50.0, 100.0, 150.0],
        "omega_fonte": 0.75,
        "amortecimento_unet": 0.20,
        "tol_fonte": 1.0e-5,
        "max_iter_fonte": 1000,
    }


def comparar(caso, nome):
    rows = []
    for metodo in ("unet_multigrid", "thomas"):
        cfg = dict(caso)
        cfg["metodo_fonte"] = metodo
        solver = SolverDifusaoAI4PDEs(**cfg)
        solver.resolver()
        rows.append(solver.resumo_resultado(nome))
    return rows


def main():
    casos = [("homogeneo", caso_homogeneo()), ("heterogeneo", caso_heterogeneo())]
    todas_cmp = []
    todas_sens = []
    for nome, cfg in casos:
        cmp_rows = comparar(cfg, nome)
        sens_rows = executar_sensibilidade(cfg)
        for row in sens_rows:
            row["Caso"] = nome
        todas_cmp.extend(cmp_rows)
        todas_sens.extend(sens_rows)
        escrever_markdown(
            OUT / f"relatorio_{nome}_comparativo.md",
            f"Relatório {nome} comparativo",
            cmp_rows,
            sens_rows,
        )
        escrever_markdown(
            OUT / f"relatorio_sensibilidade_{nome}.md",
            f"Relatório de sensibilidade {nome}",
            cmp_rows,
            sens_rows,
        )
    salvar_csv(OUT / "resultados_comparacao.csv", todas_cmp)
    salvar_csv(OUT / "resultados_sensibilidade.csv", todas_sens)
    print(f"Saídas gravadas em: {OUT}")


if __name__ == "__main__":
    main()
