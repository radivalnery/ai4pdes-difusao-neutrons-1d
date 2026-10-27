"""Experimentos reprodutiveis, CSVs e figuras; base v4.1."""
import argparse
import csv
import json
import os
import platform
import tempfile
import threading
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm, inch
    from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

from .solver import SolverDifusaoAI4PDEs
from .metrics import normalizar_forma_fluxo, erro_forma_fluxo

def criar_estrutura_saida(base_saida):
    base = os.path.abspath(base_saida)
    subpastas = ["csv", "relatorios", "figuras", "figuras_artigo", "tabelas_latex", "logs"]
    os.makedirs(base, exist_ok=True)
    caminhos = {"base": base}
    for nome in subpastas:
        caminho = os.path.join(base, nome)
        os.makedirs(caminho, exist_ok=True)
        caminhos[nome] = caminho
    return caminhos


def problemas_artigo():
    return {
        "homogeneo": {
            "nome": "Homogêneo (Lamarsh adaptado)",
            "L": 26.0,
            "materiais": [{"inicio": 0.0, "fim": 26.0, "D": 0.9, "Sigma_a": 0.065, "nuSigma_f": 0.0681}],
            "cond_esquerda": "reflexiva",
            "cond_direita": "vácuo",
            "pontos_interesse": [0.0, 13.0, 26.0],
        },
        "heterogeneo": {
            "nome": "Heterogêneo (Couto, 2003)",
            "L": 150.0,
            "materiais": [
                {"inicio": 0.0, "fim": 50.0, "D": 1.333333, "Sigma_a": 0.200000, "nuSigma_f": 0.220000},
                {"inicio": 50.0, "fim": 100.0, "D": 1.333333, "Sigma_a": 0.240000, "nuSigma_f": 0.250000},
                {"inicio": 100.0, "fim": 150.0, "D": 2.777777, "Sigma_a": 0.110000, "nuSigma_f": 0.080000},
            ],
            "cond_esquerda": "reflexiva",
            "cond_direita": "vácuo",
            "pontos_interesse": [0.0, 50.0, 100.0, 150.0],
        },
    }


def escrever_csv(caminho, linhas, fieldnames=None):
    if not linhas:
        return
    if fieldnames is None:
        fieldnames = []
        for linha in linhas:
            for chave in linha.keys():
                if chave not in fieldnames:
                    fieldnames.append(chave)
    with open(caminho, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(linhas)


def adicionar_linhas_regioes(ax, L):
    for xv in [0.0, 50.0, 100.0, float(L)]:
        if -1.0e-12 <= xv <= float(L) + 1.0e-12:
            estilo = '-' if xv in (0.0, float(L)) else '--'
            ax.axvline(xv, color='k', linestyle=estilo, linewidth=0.9, alpha=0.55)
            ax.text(
                xv,
                0.98,
                f"x={xv:g}",
                transform=ax.get_xaxis_transform(),
                rotation=90,
                va="top",
                ha="right",
                fontsize=7,
                alpha=0.75,
            )


def salvar_figura(fig, pasta_figuras, pasta_artigo, nome):
    # Valores de N proximos (100 e 104) nao podem sobrepor os rotulos.
    for ax in fig.axes:
        if ax.get_xlabel() == 'N':
            ax.xaxis.set_major_locator(MaxNLocator(nbins=7, integer=True))
    caminho_png = os.path.join(pasta_figuras, nome + ".png")
    caminho_artigo_png = os.path.join(pasta_artigo, nome + ".png")
    caminho_artigo_pdf = os.path.join(pasta_artigo, nome + ".pdf")
    fig.savefig(caminho_png, dpi=300, bbox_inches="tight")
    fig.savefig(caminho_artigo_png, dpi=300, bbox_inches="tight")
    fig.savefig(caminho_artigo_pdf, bbox_inches="tight")
    return caminho_artigo_png


def registrar_hardware(caminhos):
    info = {
        "data_hora": datetime.now().isoformat(timespec="seconds"),
        "sistema": platform.platform(),
        "processador": platform.processor() or platform.machine(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "cuda_disponivel": bool(torch.cuda.is_available()),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    with open(os.path.join(caminhos["base"], "hardware_ambiente.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    return info


def executar_solver_artigo(caso_id, caso_cfg, N, metodo, args):
    permitir_thomas = metodo == "thomas"
    solver = SolverDifusaoAI4PDEs(
        L=caso_cfg["L"],
        N=N,
        materiais=[dict(m) for m in caso_cfg["materiais"]],
        cond_esquerda=caso_cfg["cond_esquerda"],
        cond_direita=caso_cfg["cond_direita"],
        tol_k=args.tol_k,
        tol_phi=args.tol_phi,
        max_iter=args.max_iter,
        potencia_nominal=args.potencia,
        pontos_interesse=caso_cfg["pontos_interesse"],
        dispositivo_preferido="auto",
        omega_fonte=args.omega,
        amortecimento_unet=args.amortecimento,
        tol_fonte=args.tol_fonte,
        max_iter_fonte=args.max_fonte,
        metodo_fonte=metodo,
        guardar_historicos_fonte=False,
        permitir_thomas_comparacao=permitir_thomas,
        discretizacao=args.discretizacao,
    )
    solver.resolver(metodo_fonte=metodo)
    resumo = solver.resumo_resultado(caso_cfg["nome"])
    resumo.update({"caso_id": caso_id, "N": N, "metodo_id": metodo})
    return solver, resumo


def gerar_graficos_artigo(caminhos, pares, comparacoes, refinamento):
    figuras = []
    pasta_fig = caminhos["figuras"]
    pasta_art = caminhos["figuras_artigo"]

    for (caso_id, N), par in pares.items():
        np_solver = par.get("unet_multigrid")
        th_solver = par.get("thomas")
        if np_solver is None or th_solver is None:
            continue
        x = np_solver.x
        fig = Figure(figsize=(7.2, 4.4), dpi=120)
        ax = fig.add_subplot(111)
        ax.plot(x, normalizar_forma_fluxo(np_solver.phi), label="Neural Physics/U-Net", linewidth=2)
        ax.plot(x, normalizar_forma_fluxo(th_solver.phi), label="Thomas", linestyle="--", linewidth=2)
        if caso_id == "homogeneo" and np_solver.phi_ref is not None:
            ax.plot(x, normalizar_forma_fluxo(np_solver.phi_ref), label="Analítico", linestyle=":", linewidth=2)
            titulo = f"Fluxo homogêneo normalizado - N={N}"
            nome = f"fig01_fluxo_homogeneo_np_thomas_analitico_N{N}"
        else:
            adicionar_linhas_regioes(ax, np_solver.L)
            titulo = f"Fluxo heterogêneo normalizado - N={N}"
            nome = f"fig06_fluxo_heterogeneo_np_thomas_N{N}"
        ax.set_xlabel("x [cm]")
        ax.set_ylabel("Fluxo normalizado")
        ax.set_title(titulo + ' / ' + np_solver.discretizacao.upper())
        ax.grid(True, alpha=0.3)
        ax.legend()
        figuras.append((nome, salvar_figura(fig, pasta_fig, pasta_art, nome)))

        fig = Figure(figsize=(7.2, 4.4), dpi=120)
        ax = fig.add_subplot(111)
        ax.plot(range(1, len(np_solver.historico_k)+1), np_solver.historico_k, label="Neural Physics/U-Net", linewidth=2)
        ax.plot(range(1, len(th_solver.historico_k)+1), th_solver.historico_k, label="Thomas", linestyle="--", linewidth=2)
        ax.set_xlabel("Iteração externa")
        ax.set_ylabel(r"$k_{eff}$")
        ax.set_title(f"Convergência de k_eff - {caso_id}, N={N}")
        ax.grid(True, alpha=0.3)
        ax.legend()
        nome = f"fig02_convergencia_keff_{caso_id}_N{N}"
        figuras.append((nome, salvar_figura(fig, pasta_fig, pasta_art, nome)))

    # Materiais heterogêneos
    hetero_key = next((key for key in pares if key[0] == "heterogeneo"), None)
    if hetero_key:
        solver = pares[hetero_key].get("unet_multigrid") or pares[hetero_key].get("thomas")
        if solver is not None:
            fig = Figure(figsize=(7.2, 5.0), dpi=120)
            ax = fig.add_subplot(111)
            ax.plot(solver.x, solver.D_arr, label="D(x)", linewidth=2)
            ax.plot(solver.x, solver.Sigma_a_arr, label=r"Sigma_a(x)", linewidth=2)
            ax.plot(solver.x, solver.nuSigma_f_arr, label=r"nuSigma_f(x)", linewidth=2)
            adicionar_linhas_regioes(ax, solver.L)
            ax.set_xlabel("x [cm]")
            ax.set_ylabel("Parâmetros materiais")
            ax.set_title("Parâmetros materiais do caso heterogêneo")
            ax.grid(True, alpha=0.3)
            ax.legend()
            nome = "fig05_parametros_materiais_heterogeneo"
            figuras.append((nome, salvar_figura(fig, pasta_fig, pasta_art, nome)))

    if refinamento:
        for caso_id in sorted(set(r["caso_id"] for r in refinamento)):
            dados = [r for r in refinamento if r["caso_id"] == caso_id]
            dados = sorted(dados, key=lambda r: r["N"])
            if not dados:
                continue
            fig = Figure(figsize=(7.2, 4.4), dpi=120)
            ax = fig.add_subplot(111)
            ax.plot([r["N"] for r in dados], [r["erro_k_np_thomas"] for r in dados], marker="o", label='NP vs Thomas (mesma malha)')
            if caso_id == 'homogeneo':
                ax.plot([r['N'] for r in dados], [r['erro_k_np_analitico'] for r in dados],
                        marker='s', label='NP vs analitico')
            else:
                ax.plot([r['N'] for r in dados], [r['erro_k_np_couto'] for r in dados],
                        marker='s', label='NP vs Couto (2003)')
            ax.legend()
            ax.set_xlabel("N")
            ax.set_ylabel("Erro relativo em k_eff")
            ax.set_title(f"Refinamento: erro em k_eff - {caso_id}")
            ax.grid(True, alpha=0.3)
            nome = f"fig03_refinamento_erro_keff_{caso_id}"
            figuras.append((nome, salvar_figura(fig, pasta_fig, pasta_art, nome)))

            fig = Figure(figsize=(7.2, 4.4), dpi=120)
            ax = fig.add_subplot(111)
            ax.plot([r["N"] for r in dados], [r["erro_fluxo_inf_np_thomas"] for r in dados], marker="o", label="Norma infinito")
            ax.plot([r["N"] for r in dados], [r["erro_fluxo_l2_np_thomas"] for r in dados], marker="s", label="Norma L2 relativa")
            ax.set_xlabel("N")
            ax.set_ylabel("Erro de forma do fluxo")
            ax.set_title(f"Refinamento: erro de forma do fluxo - {caso_id}")
            ax.grid(True, alpha=0.3)
            ax.legend()
            nome = f"fig04_refinamento_erro_fluxo_{caso_id}"
            figuras.append((nome, salvar_figura(fig, pasta_fig, pasta_art, nome)))

    if comparacoes:
        fig = Figure(figsize=(7.2, 4.4), dpi=120)
        ax = fig.add_subplot(111)
        for caso_id in sorted(set(r["caso_id"] for r in comparacoes)):
            dados = sorted([r for r in comparacoes if r["caso_id"] == caso_id], key=lambda r: r["N"])
            Ns = [r["N"] for r in dados]
            ax.plot(Ns, [r["tempo_np_s"] for r in dados], marker="o", label=f"NP/U-Net - {caso_id}")
            ax.plot(Ns, [r["tempo_thomas_s"] for r in dados], marker="s", linestyle="--", label=f"Thomas - {caso_id}")
        todos_N = sorted(set(r["N"] for r in comparacoes))
        ax.set_xticks(todos_N)
        ax.set_xticklabels([str(n) for n in todos_N], rotation=30, ha="right")
        ax.set_xscale("linear")
        ax.set_yscale("log")
        ax.set_xlabel("N")
        ax.set_ylabel("Tempo total [s]")
        ax.set_title("Tempo computacional: Neural Physics/U-Net versus Thomas")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        nome = "fig09_tempo_np_vs_thomas"
        figuras.append((nome, salvar_figura(fig, pasta_fig, pasta_art, nome)))

    # Comparação com Couto
    dados_couto = [r for r in comparacoes if r["caso_id"] == "heterogeneo"]
    if dados_couto:
        r = sorted(dados_couto, key=lambda x: x["N"])[-1]
        fig = Figure(figsize=(6.8, 4.2), dpi=120)
        ax = fig.add_subplot(111)
        valores = [r["k_np"], r["k_thomas"], r["k_couto"]]
        ax.bar(["Neural Physics", "Thomas", "Couto (2003)"], valores)
        ax.set_ylabel(r"$k_{eff}$")
        ax.set_title(f"Comparação de k_eff com Couto (2003) - N={r['N']}")
        ax.grid(True, axis="y", alpha=0.3)
        nome = "fig08_comparacao_keff_couto_2003"
        figuras.append((nome, salvar_figura(fig, pasta_fig, pasta_art, nome)))

    return figuras


def gerar_tabelas_latex(caminhos, comparacoes, refinamento):
    pasta = caminhos["tabelas_latex"]
    def esc(v):
        if isinstance(v, float):
            return f"{v:.6e}" if abs(v) < 1e-4 else f"{v:.6f}"
        return str(v)
    with open(os.path.join(pasta, "tab_comparacao_np_thomas.tex"), "w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{lrrrrr}\n\\hline\nCaso & N & $k_{NP}$ & $k_{Thomas}$ & $\\varepsilon_k$ & Tempo NP/Thomas \\\\ \n\\hline\n")
        for r in comparacoes:
            f.write(f"{r['caso_id']} & {r['N']} & {esc(r['k_np'])} & {esc(r['k_thomas'])} & {esc(r['erro_k_np_thomas'])} & {esc(r['tempo_np_s']/max(r['tempo_thomas_s'],1e-30))} \\\\ \n")
        f.write("\\hline\n\\end{tabular}\n")
    with open(os.path.join(pasta, "tab_refinamento_malha.tex"), "w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{lrrrr}\n\\hline\nCaso & N & $\\varepsilon_k$ & $\\varepsilon_{\\phi,\\infty}$ & $\\varepsilon_{\\phi,2}$ \\\\ \n\\hline\n")
        for r in refinamento:
            f.write(f"{r['caso_id']} & {r['N']} & {esc(r['erro_k_np_thomas'])} & {esc(r['erro_fluxo_inf_np_thomas'])} & {esc(r['erro_fluxo_l2_np_thomas'])} \\\\ \n")
        f.write("\\hline\n\\end{tabular}\n")


def gerar_relatorio_experimental(caminhos, resultados, comparacoes, refinamento, figuras):
    md_path = os.path.join(caminhos["relatorios"], "relatorio_experimental.md")
    linhas = [
        "# Relatório experimental - Difusão de nêutrons 1D",
        "",
        "Escopo: comparação enxuta entre Neural Physics/U-Net multigrid, Thomas e referência Couto (2003) no caso heterogêneo.",
        'Discretizacao: ' + caminhos.get('discretizacao', '') + '. N = intervalos/elementos; nos = N+1.',
        'MEF: Galerkin P1 com massa consistente. Vacuo significa fluxo nulo na fronteira.',
        'Precisao: MDF float32; MEF float64. Os tempos nao isolam o efeito da discretizacao.',
        'As colunas convergiu_np e convergiu_thomas dos CSVs identificam resultados sem convergencia.',
        "",
        "## Comparações principais",
        "",
        "| Caso | N | k NP | k Thomas | erro k NP-Thomas | erro fluxo inf | erro fluxo L2 | Convergiram |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in comparacoes:
        linhas.append(f"| {r['caso_id']} | {r['N']} | {r['k_np']:.8f} | {r['k_thomas']:.8f} | {r['erro_k_np_thomas']:.3e} | {r['erro_fluxo_inf_np_thomas']:.3e} | {r['erro_fluxo_l2_np_thomas']:.3e} | {r['convergiu_np'] and r['convergiu_thomas']} |")
    linhas.extend([
        "",
        "## Tempos separados",
        "",
        "| Caso | N | Método | Montagem (s) | Transferência (s) | Iteração numérica (s) | Fonte fixa (s) | Total (s) |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ])
    for r in comparacoes:
        linhas.append(f"| {r['caso_id']} | {r['N']} | Neural Physics/U-Net | {r['tempo_np_montagem_s']:.4f} | {r['tempo_np_transferencia_s']:.4f} | {r['tempo_np_iteracao_s']:.4f} | {r['tempo_np_fonte_fixa_s']:.4f} | {r['tempo_np_s']:.4f} |")
        linhas.append(f"| {r['caso_id']} | {r['N']} | Thomas | {r['tempo_thomas_montagem_s']:.4f} | {r['tempo_thomas_transferencia_s']:.4f} | {r['tempo_thomas_iteracao_s']:.4f} | {r['tempo_thomas_fonte_fixa_s']:.4f} | {r['tempo_thomas_s']:.4f} |")
    linhas.extend(["", "## Figuras geradas", ""])
    for nome, caminho in figuras:
        linhas.append(f"- {nome}: `{caminho}`")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(linhas))

    if REPORTLAB_AVAILABLE:
        pdf_path = os.path.join(caminhos["relatorios"], "relatorio_experimental.pdf")
        styles = getSampleStyleSheet()
        normal = styles["BodyText"]
        title = styles["Title"]
        h2 = styles["Heading2"]
        story = [Paragraph("Relatório experimental - Difusão de nêutrons 1D", title), Spacer(1, 0.4*cm)]
        story.append(Paragraph("Escopo: Neural Physics/U-Net multigrid, Thomas e Couto (2003).", normal))
        story.append(Paragraph('Discretizacao: ' + caminhos.get('discretizacao', '')
                               + '. N = intervalos/elementos; nos = N+1. MEF: Galerkin P1, massa consistente.', normal))
        story.append(Paragraph('Precisao: MDF float32; MEF float64. Tempos nao isolam o efeito da discretizacao. '
                               'Consulte nos CSVs os indicadores de convergencia antes de usar os resultados.', normal))
        story.append(Spacer(1, 0.3*cm))
        story.append(Paragraph("Comparações principais", h2))
        data = [["Caso", "N", "k NP", "k Thomas", "erro k", "phi inf", "phi L2", "Conv."]]
        for r in comparacoes:
            data.append([r['caso_id'], r['N'], f"{r['k_np']:.8f}", f"{r['k_thomas']:.8f}", f"{r['erro_k_np_thomas']:.2e}", f"{r['erro_fluxo_inf_np_thomas']:.2e}", f"{r['erro_fluxo_l2_np_thomas']:.2e}", 'sim' if r['convergiu_np'] and r['convergiu_thomas'] else 'NAO'])
        tbl = Table(data, repeatRows=1)
        tbl.setStyle(TableStyle([
            ('GRID', (0,0), (-1,-1), 0.25, colors.grey),
            ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
            ('FONTSIZE', (0,0), (-1,-1), 7),
        ]))
        story.append(tbl)
        story.append(Spacer(1, 0.3*cm))
        story.append(Paragraph("Tempos separados", h2))
        data_tempo = [["Caso", "N", "Método", "Mont.", "Transf.", "Iter.", "Fonte", "Total"]]
        for r in comparacoes:
            data_tempo.append([
                r['caso_id'], r['N'], "NP/U-Net",
                f"{r['tempo_np_montagem_s']:.3f}",
                f"{r['tempo_np_transferencia_s']:.3f}",
                f"{r['tempo_np_iteracao_s']:.3f}",
                f"{r['tempo_np_fonte_fixa_s']:.3f}",
                f"{r['tempo_np_s']:.3f}",
            ])
            data_tempo.append([
                r['caso_id'], r['N'], "Thomas",
                f"{r['tempo_thomas_montagem_s']:.3f}",
                f"{r['tempo_thomas_transferencia_s']:.3f}",
                f"{r['tempo_thomas_iteracao_s']:.3f}",
                f"{r['tempo_thomas_fonte_fixa_s']:.3f}",
                f"{r['tempo_thomas_s']:.3f}",
            ])
        tbl_tempo = Table(data_tempo, repeatRows=1)
        tbl_tempo.setStyle(TableStyle([
            ('GRID', (0,0), (-1,-1), 0.25, colors.grey),
            ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
            ('FONTSIZE', (0,0), (-1,-1), 6),
        ]))
        story.append(tbl_tempo)
        story.append(PageBreak())
        story.append(Paragraph("Figuras em ordem controlada", h2))
        for _, caminho in figuras:
            if os.path.exists(caminho):
                story.append(Image(caminho, width=15.5*cm, height=9.0*cm))
                story.append(Spacer(1, 0.25*cm))
        doc = SimpleDocTemplate(pdf_path, pagesize=A4, leftMargin=1.6*cm, rightMargin=1.6*cm, topMargin=1.6*cm, bottomMargin=1.6*cm)
        doc.build(story)
    return md_path


def executar_experimento_artigo(args):
    """Executa uma discretizacao ou duas campanhas isoladas com consolidacao.

    MDF/MEF e a escolha da formulacao; ambos sempre executam Neural Physics
    como metodo principal e Thomas como verificacao da MESMA discretizacao.
    Campanhas diferentes nunca sobrescrevem as figuras umas das outras.
    """
    if args.discretizacao == 'ambos':
        return executar_duas_discretizacoes(args)
    caminhos = criar_estrutura_saida(args.saida)
    caminhos['discretizacao'] = args.discretizacao.upper()
    hardware = registrar_hardware(caminhos)
    with open(os.path.join(caminhos['base'], 'configuracao.json'), 'w', encoding='utf-8') as f:
        json.dump({'parametros': vars(args), 'problemas': problemas_artigo()}, f, indent=2, ensure_ascii=False)
    problemas = problemas_artigo()
    malhas_h = [int(x) for x in args.malhas_homogeneo.split(',') if x.strip()]
    malhas_he = [int(x) for x in args.malhas_heterogeneo.split(',') if x.strip()]
    if not malhas_h or not malhas_he or min(malhas_h) < 1 or min(malhas_he) < 3:
        raise ValueError('Informe malhas positivas; no heterogeneo N deve ser >= 3.')

    resultados = []
    historico_keff = []
    historico_residuo = []
    tempos = []
    pares = {}

    agenda = [("homogeneo", problemas["homogeneo"], malhas_h), ("heterogeneo", problemas["heterogeneo"], malhas_he)]
    for caso_id, caso_cfg, malhas in agenda:
        for N in malhas:
            pares[(caso_id, N)] = {}
            for metodo in ["unet_multigrid", "thomas"]:
                print(f"\n[EXPERIMENTO] caso={caso_id}, N={N}, metodo={metodo}")
                solver, resumo = executar_solver_artigo(caso_id, caso_cfg, N, metodo, args)
                pares[(caso_id, N)][metodo] = solver
                resultados.append(resumo)
                for i, k in enumerate(solver.historico_k, start=1):
                    historico_keff.append({"caso_id": caso_id, "discretizacao": args.discretizacao, "N": N, "metodo": metodo, "iteracao": i, "k_eff": float(k)})
                for i, (it_fonte, res_fonte, conv) in enumerate(zip(solver.iteracoes_fonte_fixa, solver.residuos_fonte_fixa, solver.convergiu_fonte_fixa), start=1):
                    historico_residuo.append({"caso_id": caso_id, "discretizacao": args.discretizacao, "N": N, "metodo": metodo, "chamada_fonte": i, "iteracoes_fonte": int(it_fonte), "residuo_final": float(res_fonte), "convergiu": bool(conv)})
                tr = {"caso_id": caso_id, "discretizacao": args.discretizacao, "N": N, "metodo": metodo}
                tr.update(solver.tempos_detalhados)
                tempos.append(tr)

    comparacoes = []
    refinamento = []
    fluxos_finais = []
    comparacao_couto = []
    for (caso_id, N), par in pares.items():
        np_solver = par.get("unet_multigrid")
        th_solver = par.get("thomas")
        if np_solver is None or th_solver is None:
            continue
        erro_inf, erro_l2 = erro_forma_fluxo(np_solver.phi, th_solver.phi)
        erro_k = abs(np_solver.k_eff - th_solver.k_eff) / max(abs(th_solver.k_eff), 1.0e-30)
        linha = {
            "caso_id": caso_id,
            "discretizacao": args.discretizacao,
            "N": N,
            "k_np": float(np_solver.k_eff),
            "k_thomas": float(th_solver.k_eff),
            "erro_k_np_thomas": float(erro_k),
            "erro_fluxo_inf_np_thomas": float(erro_inf),
            "erro_fluxo_l2_np_thomas": float(erro_l2),
            "tempo_np_s": float(np_solver.tempo_total),
            "tempo_thomas_s": float(th_solver.tempo_total),
            "tempo_np_montagem_s": float(np_solver.tempos_detalhados.get("tempo_montagem_total_s", 0.0)),
            "tempo_thomas_montagem_s": float(th_solver.tempos_detalhados.get("tempo_montagem_total_s", 0.0)),
            "tempo_np_transferencia_s": float(np_solver.tempos_detalhados.get("tempo_transferencia_total_s", 0.0)),
            "tempo_thomas_transferencia_s": float(th_solver.tempos_detalhados.get("tempo_transferencia_total_s", 0.0)),
            "tempo_np_iteracao_s": float(np_solver.tempos_detalhados.get("tempo_iteracao_numerica_s", 0.0)),
            "tempo_thomas_iteracao_s": float(th_solver.tempos_detalhados.get("tempo_iteracao_numerica_s", 0.0)),
            "tempo_np_fonte_fixa_s": float(np_solver.tempos_detalhados.get("tempo_fonte_fixa_s", 0.0)),
            "tempo_thomas_fonte_fixa_s": float(th_solver.tempos_detalhados.get("tempo_fonte_fixa_s", 0.0)),
            "iter_np": int(np_solver.iteracoes_totais),
            "iter_thomas": int(th_solver.iteracoes_totais),
            "convergiu_np": np_solver.convergiu,
            "convergiu_thomas": th_solver.convergiu,
            "fonte_np_convergiu": all(np_solver.convergiu_fonte_fixa),
            "k_couto": float(SolverDifusaoAI4PDEs.REFERENCIAS['heterogeneo']['k_eff']) if caso_id == "heterogeneo" else None,
        }
        if caso_id == "heterogeneo":
            k_couto = SolverDifusaoAI4PDEs.REFERENCIAS['heterogeneo']['k_eff']
            linha["erro_k_np_couto"] = abs(np_solver.k_eff - k_couto) / abs(k_couto)
            linha["erro_k_thomas_couto"] = abs(th_solver.k_eff - k_couto) / abs(k_couto)
            comparacao_couto.append(linha.copy())
        else:
            linha["k_analitico"] = float(np_solver.k_ref)
            linha["erro_k_np_analitico"] = abs(np_solver.k_eff - np_solver.k_ref) / abs(np_solver.k_ref)
            linha["erro_k_thomas_analitico"] = abs(th_solver.k_eff - th_solver.k_ref) / abs(th_solver.k_ref)
            if np_solver.phi_ref is not None:
                ei, e2 = erro_forma_fluxo(np_solver.phi, np_solver.phi_ref)
                linha["erro_fluxo_inf_np_analitico"] = ei
                linha["erro_fluxo_l2_np_analitico"] = e2
        comparacoes.append(linha)
        refinamento.append(linha.copy())
        for i, x in enumerate(np_solver.x):
            row = {"caso_id": caso_id, "discretizacao": args.discretizacao, "N": N, "x": float(x), "phi_np": float(np_solver.phi[i]), "phi_thomas": float(th_solver.phi[i])}
            if caso_id == "homogeneo" and np_solver.phi_ref is not None:
                row["phi_analitico"] = float(np_solver.phi_ref[i])
            fluxos_finais.append(row)

    escrever_csv(os.path.join(caminhos["csv"], "resultados_consolidados.csv"), resultados)
    escrever_csv(os.path.join(caminhos["csv"], "comparacao_neuralphysics_thomas.csv"), comparacoes)
    escrever_csv(os.path.join(caminhos["csv"], "comparacao_couto_2003.csv"), comparacao_couto)
    escrever_csv(os.path.join(caminhos["csv"], "refinamento_malha.csv"), refinamento)
    escrever_csv(os.path.join(caminhos["csv"], "historico_keff.csv"), historico_keff)
    escrever_csv(os.path.join(caminhos["csv"], "historico_residuo_fonte.csv"), historico_residuo)
    escrever_csv(os.path.join(caminhos["csv"], "fluxos_finais.csv"), fluxos_finais)
    escrever_csv(os.path.join(caminhos["csv"], "tempos_execucao.csv"), tempos)

    figuras = gerar_graficos_artigo(caminhos, pares, comparacoes, refinamento)
    gerar_tabelas_latex(caminhos, comparacoes, refinamento)
    gerar_relatorio_experimental(caminhos, resultados, comparacoes, refinamento, figuras)
    caminhos['_resultados'] = resultados
    caminhos['_comparacoes'] = comparacoes
    caminhos['_figuras'] = figuras

    print("\n" + "="*70)
    print("EXPERIMENTO CONCLUÍDO")
    print(f"Pasta de saída: {caminhos['base']}")
    print("Arquivos principais:")
    print(f"- CSVs: {caminhos['csv']}")
    print(f"- Figuras para artigo: {caminhos['figuras_artigo']}")
    print(f"- Relatórios: {caminhos['relatorios']}")
    print("="*70)
    return caminhos


def executar_duas_discretizacoes(args):
    """Preserva cada campanha e reune os CSVs em uma pasta comum."""
    caminhos = criar_estrutura_saida(args.saida)
    campanhas = []
    for discretizacao in ('mdf', 'mef'):
        config = argparse.Namespace(**vars(args))
        config.discretizacao = discretizacao
        config.saida = os.path.join(args.saida, discretizacao)
        campanhas.append(executar_experimento_artigo(config))
    for filename in os.listdir(campanhas[0]['csv']):
        if filename.endswith('.csv'):
            rows = []
            for campanha in campanhas:
                with open(os.path.join(campanha['csv'], filename), encoding='utf-8', newline='') as f:
                    rows.extend(csv.DictReader(f))
            escrever_csv(os.path.join(caminhos['csv'], filename), rows)
    # Consolidacao mantem o nome da discretizacao em cada linha do relatorio.
    todas_cmp = [dict(row, caso_id=row['discretizacao'].upper() + '/' + row['caso_id'])
                 for campanha in campanhas for row in campanha['_comparacoes']]
    todos_resultados = [row for campanha in campanhas for row in campanha['_resultados']]
    todas_figuras = [item for campanha in campanhas for item in campanha['_figuras']]
    caminhos['discretizacao'] = 'MDF + MEF'
    gerar_relatorio_experimental(caminhos, todos_resultados, todas_cmp, todas_cmp, todas_figuras)
    gerar_tabelas_latex(caminhos, todas_cmp, todas_cmp)
    with open(os.path.join(caminhos['relatorios'], 'indice.md'), 'w', encoding='utf-8') as f:
        f.write('# Campanha MDF + MEF\n\n')
        for nome in ('mdf', 'mef'):
            f.write(f'- [{nome.upper()}](../{nome}/relatorios/relatorio_experimental.pdf)\n')
    return caminhos


def construir_parser():
    parser = argparse.ArgumentParser(description="Difusão de nêutrons 1D - Neural Physics/U-Net, Thomas e Couto (2003).")
    grupo = parser.add_mutually_exclusive_group()
    grupo.add_argument("--gui", action="store_true", help="abre a interface gráfica Tkinter")
    grupo.add_argument("--experimento-artigo", action="store_true", help="executa o pacote experimental sem GUI")
    parser.add_argument('--discretizacao', choices=['mdf', 'mef', 'ambos'], default='mdf',
                        help='MDF ou MEF P1; ambos executa e consolida duas campanhas')
    parser.add_argument("--saida", default="resultados_artigo_v4_2", help="pasta de saída dos resultados")
    parser.add_argument("--malhas-homogeneo", default="1,13,26,52,100,104", help="lista de N para o caso homogêneo, separada por vírgulas")
    parser.add_argument("--malhas-heterogeneo", default="3,12,48,96,192,300,384", help="lista de N para o caso heterogêneo, separada por vírgulas")
    parser.add_argument("--tol-k", type=float, default=1e-6)
    parser.add_argument("--tol-phi", type=float, default=1e-5)
    parser.add_argument("--tol-fonte", type=float, default=1e-5)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--max-fonte", type=int, default=5000)
    parser.add_argument("--omega", type=float, default=0.75)
    parser.add_argument("--amortecimento", type=float, default=0.20)
    parser.add_argument("--potencia", type=float, default=100.0)
    return parser


def main():
    print("=" * 60)
    print("SOLVER DE DIFUSÃO DE NÊUTRONS 1D - v4.2 MDF/MEF")
    print("Escopo: Neural Physics/U-Net, Thomas e Couto (2003)")
    print("=" * 60)
    print(f"CUDA disponível: {torch.cuda.is_available()}")
    print("=" * 60)

    parser = construir_parser()
    args = parser.parse_args()
    if args.discretizacao == 'ambos' and not args.experimento_artigo:
        parser.error('--discretizacao ambos requer --experimento-artigo; na GUI selecione MDF ou MEF.')

    if args.experimento_artigo:
        executar_experimento_artigo(args)
        return

    # Padrão: abrir GUI, preservando o comportamento da versão anterior.
    from .gui import DifusaoGUI_AI4PDEs
    root = tk.Tk()
    app = DifusaoGUI_AI4PDEs(root)
    if args.discretizacao != 'ambos':
        app.discretizacao_var.set(args.discretizacao.upper())
    root.mainloop()

if __name__ == "__main__":
    main()
