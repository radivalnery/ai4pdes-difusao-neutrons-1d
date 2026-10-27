"""Interface Tkinter, graficos e relatorios; base v4.1 preservada."""
import csv
import json
import os
import tempfile
import threading
import queue
from html import escape
import time
from datetime import datetime

import numpy as np
import torch
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

from .campaigns import executar_campanha, AVISO_TEMPOS_PARALELOS
from .plots import desenhar_resultados

class DifusaoGUI_AI4PDEs:
    def __init__(self, root):
        self.root = root
        self.root.title("Solver de Difusão de Nêutrons 1D - Neural Physics 1D adaptado")
        self.root.state('zoomed')
        
        self.solver = None
        self.solver_thomas = None
        self.solvers = {}
        self.referencias = {}
        self.campanha = None
        self._fila_eventos = queue.Queue()
        self._fracoes = {}
        self._estados_controles = {}
        self.root.protocol('WM_DELETE_WINDOW', self.fechar_janela)
        self.executando = False
        self.resultados_tabela = []
        self.resultados_comparacao = []
        self.modo_grafico = tk.StringVar(value="fluxo")
        self.tempo_inicio_execucao = None
        
        self.problemas = {
            'Homogêneo (Lamarsh)': {
                'L': 26.0, 'N': 100,
                'materiais': [{'inicio': 0.0, 'fim': 26.0, 'D': 0.9, 
                              'Sigma_a': 0.065, 'nuSigma_f': 0.0681}],
                'esquerda': 'reflexiva', 'direita': 'vácuo',
                'potencia': 100.0,
                'pontos': [0.0, 13.0, 26.0],
                'malhas_tabela': [1, 13, 26, 52, 100, 104],
            },
            'Heterogêneo (Couto, 2003)': {
                'L': 150.0, 'N': 300,
                'materiais': [
                    {'inicio': 0.0, 'fim': 50.0, 'D': 1.333333, 
                     'Sigma_a': 0.200000, 'nuSigma_f': 0.220000},
                    {'inicio': 50.0, 'fim': 100.0, 'D': 1.333333, 
                     'Sigma_a': 0.240000, 'nuSigma_f': 0.250000},
                    {'inicio': 100.0, 'fim': 150.0, 'D': 2.777777, 
                     'Sigma_a': 0.110000, 'nuSigma_f': 0.080000}
                ],
                'esquerda': 'reflexiva', 'direita': 'vácuo',
                'potencia': 100.0,
                'pontos': [0.0, 50.0, 100.0, 150.0],
                'malhas_tabela': [3, 12, 48, 96, 192, 300, 384],
            }
        }
        
        self.materiais_atual = []
        self.pontos_interesse = [0.0, 50.0, 100.0, 150.0]
        self.potencia_nominal = 100.0
        
        self.criar_widgets()
        self.carregar_problema('Homogêneo (Lamarsh)')
    
    def criar_widgets(self):
        # Configurar grid principal
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        
        self.frame_principal = ttk.Frame(self.root)
        self.frame_principal.grid(row=0, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        self.frame_principal.grid_rowconfigure(0, weight=1)
        self.frame_principal.grid_columnconfigure(1, weight=3)
        
        # ========== PAINEL ESQUERDO ==========
        self.frame_controles = ttk.LabelFrame(self.frame_principal, text="Controles", padding=5)
        self.frame_controles.grid(row=0, column=0, sticky='nsw', padx=5, pady=5)
        self.frame_controles.grid_rowconfigure(0, weight=1)
        self.frame_controles.grid_columnconfigure(0, weight=1)
        self.settings_canvas = tk.Canvas(self.frame_controles, width=440, highlightthickness=0)
        self.settings_canvas.grid(row=0, column=0, sticky='nsew')
        scroll = ttk.Scrollbar(self.frame_controles, orient='vertical', command=self.settings_canvas.yview)
        scroll.grid(row=0, column=1, sticky='ns')
        self.settings_canvas.configure(yscrollcommand=scroll.set)
        self.painel_controles = ttk.Frame(self.settings_canvas, padding=3)
        settings_window = self.settings_canvas.create_window((0,0), window=self.painel_controles, anchor='nw')
        self.painel_controles.bind('<Configure>', lambda e: self.settings_canvas.configure(
            scrollregion=self.settings_canvas.bbox('all')))
        self.settings_canvas.bind('<Configure>', lambda e: self.settings_canvas.itemconfigure(
            settings_window, width=e.width))
        self.settings_canvas.bind('<MouseWheel>', lambda e: self.settings_canvas.yview_scroll(
            -int(e.delta/120), 'units'))
        self.root.bind('<MouseWheel>', self.rolar_parametros, add='+')
        self.painel_acoes = ttk.Frame(self.frame_controles)
        self.painel_acoes.grid(row=1, column=0, columnspan=2, sticky='ew')
        
        self.criar_controles()
        self.criar_barra_progresso()
        
        # ========== PAINEL DIREITO ==========
        self.painel_direito = ttk.Frame(self.frame_principal)
        self.painel_direito.grid(row=0, column=1, sticky=tk.N+tk.S+tk.E+tk.W, padx=5, pady=5)
        self.painel_direito.grid_rowconfigure(0, weight=1)
        self.painel_direito.grid_columnconfigure(0, weight=1)
        
        self.notebook = ttk.Notebook(self.painel_direito)
        self.notebook.grid(row=0, column=0, sticky=tk.N+tk.S+tk.E+tk.W, pady=(0, 5))
        
        # ----- Aba Gráfico -----
        self.aba_grafico = ttk.Frame(self.notebook)
        self.notebook.add(self.aba_grafico, text="Gráfico")
        self.aba_grafico.grid_rowconfigure(0, weight=1)
        self.aba_grafico.grid_rowconfigure(1, weight=0)
        self.aba_grafico.grid_columnconfigure(0, weight=1)
        
        self.fig = Figure(figsize=(10, 8), dpi=110)
        self.ax = self.fig.add_subplot(111)
        self.fig.tight_layout(pad=4.0)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.aba_grafico)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky=tk.N+tk.S+tk.E+tk.W, padx=5, pady=5)
        
        toolbar_frame = ttk.Frame(self.aba_grafico)
        toolbar_frame.grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.toolbar = NavigationToolbar2Tk(self.canvas, toolbar_frame)
        self.toolbar.update()
        
        frame_modo = ttk.Frame(self.aba_grafico)
        frame_modo.grid(row=2, column=0, sticky=tk.W, pady=5)
        ttk.Label(frame_modo, text="Visualização:").grid(row=0, column=0, sticky="w", padx=5)
        modos = [("Fluxo", "fluxo"), ("Convergência", "convergencia"),
                 ("Resíduo fonte", "residuo_fonte"), ("Tempos", "tempo_metodo"),
                 ("Forma do fluxo", "comparacao"), ("Materiais", "todos"),
                 ("Refinamento", "tabela_N"), ("Escalabilidade", "escalabilidade")]
        for i, (texto, valor) in enumerate(modos):
            ttk.Radiobutton(frame_modo, text=texto, variable=self.modo_grafico,
                           value=valor, command=self.atualizar_grafico).grid(
                               row=1+i//4, column=i%4, sticky='w', padx=5)
        
        self.frame_resultados = ttk.LabelFrame(self.aba_grafico, text="Resultados", padding=5)
        self.frame_resultados.grid(row=3, column=0, sticky=tk.W+tk.E, pady=5)
        self.texto_resultados = scrolledtext.ScrolledText(self.frame_resultados, height=3)
        self.texto_resultados.pack(fill=tk.X)
        
        self.criar_aba_execucoes()

        # ----- Aba Tabelas -----
        self.aba_tabelas = ttk.Frame(self.notebook)
        self.notebook.add(self.aba_tabelas, text="Tabelas")
        self.aba_tabelas.grid_rowconfigure(0, weight=1)
        self.aba_tabelas.grid_columnconfigure(0, weight=1)
        
        self.tabela_frame = ttk.LabelFrame(self.aba_tabelas, text="Resultados para diferentes N", padding=5)
        self.tabela_frame.grid(row=0, column=0, sticky=tk.N+tk.S+tk.E+tk.W, padx=5, pady=5)
        self.tabela_frame.grid_rowconfigure(0, weight=1)
        self.tabela_frame.grid_columnconfigure(0, weight=1)
        
        colunas = self.montar_colunas_tabela()
        self.tree = ttk.Treeview(self.tabela_frame, columns=colunas, show='headings', height=15)
        
        scrollbar = ttk.Scrollbar(self.tabela_frame, orient=tk.VERTICAL, command=self.tree.yview)
        scrollbar_x = ttk.Scrollbar(self.tabela_frame, orient=tk.HORIZONTAL, command=self.tree.xview)
        self.tree.configure(yscrollcommand=scrollbar.set, xscrollcommand=scrollbar_x.set)
        self.tree.grid(row=0, column=0, sticky=tk.N+tk.S+tk.E+tk.W)
        scrollbar.grid(row=0, column=1, sticky=tk.N+tk.S)
        scrollbar_x.grid(row=1, column=0, sticky=tk.E+tk.W)
        self.configurar_colunas_tabela()
        
        # ============================================================
        # BOTÕES DA TABELA - AQUI ESTÃO OS BOTÕES
        # ============================================================
        frame_botoes_tabela = ttk.Frame(self.tabela_frame)
        frame_botoes_tabela.grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=5)

        ttk.Label(frame_botoes_tabela, text="Valores de N:").pack(side=tk.LEFT, padx=(0, 4))
        self.N_tabela_entry = ttk.Entry(frame_botoes_tabela, width=28)
        self.N_tabela_entry.pack(side=tk.LEFT, padx=2)
        self.N_tabela_entry.insert(0, "1, 13, 26, 52, 100, 104")
        
        self.btn_gerar_tabela = ttk.Button(frame_botoes_tabela, text="📊 Gerar Tabela (N)", 
                                          command=self.gerar_tabela_variando_N_thread)
        self.btn_gerar_tabela.pack(side=tk.LEFT, padx=2)
        
        self.btn_limpar_tabela = ttk.Button(frame_botoes_tabela, text="Limpar Tabela", command=self.limpar_tabela)
        self.btn_limpar_tabela.pack(side=tk.LEFT, padx=2)
        
        self.tabela_status = ttk.Label(self.tabela_frame, text="")
        self.tabela_status.grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=2)
        
        # ----- Aba Ajuda -----
        self.aba_ajuda = ttk.Frame(self.notebook)
        self.notebook.add(self.aba_ajuda, text="Ajuda")
        self.texto_ajuda = scrolledtext.ScrolledText(self.aba_ajuda, wrap=tk.WORD, font=('Courier New', 10))
        self.texto_ajuda.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.criar_ajuda()
        
        # Status
        self.status_bar = ttk.Label(self.root, text="Pronto", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.grid(row=1, column=0, sticky=tk.W+tk.E, padx=5, pady=2)
    
    def reset_view(self):
        if self.solver and self.solver.phi is not None:
            self.ax.set_xlim(self.solver.x[0], self.solver.x[-1])
            self.ax.set_ylim(0, np.max(self.solver.phi) * 1.1)
            self.canvas.draw()
    
    def zoom_in(self):
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        cx = (xlim[0] + xlim[1]) / 2
        cy = (ylim[0] + ylim[1]) / 2
        dx = (xlim[1] - xlim[0]) * 0.25
        dy = (ylim[1] - ylim[0]) * 0.25
        self.ax.set_xlim(cx - dx, cx + dx)
        self.ax.set_ylim(cy - dy, cy + dy)
        self.canvas.draw()
    
    def zoom_out(self):
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        cx = (xlim[0] + xlim[1]) / 2
        cy = (ylim[0] + ylim[1]) / 2
        dx = (xlim[1] - xlim[0]) * 0.5
        dy = (ylim[1] - ylim[0]) * 0.5
        self.ax.set_xlim(cx - dx, cx + dx)
        self.ax.set_ylim(cy - dy, cy + dy)
        self.canvas.draw()
    
    def fit_view(self):
        if self.solver and self.solver.phi is not None:
            self.ax.set_xlim(self.solver.x[0], self.solver.x[-1])
            self.ax.set_ylim(0, np.max(self.solver.phi) * 1.1)
            self.canvas.draw()
    
    def criar_controles(self):
        row = 0
        
        ttk.Label(self.painel_controles, text="Problema Modelo:").grid(row=row, column=0, sticky=tk.W, pady=3)
        self.problema_var = tk.StringVar()
        self.problema_combo = ttk.Combobox(self.painel_controles, textvariable=self.problema_var,
                                           values=list(self.problemas.keys()), width=18)
        self.problema_combo.grid(row=row, column=1, pady=3, padx=3)
        self.problema_combo.bind('<<ComboboxSelected>>', 
                                 lambda e: self.carregar_problema(self.problema_var.get()))
        ttk.Button(self.painel_controles, text="Carregar", 
                  command=lambda: self.carregar_problema(self.problema_var.get())).grid(row=row, column=2, pady=3, padx=3)
        row += 1
        
        ttk.Separator(self.painel_controles, orient='horizontal').grid(row=row, column=0, columnspan=3, sticky=tk.W+tk.E, pady=5)
        row += 1
        
        ttk.Label(self.painel_controles, text="Domínio:", font=('Arial', 10, 'bold')).grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=3)
        row += 1
        
        ttk.Label(self.painel_controles, text="L [cm]:").grid(row=row, column=0, sticky=tk.W)
        self.L_entry = ttk.Entry(self.painel_controles, width=10)
        self.L_entry.grid(row=row, column=1, padx=3)
        self.L_entry.insert(0, "26.0")
        self.L_entry.bind('<KeyRelease>', self.atualizar_materiais_por_L)
        
        ttk.Label(self.painel_controles, text="N:").grid(row=row, column=2, sticky=tk.W)
        self.N_entry = ttk.Entry(self.painel_controles, width=8)
        self.N_entry.grid(row=row, column=3, padx=3)
        self.N_entry.insert(0, "100")
        row += 1

        ttk.Label(self.painel_controles, text="Discretização:").grid(row=row, column=0, sticky=tk.W)
        self.mdf_var = tk.BooleanVar(value=True)
        self.mef_var = tk.BooleanVar(value=False)
        selection = ttk.Frame(self.painel_controles)
        selection.grid(row=row, column=1, sticky='w')
        ttk.Checkbutton(selection, text='MDF', variable=self.mdf_var,
                        command=self.atualizar_aviso_execucao).pack(side=tk.LEFT)
        ttk.Checkbutton(selection, text='MEF', variable=self.mef_var,
                        command=self.atualizar_aviso_execucao).pack(side=tk.LEFT)
        ttk.Label(self.painel_controles, text="Precisão:").grid(row=row, column=2, sticky=tk.W)
        self.precisao_var = tk.StringVar(value='float64')
        ttk.Combobox(self.painel_controles, textvariable=self.precisao_var,
                     values=['float64', 'float32'], state='readonly', width=9).grid(row=row, column=3, padx=3)
        row += 1
        
        ttk.Label(self.painel_controles, text="Condições:", font=('Arial', 10, 'bold')).grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=3)
        row += 1
        
        ttk.Label(self.painel_controles, text="Esquerda:").grid(row=row, column=0, sticky=tk.W)
        self.esquerda_var = tk.StringVar(value="reflexiva")
        self.esquerda_combo = ttk.Combobox(self.painel_controles, textvariable=self.esquerda_var,
                                           values=['reflexiva', 'vácuo'], width=9)
        self.esquerda_combo.grid(row=row, column=1, padx=3)
        
        ttk.Label(self.painel_controles, text="Direita:").grid(row=row, column=2, sticky=tk.W)
        self.direita_var = tk.StringVar(value="vácuo")
        self.direita_combo = ttk.Combobox(self.painel_controles, textvariable=self.direita_var,
                                          values=['reflexiva', 'vácuo'], width=9)
        self.direita_combo.grid(row=row, column=3, padx=3)
        row += 1
        
        ttk.Label(self.painel_controles, text="Potência [MWT]:").grid(row=row, column=0, sticky=tk.W, pady=3)
        self.potencia_entry = ttk.Entry(self.painel_controles, width=10)
        self.potencia_entry.grid(row=row, column=1, padx=3)
        self.potencia_entry.insert(0, "100.0")
        
        ttk.Label(self.painel_controles, text="Pontos:").grid(row=row, column=2, sticky=tk.W, pady=3)
        self.pontos_entry = ttk.Entry(self.painel_controles, width=10)
        self.pontos_entry.grid(row=row, column=3, padx=3)
        self.pontos_entry.insert(0, "0, 50, 100, 150")
        row += 1
        
        ttk.Separator(self.painel_controles, orient='horizontal').grid(row=row, column=0, columnspan=4, sticky=tk.W+tk.E, pady=5)
        row += 1
        
        ttk.Label(self.painel_controles, text="Materiais:", font=('Arial', 10, 'bold')).grid(row=row, column=0, columnspan=4, sticky=tk.W, pady=3)
        row += 1
        
        self.frame_materiais = ttk.Frame(self.painel_controles)
        self.frame_materiais.grid(row=row, column=0, columnspan=4, sticky=tk.W+tk.E, pady=3)
        
        self.materiais_listbox = tk.Listbox(self.frame_materiais, height=3, width=30)
        self.materiais_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(self.frame_materiais, orient=tk.VERTICAL, command=self.materiais_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.materiais_listbox.config(yscrollcommand=scrollbar.set)
        row += 1
        
        frame_buttons = ttk.Frame(self.painel_controles)
        frame_buttons.grid(row=row, column=0, columnspan=4, pady=3)
        ttk.Button(frame_buttons, text="Adicionar", command=self.adicionar_material).pack(side=tk.LEFT, padx=2)
        ttk.Button(frame_buttons, text="Editar", command=self.editar_material).pack(side=tk.LEFT, padx=2)
        ttk.Button(frame_buttons, text="Remover", command=self.remover_material).pack(side=tk.LEFT, padx=2)
        row += 1
        
        ttk.Separator(self.painel_controles, orient='horizontal').grid(row=row, column=0, columnspan=4, sticky=tk.W+tk.E, pady=5)
        row += 1
        
        ttk.Label(self.painel_controles, text="Convergência:", font=('Arial', 10, 'bold')).grid(row=row, column=0, columnspan=4, sticky=tk.W, pady=3)
        row += 1
        
        ttk.Label(self.painel_controles, text="tol_k:").grid(row=row, column=0, sticky=tk.W)
        self.tol_k_entry = ttk.Entry(self.painel_controles, width=10)
        self.tol_k_entry.grid(row=row, column=1, padx=3)
        self.tol_k_entry.insert(0, "1e-6")
        
        ttk.Label(self.painel_controles, text="tol_phi:").grid(row=row, column=2, sticky=tk.W)
        self.tol_phi_entry = ttk.Entry(self.painel_controles, width=10)
        self.tol_phi_entry.grid(row=row, column=3, padx=3)
        self.tol_phi_entry.insert(0, "1e-5")
        row += 1
        
        ttk.Label(self.painel_controles, text="max_iter:").grid(row=row, column=0, sticky=tk.W)
        self.max_iter_entry = ttk.Entry(self.painel_controles, width=10)
        self.max_iter_entry.grid(row=row, column=1, padx=3)
        self.max_iter_entry.insert(0, "1000")
        row += 1

        ttk.Label(self.painel_controles, text="tol_fonte:").grid(row=row, column=0, sticky=tk.W)
        self.tol_fonte_entry = ttk.Entry(self.painel_controles, width=10)
        self.tol_fonte_entry.grid(row=row, column=1, padx=3)
        self.tol_fonte_entry.insert(0, "1e-5")

        ttk.Label(self.painel_controles, text="max_fonte:").grid(row=row, column=2, sticky=tk.W)
        self.max_iter_fonte_entry = ttk.Entry(self.painel_controles, width=10)
        self.max_iter_fonte_entry.grid(row=row, column=3, padx=3)
        self.max_iter_fonte_entry.insert(0, "5000")
        row += 1

        ttk.Label(self.painel_controles, text="omega:").grid(row=row, column=0, sticky=tk.W)
        self.omega_fonte_entry = ttk.Entry(self.painel_controles, width=10)
        self.omega_fonte_entry.grid(row=row, column=1, padx=3)
        self.omega_fonte_entry.insert(0, "0.75")

        ttk.Label(self.painel_controles, text="amort.:").grid(row=row, column=2, sticky=tk.W)
        self.amortecimento_entry = ttk.Entry(self.painel_controles, width=10)
        self.amortecimento_entry.grid(row=row, column=3, padx=3)
        self.amortecimento_entry.insert(0, "0.20")
        row += 1
        
        ttk.Separator(self.painel_controles, orient='horizontal').grid(row=row, column=0, columnspan=4, sticky=tk.W+tk.E, pady=5)
        row += 1

        ttk.Label(self.painel_controles, text="Computação:", font=('Arial', 10, 'bold')).grid(row=row, column=0, columnspan=4, sticky=tk.W, pady=3)
        row += 1

        ttk.Label(self.painel_controles, text="Modo:").grid(row=row, column=0, sticky=tk.W)
        self.modo_execucao_var = tk.StringVar(value='Sequencial')
        self.modo_execucao_combo = ttk.Combobox(self.painel_controles,
            textvariable=self.modo_execucao_var, values=['Sequencial', 'Paralelo'],
            width=18, state='readonly')
        self.modo_execucao_combo.grid(row=row, column=1, columnspan=3, sticky=tk.W, padx=3)
        self.modo_execucao_combo.bind('<<ComboboxSelected>>', self.atualizar_aviso_execucao)
        row += 1
        self.thomas_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.painel_controles, text='Incluir referencia Thomas',
                        variable=self.thomas_var).grid(row=row, column=0, columnspan=4, sticky='w')
        row += 1
        self.aviso_execucao_label = ttk.Label(self.painel_controles, wraplength=410, foreground='#8d4b00')
        self.aviso_execucao_label.grid(row=row, column=0, columnspan=4, sticky='ew', pady=3)
        self.atualizar_aviso_execucao()
        row += 1

        cuda_txt = "sim" if torch.cuda.is_available() else "não"
        self.dispositivo_label = ttk.Label(
            self.painel_controles,
            text=f"Dispositivo: automático | CUDA disponível: {cuda_txt}", wraplength=410
        )
        self.dispositivo_label.grid(row=row, column=0, columnspan=4, sticky=tk.W, pady=2)
        row += 1

        self.metodo_label = ttk.Label(
            self.painel_controles,
            text="Método: operadores convolucionais fixos sem treinamento", wraplength=410
        )
        self.metodo_label.grid(row=row, column=0, columnspan=4, sticky=tk.W, pady=2)
        row += 1

        ttk.Separator(self.painel_controles, orient='horizontal').grid(row=row, column=0, columnspan=4, sticky=tk.W+tk.E, pady=5)
        row += 1
        
        frame_acoes = ttk.Frame(self.painel_acoes)
        frame_acoes.pack(fill=tk.X, pady=4)
        self.btn_executar = ttk.Button(frame_acoes, text="Executar selecionados",
                                       command=self.iniciar_execucao)
        self.btn_executar.pack(side=tk.LEFT, padx=3)
        self.btn_relatorio = ttk.Button(frame_acoes, text="Relatorio", command=self.gerar_relatorio,
                                        state=tk.DISABLED)
        self.btn_relatorio.pack(side=tk.LEFT, padx=3)

    def criar_barra_progresso(self):
        self.frame_progresso = ttk.LabelFrame(self.painel_acoes, text="Progresso", padding=5)
        self.frame_progresso.pack(fill=tk.X, pady=4)
        self.progresso_var = tk.DoubleVar()
        ttk.Progressbar(self.frame_progresso, variable=self.progresso_var, maximum=100).pack(fill=tk.X)
        self.progresso_label = ttk.Label(self.frame_progresso, text="Aguardando...", wraplength=400)
        self.progresso_label.pack(fill=tk.X)

    def criar_ajuda(self):
        ajuda = """
+----------------------------------------------------------------+
| AJUDA - FORMULAS E CALCULOS                                    |
+----------------------------------------------------------------+

+----------------------------------------------------------------+
| 1. EQUACAO DE DIFUSAO DE NEUTRONS                              |
+----------------------------------------------------------------+
| -d/dx(D dphi/dx) + Sigma_a phi = (1/k_eff) nuSigma_f phi       |
+----------------------------------------------------------------+

+----------------------------------------------------------------+
| 2. MDF NO CASO HOMOGENEO                                       |
+----------------------------------------------------------------+
| Stencil interno do operador A:                                 |
| [-D/h^2,  2D/h^2 + Sigma_a,  -D/h^2]                           |
+----------------------------------------------------------------+

+----------------------------------------------------------------+
| 3. CASO HOMOGENEO: REFLEXIVA + VACUO                           |
+----------------------------------------------------------------+
| Fluxo analitico:                                               |
| phi(x) ~ cos(pi*x/(2L))                                        |
|                                                                |
| Autovalor analitico:                                           |
| k_eff = nuSigma_f / [Sigma_a + D*(pi/(2L))^2]                  |
+----------------------------------------------------------------+

+----------------------------------------------------------------+
| 4. MDF NO CASO HETEROGENEO                                     |
+----------------------------------------------------------------+
| O operador usa a forma conservativa:                           |
| -d/dx[D(x) dphi/dx]                                            |
|                                                                |
| Nas interfaces, o codigo usa coeficiente de difusao efetivo    |
| entre pontos vizinhos para preservar a corrente.               |
+----------------------------------------------------------------+

+----------------------------------------------------------------+
| 5. METODO NUMERICO                                             |
+----------------------------------------------------------------+
| Abordagem inspirada em Neural Physics/AI4PDEs:                 |
| - stencils convolucionais fixos;                               |
| - operador conservativo local em PyTorch;                      |
| - stencil conservativo para o caso heterogeneo;                |
| - suavizacao de Jacobi;                                        |
| - restricao, prolongamento e ciclo U-Net/multigrid geometrico; |
| - sem treinamento de pesos.                                    |
+----------------------------------------------------------------+

+----------------------------------------------------------------+
| 6. ERROS                                                       |
+----------------------------------------------------------------+
| erro_k   = |k_calc - k_ref| / |k_ref|                          |
| erro_phi = |phi_calc - phi_ref| / |phi_ref|                    |
|                                                                |
| Homogeneo: referencia analitica.                               |
| Heterogeneo: referencia principal de k_eff e Couto (2003).     |
+----------------------------------------------------------------+

+----------------------------------------------------------------+
| 7. MEF DE GALERKIN: ELEMENTOS LINEARES P1                       |
+----------------------------------------------------------------+
| (K + M_a) phi = M_f phi / k_eff                                |
| K_e = D/h * [[1,-1],[-1,1]]                                   |
| M_e(c) = c*h/6 * [[2,1],[1,2]]                                |
| c = Sigma_a (absorcao) ou nuSigma_f (fissao).                  |
| Massa consistente; materiais integrados por elemento/regiao.   |
| Reflexiva: corrente nula natural. Vacuo: fluxo nulo.            |
| A/F por Conv1d fixa; MEF escalonado por H^-1, sem lumping.      |
| Mesmo ciclo V e R=Hc^-1 P^T Hf nas duas discretizacoes.         |
| Thomas verifica o mesmo sistema e nao entra no ciclo V.        |
+----------------------------------------------------------------+

+----------------------------------------------------------------+
| 8. LEITURA DOS RESULTADOS                                      |
+----------------------------------------------------------------+
| N = intervalos/elementos; numero de nos = N+1.                  |
| Erro relativo pontual: N/A quando a referencia e zero.          |
| Erro global de forma: normas nodais infinito e L2 relativa.     |
| Tempos separados: montagem, transferencia e iteracao.           |
| Fonte fixa ja esta incluida no tempo de iteracao numerica.      |
+----------------------------------------------------------------+
"""
        self.texto_ajuda.insert(tk.END, ajuda)
        self.texto_ajuda.config(state=tk.DISABLED)
    
    def atualizar_materiais_por_L(self, event=None):
        try:
            L = float(self.L_entry.get())
            if self.materiais_atual:
                self.materiais_atual[-1]['fim'] = L
                self.atualizar_lista_materiais()
        except ValueError:
            pass
    
    def carregar_problema(self, nome):
        if nome not in self.problemas:
            return
        config = self.problemas[nome]
        self.L_entry.delete(0, tk.END)
        self.L_entry.insert(0, str(config['L']))
        self.N_entry.delete(0, tk.END)
        self.N_entry.insert(0, str(config['N']))
        self.esquerda_var.set(config['esquerda'])
        self.direita_var.set(config['direita'])
        self.potencia_entry.delete(0, tk.END)
        self.potencia_entry.insert(0, str(config.get('potencia', 100.0)))
        pontos_str = ", ".join(str(p) for p in config.get('pontos', [0.0, 50.0, 100.0, 150.0]))
        self.pontos_entry.delete(0, tk.END)
        self.pontos_entry.insert(0, pontos_str)
        malhas_str = ", ".join(str(n) for n in config.get('malhas_tabela', []))
        if malhas_str and hasattr(self, 'N_tabela_entry'):
            self.N_tabela_entry.delete(0, tk.END)
            self.N_tabela_entry.insert(0, malhas_str)
        self.materiais_atual = [dict(m) for m in config['materiais']]
        self.atualizar_lista_materiais()
        self.status_bar.config(text=f"Problema '{nome}' carregado")
    
    def atualizar_lista_materiais(self):
        self.materiais_listbox.delete(0, tk.END)
        for i, mat in enumerate(self.materiais_atual):
            texto = f"Região {i+1}: [{mat['inicio']:.0f}, {mat['fim']:.0f}] "
            texto += f"D={mat['D']:.4f}, Σa={mat['Sigma_a']:.4f}, νΣf={mat['nuSigma_f']:.4f}"
            self.materiais_listbox.insert(tk.END, texto)
    
    def adicionar_material(self):
        self.janela_material("Adicionar Material", None)
    
    def editar_material(self):
        idx = self.materiais_listbox.curselection()
        if idx:
            self.janela_material("Editar Material", idx[0])
        else:
            messagebox.showwarning("Aviso", "Selecione um material para editar.")
    
    def remover_material(self):
        idx = self.materiais_listbox.curselection()
        if idx:
            del self.materiais_atual[idx[0]]
            self.atualizar_lista_materiais()
            self.status_bar.config(text="Material removido")
        else:
            messagebox.showwarning("Aviso", "Selecione um material para remover.")
    
    def janela_material(self, titulo, idx):
        janela = tk.Toplevel(self.root)
        janela.title(titulo)
        janela.geometry("350x280")
        janela.transient(self.root)
        janela.grab_set()
        
        if idx is not None:
            mat = self.materiais_atual[idx]
            inicio, fim = mat['inicio'], mat['fim']
            D, Sigma_a, nuSigma_f = mat['D'], mat['Sigma_a'], mat['nuSigma_f']
        else:
            inicio, fim = 0.0, 10.0
            D, Sigma_a, nuSigma_f = 1.0, 0.1, 0.1
        
        ttk.Label(janela, text="Início [cm]:").grid(row=0, column=0, sticky=tk.W, pady=5, padx=10)
        inicio_entry = ttk.Entry(janela, width=15)
        inicio_entry.grid(row=0, column=1, pady=5, padx=10)
        inicio_entry.insert(0, str(inicio))
        
        ttk.Label(janela, text="Fim [cm]:").grid(row=1, column=0, sticky=tk.W, pady=5, padx=10)
        fim_entry = ttk.Entry(janela, width=15)
        fim_entry.grid(row=1, column=1, pady=5, padx=10)
        fim_entry.insert(0, str(fim))
        
        ttk.Label(janela, text="D [cm]:").grid(row=2, column=0, sticky=tk.W, pady=5, padx=10)
        D_entry = ttk.Entry(janela, width=15)
        D_entry.grid(row=2, column=1, pady=5, padx=10)
        D_entry.insert(0, str(D))
        
        ttk.Label(janela, text="Σa [1/cm]:").grid(row=3, column=0, sticky=tk.W, pady=5, padx=10)
        Sigma_a_entry = ttk.Entry(janela, width=15)
        Sigma_a_entry.grid(row=3, column=1, pady=5, padx=10)
        Sigma_a_entry.insert(0, str(Sigma_a))
        
        ttk.Label(janela, text="νΣf [1/cm]:").grid(row=4, column=0, sticky=tk.W, pady=5, padx=10)
        nuSigma_f_entry = ttk.Entry(janela, width=15)
        nuSigma_f_entry.grid(row=4, column=1, pady=5, padx=10)
        nuSigma_f_entry.insert(0, str(nuSigma_f))
        
        def salvar():
            try:
                novo_mat = {
                    'inicio': float(inicio_entry.get()),
                    'fim': float(fim_entry.get()),
                    'D': float(D_entry.get()),
                    'Sigma_a': float(Sigma_a_entry.get()),
                    'nuSigma_f': float(nuSigma_f_entry.get())
                }
                if novo_mat['inicio'] >= novo_mat['fim']:
                    messagebox.showerror("Erro", "Início deve ser menor que fim.")
                    return
                if idx is not None:
                    self.materiais_atual[idx] = novo_mat
                else:
                    self.materiais_atual.append(novo_mat)
                    self.materiais_atual.sort(key=lambda x: x['inicio'])
                self.atualizar_lista_materiais()
                janela.destroy()
                self.status_bar.config(text="Material salvo")
            except ValueError:
                messagebox.showerror("Erro", "Valores inválidos. Use números.")
        
        ttk.Button(janela, text="Salvar", command=salvar).grid(row=5, column=0, columnspan=2, pady=20)
    
    def limpar_tabela(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.resultados_tabela = []
        self.tabela_status.config(text="Tabela limpa")

    def montar_colunas_tabela(self):
        return ('Discr.', 'N', 'x', 'k_eff', 'Iterações', 'Erro_k (%)', 'Fluxo',
                'Erro fluxo', 'Conv.', 'Fonte conv.', 'Falhas fonte', 'Tempo (s)')

    def configurar_colunas_tabela(self):
        colunas = self.montar_colunas_tabela()
        self.tree.configure(columns=colunas)
        for col in colunas:
            self.tree.heading(col, text=col)
            largura = 105
            if col in ("Fluxo", "Erro fluxo", "Tempo (s)", "Falhas fonte"):
                largura = 120
            elif col == "x":
                largura = 80
            self.tree.column(col, width=largura, anchor='center', stretch=False)

    def calcular_fluxos_pontos_tabela(self, solver, pontos):
        fluxos = {}
        phi_tabela = solver.phi_normalizado if solver.phi_normalizado is not None else solver.phi
        for ponto in pontos:
            fluxo = float(np.interp(ponto, solver.x, phi_tabela))
            fluxos[ponto] = fluxo
        return fluxos

    def valores_linhas_tabela(self, resultado, pontos):
        linhas = []
        fluxos = resultado.get('fluxos_pontos', {})
        erros_fluxo = resultado.get('erro_fluxo_pontos', {})
        for ponto in pontos:
            erro_fluxo = erros_fluxo.get(ponto)
            linhas.append((
                resultado['discretizacao'],
                str(resultado['N']),
                f"{ponto:g}",
                f"{resultado['k_eff']:.8f}",
                str(resultado['iteracoes']),
                f"{(resultado['erro_k'] * 100 if resultado['erro_k'] is not None else 0.0):.4e}",
                f"{fluxos.get(ponto, 0.0):.6e}",
                f"{erro_fluxo:.4e}" if erro_fluxo is not None else "N/A",
                "sim" if resultado.get('convergiu', True) else "nao",
                "sim" if resultado.get('fonte_fixa_convergiu', True) else "não",
                str(resultado.get('chamadas_fonte_nao_convergidas', 0)),
                f"{resultado.get('tempo_total', 0.0):.4f}",
            ))
        return linhas

    def calcular_erros_fluxo_tabela(self, resultados, pontos):
        self.referencias_fluxo_tabela = {}
        for disc in dict.fromkeys(r['discretizacao'] for r in resultados):
            group = [r for r in resultados if r['discretizacao']==disc]
            self._calcular_erros_grupo(group,pontos)
            self.referencias_fluxo_tabela[disc] = self.referencia_fluxo_tabela

    def _calcular_erros_grupo(self, resultados, pontos):
        """Erro percentual por ponto para a tabela de refinamento.

        Homogêneo: usa a solução analítica normalizada por potência.
        Heterogêneo: usa a maior malha calculada como referência interna.
        """
        if not resultados:
            return
        if all(r.get('is_homogeneo', False) for r in resultados):
            for resultado in resultados:
                erros = {}
                fluxos = resultado.get('fluxos_pontos', {})
                refs = resultado.get('fluxos_ref_analitico', {})
                limite_zero = 1e-12 * max([abs(v) for v in refs.values()] or [1.])
                for ponto in pontos:
                    fluxo = fluxos.get(ponto)
                    fluxo_ref = refs.get(ponto)
                    if fluxo is None or fluxo_ref is None:
                        erros[ponto] = None
                    elif abs(fluxo_ref) > limite_zero:
                        erros[ponto] = 100.0 * abs(fluxo - fluxo_ref) / abs(fluxo_ref)
                    else:
                        erros[ponto] = None
                resultado['erro_fluxo_pontos'] = erros
            self.referencia_fluxo_tabela = 'analítica'
            return

        validos = [r for r in resultados if r.get('convergiu', True)]
        if not validos:
            for resultado in resultados:
                resultado['erro_fluxo_pontos'] = {p:None for p in pontos}
            self.referencia_fluxo_tabela = 'indisponivel'
            return
        referencia = max(validos, key=lambda item: item['N'])
        fluxos_ref = referencia.get('fluxos_pontos', {})
        limite_zero = 1e-12 * max([abs(v) for v in fluxos_ref.values()] or [1.])
        for resultado in resultados:
            erros = {}
            for ponto in pontos:
                fluxo = resultado.get('fluxos_pontos', {}).get(ponto)
                fluxo_ref = fluxos_ref.get(ponto)
                if fluxo is None or fluxo_ref is None:
                    erros[ponto] = None
                elif abs(fluxo_ref) > limite_zero:
                    erros[ponto] = 100.0 * abs(fluxo - fluxo_ref) / abs(fluxo_ref)
                else:
                    erros[ponto] = None
            resultado['erro_fluxo_pontos'] = erros
        self.referencia_fluxo_tabela = referencia['N']

    def calcular_fluxos_referencia_analitica(self, solver, pontos):
        if solver.phi_ref is None:
            return {}
        values = solver.fluxo_analitico_pontos(pontos, potencia=True)
        return dict(zip(pontos, values))

    def valores_resumo_tabela(self, resultado):
        return [
            str(resultado['N']),
            f"{resultado['k_eff']:.8f}",
            str(resultado['iteracoes']),
            f"{(resultado['erro_k'] * 100 if resultado['erro_k'] is not None else 0.0):.4e}",
        ]

    def formatar_tempo(self, segundos):
        segundos = max(0, int(segundos))
        horas = segundos // 3600
        minutos = (segundos % 3600) // 60
        seg = segundos % 60
        return f"{horas:02d}:{minutos:02d}:{seg:02d}"

    def obter_lista_N_tabela(self):
        texto = self.N_tabela_entry.get()
        minimo = max(1, len(self.materiais_atual))
        valores = []
        for item in texto.split(','):
            item = item.strip()
            if not item:
                continue
            n = int(item)
            if n < minimo:
                raise ValueError(
                    f"Todos os valores de N devem ser maiores ou iguais ao número de regiões ({minimo})."
                )
            valores.append(n)
        if not valores:
            raise ValueError("Informe pelo menos um valor de N.")
        return sorted(set(valores))

    def obter_interfaces_materiais(self):
        if not self.materiais_atual:
            return []
        try:
            L = float(self.L_entry.get())
        except Exception:
            L = None
        interfaces = []
        for mat in self.materiais_atual:
            fim = float(mat['fim'])
            if L is not None and abs(fim - L) < 1.0e-12:
                continue
            interfaces.append(fim)
        return sorted(set(interfaces))

    def obter_interfaces_materiais_solver(self):
        if self.solver is None or not getattr(self.solver, "materiais", None):
            return []
        interfaces = []
        for mat in self.solver.materiais:
            fim = float(mat['fim'])
            if abs(fim - float(self.solver.L)) < 1.0e-12:
                continue
            interfaces.append(fim)
        return sorted(set(interfaces))
    
    def criar_aba_execucoes(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text='Execucoes')
        tab.grid_rowconfigure(1, weight=1)
        tab.grid_columnconfigure(0, weight=1)
        self.contexto_execucao_var = tk.StringVar(value='Nenhuma execucao iniciada.')
        ttk.Label(tab, textvariable=self.contexto_execucao_var, wraplength=680).grid(
            row=0, column=0, columnspan=2, sticky='ew', padx=5, pady=5)
        columns = ('Discr.','N','Metodo','Estado','It.','k_eff','Montagem (s)',
                   'Transf. (s)','Iteracao (s)','Total (s)')
        self.exec_tree = ttk.Treeview(tab, columns=columns, show='headings')
        for col in columns:
            self.exec_tree.heading(col, text=col)
            self.exec_tree.column(col, width=115 if col not in ('N','It.') else 55, stretch=False, anchor='center')
        self.exec_tree.grid(row=1, column=0, sticky='nsew')
        ttk.Scrollbar(tab, orient='vertical', command=self.exec_tree.yview).grid(row=1,column=1,sticky='ns')
        sx = ttk.Scrollbar(tab, orient='horizontal', command=self.exec_tree.xview)
        sx.grid(row=2,column=0,sticky='ew')
        self.exec_tree.configure(xscrollcommand=sx.set)
        # O scrollbar vertical precisa receber tambem as atualizacoes da arvore.
        vertical = tab.grid_slaves(row=1,column=1)[0]
        self.exec_tree.configure(yscrollcommand=vertical.set)

    def fechar_janela(self):
        if self.executando:
            messagebox.showwarning('Execucao em andamento', 'Aguarde a campanha terminar antes de fechar.')
        else:
            self.root.destroy()

    def rolar_parametros(self, event):
        widget = self.root.winfo_containing(event.x_root,event.y_root)
        while widget is not None:
            if widget is self.painel_controles:
                self.settings_canvas.yview_scroll(-int(event.delta/120),'units')
                return 'break'
            widget = getattr(widget,'master',None)

    def selecionar_discretizacoes(self, valores):
        values = {str(v).lower() for v in valores}
        self.mdf_var.set('mdf' in values)
        self.mef_var.set('mef' in values)
        self.atualizar_aviso_execucao()

    def obter_discretizacoes(self):
        values = tuple(d for d,var in (('mdf', self.mdf_var), ('mef', self.mef_var)) if var.get())
        if not values:
            raise ValueError('Marque pelo menos uma discretizacao: MDF ou MEF.')
        return values

    def atualizar_aviso_execucao(self, event=None):
        if not hasattr(self, 'aviso_execucao_label'):
            return
        if self.modo_execucao_var.get() == 'Paralelo':
            text = (AVISO_TEMPOS_PARALELOS if self.mdf_var.get() and self.mef_var.get()
                    else 'Um unico perfil sera executado sequencialmente.')
        else:
            text = 'MDF/MEF selecionados serao executados um apos o outro.'
        self.aviso_execucao_label.configure(text=text)

    def _bloquear_controles(self, bloquear):
        if bloquear:
            self._estados_controles = {}
            def visit(parent):
                for w in parent.winfo_children():
                    if isinstance(w, (ttk.Entry, ttk.Combobox, ttk.Checkbutton, ttk.Button, tk.Listbox)):
                        self._estados_controles[w] = str(w.cget('state'))
                        w.configure(state='disabled')
                    visit(w)
            visit(self.painel_controles)
        else:
            for widget, state in self._estados_controles.items():
                widget.configure(state=state)
            self._estados_controles = {}
        state = tk.DISABLED if bloquear else tk.NORMAL
        for widget in (self.btn_executar, self.btn_gerar_tabela, self.btn_limpar_tabela, self.N_tabela_entry):
            widget.configure(state=state)
        self.btn_relatorio.configure(state=(tk.NORMAL if not bloquear and self.solver is not None else tk.DISABLED))

    @staticmethod
    def _id_execucao(key):
        return '|'.join(map(str,key))

    def _processar_evento(self, event):
        key = event['chave']
        iid = self._id_execucao(key)
        values = list(self.exec_tree.item(iid,'values'))
        if event['tipo'] == 'inicio':
            values[3] = 'Executando'
            self._inicios[key] = event['inicio']
            self._ativos.add(key)
        elif event['tipo'] == 'progresso':
            values[4], values[5] = str(event['iteracao']), f"{event['k_eff']:.8f}"
            self._fracoes[key] = min(.99, event['iteracao']/event['max_iter'])
        elif event['tipo'] == 'resultado':
            solver = event['solver']
            values[3] = 'Convergiu' if solver.convergiu else 'NAO convergiu'
            values[4], values[5] = str(solver.iteracoes_totais), f'{solver.k_eff:.8f}'
            values[6:] = [f'{solver.tempos_detalhados[k]:.4f}' for k in
                          ('tempo_montagem_total_s','tempo_transferencia_total_s',
                           'tempo_iteracao_numerica_s','tempo_total_s')]
            self._fracoes[key] = 1.
            self._ativos.discard(key)
        elif event['tipo'] == 'erro':
            values[3] = 'Erro: '+event['mensagem']
            self._fracoes[key] = 1.
            self._ativos.discard(key)
        self.exec_tree.item(iid, values=values)

    def _consultar_eventos(self):
        for _ in range(200):
            try:
                event = self._fila_eventos.get_nowait()
            except queue.Empty:
                break
            if event['tipo'] == 'fim':
                self._concluir_campanha(event['campanha'])
                return
            if event['tipo'] == 'erro_fatal':
                self.executando = False
                self._bloquear_controles(False)
                messagebox.showerror('Erro na campanha', event['mensagem'])
                return
            self._processar_evento(event)
        elapsed = time.perf_counter()-self.tempo_inicio_execucao
        for key in self._ativos:
            iid = self._id_execucao(key)
            values = list(self.exec_tree.item(iid,'values'))
            values[-1] = f'{time.perf_counter()-self._inicios[key]:.1f} (em curso)'
            self.exec_tree.item(iid, values=values)
        percent = 100*sum(self._fracoes.values())/max(len(self._fracoes),1)
        self.progresso_var.set(percent)
        self.frame_progresso.configure(text=f'Processamento {percent:.1f}% | tempo {self.formatar_tempo(elapsed)}')
        active = ', '.join(d.upper()+'/N='+str(n)+('/Thomas' if m=='thomas' else '/NP')
                           for d,n,m in sorted(self._ativos))
        self.progresso_label.configure(text=active or 'Preparando proxima execucao...')
        if self.executando:
            self.root.after(50, self._consultar_eventos)

    def obter_configuracao_solver(self, N=None):
        L = float(self.L_entry.get())
        N = int(self.N_entry.get()) if N is None else N
        esquerda = self.esquerda_var.get()
        direita = self.direita_var.get()
        tol_k = float(self.tol_k_entry.get())
        tol_phi = float(self.tol_phi_entry.get())
        max_iter = int(self.max_iter_entry.get())
        tol_fonte = float(self.tol_fonte_entry.get())
        max_iter_fonte = int(self.max_iter_fonte_entry.get())
        omega_fonte = float(self.omega_fonte_entry.get())
        amortecimento_unet = float(self.amortecimento_entry.get())
        potencia = float(self.potencia_entry.get())
        pontos_str = self.pontos_entry.get()
        pontos = [float(p.strip()) for p in pontos_str.split(',') if p.strip()]

        if not self.materiais_atual:
            raise ValueError("Defina pelo menos um material.")
        min_N = max(1, len(self.materiais_atual))
        if N < min_N:
            raise ValueError(f"N deve ser maior ou igual ao número de regiões ({min_N}).")
        if not (0.0 < omega_fonte <= 1.0):
            raise ValueError("omega deve estar no intervalo 0 < omega <= 1.")
        if not (0.0 <= amortecimento_unet <= 1.0):
            raise ValueError("amortecimento deve estar no intervalo 0 <= amort. <= 1.")
        if tol_fonte <= 0.0 or max_iter_fonte < 1:
            raise ValueError("tol_fonte deve ser positiva e max_fonte deve ser maior que zero.")

        return {
            "L": L,
            "discretizacao": self.obter_discretizacoes()[0],
            "precisao": self.precisao_var.get(),
            "N": N,
            "materiais": [dict(m) for m in self.materiais_atual],
            "cond_esquerda": esquerda,
            "cond_direita": direita,
            "tol_k": tol_k,
            "tol_phi": tol_phi,
            "max_iter": max_iter,
            "potencia_nominal": potencia,
            "pontos_interesse": pontos,
            "dispositivo_preferido": "auto",
            "omega_fonte": omega_fonte,
            "amortecimento_unet": amortecimento_unet,
            "tol_fonte": tol_fonte,
            "max_iter_fonte": max_iter_fonte,
        }

    def salvar_csv(self, caminho, linhas):
        if not linhas:
            with open(caminho, 'w', encoding='utf-8'):
                pass
            return
        with open(caminho, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(linhas[0].keys()))
            writer.writeheader()
            writer.writerows(linhas)

    def iniciar_execucao(self):
        self._iniciar_campanha(para_tabela=False)

    def gerar_tabela_variando_N_thread(self):
        self._iniciar_campanha(para_tabela=True)

    def _iniciar_campanha(self, para_tabela):
        if self.executando:
            return
        try:
            discs = self.obter_discretizacoes()
            malhas = self.obter_lista_N_tabela() if para_tabela else None
            config = self.obter_configuracao_solver(N=malhas[0] if malhas else None)
            mode = self.modo_execucao_var.get().lower()
            thomas = self.thomas_var.get()
            if mode == 'paralelo' and len(discs) > 1:
                if not messagebox.askyesno('Tempos em execucao paralela', AVISO_TEMPOS_PARALELOS+'\n\nContinuar?'):
                    return
        except Exception as exc:
            messagebox.showerror('Configuracao', str(exc))
            return
        self.nome_problema_execucao = self.problema_var.get()
        self.campanha = None
        self.solvers, self.referencias = {}, {}
        self.solver, self.solver_thomas = None, None
        self.resultados_comparacao = []
        self.limpar_tabela()
        self.ax.clear()
        self.canvas.draw()
        self.texto_resultados.delete('1.0',tk.END)
        self.executando = True
        self._bloquear_controles(True)
        self.tempo_inicio_execucao = time.perf_counter()
        self.progresso_var.set(0)
        self.progresso_label.configure(text='Iniciando...')
        self._fila_eventos = queue.Queue()
        self._fracoes, self._inicios, self._ativos = {}, {}, set()
        for item in self.exec_tree.get_children():
            self.exec_tree.delete(item)
        methods = ('unet_multigrid','thomas') if thomas else ('unet_multigrid',)
        for disc in discs:
            for n in malhas or [config['N']]:
                for method in methods:
                    key = (disc,n,method)
                    self._fracoes[key] = 0.
                    self.exec_tree.insert('',tk.END,iid=self._id_execucao(key),
                        values=(disc.upper(),n,'Thomas' if method=='thomas' else 'Neural Physics',
                                'Aguardando','','','','','',''))
        self.contexto_execucao_var.set(
            ' + '.join(d.upper() for d in discs)+' | N='+str(malhas or [config['N']])+
            ' | '+mode+' | parametros comuns capturados no inicio')
        self.status_bar.configure(text='Executando configuracoes selecionadas...')
        self.tabela_status.configure(text='Executando '+('tabela N' if para_tabela else 'N atual'))
        events = self._fila_eventos
        def worker():
            try:
                campaign = executar_campanha(config, discs, modo=mode, incluir_thomas=thomas,
                                             malhas=malhas, evento=events.put)
                events.put(dict(tipo='fim', campanha=campaign))
            except Exception as exc:
                events.put(dict(tipo='erro_fatal', mensagem=str(exc)))
        threading.Thread(target=worker, daemon=True).start()
        self.root.after(50, self._consultar_eventos)

    def _resultado_tabela(self, solver):
        points = solver.pontos_interesse
        return dict(N=solver.N, discretizacao=solver.discretizacao.upper(), precisao=solver.precisao,
                    nos=solver.N+1, k_eff=solver.k_eff, iteracoes=solver.iteracoes_totais,
                    erro_k=solver.erro_k, fluxos_pontos=self.calcular_fluxos_pontos_tabela(solver,points),
                    fluxos_ref_analitico=self.calcular_fluxos_referencia_analitica(solver,points),
                    is_homogeneo=solver.is_homogeneo, convergiu=solver.convergiu,
                    erro_fluxo_iterativo=solver.erro_phi_iterativo,
                    fonte_fixa_convergiu=bool(solver.convergiu_fonte_fixa) and all(solver.convergiu_fonte_fixa),
                    chamadas_fonte_nao_convergidas=sum(not ok for ok in solver.convergiu_fonte_fixa),
                    bateu_max_iter_externo=not solver.convergiu, tempo_total=solver.tempo_total)

    def _concluir_campanha(self, campaign):
        self.campanha = campaign
        self.resultados_comparacao = campaign.resumos()
        for disc in campaign.discretizacoes:
            n = max(campaign.malhas)
            solver = campaign.solvers.get((disc,n,'unet_multigrid'))
            if solver is not None:
                self.solvers[disc] = solver
                ref = campaign.solvers.get((disc,n,'thomas'))
                if ref is not None:
                    self.referencias[disc] = ref
        self.solver = next(iter(self.solvers.values()),None)
        self.solver_thomas = self.referencias.get(self.solver.discretizacao) if self.solver else None
        self.resultados_tabela = [self._resultado_tabela(campaign.solvers[key])
                                 for key in campaign.chaves()
                                 if key in campaign.solvers and key[2]=='unet_multigrid']
        self.executando = False
        self._bloquear_controles(False)
        points = campaign.configuracao['pontos_interesse']
        self.calcular_erros_fluxo_tabela(self.resultados_tabela, points)
        self.preencher_tabela_resultados(points)
        table_rows = [dict(zip(self.montar_colunas_tabela(),line))
                      for row in self.resultados_tabela for line in self.valores_linhas_tabela(row,points)]
        export_error = None
        try:
            self.salvar_csv('resultados_comparacao.csv', self.resultados_comparacao)
            self.salvar_csv('resultados_refinamento.csv',table_rows)
            with open('configuracao_execucao.json','w',encoding='utf-8') as stream:
                json.dump(campaign.metadados(), stream, indent=2, ensure_ascii=False)
        except OSError as exc:
            export_error = str(exc)
        self.contexto_execucao_var.set(
            ' + '.join(d.upper() for d in campaign.discretizacoes)+
            ' | '+campaign.modo_efetivo+' | '+campaign.aviso_tempos)
        self.atualizar_grafico()
        self.atualizar_resultados()
        self.progresso_var.set(100)
        self.frame_progresso.configure(text='Processamento encerrado | tempo '+self.formatar_tempo(campaign.tempo_parede))
        failures = sum(not s.convergiu for s in campaign.solvers.values())+len(campaign.erros)
        state = f'Finalizado: {len(campaign.solvers)} resolucoes; {failures} falha(s)/nao convergida(s).'
        self.progresso_label.configure(text=state)
        self.status_bar.configure(text=state)
        self.tabela_status.configure(text=f'{len(table_rows)} linhas (discretizacao, N, x).')
        if self.solver is not None:
            self.dispositivo_label.configure(text='Dispositivo: '+self.solver.nome_dispositivo)
        if failures:
            messagebox.showwarning('Campanha com resultados incompletos',
                state+'\nConsulte a aba Execucoes. Resultados nao convergidos nao validam a comparacao.')
        if export_error:
            messagebox.showwarning('Falha ao salvar CSV/JSON',
                'Os resultados continuam disponiveis na interface.\n'+export_error)

    def preencher_tabela_resultados(self, pontos):
        for item in self.tree.get_children():
            self.tree.delete(item)
        for resultado in self.resultados_tabela:
            for linha in self.valores_linhas_tabela(resultado,pontos):
                self.tree.insert('',tk.END,values=linha)

    def atualizar_grafico(self):
        if not self.solvers:
            return
        for other in list(self.fig.axes):
            if other is not self.ax:
                other.remove()
        self.ax.clear()
        desenhar_resultados(self.ax,self.solvers,self.referencias,self.modo_grafico.get(),
                            self.campanha.aviso_tempos if self.campanha else '',
                            self.resultados_tabela)
        self.fig.tight_layout(pad=3.)
        self.canvas.draw()

    def atualizar_resultados(self):
        if not self.campanha:
            return
        lines = [self.campanha.aviso_tempos,
                 f'Tempo de parede da campanha: {self.campanha.tempo_parede:.3f} s (nao e soma dos tempos individuais).']
        for row in self.resultados_comparacao:
            lines.append(f"{row['Discretizacao']} / N={row['N']} / {row['Método']}: "+
                         f"k={row['k_eff']:.8f}; iteracoes={row['Iter. externas']}; "+
                         f"convergiu={'nao' if row['Bateu max_iter externo'] else 'sim'}; "+
                         f"montagem={row['tempo_montagem_total_s']:.4f} s; "+
                         f"transferencia={row['tempo_transferencia_total_s']:.4f} s; "+
                         f"iteracao={row['tempo_iteracao_numerica_s']:.4f} s; total={row['Tempo (s)']:.4f} s")
        for (d,n,m),error in self.campanha.erros.items():
            lines.append(f'ERRO: {d.upper()} / N={n} / {m}: {error}')
        self.texto_resultados.delete('1.0',tk.END)
        self.texto_resultados.insert(tk.END,'\n'.join(lines))

    def gerar_relatorio(self):
        if self.solver is None or self.solver.phi is None:
            messagebox.showerror("Erro", "Execute o solver primeiro.")
            return
        if not REPORTLAB_AVAILABLE:
            arquivo = filedialog.asksaveasfilename(
                defaultextension=".md",
                filetypes=[("Markdown files", "*.md")],
                title="Salvar Relatório Markdown"
            )
            if arquivo:
                self.gerar_relatorio_markdown(arquivo)
            return

        try:
            arquivo = filedialog.asksaveasfilename(
                defaultextension=".pdf",
                filetypes=[("PDF files", "*.pdf")],
                title="Salvar Relatório PDF"
            )
            if not arquivo:
                return

            doc = SimpleDocTemplate(
                arquivo, pagesize=A4,
                leftMargin=1.5*cm, rightMargin=1.5*cm,
                topMargin=1.5*cm, bottomMargin=1.5*cm
            )
            # A fonte fornecida pelo Matplotlib cobre phi/Sigma e e incorporada
            # no PDF, sem depender de fontes Symbol instaladas no computador.
            from matplotlib.font_manager import FontProperties, findfont
            from reportlab.pdfbase import pdfmetrics
            from reportlab.pdfbase.ttfonts import TTFont
            for name, weight in (('NPReport','normal'),('NPReport-Bold','bold')):
                if name not in pdfmetrics.getRegisteredFontNames():
                    font_path = findfont(FontProperties(family='DejaVu Sans',weight=weight))
                    pdfmetrics.registerFont(TTFont(name,font_path))
            pdfmetrics.registerFontFamily('NPReport', normal='NPReport', bold='NPReport-Bold',
                                          italic='NPReport',boldItalic='NPReport-Bold')
            styles = getSampleStyleSheet()
            for style in styles.byName.values():
                if not isinstance(style,ParagraphStyle):
                    continue
                bold = 'Bold' in style.fontName or style.name.startswith(('Heading','Title'))
                style.fontName = 'NPReport-Bold' if bold else 'NPReport'
            styles['Heading2'].keepWithNext = True
            styles['Heading3'].keepWithNext = True
            story = []
            imagens_temp = []
            is_homogeneo = self.solver.is_homogeneo

            titulo_style = ParagraphStyle(
                'CustomTitle', parent=styles['Heading1'],
                fontSize=22, leading=26, spaceAfter=20, alignment=1
            )
            tabela_cell_style = ParagraphStyle(
                'TabelaCell',
                parent=styles['Normal'],
                fontSize=7,
                leading=8,
                alignment=1,
                wordWrap='CJK',
            )
            tabela_header_style = ParagraphStyle(
                'TabelaHeader',
                parent=tabela_cell_style,
                fontName='NPReport-Bold',
                textColor=colors.whitesmoke,
            )

            def celulas_tabela(dados, font_size=7):
                body_style = ParagraphStyle(
                    f'TabelaCell{font_size}',
                    parent=tabela_cell_style,
                    fontSize=font_size,
                    leading=font_size + 1,
                )
                header_style = ParagraphStyle(
                    f'TabelaHeader{font_size}',
                    parent=tabela_header_style,
                    fontSize=font_size,
                    leading=font_size + 1,
                )
                linhas = []
                for i, linha in enumerate(dados):
                    estilo = header_style if i == 0 else body_style
                    linhas.append([Paragraph(str(valor), estilo) for valor in linha])
                return linhas

            def estilo_tabela(tabela):
                tabela.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2C3E50')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'NPReport-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 7),
                    ('FONTSIZE', (0, 1), (-1, -1), 7),
                    ('LEADING', (0, 0), (-1, -1), 8),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 4),
                    ('TOPPADDING', (0, 0), (-1, -1), 3),
                    ('BOTTOMPADDING', (0, 1), (-1, -1), 3),
                    ('LEFTPADDING', (0, 0), (-1, -1), 3),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 3),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#ECF0F1')),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ]))
                return tabela

            def salvar_figura(tipo):
                fig = Figure(figsize=(9,5), dpi=150)
                ax = fig.add_subplot(111)
                desenhar_resultados(ax, self.solvers, self.referencias, tipo,
                                    self.campanha.aviso_tempos if self.campanha else '',
                                    self.resultados_tabela)
                fig.tight_layout()
                temp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                temp.close()
                fig.savefig(temp.name,dpi=150,bbox_inches='tight')
                imagens_temp.append(temp.name)
                return temp.name

            story.append(Paragraph("Relatório do Solver de Difusão de Nêutrons 1D", titulo_style))
            story.append(Paragraph('Neural Physics/U-Net multigrid / ' + ' + '.join(d.upper() for d in self.solvers), styles['Heading2']))
            story.append(Paragraph(f"Gerado em: {datetime.now().strftime('%d/%m/%Y às %H:%M')}", styles['Normal']))
            story.append(Spacer(1, 20))

            story.append(Paragraph("Sumário", styles['Heading2']))
            for item in [
                "1. Metodologia computacional",
                "2. Configuração do problema",
                "3. Configuração computacional",
                "4. Resultados das configuracoes",
                "5. Parâmetros materiais",
                "6. Comparação analítica ou referência Couto (2003)",
                "7. Convergência de k_eff",
                "8. Tabela variando N",
                "9. Discussão automática curta",
                "10. Conclusão parcial",
            ]:
                story.append(Paragraph(item, styles['Normal']))
            story.append(PageBreak())

            story.append(Paragraph("1. Metodologia Computacional", styles['Heading2']))
            for solver in self.solvers.values():
                story.append(Paragraph(solver.descricao_discretizacao(), styles['Normal']))
            if self.campanha:
                story.append(Paragraph(escape(self.campanha.aviso_tempos), styles['Normal']))
                for (d,n,m), error in self.campanha.erros.items():
                    story.append(Paragraph(escape(f'FALHA: {d.upper()}, N={n}, {m}: {error}'), styles['Normal']))
            story.append(Paragraph(
                "O resolvedor implementa uma abordagem inspirada em Neural Physics/AI4PDEs, "
                "usando operadores discretos como stencils convolucionais com pesos fixos, sem treinamento.",
                styles['Normal']
            ))
            story.append(Paragraph(
                "Neste trabalho, o termo Neural Physics não se refere ao treinamento de uma rede neural para aproximar a solução. "
                "Ele se refere à implementação de operadores numéricos discretos por meio de operações típicas de bibliotecas "
                "de inteligência artificial, como convoluções, pooling e interpolação. Os pesos são fixos e definidos pela "
                "discretização física do problema.",
                styles['Normal']
            ))
            story.append(Paragraph(
                "O problema de fonte fixa Aφ = S é tratado por aplicação de um operador conservativo local em PyTorch, "
                "suavização iterativa de Jacobi, restrição compatível com a discretização, prolongamento por interpolação "
                "linear e operadores rediscretizados em um ciclo U-Net/multigrid geométrico. "
                "No nível mais grosso, a correção também é obtida por suavizações de Jacobi, "
                "sem solver direto clássico dentro do método U-Net/multigrid.",
                styles['Normal']
            ))
            story.append(Paragraph(
                "A normalização por máximo é usada apenas para fixar a escala do autovetor durante a iteração de potência. "
                "Como o problema de autovalor determina o fluxo apenas até uma constante multiplicativa, essa normalização "
                "não altera o autovalor. Para interpretação física, o fluxo pode ser posteriormente normalizado por potência.",
                styles['Normal']
            ))
            story.append(Spacer(1, 12))

            story.append(Paragraph("2. Configuração do Problema", styles['Heading2']))
            dados = [
                ["Parâmetro", "Valor"],
                ["Comprimento (L)", f"{self.solver.L:.2f} cm"],
                ["Discretizacoes executadas", ' + '.join(d.upper() for d in self.solvers)],
                ["Malhas executadas", str(list(self.campanha.malhas)) if self.campanha else str(self.solver.N)],
                ["Modo de execucao", self.campanha.modo_efetivo if self.campanha else 'sequencial'],
                ["N dos graficos de fluxo", str(self.solver.N)],
                ["Intervalos / elementos (N)", str(self.solver.N)],
                ["Nós", str(self.solver.N + 1)],
                ["Precisão dos operadores", str(self.solver.dtype)],
                ["Tipo", "Homogêneo" if is_homogeneo else "Heterogêneo"],
                ["Condição Esquerda", self.solver.cond_esquerda.capitalize()],
                ["Condição Direita", self.solver.cond_direita.capitalize()],
                ["Potência Nominal", f"{self.solver.potencia_nominal} MWT"],
                ["Tolerância k", f"{self.solver.tol_k:.1e}"],
                ["Tolerância φ", f"{self.solver.tol_phi:.1e}"],
                ["Tolerância fonte fixa", f"{self.solver.tol_fonte:.1e}"],
                ["Máx. iterações fonte fixa (solver principal)", f"{self.solver.max_iter_fonte}"],
                ["Omega fonte fixa", f"{self.solver.omega_fonte:.3f}"],
                ["Fator de relaxação da correção multigrid", f"{self.solver.amortecimento_unet:.3f}"],
                ["Método", "Neural Physics: stencils locais por banco Conv1d fixo + U-Net/multigrid"],
                ["Transferências", "P linear; R=Hc^-1 P^T Hf (mesmas rotinas MDF/MEF)"],
                ["Ciclo / suavização", "V; 2 pré, 2 pós, 20 Jacobi no grosso; até 6 níveis"],
                ["Pesos treináveis", "Não"],
                ["Relação com AI4PDEs", "Adaptação 1D local da filosofia do pacote"]
            ]
            story.append(estilo_tabela(Table(celulas_tabela(dados, font_size=7), colWidths=[4.5*cm, 11.5*cm])))
            story.append(Spacer(1, 20))

            story.append(Paragraph("3. Configuração Computacional", styles['Heading2']))
            if self.campanha:
                story.append(Paragraph(
                    f'Tempo de parede da campanha: {self.campanha.tempo_parede:.6f} s. '
                    'Nao e a soma dos tempos individuais e nao inclui relatorio/graficos.', styles['Normal']))
            story.append(Paragraph(f'Detalhe do perfil {self.solver.discretizacao.upper()}, N={self.solver.N}:', styles['Normal']))
            dados_comp = [
                ["Item", "Valor"],
                ["CPU/GPU usada", f"{self.solver.device.type.upper()} - {self.solver.nome_dispositivo}"],
                ["PyTorch disponível", f"Sim ({torch.__version__})"],
                ["CUDA disponível", "Sim" if self.solver.cuda_disponivel else "Não"],
                ["Método executado", self.solver.metodo_executado],
                ["Tempo de montagem", f"{self.solver.tempos_detalhados.get('tempo_montagem_total_s', 0.0):.6f} s"],
                ["Tempo de transferência", f"{self.solver.tempos_detalhados.get('tempo_transferencia_total_s', 0.0):.6f} s"],
                ["Tempo de iteração numérica", f"{self.solver.tempos_detalhados.get('tempo_iteracao_numerica_s', 0.0):.6f} s"],
                ["Tempo da fonte fixa", f"{self.solver.tempos_detalhados.get('tempo_fonte_fixa_s', 0.0):.6f} s"],
                ["Tempo total", f"{self.solver.tempo_total:.6f} s"],
                ["Tempo médio por iteração", f"{self.solver.tempo_medio_iteracao:.6f} s"],
            ]
            story.append(estilo_tabela(Table(celulas_tabela(dados_comp, font_size=7), colWidths=[4.5*cm, 11.5*cm])))
            story.append(Spacer(1,10))
            fases = [['Perfil','N','Montagem (s)','Transf. (s)','Iteracao (s)','Total (s)']]
            for r in self.resultados_comparacao:
                perfil = r['Discretizacao']+(' / Thomas' if r['metodo_id']=='thomas' else ' / NP')
                fases.append([perfil,r['N']]+[f"{r[k]:.5f}" for k in
                    ('tempo_montagem_total_s','tempo_transferencia_total_s','tempo_iteracao_numerica_s','Tempo (s)')])
            story.append(estilo_tabela(Table(celulas_tabela(fases,7),
                colWidths=[3.1*cm,1.1*cm,2.9*cm,2.9*cm,2.9*cm,2.9*cm], repeatRows=1)))
            story.append(PageBreak())

            story.append(Paragraph("4. Resultados das Configuracoes", styles['Heading2']))
            story.append(Paragraph(f'Detalhe de {self.solver.discretizacao.upper()}, N={self.solver.N}; todos os perfis constam na tabela abaixo.', styles['Normal']))
            story.append(Paragraph(f"<b>k_eff = {self.solver.k_eff:.8f}</b>", styles['Normal']))
            if self.solver.k_ref:
                story.append(Paragraph(f"Referência: {self.solver.k_ref:.8f} ({self.solver.fonte_ref})", styles['Normal']))
                story.append(Paragraph(f"Erro relativo: {self.solver.erro_k*100:.4f}%", styles['Normal']))
            story.append(Paragraph(f"Iterações externas: {self.solver.iteracoes_totais}", styles['Normal']))
            story.append(Paragraph(f"Convergência: {'sim' if self.solver.convergiu else 'não'}", styles['Normal']))
            if self.solver.iteracoes_fonte_fixa:
                story.append(Paragraph(
                    f"Iterações médias do resolvedor de fonte fixa: {np.mean(self.solver.iteracoes_fonte_fixa):.2f}",
                    styles['Normal']
                ))
                story.append(Paragraph(
                    f"Último resíduo de fonte fixa: {self.solver.residuos_fonte_fixa[-1]:.3e}",
                    styles['Normal']
                ))
                n_falhas_fonte = int(sum(1 for ok in self.solver.convergiu_fonte_fixa if not ok))
                story.append(Paragraph(
                    f"Fonte fixa convergiu em todas as chamadas: {'sim' if n_falhas_fonte == 0 else 'não'}",
                    styles['Normal']
                ))
                if n_falhas_fonte:
                    story.append(Paragraph(
                        f"<b>Aviso:</b> {n_falhas_fonte} chamada(s) do resolvedor U-Net/multigrid "
                        f"atingiram max_iter_fonte={self.solver.max_iter_fonte} antes da tolerância.",
                        styles['Normal']
                    ))
            story.append(Spacer(1, 12))
            story.append(Image(salvar_figura("fluxo"), width=15*cm, height=8*cm))
            if self.resultados_comparacao:
                story.append(Spacer(1, 12))
                story.append(Paragraph("4.1 Comparacao das configuracoes", styles['Heading2']))
                story.append(Paragraph('Erros globais de forma: Neural Physics versus Thomas na mesma discretizacao e malha. '
                    'N/A significa referencia ausente ou resolucao nao convergida.', styles['Normal']))
                dados_cmp = [['Discr.','N','Metodo','k_eff','Erro k (%)','Phi inf','Phi L2','Iter.','Conv.','Total (s)']]
                for r in self.resultados_comparacao:
                    def number(value):
                        return f'{value:.3e}' if value is not None else 'N/A'
                    dados_cmp.append([r['Discretizacao'], r['N'],
                        'Thomas' if r['metodo_id']=='thomas' else 'Neural Physics',
                        f"{r['k_eff']:.8f}", number(r['Erro k (%)']),
                        number(r['erro_fluxo_global_inf']), number(r['erro_fluxo_global_L2']),
                        r['Iter. externas'], 'nao' if r['Bateu max_iter externo'] else 'sim',
                        f"{r['Tempo (s)']:.4f}"])
                story.append(estilo_tabela(Table(celulas_tabela(dados_cmp,6.1),
                    colWidths=[1.0*cm,.7*cm,2.1*cm,1.9*cm,1.8*cm,1.6*cm,1.6*cm,1.0*cm,.9*cm,1.4*cm],
                    repeatRows=1)))
                story.append(Image(salvar_figura("tempo_metodo"), width=15*cm, height=7*cm))
            story.append(PageBreak())

            story.append(Paragraph("5. Parâmetros Materiais", styles['Heading2']))
            dados_materiais = [["Região", "Início", "Fim", "D", "Σa", "νΣf"]]
            for i, mat in enumerate(self.solver.materiais):
                dados_materiais.append([
                    str(i+1), f"{mat['inicio']:.2f}", f"{mat['fim']:.2f}",
                    f"{mat['D']:.4f}", f"{mat['Sigma_a']:.4f}", f"{mat['nuSigma_f']:.4f}"
                ])
            story.append(estilo_tabela(Table(
                celulas_tabela(dados_materiais, font_size=7),
                colWidths=[1.5*cm, 2*cm, 2*cm, 2*cm, 2*cm, 2*cm]
            )))
            story.append(Spacer(1, 12))
            story.append(Image(salvar_figura("materiais"), width=15*cm, height=8*cm))
            story.append(PageBreak())

            story.append(Paragraph("6. Comparação Analítica ou Referência Couto (2003)", styles['Heading2']))
            if is_homogeneo:
                story.append(Paragraph(
                    "Para o caso homogêneo com condição reflexiva em x=0 e vácuo em x=L, "
                    "a referência usada é φ(x) ~ cos(πx/2L) e "
                    "k_eff = νΣf/[Σa + D(π/2L)^2].",
                    styles['Normal']
                ))
            else:
                story.append(Paragraph(
                    "Para o caso heterogêneo, Couto (2003) é usado como referência apenas para k_eff "
                    f"(k_eff = {self.solver.k_ref:.5f}). Não há comparação de fluxo com Couto (2003) "
                    "neste relatório, pois não foram fornecidos valores de fluxo de referência.",
                    styles['Normal']
                ))
            story.append(Spacer(1, 10))
            story.append(Image(salvar_figura("comparacao"), width=15*cm, height=8*cm))

            if self.solver.is_homogeneo:
                story.append(Paragraph('Fluxo/maximo e erro relativo a formula analitica nos pontos:',styles['Heading3']))
                dados_pontos = [['Discr.','N','x (cm)','Fluxo/maximo','Erro relativo']]
                for disc, solver in self.solvers.items():
                    for point, error in solver.erro_phi_pontos.items():
                        value = float(np.interp(point,solver.x,solver.phi))
                        dados_pontos.append([disc.upper(),solver.N,f'{point:g}',f'{value:.6e}',
                                             f'{error:.4e}' if error is not None else 'N/A (ref.=0)'])
                story.append(estilo_tabela(Table(celulas_tabela(dados_pontos,7),
                    colWidths=[2*cm,1*cm,2*cm,4*cm,4*cm],repeatRows=1)))

            story.append(PageBreak())
            story.append(Paragraph("7. Convergência de k_eff", styles['Heading2']))
            story.append(Image(salvar_figura("convergencia"), width=15*cm, height=8*cm))

            story.append(Spacer(1, 20))
            story.append(Paragraph("8. Resultados para Diferentes Malhas (N)", styles['Heading2']))
            if self.resultados_tabela:
                story.append(Paragraph('Discretização da tabela: '
                                       + ' + '.join(dict.fromkeys(r['discretizacao'] for r in self.resultados_tabela)), styles['Normal']))
                cabecalho = ["Discr.","N", "x", "k_eff", "Iter.", "Erro_k (%)", "Fluxo",
                             "Erro fluxo","Conv.", "Fonte<br/>conv.", "Falhas", "Tempo<br/>(s)"]
                dados_tabela = [cabecalho]
                for r in self.resultados_tabela:
                    fluxos = r.get('fluxos_pontos', {})
                    erros_fluxo = r.get('erro_fluxo_pontos', {})
                    for ponto in fluxos.keys():
                        erro_fluxo = erros_fluxo.get(ponto)
                        dados_tabela.append([
                            r['discretizacao'], str(r['N']),
                            f"{ponto:g}",
                            f"{r['k_eff']:.8f}",
                            str(r['iteracoes']),
                            f"{r['erro_k']*100:.4e}" if r['erro_k'] is not None else "N/A",
                            f"{fluxos.get(ponto, 0.0):.6e}",
                            f"{erro_fluxo:.4e}" if erro_fluxo is not None else "N/A",
                            "sim" if r.get('convergiu', True) else "nao",
                            "sim" if r.get('fonte_fixa_convergiu', True) else "não",
                            str(r.get('chamadas_fonte_nao_convergidas', 0)),
                            f"{r.get('tempo_total', 0.0):.4f}",
                        ])
                col_widths = [1.0*cm,.8*cm,.9*cm,2.1*cm,1.1*cm,1.6*cm,
                              2.2*cm,1.8*cm,.95*cm,1.5*cm,1.15*cm,1.5*cm]
                story.append(estilo_tabela(Table(
                    celulas_tabela(dados_tabela, font_size=6.5),
                    colWidths=col_widths,
                    repeatRows=1
                )))
                story.append(Spacer(1, 5))
                k_ref_str = f"{self.solver.k_ref:.8f}" if self.solver.k_ref else "N/A"
                story.append(Paragraph(f"Erro_k = |k_eff - k_ref| / |k_ref|  (k_ref = {k_ref_str})", styles['Normal']))
                n_ref_fluxo = getattr(self, 'referencia_fluxo_tabela', None)
                ref_fluxo_txt = '; '.join(d+': '+('N='+str(v) if isinstance(v,int) else str(v))
                                          for d,v in self.referencias_fluxo_tabela.items())
                if is_homogeneo:
                    story.append(Paragraph(
                        "Erro fluxo: desvio relativo percentual por posição em relação à solução analítica "
                        "normalizada por potência.",
                        styles['Normal']
                    ))
                else:
                    story.append(Paragraph(
                        "Erro fluxo: desvio relativo percentual por posicao usando a maior malha convergida de CADA discretizacao "
                        f"como referência interna ({ref_fluxo_txt}). Couto (2003) é usado somente como referência de k_eff.",
                        styles['Normal']
                    ))
                story.append(Spacer(1, 10))
                story.append(Image(salvar_figura("tabela_N"), width=15*cm, height=8*cm))
                story.append(Paragraph("8.1 Escalabilidade N vs tempo", styles['Heading2']))
                story.append(Image(salvar_figura("escalabilidade"), width=15*cm, height=8*cm))
            else:
                story.append(Paragraph("Nenhum dado disponível para a tabela variando N.", styles['Normal']))

            story.append(PageBreak())
            story.append(Paragraph("9. Discussão automática curta", styles['Heading2']))
            story.append(Paragraph(
                "Para problemas 1D, o método de Thomas é o resolvedor clássico mais natural para sistemas tridiagonais. "
                "A formulação Neural Physics aqui avaliada não tem como objetivo superar Thomas em 1D, mas validar uma "
                "implementação determinística baseada em operadores convolucionais fixos e operações multiescala compatíveis "
                "com bibliotecas de IA.",
                styles['Normal']
            ))
            story.append(Spacer(1, 12))
            story.append(Paragraph("10. Conclusão Parcial", styles['Heading2']))
            for disc, solver in self.solvers.items():
                state = 'convergiu' if solver.convergiu else 'NAO convergiu'
                story.append(Paragraph(f'{disc.upper()}, N={solver.N}: {state}; k_eff={solver.k_eff:.8f}. '
                    'Neural Physics/U-Net multigrid com pesos fixos, sem treinamento.', styles['Normal']))
            if self.campanha:
                story.append(Paragraph(escape(self.campanha.aviso_tempos), styles['Normal']))

            doc.build(story)

            for fig_temp in imagens_temp:
                if os.path.exists(fig_temp):
                    os.remove(fig_temp)

            messagebox.showinfo("Sucesso", f"Relatório salvo em:\n{arquivo}")
            self.status_bar.config(text=f"Relatório salvo: {arquivo}")

        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao gerar relatório:\n{str(e)}")
            self.status_bar.config(text="Erro ao gerar relatório")
            import traceback
            traceback.print_exc()

    def gerar_relatorio_markdown(self, arquivo):
        if self.campanha is None:
            return
        meta = self.campanha.metadados()
        lines = ['# Relatorio comparativo de difusao de neutrons 1D', '',
                 '## Execucao registrada', self.campanha.aviso_tempos,
                 f"- Discretizacoes: {', '.join(d.upper() for d in self.campanha.discretizacoes)}",
                 f"- Modo solicitado/efetivo: {self.campanha.modo_solicitado}/{self.campanha.modo_efetivo}",
                 f"- Tempo de parede da campanha: {self.campanha.tempo_parede:.6f} s",
                 '## Parametros comuns', '```json', json.dumps(meta['configuracao'],indent=2,ensure_ascii=False), '```',
                 '## Metodo', 'Mesmo banco Conv1d fixo e ciclo V/U-Net, sem treinamento. '
                 'MEF P1 usa massa consistente. Thomas e somente referencia externa.']
        for solver in self.solvers.values():
            lines.append('- '+solver.descricao_discretizacao())
        lines += ['', '## Resultados', '| Discr. | N | Metodo | k_eff | Iteracoes | Convergiu | Erro forma inf vs Thomas | Erro forma L2 vs Thomas |',
                  '|---|---:|---|---:|---:|---|---:|---:|']
        def fmt(value):
            return f'{value:.4e}' if value is not None else 'N/A'
        for r in self.resultados_comparacao:
            lines.append(f"| {r['Discretizacao']} | {r['N']} | {r['Método']} | {r['k_eff']:.8f} | "+
                         f"{r['Iter. externas']} | {not r['Bateu max_iter externo']} | "+
                         f"{fmt(r['erro_fluxo_global_inf'])} | {fmt(r['erro_fluxo_global_L2'])} |")
        lines += ['', '## Tempos separados',
                  '| Discr. | N | Metodo | Montagem (s) | Transferencia (s) | Iteracao (s) | Total (s) |',
                  '|---|---:|---|---:|---:|---:|---:|']
        for r in self.resultados_comparacao:
            lines.append(f"| {r['Discretizacao']} | {r['N']} | {r['Método']} | "+
                ' | '.join(f"{r[k]:.6f}" for k in ('tempo_montagem_total_s','tempo_transferencia_total_s',
                           'tempo_iteracao_numerica_s','Tempo (s)'))+' |')
        lines += ['', 'Fonte fixa e parte da iteracao; nao somar novamente. '
                  'Couto (2003) referencia k_eff heterogeneo, nao fluxo.', '## Fluxo por (discretizacao, N, x)',
                  '| '+' | '.join(self.montar_colunas_tabela())+' |',
                  '| '+' | '.join(['---']*len(self.montar_colunas_tabela()))+' |']
        for row in self.resultados_tabela:
            for values in self.valores_linhas_tabela(row,self.campanha.configuracao['pontos_interesse']):
                lines.append('| '+' | '.join(map(str,values))+' |')
        for error in meta['erros']:
            lines.append('- FALHA: '+str(error))
        with open(arquivo,'w',encoding='utf-8') as stream:
            stream.write('\n'.join(lines))
        messagebox.showinfo('Relatorio', f'Relatorio salvo em:\n{arquivo}')
