"""
Solver de difusão de nêutrons 1D para comparação ENMC 2026.

Versão única gerada a partir do pacote modular neutron_ai4pdes.
Implementa:
- operadores convolucionais fixos sem treinamento;
- arquitetura algorítmica inspirada em U-Net/multigrid;
- método clássico de Thomas para sistemas tridiagonais;
- comparação entre resolvedores;
- análise de sensibilidade;
- interface Tkinter e relatórios PDF/Markdown.
"""

import csv
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

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm, inch
    from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False




METODO_AI4PDES_1D = (
    "Adaptação 1D da filosofia AI4PDEs: operador discreto por stencil/"
    "convolução fixa em PyTorch e correção multiescala sem treinamento."
)


def normalizar_condicao_contorno(condicao):
    return str(condicao).strip().lower().replace('á', 'a')


def calcular_k_eff_analitico(D, Sigma_a, nuSigma_f, L, cond_esq, cond_dir):
    cond_esq = normalizar_condicao_contorno(cond_esq)
    cond_dir = normalizar_condicao_contorno(cond_dir)
    if cond_esq == 'reflexiva' and cond_dir == 'vacuo':
        B2 = (np.pi / (2 * L)) ** 2
    elif cond_esq == 'vacuo' and cond_dir == 'reflexiva':
        B2 = (np.pi / (2 * L)) ** 2
    elif cond_esq == 'vacuo' and cond_dir == 'vacuo':
        B2 = (np.pi / L) ** 2
    elif cond_esq == 'reflexiva' and cond_dir == 'reflexiva':
        B2 = 0.0
    else:
        raise ValueError(f"Condições de contorno inválidas: {cond_esq}, {cond_dir}")
    denominador = Sigma_a + D * B2
    if denominador <= 0:
        return 1.0
    return nuSigma_f / denominador


def fluxo_analitico_homogeneo(x, L, cond_esq='reflexiva', cond_dir='vácuo'):
    cond_esq = normalizar_condicao_contorno(cond_esq)
    cond_dir = normalizar_condicao_contorno(cond_dir)
    if cond_esq == 'reflexiva' and cond_dir == 'vacuo':
        return np.cos(np.pi * x / (2 * L))
    elif cond_esq == 'vacuo' and cond_dir == 'reflexiva':
        return np.sin(np.pi * x / (2 * L))
    elif cond_esq == 'vacuo' and cond_dir == 'vacuo':
        return np.sin(np.pi * x / L)
    elif cond_esq == 'reflexiva' and cond_dir == 'reflexiva':
        return np.ones_like(x)
    else:
        raise ValueError(f"Condições de contorno inválidas: {cond_esq}, {cond_dir}")


# ============================================================================
# 3. MODELO NEURAL PHYSICS
# ============================================================================


def resolver_tridiagonal_thomas(lower, diag, upper, rhs):
    """
    Resolve sistema tridiagonal Ax = rhs pelo método de Thomas.

    lower: tamanho n-1
    diag: tamanho n
    upper: tamanho n-1
    rhs: tamanho n
    retorna x
    """
    a = np.asarray(lower, dtype=float).copy()
    b = np.asarray(diag, dtype=float).copy()
    c = np.asarray(upper, dtype=float).copy()
    d = np.asarray(rhs, dtype=float).copy()

    n = b.size
    if a.size != n - 1 or c.size != n - 1 or d.size != n:
        raise ValueError("Dimensões incompatíveis para sistema tridiagonal.")

    eps = 1.0e-14
    if abs(b[0]) < eps:
        raise ZeroDivisionError("Pivô nulo ou muito pequeno no método de Thomas.")

    for i in range(1, n):
        m = a[i - 1] / b[i - 1]
        b[i] -= m * c[i - 1]
        d[i] -= m * d[i - 1]
        if abs(b[i]) < eps:
            raise ZeroDivisionError("Pivô nulo ou muito pequeno no método de Thomas.")

    x = np.zeros(n, dtype=float)
    x[-1] = d[-1] / b[-1]
    for i in range(n - 2, -1, -1):
        x[i] = (d[i] - c[i] * x[i + 1]) / b[i]
    return x


class OperadorDifusao1D(nn.Module):
    """
    Operador A = -d/dx(D dphi/dx) + Sigma_a phi em forma Neural Physics.

    Para meio homogêneo, o stencil interno é aplicado como uma Conv1d fixa:

        [-D/h², 2D/h² + Sigma_a, -D/h²].

    Para meio heterogêneo, o operador é aplicado na forma conservativa usando
    D_{i+1/2}. A estrutura continua sendo local, stencilada e sem treinamento.
    """

    def __init__(self, D_arr, Sigma_a_arr, h, cond_esquerda='reflexiva', cond_direita='vácuo'):
        super().__init__()
        self.h = float(h)
        self.cond_esquerda = cond_esquerda
        self.cond_direita = cond_direita

        D_np = np.asarray(D_arr, dtype=np.float32)
        Sigma_np = np.asarray(Sigma_a_arr, dtype=np.float32)
        self.register_buffer("D", torch.tensor(D_np, dtype=torch.float32))
        self.register_buffer("Sigma_a", torch.tensor(Sigma_np, dtype=torch.float32))

        D_left = D_np[:-1]
        D_right = D_np[1:]
        D_half = np.zeros_like(D_left, dtype=np.float32)
        mask = (D_left + D_right) > 0.0
        D_half[mask] = 2.0 * D_left[mask] * D_right[mask] / (D_left[mask] + D_right[mask])
        self.register_buffer("D_half", torch.tensor(D_half, dtype=torch.float32))

        self.homogeneo = bool(np.allclose(D_np, D_np[0]) and np.allclose(Sigma_np, Sigma_np[0]))

        h2 = self.h * self.h
        k_esq = -float(D_np[0]) / h2
        k_centro = (2.0 * float(D_np[0]) / h2) + float(Sigma_np[0])
        k_dir = -float(D_np[0]) / h2
        kernel = torch.tensor([[[k_esq, k_centro, k_dir]]], dtype=torch.float32)

        self.conv_A_homogeneo = nn.Conv1d(1, 1, kernel_size=3, padding=1, padding_mode='replicate', bias=False)
        with torch.no_grad():
            self.conv_A_homogeneo.weight.data = kernel
        for param in self.conv_A_homogeneo.parameters():
            param.requires_grad = False

    def _vetor(self, phi):
        if phi.ndim == 3:
            return phi.view(-1)
        return phi

    def aplicar_contorno_fluxo(self, phi):
        phi = phi.clone()
        # Condicoes de Dirichlet sao impostas diretamente. A condicao
        # reflexiva ja esta representada na linha de fronteira do operador A,
        # portanto nao se deve sobrescrever phi[0] = phi[1] a cada suavizacao.
        if self.cond_esquerda == 'vácuo':
            phi[0] = 0.0

        if self.cond_direita == 'vácuo':
            phi[-1] = 0.0
        return phi

    def diagonal(self):
        n = self.D.numel()
        h2 = self.h * self.h
        diag = torch.zeros(n, device=self.D.device, dtype=torch.float32)
        diag[1:-1] = (self.D_half[:-1] + self.D_half[1:]) / h2 + self.Sigma_a[1:-1]

        if self.cond_esquerda == 'vácuo':
            diag[0] = 1.0
        else:
            diag[0] = 2.0 * self.D[0] / h2 + self.Sigma_a[0]

        if self.cond_direita == 'vácuo':
            diag[-1] = 1.0
        else:
            diag[-1] = 2.0 * self.D[-1] / h2 + self.Sigma_a[-1]
        return torch.clamp(diag, min=1.0e-20)

    def matriz_tridiagonal_numpy(self, rhs=None):
        """Monta lower, diag, upper e rhs pela mesma discretização do operador."""
        D = self.D.detach().cpu().numpy().astype(float)
        Sigma_a = self.Sigma_a.detach().cpu().numpy().astype(float)
        D_half = self.D_half.detach().cpu().numpy().astype(float)
        n = D.size
        h2 = self.h * self.h

        lower = np.zeros(n - 1, dtype=float)
        diag = np.zeros(n, dtype=float)
        upper = np.zeros(n - 1, dtype=float)
        rhs_np = np.zeros(n, dtype=float) if rhs is None else np.asarray(rhs, dtype=float).copy()

        for i in range(1, n - 1):
            lower[i - 1] = -D_half[i - 1] / h2
            diag[i] = (D_half[i - 1] + D_half[i]) / h2 + Sigma_a[i]
            upper[i] = -D_half[i] / h2

        if self.cond_esquerda == 'vácuo':
            diag[0] = 1.0
            upper[0] = 0.0
            rhs_np[0] = 0.0
        elif self.cond_esquerda == 'reflexiva':
            diag[0] = 2.0 * D[0] / h2 + Sigma_a[0]
            upper[0] = -2.0 * D[0] / h2

        if self.cond_direita == 'vácuo':
            diag[-1] = 1.0
            lower[-1] = 0.0
            rhs_np[-1] = 0.0
        elif self.cond_direita == 'reflexiva':
            lower[-1] = -2.0 * D[-1] / h2
            diag[-1] = 2.0 * D[-1] / h2 + Sigma_a[-1]

        return lower, diag, upper, rhs_np

    def forward(self, phi):
        phi = self._vetor(phi).to(self.D.device).float()
        h2 = self.h * self.h

        if self.homogeneo:
            out = self.conv_A_homogeneo(phi.view(1, 1, -1)).view(-1)
        else:
            out = torch.zeros_like(phi)
            grad = (phi[1:] - phi[:-1]) / self.h
            corrente = self.D_half * grad
            out[1:-1] = -(corrente[1:] - corrente[:-1]) / self.h + self.Sigma_a[1:-1] * phi[1:-1]

        if self.cond_esquerda == 'vácuo':
            out[0] = phi[0]
        elif self.cond_esquerda == 'reflexiva':
            out[0] = (2.0 * self.D[0] / h2 + self.Sigma_a[0]) * phi[0] - (2.0 * self.D[0] / h2) * phi[1]

        if self.cond_direita == 'vácuo':
            out[-1] = phi[-1]
        elif self.cond_direita == 'reflexiva':
            out[-1] = -(2.0 * self.D[-1] / h2) * phi[-2] + (2.0 * self.D[-1] / h2 + self.Sigma_a[-1]) * phi[-1]

        return out


class SolverFonteFixaUNet1D(nn.Module):
    """
    Resolve A phi = S com uma arquitetura algorítmica inspirada em U-Net/multigrid.

    Não há treinamento. As operações são:
    - aplicação do operador A por stencil/convolução fixa;
    - suavização por Jacobi ponderado;
    - restrição por AvgPool1d;
    - prolongamento por interpolação linear;
    - correção residual em múltiplas escalas.

    Essa é a parte que materializa a filosofia Neural Physics/AI4PDEs no código.
    """

    def __init__(self, operador_A, omega=0.75, amortecimento_unet=0.20):
        super().__init__()
        self.operador_A = operador_A
        self.omega = float(omega)
        self.amortecimento_unet = float(amortecimento_unet)
        self.restricao = nn.AvgPool1d(kernel_size=2, stride=2)

    def _corrigir_contorno_rhs(self, rhs):
        rhs = rhs.clone()
        # Apenas contornos de vacuo/Dirichlet recebem RHS nulo. Em contorno
        # reflexivo, a equacao discretizada na fronteira ainda possui fonte.
        if self.operador_A.cond_esquerda == 'vácuo':
            rhs[0] = 0.0
        if self.operador_A.cond_direita == 'vácuo':
            rhs[-1] = 0.0
        return rhs

    def _suavizar_jacobi(self, phi, rhs, diag, n_passos):
        for _ in range(n_passos):
            residuo = rhs - self.operador_A(phi)
            phi = phi + self.omega * residuo / diag
            phi = self.operador_A.aplicar_contorno_fluxo(phi)
        return phi

    def _correcao_unet(self, residuo, diag):
        n = residuo.numel()
        r = (residuo / diag).view(1, 1, -1)
        correcoes = []

        atual = r
        for _ in range(3):
            if atual.shape[-1] < 8:
                break
            atual = self.restricao(atual)
            correcoes.append(atual)

        if not correcoes:
            return torch.zeros(n, device=residuo.device)

        correcao = torch.zeros_like(r)
        for nivel in reversed(correcoes):
            up = F.interpolate(nivel, size=n, mode='linear', align_corners=False)
            correcao = correcao + up
        return correcao.view(-1)

    def resolver(self, rhs, chute=None, tol=1.0e-5, max_iter=5000):
        rhs = self._corrigir_contorno_rhs(rhs.float())
        if chute is None:
            phi = torch.zeros_like(rhs)
        else:
            phi = chute.clone().float()
        phi = self.operador_A.aplicar_contorno_fluxo(phi)

        diag = self.operador_A.diagonal()
        norma_rhs = max(float(torch.linalg.vector_norm(rhs, ord=float('inf')).item()), 1.0)
        ultimo_residuo = float('inf')
        historico_residuo = []

        for it in range(1, max_iter + 1):
            phi = self._suavizar_jacobi(phi, rhs, diag, n_passos=3)
            residuo = rhs - self.operador_A(phi)
            ultimo_residuo = float(torch.linalg.vector_norm(residuo, ord=float('inf')).item())
            historico_residuo.append(ultimo_residuo / norma_rhs)
            if ultimo_residuo / norma_rhs < tol:
                return phi, it, ultimo_residuo, historico_residuo

            correcao = self._correcao_unet(residuo, diag)
            phi = phi + self.amortecimento_unet * correcao
            phi = self.operador_A.aplicar_contorno_fluxo(phi)
            phi = self._suavizar_jacobi(phi, rhs, diag, n_passos=2)

        return phi, max_iter, ultimo_residuo, historico_residuo


class SolverDifusaoAI4PDEs:
    REFERENCIAS = {
        'homogeneo': {'k_eff': None, 'fonte': 'Analítica'},
        'heterogeneo': {'k_eff': 1.09506, 'fonte': 'Nozimar'}
    }
    
    def __init__(self, L, N, materiais, cond_esquerda='reflexiva', cond_direita='vácuo',
                 tol_k=1e-6, tol_phi=1e-5, max_iter=1000, potencia_nominal=100.0,
                 pontos_interesse=None, progress_callback=None, dispositivo_preferido='auto',
                 omega_fonte=0.75, amortecimento_unet=0.20,
                 tol_fonte=None, max_iter_fonte=5000,
                 metodo_fonte="unet_multigrid", guardar_historicos_fonte=False):
        
        self.L = L
        self.N = N
        self.h = L / N
        self.x = np.linspace(0, L, N+1)
        self.cond_esquerda = cond_esquerda
        self.cond_direita = cond_direita
        self.tol_k = tol_k
        self.tol_phi = tol_phi
        self.max_iter = max_iter
        self.potencia_nominal = potencia_nominal
        self.pontos_interesse = pontos_interesse or [0.0, 50.0, 100.0, 150.0]
        self.progress_callback = progress_callback
        self.dispositivo_preferido = dispositivo_preferido
        self.omega_fonte = float(omega_fonte)
        self.amortecimento_unet = float(amortecimento_unet)
        self.tol_fonte = float(tol_phi if tol_fonte is None else tol_fonte)
        self.max_iter_fonte = int(max_iter_fonte)
        self.metodo_fonte = metodo_fonte
        self.guardar_historicos_fonte = guardar_historicos_fonte
        
        self.materiais = materiais
        self.D_arr = np.zeros(N+1)
        self.Sigma_a_arr = np.zeros(N+1)
        self.nuSigma_f_arr = np.zeros(N+1)
        
        for i, x in enumerate(self.x):
            for mat in materiais:
                if mat['inicio'] <= x <= mat['fim']:
                    self.D_arr[i] = mat['D']
                    self.Sigma_a_arr[i] = mat['Sigma_a']
                    self.nuSigma_f_arr[i] = mat['nuSigma_f']
                    break
        
        self.cuda_disponivel = bool(torch.cuda.is_available())
        if dispositivo_preferido == 'cpu':
            self.device = torch.device('cpu')
        elif dispositivo_preferido == 'gpu' and self.cuda_disponivel:
            self.device = torch.device('cuda')
        elif dispositivo_preferido == 'gpu' and not self.cuda_disponivel:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda' if self.cuda_disponivel else 'cpu')

        self.nome_dispositivo = self.obter_nome_dispositivo()
        self.metodo_executado = (
            "AI4PDEs 1D adaptado com PyTorch CUDA"
            if self.device.type == 'cuda'
            else "AI4PDEs 1D adaptado com PyTorch CPU"
        )
        self.modelo_A = None
        self.solver_fonte_fixa = None
        self.iteracoes_fonte_fixa = []
        self.residuos_fonte_fixa = []
        self.historico_residuo_fonte_ultima_chamada = []
        self.historicos_residuo_fonte = []
        
        self.k_eff = None
        self.phi = None
        self.phi_normalizado = None
        self.historico_k = []
        self.historico_phi = []
        self.tempos_iteracao = []
        self.tempo_total = 0.0
        self.tempo_medio_iteracao = 0.0
        
        self.iteracoes_totais = 0
        self.convergiu = False
        self.erro_k = 0.0
        self.erro_phi_iterativo = 0.0
        self.erro_phi_max = 0.0
        self.erro_phi_pontos = {}
        self.k_ref = None
        self.fonte_ref = None
        self.phi_ref = None
        self.is_homogeneo = len(materiais) == 1

    @staticmethod
    def nome_metodo(metodo):
        nomes = {
            "unet_multigrid": "U-Net/multigrid sem treinamento",
            "thomas": "Thomas clássico",
        }
        return nomes.get(metodo, metodo)

    def obter_nome_dispositivo(self):
        if self.device.type == 'cuda':
            try:
                return torch.cuda.get_device_name(0)
            except Exception:
                return "GPU CUDA"
        return platform.processor() or platform.machine() or "CPU"
    
    def criar_operador_A(self):
        self.modelo_A = OperadorDifusao1D(
            self.D_arr,
            self.Sigma_a_arr,
            self.h,
            self.cond_esquerda,
            self.cond_direita,
        ).to(self.device)
        self.solver_fonte_fixa = SolverFonteFixaUNet1D(
            self.modelo_A,
            omega=self.omega_fonte,
            amortecimento_unet=self.amortecimento_unet,
        ).to(self.device)
        self.metodo_executado = (
            "Adaptação 1D Neural Physics/AI4PDEs: operadores convolucionais fixos "
            "e correção multiescala sem treinamento"
        )
    
    def resolver_fonte_fixa_unet(self, S, chute=None):
        if self.solver_fonte_fixa is None:
            raise RuntimeError("O resolvedor de fonte fixa ainda não foi criado.")
        S_tensor = S.to(self.device).float() if torch.is_tensor(S) else torch.tensor(S, device=self.device, dtype=torch.float32)
        chute_tensor = None
        if chute is not None:
            chute_tensor = chute.to(self.device).float() if torch.is_tensor(chute) else torch.tensor(chute, device=self.device, dtype=torch.float32)
        phi, n_iter, residuo, historico = self.solver_fonte_fixa.resolver(
            S_tensor,
            chute=chute_tensor,
            tol=self.tol_fonte,
            max_iter=self.max_iter_fonte,
        )
        self.iteracoes_fonte_fixa.append(int(n_iter))
        self.residuos_fonte_fixa.append(float(residuo))
        self.historico_residuo_fonte_ultima_chamada = list(historico)
        if self.guardar_historicos_fonte:
            self.historicos_residuo_fonte.append(list(historico))
        return phi

    def resolver_fonte_fixa_thomas(self, S):
        if self.modelo_A is None:
            raise RuntimeError("O operador A ainda não foi criado.")
        S_np = S.detach().cpu().numpy() if torch.is_tensor(S) else np.asarray(S, dtype=float)
        lower, diag, upper, rhs = self.modelo_A.matriz_tridiagonal_numpy(S_np)
        phi_np = resolver_tridiagonal_thomas(lower, diag, upper, rhs)
        phi_tensor = torch.tensor(phi_np, device=self.device, dtype=torch.float32)
        residuo_tensor = torch.tensor(rhs, device=self.device, dtype=torch.float32) - self.modelo_A(phi_tensor)
        residuo = float(torch.linalg.vector_norm(residuo_tensor, ord=float('inf')).item())
        norma_rhs = max(float(np.linalg.norm(rhs, ord=np.inf)), 1.0)
        self.iteracoes_fonte_fixa.append(1)
        self.residuos_fonte_fixa.append(residuo)
        self.historico_residuo_fonte_ultima_chamada = [residuo / norma_rhs]
        if self.guardar_historicos_fonte:
            self.historicos_residuo_fonte.append([residuo / norma_rhs])
        return phi_tensor

    def resolver_fonte_fixa(self, S, chute=None, metodo=None):
        metodo = metodo or self.metodo_fonte
        if metodo == "unet_multigrid":
            return self.resolver_fonte_fixa_unet(S, chute=chute)
        if metodo == "thomas":
            return self.resolver_fonte_fixa_thomas(S)
        raise ValueError(f"Método de fonte fixa desconhecido: {metodo}")
    
    def calcular_integral(self, f):
        integral = 0.5 * (f[0] + f[-1])
        integral += np.sum(f[1:-1])
        integral *= self.h
        return integral
    
    def calcular_referencias(self):
        if self.is_homogeneo:
            D = self.D_arr[0]
            Sigma_a = self.Sigma_a_arr[0]
            nuSigma_f = self.nuSigma_f_arr[0]
            self.k_ref = calcular_k_eff_analitico(D, Sigma_a, nuSigma_f, self.L,
                                                  self.cond_esquerda, self.cond_direita)
            self.fonte_ref = 'Analítica (fórmula)'
            self.phi_ref = fluxo_analitico_homogeneo(self.x, self.L, 
                                                     self.cond_esquerda, self.cond_direita)
            if np.max(self.phi_ref) > 0:
                self.phi_ref = self.phi_ref / np.max(self.phi_ref)
        else:
            self.k_ref = self.REFERENCIAS['heterogeneo']['k_eff']
            self.fonte_ref = self.REFERENCIAS['heterogeneo']['fonte']
            self.phi_ref = None
    
    def normalizar_potencia(self, phi):
        P_atual = 3.2e-11 * np.sum(self.nuSigma_f_arr * phi * self.h)
        if P_atual > 0:
            return phi * (self.potencia_nominal / P_atual)
        return phi
    
    def resolver(self, metodo_fonte=None):
        if metodo_fonte is not None:
            self.metodo_fonte = metodo_fonte
        self.criar_operador_A()
        self.calcular_referencias()
        n = self.N + 1
        nuSigma_f = self.nuSigma_f_arr
        phi = np.ones(n)
        k_eff = 1.0
        self.historico_k = []
        self.historico_phi = []
        self.tempos_iteracao = []
        self.iteracoes_fonte_fixa = []
        self.residuos_fonte_fixa = []
        self.historico_residuo_fonte_ultima_chamada = []
        self.historicos_residuo_fonte = []
        self.convergiu = False
        self.metodo_executado = (
            f"{self.nome_metodo(self.metodo_fonte)} aplicado ao problema de fonte fixa"
        )
        
        print("\n" + "="*60)
        print("SOLVER DE DIFUSÃO DE NÊUTRONS COM AI4PDEs 1D ADAPTADO")
        print("="*60)
        print(f"Malha: {self.N} intervalos, h = {self.h:.4f} cm")
        print(f"Tipo: {'Homogêneo' if self.is_homogeneo else 'Heterogêneo'}")
        print(f"Método: {self.metodo_executado}")
        print(f"Dispositivo: {self.nome_dispositivo}")
        if self.k_ref:
            print(f"k_ref: {self.k_ref:.8f} ({self.fonte_ref})")
        print("="*60)
        
        tempo_inicio_solver = time.perf_counter()
        
        for iteracao in range(1, self.max_iter + 1):
            inicio_iter = time.perf_counter()
            S = (1.0 / k_eff) * nuSigma_f * phi
            S_tensor = torch.tensor(S, device=self.device, dtype=torch.float32)
            phi_novo_tensor = self.resolver_fonte_fixa(S_tensor, chute=phi, metodo=self.metodo_fonte)
            phi_novo = phi_novo_tensor.cpu().numpy()
            
            integral_novo = self.calcular_integral(nuSigma_f * phi_novo)
            integral_antigo = self.calcular_integral(nuSigma_f * phi)
            if abs(integral_antigo) < 1e-30:
                raise ValueError("Integral da fonte muito pequena")
            k_eff_novo = k_eff * (integral_novo / integral_antigo)
            
            max_phi = np.max(phi_novo)
            if max_phi > 0:
                phi_novo = phi_novo / max_phi
            
            erro_k = abs(k_eff_novo - k_eff) / abs(k_eff_novo) if abs(k_eff_novo) > 0 else 1.0
            erro_phi = np.max(np.abs(phi_novo - phi)) / (np.max(np.abs(phi_novo)) + 1e-15)
            self.erro_phi_iterativo = float(erro_phi)
            
            self.historico_k.append(k_eff_novo)
            if iteracao % 10 == 0 or iteracao == 1:
                self.historico_phi.append(phi_novo.copy())
            self.tempos_iteracao.append(time.perf_counter() - inicio_iter)
            
            phi = phi_novo
            k_eff = k_eff_novo
            
            if self.progress_callback:
                self.progress_callback(iteracao, self.max_iter, k_eff, erro_k, erro_phi)
            
            if iteracao % 10 == 0 or iteracao == 1:
                print(f"Iteração {iteracao:4d}: k_eff = {k_eff:.8f}, "
                      f"erro_k = {erro_k:.2e}, erro_phi = {erro_phi:.2e},"
                      f"Iterações phi: {self.iteracoes_fonte_fixa[-1]}")

            if erro_k < self.tol_k and erro_phi < self.tol_phi:
                self.convergiu = True
                self.iteracoes_totais = iteracao
                print(f"\nCONVERGÊNCIA ALCANÇADA em {iteracao} iterações!")
                print(f"k_eff = {k_eff:.8f}")
                break
        
        if not self.convergiu:
            self.iteracoes_totais = self.max_iter
            print(f"\nNão convergiu após {self.max_iter} iterações")
        
        self.tempo_total = time.perf_counter() - tempo_inicio_solver
        self.tempo_medio_iteracao = (
            self.tempo_total / max(self.iteracoes_totais, 1)
            if self.iteracoes_totais else 0.0
        )
        self.k_eff = k_eff
        self.phi = phi
        
        if self.k_ref:
            self.erro_k = abs(self.k_eff - self.k_ref) / abs(self.k_ref)
        
        if self.phi_ref is not None:
            for ponto in self.pontos_interesse:
                idx = np.argmin(np.abs(self.x - ponto))
                if idx < len(self.phi) and idx < len(self.phi_ref):
                    val_calc = self.phi[idx]
                    val_ref = self.phi_ref[idx]
                    if abs(val_ref) > 1e-15:
                        self.erro_phi_pontos[ponto] = abs(val_calc - val_ref) / abs(val_ref)
                    else:
                        self.erro_phi_pontos[ponto] = abs(val_calc - val_ref)
        
        self.erro_phi_max = (
            float(np.max(list(self.erro_phi_pontos.values())))
            if self.erro_phi_pontos else 0.0
        )
        self.phi_normalizado = self.normalizar_potencia(phi)
        return k_eff, self.phi_normalizado

    def resumo_resultado(self, caso=None):
        return {
            "Caso": caso or ("Homogêneo" if self.is_homogeneo else "Heterogêneo"),
            "Método": self.nome_metodo(self.metodo_fonte),
            "N": self.N,
            "k_eff": self.k_eff,
            "Referência": self.k_ref,
            "Erro k (%)": self.erro_k * 100.0 if self.erro_k is not None else None,
            "Iter. externas": self.iteracoes_totais,
            "Iter. fonte média": float(np.mean(self.iteracoes_fonte_fixa)) if self.iteracoes_fonte_fixa else 0.0,
            "Resíduo final": float(self.residuos_fonte_fixa[-1]) if self.residuos_fonte_fixa else None,
            "Tempo (s)": self.tempo_total,
        }


def executar_comparacao_resolvedores(config_base, metodos=("unet_multigrid", "thomas")):
    resultados = []
    solvers = []
    for metodo in metodos:
        cfg = dict(config_base)
        cfg["metodo_fonte"] = metodo
        solver = SolverDifusaoAI4PDEs(**cfg)
        solver.resolver()
        resultados.append(solver.resumo_resultado())
        solvers.append(solver)
    return resultados, solvers


def executar_sensibilidade(config_base, tol_fonte_values=None, omega_values=None, amortecimento_values=None):
    tol_fonte_values = tol_fonte_values or [1.0e-4, 1.0e-5, 1.0e-6]
    omega_values = omega_values or [config_base.get("omega_fonte", 0.75)]
    amortecimento_values = amortecimento_values or [config_base.get("amortecimento_unet", 0.20)]

    resultados = []
    for tol_fonte in tol_fonte_values:
        for omega in omega_values:
            for amortecimento in amortecimento_values:
                cfg = dict(config_base)
                cfg.update({
                    "metodo_fonte": "unet_multigrid",
                    "tol_fonte": tol_fonte,
                    "omega_fonte": omega,
                    "amortecimento_unet": amortecimento,
                })
                solver = SolverDifusaoAI4PDEs(**cfg)
                solver.resolver()
                row = solver.resumo_resultado()
                row.update({
                    "tol_fonte": tol_fonte,
                    "omega": omega,
                    "amortecimento": amortecimento,
                })
                resultados.append(row)
    return resultados


# ============================================================================
# 4. INTERFACE GRÁFICA
# ============================================================================


class DifusaoGUI_AI4PDEs:
    def __init__(self, root):
        self.root = root
        self.root.title("Solver de Difusão de Nêutrons 1D - AI4PDEs 1D adaptado")
        self.root.state('zoomed')
        
        self.solver = None
        self.executando = False
        self.resultados_tabela = []
        self.resultados_comparacao = []
        self.resultados_sensibilidade = []
        self.modo_grafico = tk.StringVar(value="fluxo")
        self.tempo_inicio_execucao = None
        
        self.problemas = {
            'Homogêneo (Lamarsh)': {
                'L': 26.0, 'N': 100,
                'materiais': [{'inicio': 0.0, 'fim': 26.0, 'D': 0.9, 
                              'Sigma_a': 0.065, 'nuSigma_f': 0.0681}],
                'esquerda': 'reflexiva', 'direita': 'vácuo',
                'potencia': 100.0,
                'pontos': [0.0, 13.0, 26.0]
            },
            'Heterogêneo (Nozimar)': {
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
                'pontos': [0.0, 50.0, 100.0, 150.0]
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
        self.painel_controles = ttk.LabelFrame(self.frame_principal, text="Controles", padding=8)
        self.painel_controles.grid(row=0, column=0, sticky=tk.N+tk.S+tk.W, padx=5, pady=5)
        self.painel_controles.grid_rowconfigure(30, weight=1)
        
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
        ttk.Label(frame_modo, text="Visualização:").pack(side=tk.LEFT, padx=5)
        modos = [("Fluxo", "fluxo"), ("Convergência", "convergencia"),
                 ("Resíduo fonte", "residuo_fonte"), ("Tempo", "tempo_metodo"),
                 ("Comparação", "comparacao"), ("Materiais", "todos")]
        for texto, valor in modos:
            ttk.Radiobutton(frame_modo, text=texto, variable=self.modo_grafico,
                           value=valor, command=self.atualizar_grafico).pack(side=tk.LEFT, padx=5)
        
        self.frame_resultados = ttk.LabelFrame(self.aba_grafico, text="Resultados", padding=5)
        self.frame_resultados.grid(row=3, column=0, sticky=tk.W+tk.E, pady=5)
        self.texto_resultados = scrolledtext.ScrolledText(self.frame_resultados, height=3)
        self.texto_resultados.pack(fill=tk.X)
        
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
        self.N_tabela_entry.insert(0, "10, 50, 100, 200, 300, 500")
        
        self.btn_gerar_tabela = ttk.Button(frame_botoes_tabela, text="📊 Gerar Tabela (N)", 
                                          command=self.gerar_tabela_variando_N_thread)
        self.btn_gerar_tabela.pack(side=tk.LEFT, padx=2)
        
        ttk.Button(frame_botoes_tabela, text="Limpar Tabela", 
                  command=self.limpar_tabela).pack(side=tk.LEFT, padx=2)
        
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

        ttk.Label(self.painel_controles, text="Resolvedor:").grid(row=row, column=0, sticky=tk.W)
        self.metodo_fonte_var = tk.StringVar(value="unet_multigrid")
        self.metodo_fonte_combo = ttk.Combobox(
            self.painel_controles,
            textvariable=self.metodo_fonte_var,
            values=[
                "unet_multigrid",
                "thomas",
                "ambos",
            ],
            width=18,
            state="readonly",
        )
        self.metodo_fonte_combo.grid(row=row, column=1, columnspan=3, sticky=tk.W, padx=3)
        row += 1

        cuda_txt = "sim" if torch.cuda.is_available() else "não"
        self.dispositivo_label = ttk.Label(
            self.painel_controles,
            text=f"Dispositivo: automático | CUDA disponível: {cuda_txt}"
        )
        self.dispositivo_label.grid(row=row, column=0, columnspan=4, sticky=tk.W, pady=2)
        row += 1

        self.metodo_label = ttk.Label(
            self.painel_controles,
            text="Método: operadores convolucionais fixos sem treinamento"
        )
        self.metodo_label.grid(row=row, column=0, columnspan=4, sticky=tk.W, pady=2)
        row += 1

        ttk.Separator(self.painel_controles, orient='horizontal').grid(row=row, column=0, columnspan=4, sticky=tk.W+tk.E, pady=5)
        row += 1
        
        ttk.Label(self.painel_controles, text="Ações:", font=('Arial', 10, 'bold')).grid(row=row, column=0, columnspan=4, sticky=tk.W, pady=3)
        row += 1
        
        frame_acoes = ttk.Frame(self.painel_controles)
        frame_acoes.grid(row=row, column=0, columnspan=4, pady=3)
        self.btn_executar = ttk.Button(frame_acoes, text="▶ Executar", command=self.iniciar_execucao, width=12)
        self.btn_executar.pack(side=tk.LEFT, padx=2)
        self.btn_relatorio = ttk.Button(frame_acoes, text="📄 Relatório", command=self.gerar_relatorio, width=12)
        self.btn_relatorio.pack(side=tk.LEFT, padx=2)
        self.btn_relatorio.config(state=tk.DISABLED)
        self.btn_sensibilidade = ttk.Button(
            frame_acoes, text="Sensibilidade", command=self.gerar_sensibilidade_thread, width=14
        )
        self.btn_sensibilidade.pack(side=tk.LEFT, padx=2)
        row += 1
    
    def criar_barra_progresso(self):
        self.frame_progresso = ttk.LabelFrame(self.painel_controles, text="Progresso", padding=5)
        self.frame_progresso.grid(row=30, column=0, columnspan=4, sticky=tk.W+tk.E, pady=5)
        
        self.progresso_var = tk.DoubleVar()
        self.progresso_bar = ttk.Progressbar(self.frame_progresso, variable=self.progresso_var, 
                                            maximum=100, length=200)
        self.progresso_bar.pack(fill=tk.X, pady=2)
        self.progresso_label = ttk.Label(self.frame_progresso, text="Aguardando...")
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
| 2. DISCRETIZACAO NO CASO HOMOGENEO                             |
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
| 4. CASO HETEROGENEO                                            |
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
| - Conv1d para o caso homogeneo;                                |
| - stencil conservativo para o caso heterogeneo;                |
| - suavizacao de Jacobi;                                        |
| - restricao e prolongamento em estrutura tipo U-Net/multigrid; |
| - sem treinamento de pesos.                                    |
+----------------------------------------------------------------+

+----------------------------------------------------------------+
| 6. ERROS                                                       |
+----------------------------------------------------------------+
| erro_k   = |k_calc - k_ref| / |k_ref|                          |
| erro_phi = |phi_calc - phi_ref| / |phi_ref|                    |
|                                                                |
| Homogeneo: referencia analitica.                               |
| Heterogeneo: referencia principal de k_eff e Nozimar.          |
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
        return ('N', 'x', 'k_eff', 'Iterações', 'Erro_k (%)', 'Fluxo', 'Erro fluxo', 'Tempo (s)')

    def configurar_colunas_tabela(self):
        colunas = self.montar_colunas_tabela()
        self.tree.configure(columns=colunas)
        for col in colunas:
            self.tree.heading(col, text=col)
            largura = 105
            if col in ("Fluxo", "Erro fluxo", "Tempo (s)"):
                largura = 120
            elif col == "x":
                largura = 80
            self.tree.column(col, width=largura, anchor='center', stretch=False)

    def calcular_fluxos_pontos_tabela(self, solver, pontos):
        fluxos = {}
        phi_tabela = solver.phi_normalizado if solver.phi_normalizado is not None else solver.phi
        for ponto in pontos:
            idx = int(np.argmin(np.abs(solver.x - ponto)))
            fluxo = float(phi_tabela[idx]) if idx < len(phi_tabela) else 0.0
            fluxos[ponto] = fluxo
        return fluxos

    def valores_linhas_tabela(self, resultado, pontos):
        linhas = []
        fluxos = resultado.get('fluxos_pontos', {})
        erros_fluxo = resultado.get('erro_fluxo_pontos', {})
        for ponto in pontos:
            erro_fluxo = erros_fluxo.get(ponto)
            linhas.append((
                str(resultado['N']),
                f"{ponto:g}",
                f"{resultado['k_eff']:.8f}",
                str(resultado['iteracoes']),
                f"{(resultado['erro_k'] * 100 if resultado['erro_k'] is not None else 0.0):.4e}",
                f"{fluxos.get(ponto, 0.0):.6e}",
                f"{erro_fluxo:.4e}" if erro_fluxo is not None else "N/A",
                f"{resultado.get('tempo_total', 0.0):.4f}",
            ))
        return linhas

    def calcular_erros_fluxo_nozimar(self, resultados, pontos):
        """Erro percentual por ponto, como na Tabela III.3.2 de Nozimar.

        A tese compara o fluxo de cada malha com uma solução de referência.
        Nesta implementação, a maior malha calculada na tabela é usada como
        referência para os fluxos nos pontos selecionados.
        """
        if not resultados:
            return
        referencia = max(resultados, key=lambda item: item['N'])
        fluxos_ref = referencia.get('fluxos_pontos', {})
        for resultado in resultados:
            erros = {}
            for ponto in pontos:
                fluxo = resultado.get('fluxos_pontos', {}).get(ponto)
                fluxo_ref = fluxos_ref.get(ponto)
                if fluxo is None or fluxo_ref is None:
                    erros[ponto] = None
                elif abs(fluxo_ref) > 1e-30:
                    erros[ponto] = 100.0 * abs(fluxo - fluxo_ref) / abs(fluxo_ref)
                else:
                    erros[ponto] = abs(fluxo - fluxo_ref)
            resultado['erro_fluxo_pontos'] = erros
        self.referencia_fluxo_tabela = referencia['N']

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
    
    def atualizar_progresso(self, iteracao, max_iter, k_eff, erro_k, erro_phi):
        progresso = (iteracao / max_iter) * 100
        self.progresso_var.set(progresso)
        if self.tempo_inicio_execucao is None:
            decorrido = 0.0
        else:
            decorrido = time.perf_counter() - self.tempo_inicio_execucao
        self.frame_progresso.config(
            text=f"Progresso {progresso:.1f}% | tempo = {self.formatar_tempo(decorrido)}"
        )
        self.progresso_label.config(
            text=(
                f"It {iteracao}/{max_iter} | k={k_eff:.8f} | "
                f"ek={erro_k:.2e} | ephi={erro_phi:.2e}"
            )
        )
        self.root.update_idletasks()
    
    def obter_configuracao_solver(self):
        L = float(self.L_entry.get())
        N = int(self.N_entry.get())
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
            return
        with open(caminho, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(linhas[0].keys()))
            writer.writeheader()
            writer.writerows(linhas)

    def iniciar_execucao(self):
        if self.executando:
            return
        
        try:
            config = self.obter_configuracao_solver()
            metodo_selecionado = self.metodo_fonte_var.get()
            
            self.resultados_tabela = []
            self.resultados_comparacao = []
            self.executando = True
            self.btn_executar.config(state=tk.DISABLED)
            self.btn_relatorio.config(state=tk.DISABLED)
            self.btn_gerar_tabela.config(state=tk.DISABLED)
            self.btn_sensibilidade.config(state=tk.DISABLED)
            self.progresso_var.set(0)
            self.progresso_label.config(text="Iniciando...")
            self.frame_progresso.config(text="Progresso 0.0% | tempo = 00:00:00")
            self.tempo_inicio_execucao = time.perf_counter()
            
            metodo_principal = "unet_multigrid" if metodo_selecionado == "ambos" else metodo_selecionado
            config_principal = dict(config)
            config_principal["metodo_fonte"] = metodo_principal
            config_principal["progress_callback"] = self.atualizar_progresso
            self.solver = SolverDifusaoAI4PDEs(**config_principal)
            self.dispositivo_label.config(
                text=f"Dispositivo em uso: {self.solver.nome_dispositivo}"
            )
            self.metodo_label.config(text=f"Método: {self.solver.metodo_executado}")
            
            self.status_bar.config(text="Executando solver...")
            
            def executar():
                try:
                    k_eff, phi = self.solver.resolver(metodo_fonte=metodo_principal)
                    if metodo_selecionado == "ambos":
                        self.resultados_comparacao = [self.solver.resumo_resultado(self.problema_var.get())]
                        cfg_thomas = dict(config)
                        cfg_thomas["metodo_fonte"] = "thomas"
                        solver_thomas = SolverDifusaoAI4PDEs(**cfg_thomas)
                        solver_thomas.resolver()
                        self.resultados_comparacao.append(solver_thomas.resumo_resultado(self.problema_var.get()))
                        self.salvar_csv("resultados_comparacao.csv", self.resultados_comparacao)
                    self.root.after(0, self.finalizar_execucao, k_eff, phi, None)
                except Exception as e:
                    self.root.after(0, self.finalizar_execucao, None, None, str(e))
            
            thread = threading.Thread(target=executar)
            thread.daemon = True
            thread.start()
            
        except Exception as e:
            self.executando = False
            self.btn_executar.config(state=tk.NORMAL)
            self.btn_gerar_tabela.config(state=tk.NORMAL)
            self.btn_sensibilidade.config(state=tk.NORMAL)
            messagebox.showerror("Erro", f"Erro na configuração:\n{str(e)}")
            self.status_bar.config(text="Erro na configuração")
    
    def finalizar_execucao(self, k_eff, phi, erro):
        self.executando = False
        self.btn_executar.config(state=tk.NORMAL)
        self.btn_gerar_tabela.config(state=tk.NORMAL)
        self.btn_sensibilidade.config(state=tk.NORMAL)
        
        if erro:
            messagebox.showerror("Erro", f"Erro na execução:\n{erro}")
            self.status_bar.config(text="Erro na execução")
            return
        
        self.btn_relatorio.config(state=tk.NORMAL)
        self.atualizar_grafico()
        self.atualizar_resultados()
        self.status_bar.config(text=f"Concluído! k_eff = {k_eff:.8f}")
        tempo_txt = self.formatar_tempo(self.solver.tempo_total if self.solver else 0.0)
        self.frame_progresso.config(text=f"Progresso 100.0% | tempo = {tempo_txt}")
        self.progresso_label.config(
            text=(
                f"Concluído | k={k_eff:.8f} | "
                f"eref={self.solver.erro_k:.2e} | "
                f"ephi={self.solver.erro_phi_max:.2e}"
            )
        )

    def gerar_sensibilidade_thread(self):
        if self.executando:
            messagebox.showwarning("Aviso", "Aguarde a execução atual terminar.")
            return
        try:
            config = self.obter_configuracao_solver()
        except Exception as e:
            messagebox.showerror("Erro", f"Erro na configuração da sensibilidade:\n{str(e)}")
            return

        self.executando = True
        self.btn_executar.config(state=tk.DISABLED)
        self.btn_gerar_tabela.config(state=tk.DISABLED)
        self.btn_sensibilidade.config(state=tk.DISABLED)
        self.tabela_status.config(text="Executando análise de sensibilidade...")
        self.status_bar.config(text="Executando análise de sensibilidade...")

        def executar():
            try:
                resultados = executar_sensibilidade(config)
                for row in resultados:
                    row["Caso"] = self.problema_var.get()
                self.resultados_sensibilidade = resultados
                self.salvar_csv("resultados_sensibilidade.csv", resultados)
                self.root.after(
                    0,
                    lambda: self.tabela_status.config(
                        text=f"Sensibilidade concluída: {len(resultados)} execuções."
                    )
                )
                self.root.after(0, lambda: self.status_bar.config(text="Sensibilidade concluída."))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Erro", f"Erro na sensibilidade:\n{str(e)}"))
                self.root.after(0, lambda: self.status_bar.config(text="Erro na sensibilidade."))
            finally:
                self.executando = False
                self.root.after(0, lambda: self.btn_executar.config(state=tk.NORMAL))
                self.root.after(0, lambda: self.btn_gerar_tabela.config(state=tk.NORMAL))
                self.root.after(0, lambda: self.btn_sensibilidade.config(state=tk.NORMAL))

        thread = threading.Thread(target=executar)
        thread.daemon = True
        thread.start()
    
    # ================================================================
    # FUNÇÃO DO BOTÃO GERAR TABELA
    # ================================================================
    def gerar_tabela_variando_N_thread(self):
        if self.executando:
            messagebox.showwarning("Aviso", "Aguarde a execução atual terminar.")
            return
        
        if not self.materiais_atual:
            messagebox.showwarning("Aviso", "Defina pelo menos um material.")
            return

        try:
            L = float(self.L_entry.get())
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
            N_list = self.obter_lista_N_tabela()
            materiais = [dict(m) for m in self.materiais_atual]
            metodo_tabela = self.metodo_fonte_var.get()
            if metodo_tabela == "ambos":
                metodo_tabela = "unet_multigrid"
            if not pontos:
                messagebox.showwarning("Aviso", "Informe pelo menos um ponto para a tabela.")
                return
            if not (0.0 < omega_fonte <= 1.0):
                messagebox.showwarning("Aviso", "omega deve estar no intervalo 0 < omega <= 1.")
                return
            if not (0.0 <= amortecimento_unet <= 1.0):
                messagebox.showwarning("Aviso", "amortecimento deve estar no intervalo 0 <= amort. <= 1.")
                return
            if tol_fonte <= 0.0 or max_iter_fonte < 1:
                messagebox.showwarning("Aviso", "tol_fonte deve ser positiva e max_fonte deve ser maior que zero.")
                return
        except Exception as e:
            messagebox.showerror("Erro", f"Erro na configuração da tabela:\n{str(e)}")
            return
        
        self.configurar_colunas_tabela()
        self.limpar_tabela()
        self.btn_gerar_tabela.config(state=tk.DISABLED)
        self.tabela_status.config(text="Gerando tabela...")
        self.status_bar.config(text="Gerando tabela variando N...")
        
        def gerar():
            try:
                self.resultados_tabela = []
                
                for N in N_list:
                    self.root.after(0, lambda n=N: self.tabela_status.config(text=f"Processando N={n}..."))
                    
                    solver = SolverDifusaoAI4PDEs(
                        L=L, N=N, materiais=materiais,
                        cond_esquerda=esquerda, cond_direita=direita,
                        tol_k=tol_k, tol_phi=tol_phi, max_iter=max_iter,
                        potencia_nominal=potencia, pontos_interesse=pontos,
                        dispositivo_preferido='auto',
                        omega_fonte=omega_fonte,
                        amortecimento_unet=amortecimento_unet,
                        tol_fonte=tol_fonte,
                        max_iter_fonte=max_iter_fonte,
                        metodo_fonte=metodo_tabela
                    )
                    k_eff, phi = solver.resolver(metodo_fonte=metodo_tabela)
                    fluxos_pontos = self.calcular_fluxos_pontos_tabela(solver, pontos)
                    
                    resultado = {
                        'N': N,
                        'k_eff': k_eff,
                        'iteracoes': solver.iteracoes_totais,
                        'erro_k': solver.erro_k,
                        'fluxos_pontos': fluxos_pontos,
                        'erro_fluxo_iterativo': solver.erro_phi_iterativo,
                        'tempo_total': solver.tempo_total
                    }
                    self.resultados_tabela.append(resultado)
                
                self.calcular_erros_fluxo_nozimar(self.resultados_tabela, pontos)
                self.salvar_csv("resultados_refinamento.csv", self.resultados_tabela)
                self.root.after(0, lambda: self.preencher_tabela_resultados(list(pontos)))
                self.root.after(
                    0,
                    lambda: self.tabela_status.config(
                        text=f"Concluído! {len(self.resultados_tabela)} resultados. Ref. fluxo: N={self.referencia_fluxo_tabela}"
                    )
                )
                self.root.after(0, lambda: self.status_bar.config(text="Tabela concluída."))
                
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Erro", f"Erro ao gerar tabela:\n{str(e)}"))
                self.root.after(0, lambda: self.tabela_status.config(text="Erro na geração."))
            finally:
                self.root.after(0, lambda: self.btn_gerar_tabela.config(state=tk.NORMAL))
        
        thread = threading.Thread(target=gerar)
        thread.daemon = True
        thread.start()
    
    def preencher_tabela_resultados(self, pontos):
        for item in self.tree.get_children():
            self.tree.delete(item)
        for resultado in self.resultados_tabela:
            for linha in self.valores_linhas_tabela(resultado, pontos):
                self.tree.insert('', tk.END, values=linha)

    def atualizar_grafico(self):
        if self.solver is None or self.solver.phi is None:
            return
        
        self.ax.clear()
        modo = self.modo_grafico.get()
        x = self.solver.x
        phi = self.solver.phi_normalizado if self.solver.phi_normalizado is not None else self.solver.phi
        
        if modo == "fluxo":
            self.ax.plot(x, phi, 'b-', linewidth=2, label='AI4PDEs 1D adaptado')
            self.ax.set_xlabel('x [cm]')
            self.ax.set_ylabel('Fluxo φ(x)')
            self.ax.set_title('Distribuição do Fluxo')
            self.ax.grid(True, alpha=0.3)
            self.ax.legend()
        
        elif modo == "convergencia":
            self.ax.plot(self.solver.historico_k, 'b-', linewidth=2)
            self.ax.set_xlabel('Iteração')
            self.ax.set_ylabel('k_eff')
            self.ax.set_title('Convergência do k_eff')
            self.ax.grid(True, alpha=0.3)
            if self.solver.historico_k:
                self.ax.axhline(y=self.solver.k_eff, color='r', linestyle='--',
                               label=f'Final = {self.solver.k_eff:.6f}')
                if self.solver.k_ref:
                    self.ax.axhline(y=self.solver.k_ref, color='g', linestyle=':',
                                   label=f'Ref = {self.solver.k_ref:.6f}')
                self.ax.legend()

        elif modo == "residuo_fonte":
            hist = self.solver.historico_residuo_fonte_ultima_chamada
            if hist:
                self.ax.semilogy(range(1, len(hist) + 1), hist, 'm-', linewidth=2)
                self.ax.set_title('Convergência do resolvedor de fonte fixa')
                self.ax.set_xlabel('Iteração interna')
                self.ax.set_ylabel('Resíduo relativo')
                self.ax.grid(True, which='both', alpha=0.3)
            else:
                self.ax.text(0.5, 0.5, 'Sem histórico de resíduo disponível',
                             transform=self.ax.transAxes, ha='center', va='center')

        elif modo == "tempo_metodo":
            dados = self.resultados_comparacao or ([self.solver.resumo_resultado(self.problema_var.get())] if self.solver else [])
            if dados:
                labels = [d["Método"] for d in dados]
                tempos = [d["Tempo (s)"] for d in dados]
                self.ax.bar(labels, tempos, color=['#2c7fb8', '#f03b20', '#31a354'][:len(labels)])
                self.ax.set_ylabel('Tempo total [s]')
                self.ax.set_title('Comparação de tempo por método')
                self.ax.tick_params(axis='x', rotation=15)
                self.ax.grid(True, axis='y', alpha=0.3)
            else:
                self.ax.text(0.5, 0.5, 'Execute a comparação entre métodos',
                             transform=self.ax.transAxes, ha='center', va='center')
        
        elif modo == "comparacao":
            self.ax.plot(x, phi, 'b-', linewidth=2, label='AI4PDEs 1D adaptado')
            if self.solver.is_homogeneo and self.solver.phi_ref is not None:
                phi_ref = self.solver.phi_ref
                if np.max(phi_ref) > 0 and np.max(phi) > 0:
                    phi_ref = phi_ref / np.max(phi_ref) * np.max(phi)
                self.ax.plot(x, phi_ref, 'r--', label='Analítica', alpha=0.7, linewidth=2)
                erro_rel = np.max(np.abs(phi - phi_ref)) / (np.max(np.abs(phi)) + 1e-15)
                self.ax.set_title(f'Comparação: Erro = {erro_rel*100:.4f}%')
            else:
                self.ax.text(0.05, 0.95, 'Problema Heterogêneo\n(Ref: Nozimar 1.09506)',
                           transform=self.ax.transAxes, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                self.ax.set_title('Solução Numérica (Heterogêneo)')
            self.ax.set_xlabel('x [cm]')
            self.ax.set_ylabel('Fluxo φ(x)')
            self.ax.grid(True, alpha=0.3)
            self.ax.legend()
        
        elif modo == "todos":
            self.ax.plot(x, self.solver.D_arr, 'r-', label='D', linewidth=2)
            self.ax.plot(x, self.solver.Sigma_a_arr, 'g-', label='Σa', linewidth=2)
            self.ax.plot(x, self.solver.nuSigma_f_arr, 'b-', label='νΣf', linewidth=2)
            for interface in self.obter_interfaces_materiais():
                self.ax.axvline(interface, color='k', linestyle='--', linewidth=1.2, alpha=0.7)
                self.ax.text(interface, self.ax.get_ylim()[1] * 0.95, f"x={interface:g}",
                             rotation=90, va='top', ha='right', fontsize=9)
            self.ax.set_xlabel('x [cm]')
            self.ax.set_ylabel('Parâmetros materiais')
            self.ax.set_title('Parâmetros Materiais e Interfaces')
            self.ax.grid(True, alpha=0.3)
            self.ax.legend()
        
        self.fig.tight_layout(pad=3.0)
        self.canvas.draw()
    
    def atualizar_resultados(self):
        if self.solver is None:
            return
        
        texto = f"k_eff = {self.solver.k_eff:.8f}\n"
        if self.solver.k_ref:
            texto += f"Referência: {self.solver.k_ref:.8f} ({self.solver.fonte_ref})\n"
            texto += f"Erro relativo: {self.solver.erro_k:.2e} ({self.solver.erro_k*100:.4f}%)\n"
        texto += f"Iterações: {self.solver.iteracoes_totais}\n"
        texto += f"Convergência: {'✅ Sim' if self.solver.convergiu else '❌ Não'}\n"
        texto += f"Tempo total: {self.solver.tempo_total:.4f}s\n"
        texto += f"Tempo médio/iteração: {self.solver.tempo_medio_iteracao:.4f}s\n"
        texto += f"Dispositivo usado: {self.solver.nome_dispositivo}\n"
        texto += f"Tipo: {'Homogêneo' if self.solver.is_homogeneo else 'Heterogêneo'}\n"
        if self.solver.iteracoes_fonte_fixa:
            texto += f"Iter. médias fonte fixa: {np.mean(self.solver.iteracoes_fonte_fixa):.1f}\n"
            texto += f"Último resíduo fonte fixa: {self.solver.residuos_fonte_fixa[-1]:.3e}\n"
            texto += f"10 iterações {self.solver.iteracoes_fonte_fixa[0:10]} (últimas 10)\n"
        texto += f"Método: {self.solver.metodo_executado}\n"
        
        self.texto_resultados.delete(1.0, tk.END)
        self.texto_resultados.insert(tk.END, texto)
    
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
            styles = getSampleStyleSheet()
            story = []
            imagens_temp = []
            is_homogeneo = self.solver.is_homogeneo

            titulo_style = ParagraphStyle(
                'CustomTitle', parent=styles['Heading1'],
                fontSize=22, spaceAfter=20, alignment=1
            )

            def estilo_tabela(tabela):
                tabela.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2C3E50')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 9),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#ECF0F1')),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ]))
                return tabela

            def salvar_figura(tipo):
                x = self.solver.x
                phi = self.solver.phi_normalizado if self.solver.phi_normalizado is not None else self.solver.phi
                fig = Figure(figsize=(9, 5), dpi=150)
                ax = fig.add_subplot(111)

                if tipo == "fluxo":
                    ax.plot(x, phi, 'b-', linewidth=2, label='AI4PDEs 1D adaptado')
                    ax.set_title('Distribuição do Fluxo')
                    ax.set_xlabel('x [cm]')
                    ax.set_ylabel('Fluxo φ(x)')
                    ax.legend()

                elif tipo == "convergencia":
                    ax.plot(self.solver.historico_k, 'b-', linewidth=2, label='k_eff por iteração')
                    ax.axhline(y=self.solver.k_eff, color='r', linestyle='--',
                               label=f'Final = {self.solver.k_eff:.6f}')
                    if self.solver.k_ref:
                        ax.axhline(y=self.solver.k_ref, color='g', linestyle=':',
                                   label=f'Ref = {self.solver.k_ref:.6f}')
                    ax.set_title('Convergência de k_eff')
                    ax.set_xlabel('Iteração')
                    ax.set_ylabel('k_eff')
                    ax.legend()

                elif tipo == "residuo_fonte":
                    hist = self.solver.historico_residuo_fonte_ultima_chamada
                    if hist:
                        ax.semilogy(range(1, len(hist) + 1), hist, 'm-', linewidth=2)
                    ax.set_title('Convergência do resolvedor de fonte fixa')
                    ax.set_xlabel('Iteração interna')
                    ax.set_ylabel('Resíduo relativo')

                elif tipo == "tempo_metodo":
                    dados_tempo = self.resultados_comparacao or [self.solver.resumo_resultado(self.problema_var.get())]
                    labels = [d["Método"] for d in dados_tempo]
                    tempos = [d["Tempo (s)"] for d in dados_tempo]
                    ax.bar(labels, tempos, color=['#2c7fb8', '#f03b20', '#31a354'][:len(labels)])
                    ax.set_title('Comparação de tempo por método')
                    ax.set_ylabel('Tempo total [s]')
                    ax.tick_params(axis='x', rotation=15)

                elif tipo == "materiais":
                    ax.plot(x, self.solver.D_arr, 'r-', label='D', linewidth=2)
                    ax.plot(x, self.solver.Sigma_a_arr, 'g-', label='Σa', linewidth=2)
                    ax.plot(x, self.solver.nuSigma_f_arr, 'b-', label='νΣf', linewidth=2)
                    for interface in self.obter_interfaces_materiais():
                        ax.axvline(interface, color='k', linestyle='--', linewidth=1.2, alpha=0.7)
                        ax.text(interface, ax.get_ylim()[1] * 0.95, f"x={interface:g}",
                                rotation=90, va='top', ha='right', fontsize=8)
                    ax.set_title('Parâmetros Materiais e Interfaces')
                    ax.set_xlabel('x [cm]')
                    ax.set_ylabel('Parâmetros materiais')
                    ax.legend()

                elif tipo == "comparacao":
                    ax.plot(x, phi, 'b-', linewidth=2, label='AI4PDEs 1D adaptado')
                    if self.solver.is_homogeneo and self.solver.phi_ref is not None:
                        phi_ref = self.solver.phi_ref
                        if np.max(phi_ref) > 0 and np.max(phi) > 0:
                            phi_ref = phi_ref / np.max(phi_ref) * np.max(phi)
                        ax.plot(x, phi_ref, 'r--', linewidth=2, label='Analítica')
                        ax.set_title('Comparação com a Solução Analítica')
                    else:
                        ax.set_title('Resultado Heterogêneo para Comparação com Nozimar')
                        ax.text(0.05, 0.95, f"k_ref Nozimar = {self.solver.k_ref:.5f}",
                                transform=ax.transAxes, va='top',
                                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                    ax.set_xlabel('x [cm]')
                    ax.set_ylabel('Fluxo φ(x)')
                    ax.legend()

                elif tipo == "tabela_N" and self.resultados_tabela:
                    Ns = [r['N'] for r in self.resultados_tabela]
                    ks = [r['k_eff'] for r in self.resultados_tabela]
                    erros = [r['erro_k'] * 100.0 for r in self.resultados_tabela]
                    ax.plot(Ns, ks, 'bo-', linewidth=2, label='k_eff')
                    ax.set_xlabel('N')
                    ax.set_ylabel('k_eff')
                    ax.set_title('Refinamento de Malha')
                    ax2 = ax.twinx()
                    ax2.plot(Ns, erros, 'rs--', linewidth=1.5, label='Erro_k (%)')
                    ax2.set_ylabel('Erro_k (%)')
                    linhas, labels = ax.get_legend_handles_labels()
                    linhas2, labels2 = ax2.get_legend_handles_labels()
                    ax.legend(linhas + linhas2, labels + labels2, loc='best')

                elif tipo == "escalabilidade" and self.resultados_tabela:
                    Ns = [r['N'] for r in self.resultados_tabela]
                    tempos = [r.get('tempo_total', 0.0) for r in self.resultados_tabela]
                    ax.loglog(Ns, tempos, 'ko-', linewidth=2)
                    ax.set_xlabel('N')
                    ax.set_ylabel('Tempo total [s]')
                    ax.set_title('Escalabilidade: N vs tempo')

                ax.grid(True, alpha=0.3)
                fig.tight_layout()
                temp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                temp.close()
                fig.savefig(temp.name, dpi=150, bbox_inches='tight')
                imagens_temp.append(temp.name)
                return temp.name

            story.append(Paragraph("Relatório do Solver de Difusão de Nêutrons 1D", titulo_style))
            story.append(Paragraph("Neural Physics/AI4PDEs com stencils convolucionais fixos", styles['Heading2']))
            story.append(Paragraph(f"Gerado em: {datetime.now().strftime('%d/%m/%Y às %H:%M')}", styles['Normal']))
            story.append(Spacer(1, 20))

            story.append(Paragraph("Sumário", styles['Heading2']))
            for item in [
                "1. Metodologia computacional",
                "2. Configuração do problema",
                "3. Configuração computacional",
                "4. Resultado do caso principal",
                "5. Parâmetros materiais",
                "6. Comparação analítica ou referência Nozimar",
                "7. Convergência de k_eff",
                "8. Tabela variando N",
                "9. Conclusão parcial",
            ]:
                story.append(Paragraph(item, styles['Normal']))
            story.append(PageBreak())

            story.append(Paragraph("1. Metodologia Computacional", styles['Heading2']))
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
                "O problema de fonte fixa Aφ = S é tratado por aplicação do operador A por stencil/Conv1d, "
                "suavização iterativa de Jacobi, restrição por AvgPool1d e prolongamento por interpolação "
                "linear em uma estrutura do tipo U-Net/multigrid.",
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
                ["Número de Pontos (N)", str(self.solver.N)],
                ["Tipo", "Homogêneo" if is_homogeneo else "Heterogêneo"],
                ["Condição Esquerda", self.solver.cond_esquerda.capitalize()],
                ["Condição Direita", self.solver.cond_direita.capitalize()],
                ["Potência Nominal", f"{self.solver.potencia_nominal} MWT"],
                ["Tolerância k", f"{self.solver.tol_k:.1e}"],
                ["Tolerância φ", f"{self.solver.tol_phi:.1e}"],
                ["Tolerância fonte fixa", f"{self.solver.tol_fonte:.1e}"],
                ["Máx. iterações fonte fixa", f"{self.solver.max_iter_fonte}"],
                ["Omega fonte fixa", f"{self.solver.omega_fonte:.3f}"],
                ["Amortecimento multiescala", f"{self.solver.amortecimento_unet:.3f}"],
                ["Método", "Neural Physics: stencil + Conv1d + U-Net/multigrid"],
                ["Pesos treináveis", "Não"],
                ["Relação com AI4PDEs", "Adaptação 1D local da filosofia do pacote"]
            ]
            story.append(estilo_tabela(Table(dados, colWidths=[4*cm, 10*cm])))
            story.append(Spacer(1, 20))

            story.append(Paragraph("3. Configuração Computacional", styles['Heading2']))
            dados_comp = [
                ["Item", "Valor"],
                ["CPU/GPU usada", f"{self.solver.device.type.upper()} - {self.solver.nome_dispositivo}"],
                ["PyTorch disponível", f"Sim ({torch.__version__})"],
                ["CUDA disponível", "Sim" if self.solver.cuda_disponivel else "Não"],
                ["Método executado", self.solver.metodo_executado],
                ["Tempo total", f"{self.solver.tempo_total:.6f} s"],
                ["Tempo médio por iteração", f"{self.solver.tempo_medio_iteracao:.6f} s"],
            ]
            story.append(estilo_tabela(Table(dados_comp, colWidths=[5*cm, 9*cm])))
            story.append(PageBreak())

            story.append(Paragraph("4. Resultado do Caso Principal", styles['Heading2']))
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
            story.append(Spacer(1, 12))
            story.append(Image(salvar_figura("fluxo"), width=15*cm, height=8*cm))
            if self.resultados_comparacao:
                story.append(Spacer(1, 12))
                story.append(Paragraph("4.1 Comparação entre resolvedores de fonte fixa", styles['Heading2']))
                cab = ["Caso", "Método", "N", "k_eff", "Referência", "Erro k (%)", "Iter. externas", "Iter. fonte média", "Resíduo final", "Tempo (s)"]
                dados_cmp = [cab]
                for r in self.resultados_comparacao:
                    dados_cmp.append([
                        r["Caso"], r["Método"], str(r["N"]), f"{r['k_eff']:.8f}",
                        f"{r['Referência']:.8f}" if r["Referência"] else "N/A",
                        f"{r['Erro k (%)']:.4e}" if r["Erro k (%)"] is not None else "N/A",
                        str(r["Iter. externas"]), f"{r['Iter. fonte média']:.2f}",
                        f"{r['Resíduo final']:.3e}" if r["Resíduo final"] is not None else "N/A",
                        f"{r['Tempo (s)']:.4f}",
                    ])
                story.append(estilo_tabela(Table(dados_cmp)))
                story.append(Image(salvar_figura("tempo_metodo"), width=15*cm, height=7*cm))
            story.append(PageBreak())

            story.append(Paragraph("5. Parâmetros Materiais", styles['Heading2']))
            dados_materiais = [["Região", "Início", "Fim", "D", "Σa", "νΣf"]]
            for i, mat in enumerate(self.materiais_atual):
                dados_materiais.append([
                    str(i+1), f"{mat['inicio']:.2f}", f"{mat['fim']:.2f}",
                    f"{mat['D']:.4f}", f"{mat['Sigma_a']:.4f}", f"{mat['nuSigma_f']:.4f}"
                ])
            story.append(estilo_tabela(Table(
                dados_materiais,
                colWidths=[1.5*cm, 2*cm, 2*cm, 2*cm, 2*cm, 2*cm]
            )))
            story.append(Spacer(1, 12))
            story.append(Image(salvar_figura("materiais"), width=15*cm, height=8*cm))
            story.append(PageBreak())

            story.append(Paragraph("6. Comparação Analítica ou Referência Nozimar", styles['Heading2']))
            if is_homogeneo:
                story.append(Paragraph(
                    "Para o caso homogêneo com condição reflexiva em x=0 e vácuo em x=L, "
                    "a referência usada é φ(x) ~ cos(πx/2L) e "
                    "k_eff = νΣf/[Σa + D(π/2L)^2].",
                    styles['Normal']
                ))
            else:
                story.append(Paragraph(
                    "Para o caso heterogêneo, a comparação de referência usa o valor de Nozimar "
                    f"k_eff = {self.solver.k_ref:.5f}.",
                    styles['Normal']
                ))
            story.append(Spacer(1, 10))
            story.append(Image(salvar_figura("comparacao"), width=15*cm, height=8*cm))

            if self.solver.erro_phi_pontos:
                story.append(Spacer(1, 10))
                story.append(Paragraph("Erros do Fluxo nos Pontos de Interesse:", styles['Heading3']))
                dados_pontos = [["Ponto (cm)", "Fluxo", "Erro"]]
                for ponto, erro in self.solver.erro_phi_pontos.items():
                    idx = np.argmin(np.abs(self.solver.x - ponto))
                    fluxo_val = self.solver.phi[idx] if idx < len(self.solver.phi) else 0.0
                    dados_pontos.append([f"{ponto:.2f}", f"{fluxo_val:.6e}", f"{erro:.4e}"])
                story.append(estilo_tabela(Table(dados_pontos, colWidths=[3*cm, 3*cm, 3*cm])))

            story.append(PageBreak())
            story.append(Paragraph("7. Convergência de k_eff", styles['Heading2']))
            story.append(Image(salvar_figura("convergencia"), width=15*cm, height=8*cm))
            story.append(Paragraph("7.1 Convergência do resolvedor de fonte fixa", styles['Heading2']))
            story.append(Image(salvar_figura("residuo_fonte"), width=15*cm, height=8*cm))

            story.append(Spacer(1, 20))
            story.append(Paragraph("8. Resultados para Diferentes Malhas (N)", styles['Heading2']))
            if self.resultados_tabela:
                cabecalho = ["N", "x", "k_eff", "Iterações", "Erro_k (%)", "Fluxo", "Erro fluxo", "Tempo (s)"]
                dados_tabela = [cabecalho]
                for r in self.resultados_tabela:
                    fluxos = r.get('fluxos_pontos', {})
                    erros_fluxo = r.get('erro_fluxo_pontos', {})
                    for ponto in fluxos.keys():
                        erro_fluxo = erros_fluxo.get(ponto)
                        dados_tabela.append([
                            str(r['N']),
                            f"{ponto:g}",
                            f"{r['k_eff']:.8f}",
                            str(r['iteracoes']),
                            f"{r['erro_k']*100:.4e}" if r['erro_k'] else "—",
                            f"{fluxos.get(ponto, 0.0):.6e}",
                            f"{erro_fluxo:.4e}" if erro_fluxo is not None else "N/A",
                            f"{r.get('tempo_total', 0.0):.4f}",
                        ])
                col_widths = [0.8*cm, 0.8*cm, 2.1*cm, 1.4*cm, 1.7*cm, 2.2*cm, 1.9*cm, 1.6*cm]
                story.append(estilo_tabela(Table(
                    dados_tabela,
                    colWidths=col_widths
                )))
                story.append(Spacer(1, 5))
                k_ref_str = f"{self.solver.k_ref:.8f}" if self.solver.k_ref else "1.09506"
                story.append(Paragraph(f"Erro_k = |k_eff - k_ref| / |k_ref|  (k_ref = {k_ref_str})", styles['Normal']))
                n_ref_fluxo = getattr(self, 'referencia_fluxo_tabela', None)
                ref_fluxo_txt = f"N={n_ref_fluxo}" if n_ref_fluxo else "maior N calculado"
                story.append(Paragraph(
                    "Erro fluxo: desvio relativo percentual por posição, como na Tabela III.3.2 de Nozimar, "
                    f"100*|phi_N(x) - phi_ref(x)|/|phi_ref(x)|, com phi_ref obtido em {ref_fluxo_txt}.",
                    styles['Normal']
                ))
                story.append(Spacer(1, 10))
                story.append(Image(salvar_figura("tabela_N"), width=15*cm, height=8*cm))
                story.append(Paragraph("8.1 Escalabilidade N vs tempo", styles['Heading2']))
                story.append(Image(salvar_figura("escalabilidade"), width=15*cm, height=8*cm))
            else:
                story.append(Paragraph("Nenhum dado disponível para a tabela variando N.", styles['Normal']))

            if self.resultados_sensibilidade:
                story.append(PageBreak())
                story.append(Paragraph("9. Análise de Sensibilidade", styles['Heading2']))
                cab = ["Caso", "tol_fonte", "omega", "amort.", "k_eff", "Erro k (%)", "Iter. externas", "Iter. fonte média", "Resíduo final", "Tempo (s)"]
                dados_sens = [cab]
                for r in self.resultados_sensibilidade:
                    dados_sens.append([
                        r["Caso"], f"{r['tol_fonte']:.1e}", f"{r['omega']:.2f}", f"{r['amortecimento']:.2f}",
                        f"{r['k_eff']:.8f}",
                        f"{r['Erro k (%)']:.4e}" if r["Erro k (%)"] is not None else "N/A",
                        str(r["Iter. externas"]), f"{r['Iter. fonte média']:.2f}",
                        f"{r['Resíduo final']:.3e}" if r["Resíduo final"] is not None else "N/A",
                        f"{r['Tempo (s)']:.4f}",
                    ])
                story.append(estilo_tabela(Table(dados_sens)))

            story.append(PageBreak())
            story.append(Paragraph("10. Discussão automática curta", styles['Heading2']))
            story.append(Paragraph(
                "Para problemas 1D, o método de Thomas é o resolvedor clássico mais natural para sistemas tridiagonais. "
                "A formulação Neural Physics aqui avaliada não tem como objetivo superar Thomas em 1D, mas validar uma "
                "implementação determinística baseada em operadores convolucionais fixos e operações multiescala compatíveis "
                "com bibliotecas de IA.",
                styles['Normal']
            ))
            story.append(Spacer(1, 12))
            story.append(Paragraph("11. Conclusão Parcial", styles['Heading2']))
            status = "convergiu com sucesso" if self.solver.convergiu else "não convergiu"
            conclusao = (
                f"O solver próprio inspirado em Neural Physics/AI4PDEs {status} para o problema "
                f"{'homogêneo' if is_homogeneo else 'heterogêneo'}. "
                f"k_eff = {self.solver.k_eff:.8f}."
            )
            if self.solver.k_ref:
                conclusao += f" Erro relativo: {self.solver.erro_k*100:.4f}%."
            conclusao += (
                " O cálculo foi realizado por uma adaptação 1D inspirada na filosofia Neural Physics/AI4PDEs, "
                "com operadores convolucionais fixos e sem treinamento."
            )
            story.append(Paragraph(conclusao, styles['Normal']))

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
        linhas = [
            "# Relatório do Solver de Difusão de Nêutrons 1D",
            "",
            "## Metodologia computacional",
            "Neste trabalho, o termo Neural Physics não se refere ao treinamento de uma rede neural para aproximar a solução. "
            "Ele se refere à implementação de operadores numéricos discretos por meio de operações típicas de bibliotecas "
            "de inteligência artificial, como convoluções, pooling e interpolação. Os pesos são fixos e definidos pela "
            "discretização física do problema.",
            "",
            "A normalização por máximo é usada apenas para fixar a escala do autovetor durante a iteração de potência. "
            "Como o problema de autovalor determina o fluxo apenas até uma constante multiplicativa, essa normalização "
            "não altera o autovalor. Para interpretação física, o fluxo pode ser posteriormente normalizado por potência.",
            "",
            "## Resultado principal",
            f"- k_eff: {self.solver.k_eff:.8f}",
            f"- Referência: {self.solver.k_ref if self.solver.k_ref else 'N/A'}",
            f"- Erro k (%): {self.solver.erro_k * 100.0:.4e}",
            f"- Iterações externas: {self.solver.iteracoes_totais}",
            f"- Iterações médias fonte fixa: {np.mean(self.solver.iteracoes_fonte_fixa):.2f}" if self.solver.iteracoes_fonte_fixa else "- Iterações médias fonte fixa: N/A",
            f"- Resíduo final: {self.solver.residuos_fonte_fixa[-1]:.3e}" if self.solver.residuos_fonte_fixa else "- Resíduo final: N/A",
            f"- Tempo total (s): {self.solver.tempo_total:.6f}",
            "",
        ]
        if self.resultados_comparacao:
            linhas.extend([
                "## Comparação entre resolvedores de fonte fixa",
                "",
                "| Caso | Método | N | k_eff | Referência | Erro k (%) | Iter. externas | Iter. fonte média | Resíduo final | Tempo (s) |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ])
            for r in self.resultados_comparacao:
                linhas.append(
                    f"| {r['Caso']} | {r['Método']} | {r['N']} | {r['k_eff']:.8f} | "
                    f"{r['Referência'] if r['Referência'] else 'N/A'} | "
                    f"{r['Erro k (%)'] if r['Erro k (%)'] is not None else 'N/A'} | "
                    f"{r['Iter. externas']} | {r['Iter. fonte média']:.2f} | "
                    f"{r['Resíduo final'] if r['Resíduo final'] is not None else 'N/A'} | {r['Tempo (s)']:.4f} |"
                )
        if self.resultados_sensibilidade:
            linhas.extend([
                "",
                "## Análise de sensibilidade",
                "",
                "| Caso | tol_fonte | omega | amortecimento | k_eff | Erro k (%) | Iter. externas | Iter. fonte média | Resíduo final | Tempo (s) |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ])
            for r in self.resultados_sensibilidade:
                linhas.append(
                    f"| {r['Caso']} | {r['tol_fonte']:.1e} | {r['omega']:.2f} | {r['amortecimento']:.2f} | "
                    f"{r['k_eff']:.8f} | {r['Erro k (%)'] if r['Erro k (%)'] is not None else 'N/A'} | "
                    f"{r['Iter. externas']} | {r['Iter. fonte média']:.2f} | "
                    f"{r['Resíduo final'] if r['Resíduo final'] is not None else 'N/A'} | {r['Tempo (s)']:.4f} |"
                )
        linhas.extend([
            "",
            "## Discussão automática curta",
            "Para problemas 1D, o método de Thomas é o resolvedor clássico mais natural para sistemas tridiagonais. "
            "A formulação Neural Physics aqui avaliada não tem como objetivo superar Thomas em 1D, mas validar uma "
            "implementação determinística baseada em operadores convolucionais fixos e operações multiescala compatíveis "
            "com bibliotecas de IA.",
        ])
        with open(arquivo, "w", encoding="utf-8") as f:
            f.write("\n".join(linhas))
        messagebox.showinfo("Sucesso", f"Relatório Markdown salvo em:\n{arquivo}")



def main():
    print("=" * 60)
    print("SOLVER DE DIFUSÃO DE NÊUTRONS 1D")
    print("Neural Physics: operadores convolucionais fixos sem treinamento")
    print("=" * 60)
    print(f"CUDA disponível: {torch.cuda.is_available()}")
    print("=" * 60)

    root = tk.Tk()
    DifusaoGUI_AI4PDEs(root)
    root.mainloop()


if __name__ == "__main__":
    main()
