"""Operadores discretos e resolvedores de fonte fixa.

O termo Neural Physics usado neste projeto não indica treinamento de uma rede
neural. Ele indica a implementação de operadores numéricos fixos por operações
típicas de bibliotecas de IA, como convolução, pooling e interpolação.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class ResultadoFonteFixa:
    phi: torch.Tensor
    n_iter: int
    residuo: float
    historico_residuo: list
    convergiu: bool


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


class SolverFonteFixaMultiescala1D(nn.Module):
    """
    Resolve A phi = S com correção multiescala por pooling/interpolação.

    Não há treinamento. As operações são:
    - aplicação do operador A por stencil/convolução fixa;
    - suavização por Jacobi ponderado;
    - restrição por AvgPool1d;
    - prolongamento por interpolação linear;
    - correção residual em múltiplas escalas por pooling/interpolação.

    Este bloco não implementa um V-cycle geométrico completo; ele é um
    pré-condicionador algorítmico determinístico sem treinamento.
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
                return ResultadoFonteFixa(phi, it, ultimo_residuo, historico_residuo, True)

            correcao = self._correcao_unet(residuo, diag)
            phi = phi + self.amortecimento_unet * correcao
            phi = self.operador_A.aplicar_contorno_fluxo(phi)
            phi = self._suavizar_jacobi(phi, rhs, diag, n_passos=2)

        return ResultadoFonteFixa(phi, max_iter, ultimo_residuo, historico_residuo, False)


# Alias mantido para compatibilidade com versões anteriores.
SolverFonteFixaUNet1D = SolverFonteFixaMultiescala1D
