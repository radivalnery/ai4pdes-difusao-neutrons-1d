"""Caminho convolucional comum aos stencils MDF e MEF, sem treinamento.

As linhas de A e F sao representadas por tres coeficientes locais. Uma unica
Conv1d fixa extrai (phi[i-1], phi[i], phi[i+1]); os coeficientes fisicos pesam
esses canais. No heterogeneo eles variam no espaco: nao se pressupoe um unico
kernel fisico global. Tanto absorcao/difusao quanto fissao usam este caminho.

H e a medida nodal positiva (meia soma dos comprimentos adjacentes). Trabalhamos
com residuos por unidade de comprimento. Para MEF, A=H^-1(K+Ma), F=H^-1 Mf;
isso e apenas escalonamento de linhas, NAO diagonalizacao da massa consistente.
"""

import numpy as np
import torch
import torch.nn as nn


def medidas_nodais(x):
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or len(x) < 2 or not np.all(np.isfinite(x)) or np.any(np.diff(x) <= 0):
        raise ValueError('A malha requer pelo menos dois nos finitos e crescentes.')
    dx = np.diff(x)
    return np.r_[dx[0]/2, (dx[:-1]+dx[1:])/2, dx[-1]/2]


class OperadorStencil1D(nn.Module):
    """Aplicacao comum das tres diagonais, contornos e conversao para Thomas."""

    def __init__(self, x, cond_esquerda, cond_direita, dtype=torch.float64):
        super().__init__()
        if dtype not in (torch.float32, torch.float64):
            raise ValueError('Precisao deve ser float32 ou float64.')
        if any(c not in ('reflexiva', 'vácuo') for c in (cond_esquerda, cond_direita)):
            raise ValueError('Contorno deve ser reflexiva ou vacuo (fluxo nulo).')
        self.x_numpy = np.asarray(x, dtype=float).copy()
        self.cond_esquerda, self.cond_direita = cond_esquerda, cond_direita
        self.register_buffer('x', torch.tensor(self.x_numpy, dtype=dtype))
        self.register_buffer('medida', torch.tensor(medidas_nodais(x), dtype=dtype))
        self.convolucao = nn.Conv1d(1, 3, 3, padding=1, bias=False, dtype=dtype)
        with torch.no_grad():
            self.convolucao.weight.copy_(torch.eye(3, dtype=dtype).reshape(3, 1, 3))
        self.convolucao.requires_grad_(False)

    def definir_stencils(self, lower, diag, upper, flower, fdiag, fupper):
        # As bandas externas nao existem; padding nulo fecha o banco de filtros.
        a = np.column_stack((np.r_[0., lower], diag, np.r_[upper, 0.]))
        f = np.column_stack((np.r_[0., flower], fdiag, np.r_[fupper, 0.]))
        for i, cond in ((0, self.cond_esquerda), (-1, self.cond_direita)):
            if cond == 'vácuo':
                a[i, :] = (0., 1., 0.)
                if i == 0:
                    a[1, 0] = 0.
                else:
                    a[-2, 2] = 0.
        if not np.all(np.isfinite(a)) or np.any(a[:, 1] <= 0):
            raise ValueError('Stencil A invalido ou diagonal nao positiva.')
        if not np.all(np.isfinite(f)) or np.any(f < 0):
            raise ValueError('Stencil de fissao invalido.')
        self.register_buffer('a_coef', torch.tensor(a, dtype=self.x.dtype))
        self.register_buffer('f_coef', torch.tensor(f, dtype=self.x.dtype))
        # Integral da producao: 1^T H F phi, sem copiar dados da GPU por iteracao.
        H = self.medida.double().numpy()
        f = self.f_coef.double().numpy()
        self.pesos_producao_numpy = H*f[:, 1]
        self.pesos_producao_numpy[:-1] += H[1:]*f[1:, 0]
        self.pesos_producao_numpy[1:] += H[:-1]*f[:-1, 2]

    def _aplicar_stencil(self, phi, coef):
        phi = phi.reshape(-1).to(device=coef.device, dtype=coef.dtype)
        vizinhos = self.convolucao(phi.reshape(1, 1, -1)).squeeze(0).transpose(0, 1)
        return torch.sum(coef * vizinhos, dim=1)

    def forward(self, phi):
        return self._aplicar_stencil(phi, self.a_coef)

    def aplicar_fissao(self, phi):
        return self._aplicar_stencil(phi, self.f_coef)

    def aplicar_fissao_numpy(self, phi):
        """Auxiliar de inspecao; a resolucao principal usa aplicar_fissao/Conv1d."""
        with torch.no_grad():
            t = torch.as_tensor(phi, device=self.x.device, dtype=self.x.dtype)
            return self.aplicar_fissao(t).cpu().numpy()

    def diagonal(self):
        return self.a_coef[:, 1]

    def aplicar_contorno_fluxo(self, phi):
        phi = phi.clone()
        if self.cond_esquerda == 'vácuo':
            phi[0] = 0.
        if self.cond_direita == 'vácuo':
            phi[-1] = 0.
        return phi

    def matriz_tridiagonal_numpy(self, rhs=None):
        a = self.a_coef.detach().cpu().numpy()
        rhs = np.zeros(len(a)) if rhs is None else np.asarray(rhs, dtype=float).copy()
        if self.cond_esquerda == 'vácuo':
            rhs[0] = 0.
        if self.cond_direita == 'vácuo':
            rhs[-1] = 0.
        return a[1:, 0].copy(), a[:, 1].copy(), a[:-1, 2].copy(), rhs


class TransferenciaNodal1D(nn.Module):
    """P linear e R=Hc^-1 P^T Hf, iguais para MDF e MEF escalonados.

    Em malhas uniformes 2:1, R coincide com full weighting no interior.
    Para residuos MEF integrados a mesma expressao equivale exatamente a P^T.
    As coordenadas reais evitam encurtar o dominio quando N e impar.
    """

    def __init__(self, fino, grosso):
        super().__init__()
        right = torch.searchsorted(grosso, fino).clamp(1, grosso.numel()-1)
        left = right-1
        weight = (fino-grosso[left])/(grosso[right]-grosso[left])
        self.n_grosso = grosso.numel()
        for name, value in [('left', left), ('right', right), ('weight', weight),
                            ('hf', torch.as_tensor(medidas_nodais(fino.cpu().numpy()), dtype=fino.dtype, device=fino.device)),
                            ('hc', torch.as_tensor(medidas_nodais(grosso.cpu().numpy()), dtype=grosso.dtype, device=grosso.device))]:
            self.register_buffer(name, value)

    def prolongar(self, phi):
        return (1-self.weight)*phi[self.left] + self.weight*phi[self.right]

    def restringir(self, residual):
        result = residual.new_zeros(self.n_grosso)
        weighted = self.hf*residual
        result.index_add_(0, self.left, (1-self.weight)*weighted)
        result.index_add_(0, self.right, self.weight*weighted)
        return result/self.hc
