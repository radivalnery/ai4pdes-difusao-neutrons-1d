"""Stencils MDF e ciclo V/U-Net UNICO para MDF/MEF, sem treinamento."""
import numpy as np
import torch
import torch.nn as nn
from .stencils import OperadorStencil1D, TransferenciaNodal1D, medidas_nodais
from dataclasses import dataclass

@dataclass
class ResultadoFonteFixa:
    """Resultado diagnóstico de uma chamada do resolvedor U-Net/multigrid."""
    phi: torch.Tensor
    iteracoes: int
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


class OperadorDifusao1D(OperadorStencil1D):
    """MDF conservativo: mesmas bandas da v4.1 na malha fina uniforme.

    Nos niveis grossos nao uniformes, fluxos entre nos sao divididos por H_i.
    A media harmonica de D nas faces e preservada. A fonte MDF e diagonal.
    O banco Conv1d e compartilhado com MEF, inclusive para a fissao.
    """

    def __init__(self, D_arr, Sigma_a_arr, h, cond_esquerda='reflexiva',
                 cond_direita='vácuo', *, x=None, materiais=None,
                 nuSigma_f_arr=None, dtype=torch.float64):
        D = np.asarray(D_arr, dtype=float)
        Sa = np.asarray(Sigma_a_arr, dtype=float)
        x = np.arange(len(D))*h if x is None else np.asarray(x, dtype=float)
        super().__init__(x, cond_esquerda, cond_direita, dtype)
        if len(D) != len(x) or Sa.shape != D.shape or np.any(D <= 0) or np.any(Sa < 0):
            raise ValueError('Coeficientes MDF invalidos.')
        self.h = float(h)
        self.materiais = tuple(dict(m) for m in (materiais or ()))
        nf = np.zeros_like(D) if nuSigma_f_arr is None else np.asarray(nuSigma_f_arr, dtype=float)
        if nf.shape != D.shape or np.any(nf < 0):
            raise ValueError('Fissao MDF invalida.')
        face = 2*D[:-1]*D[1:]/(D[:-1]+D[1:])
        H = medidas_nodais(x)
        conductance = face/np.diff(x)
        lower, upper = -conductance/H[1:], -conductance/H[:-1]
        diag = Sa.copy()
        diag[1:] -= lower
        diag[:-1] -= upper
        for name, value in [('D', D), ('Sigma_a', Sa), ('D_half', face), ('nuSigma_f', nf)]:
            self.register_buffer(name, torch.tensor(value, dtype=dtype))
        self.definir_stencils(lower, diag, upper, np.zeros_like(lower), nf, np.zeros_like(upper))

    def criar_nivel_grosso(self, x):
        if self.materiais:
            values = []
            for point in x:
                matches = [m for m in self.materiais if m['inicio'] <= point <= m['fim']]
                if not matches:
                    raise ValueError('Material MDF ausente na malha grossa.')
                values.append([matches[0][k] for k in ('D', 'Sigma_a', 'nuSigma_f')])
            D, Sa, nf = np.asarray(values).T
        else:
            D, Sa, nf = [np.interp(x, self.x_numpy, t.detach().cpu().numpy())
                         for t in (self.D, self.Sigma_a, self.nuSigma_f)]
        return OperadorDifusao1D(D, Sa, np.diff(x)[0], self.cond_esquerda,
                                 self.cond_direita, x=x, materiais=self.materiais,
                                 nuSigma_f_arr=nf, dtype=self.x.dtype)


class SolverFonteFixaUNet1D(nn.Module):
    """
    Resolve A phi = S com um ciclo U-Net/multigrid geométrico 1D.

    Não há treinamento. As operações são:
    - aplicação do operador A por stencil/convolução fixa;
    - suavização por Jacobi ponderado;
    - restricao ponderada R=Hc^-1 P^T Hf para malhas grossas;
    - prolongamento por interpolação linear;
    - operadores rediscretizados em todos os níveis;
    - solução no nível mais grosso e correção nos níveis finos.

    Essa é a parte que materializa a filosofia Neural Physics/AI4PDEs no código.
    """

    def __init__(self, operador_A, omega=0.75, amortecimento_unet=0.20,
                 max_niveis=6, min_pontos_grosso=5, pre_suavizacoes=2,
                 post_suavizacoes=2, suavizacoes_grosso=20):
        super().__init__()
        self.operador_A = operador_A
        self.omega = float(omega)
        self.amortecimento_unet = float(amortecimento_unet)
        self.max_niveis = int(max_niveis)
        self.min_pontos_grosso = int(min_pontos_grosso)
        self.pre_suavizacoes = int(pre_suavizacoes)
        self.post_suavizacoes = int(post_suavizacoes)
        self.suavizacoes_grosso = int(suavizacoes_grosso)
        self.operadores = nn.ModuleList([operador_A])
        self._construir_hierarquia()

    def _construir_hierarquia(self):
        """Mesmas malhas aninhadas para as duas formulacoes; so A/F diferem."""
        self.transferencias = nn.ModuleList()
        atual = self.operador_A
        for _ in range(1, self.max_niveis):
            x = atual.x_numpy
            if len(x) < 2*self.min_pontos_grosso-1:
                break
            selected = set(range(0, len(x), 2)) | {len(x)-1}
            for mat in atual.materiais:
                for key in ('inicio', 'fim'):
                    selected.update(np.flatnonzero(np.isclose(x, mat[key], atol=1e-12, rtol=0)).tolist())
            indices = sorted(selected)
            if len(indices) >= len(x):
                break
            coarse = atual.criar_nivel_grosso(x[indices]).to(atual.x.device)
            self.transferencias.append(TransferenciaNodal1D(atual.x, coarse.x))
            self.operadores.append(coarse)
            atual = coarse

    def _corrigir_contorno_rhs(self, rhs):
        rhs = rhs.clone()
        # Apenas contornos de vacuo/Dirichlet recebem RHS nulo. Em contorno
        # reflexivo, a equacao discretizada na fronteira ainda possui fonte.
        if self.operador_A.cond_esquerda == 'vácuo':
            rhs[0] = 0.0
        if self.operador_A.cond_direita == 'vácuo':
            rhs[-1] = 0.0
        return rhs

    def _suavizar_jacobi(self, operador, phi, rhs, diag, n_passos):
        for _ in range(n_passos):
            residuo = rhs - operador(phi)
            phi = phi + self.omega * residuo / diag
            phi = operador.aplicar_contorno_fluxo(phi)
        return phi

    def _restringir(self, fine, nivel=0):
        residual = self.transferencias[nivel].restringir(fine)
        return self.operadores[nivel+1].aplicar_contorno_fluxo(residual)

    def _prolongar(self, coarse, tamanho_fino, nivel=0):
        return self.transferencias[nivel].prolongar(coarse)

    def _resolver_grosso(self, operador, rhs, chute=None):
        """Resolve o nível mais grosso por Jacobi, sem solver direto clássico."""
        diag = operador.diagonal()
        phi = torch.zeros_like(rhs) if chute is None else chute.clone()
        phi = operador.aplicar_contorno_fluxo(phi)
        return self._suavizar_jacobi(operador, phi, rhs, diag, self.suavizacoes_grosso)

    def _ciclo_unet_multigrid(self, nivel, phi, rhs):
        operador = self.operadores[nivel]
        diag = operador.diagonal()

        if nivel == len(self.operadores) - 1:
            return self._resolver_grosso(operador, rhs, chute=phi)

        phi = self._suavizar_jacobi(operador, phi, rhs, diag, self.pre_suavizacoes)
        residuo = rhs - operador(phi)
        residuo_grosso = self._restringir(residuo, nivel)
        erro_grosso = torch.zeros_like(residuo_grosso)
        erro_grosso = self._ciclo_unet_multigrid(nivel + 1, erro_grosso, residuo_grosso)
        erro_fino = self._prolongar(erro_grosso, phi.numel(), nivel)
        phi = phi + self.amortecimento_unet * erro_fino
        phi = operador.aplicar_contorno_fluxo(phi)
        phi = self._suavizar_jacobi(operador, phi, rhs, diag, self.post_suavizacoes)
        return phi

    def resolver(self, rhs, chute=None, tol=1.0e-5, max_iter=5000):
        dtype = self.operador_A.diagonal().dtype
        rhs = self._corrigir_contorno_rhs(rhs.to(dtype=dtype))
        if chute is None:
            phi = torch.zeros_like(rhs)
        else:
            phi = chute.clone().to(dtype=dtype)
        phi = self.operador_A.aplicar_contorno_fluxo(phi)

        norma_rhs = max(float(torch.linalg.vector_norm(rhs, ord=float('inf')).item()),
                        torch.finfo(dtype).tiny)
        ultimo_residuo = float('inf')
        historico_residuo = []

        for it in range(1, max_iter + 1):
            phi = self._ciclo_unet_multigrid(0, phi, rhs)
            residuo = rhs - self.operador_A(phi)
            ultimo_residuo = float(torch.linalg.vector_norm(residuo, ord=float('inf')).item())
            historico_residuo.append(ultimo_residuo / norma_rhs)
            if ultimo_residuo / norma_rhs < tol:
                return ResultadoFonteFixa(phi, it, ultimo_residuo, historico_residuo, True)

        return ResultadoFonteFixa(phi, max_iter, ultimo_residuo, historico_residuo, False)
