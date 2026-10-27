"""Difusao de neutrons 1D - v4.2 MDF/MEF.

Arquivo unico gerado de src/neutron_ai4pdes; usa as MESMAS funcoes dos modulos.
GUI: python difusao_neutrons_neuralphysics_1d_mdf_mef_v4_2.py
Experimento: acrescente --experimento-artigo --discretizacao mef
MDF e MEF escolhem a discretizacao; U-Net/multigrid e o resolvedor principal.
Mesmo banco Conv1d, precisao, transferencias e ciclo V para MDF e MEF.
GUI: marque MDF/MEF e escolha sequencial/paralelo; parametros sao comuns.
Tempos paralelos medem concorrencia, nao desempenho isolado.
Thomas e referencia opcional e nao participa de nenhum nivel do multigrid.
As versoes antigas e o backup continuam disponiveis separadamente.
"""


# ============================================================================
# 1. DESCRICAO DO METODO
# ============================================================================

"""Constantes descritivas do método AI4PDEs 1D adaptado."""

METODO_AI4PDES_1D = (
    "Adaptação 1D da filosofia Neural Physics/AI4PDEs: discretização MDF ou MEF, "
    "operadores locais fixos em PyTorch e ciclo U-Net/multigrid sem treinamento."
)


# ============================================================================
# 2. SOLUCOES ANALITICAS
# ============================================================================

"""Soluções e parâmetros de referência para os problemas 1D."""

import numpy as np


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


# ============================================================================
# 3. METRICAS DO FLUXO
# ============================================================================

"""Metricas globais de forma do fluxo; independem da amplitude de potencia."""
import numpy as np

def normalizar_forma_fluxo(phi):
    arr = np.asarray(phi, dtype=float).copy()
    m = np.max(np.abs(arr))
    if m > 0:
        arr /= m
    return arr


def erro_forma_fluxo(phi_calc, phi_ref):
    """Normas discretas nodais apos normalizar cada vetor pelo proprio maximo.

    Retorna max|a-b| e ||a-b||_2/||b||_2. A norma L2 e vetorial;
    nao e a integral continua do erro entre funcoes de elementos finitos.
    """
    a = normalizar_forma_fluxo(phi_calc)
    b = normalizar_forma_fluxo(phi_ref)
    if a.size != b.size:
        raise ValueError("Fluxos com tamanhos diferentes no cálculo do erro de forma.")
    erro_inf = float(np.linalg.norm(a - b, ord=np.inf))
    denom = float(np.linalg.norm(b, ord=2))
    erro_l2 = float(np.linalg.norm(a - b, ord=2) / denom) if denom > 0 else float(np.linalg.norm(a - b, ord=2))
    return erro_inf, erro_l2


# ============================================================================
# 4. CONVOLUCOES E TRANSFERENCIAS COMUNS MDF/MEF
# ============================================================================

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


# ============================================================================
# 5. OPERADOR MDF E CICLO MULTIGRID UNICO
# ============================================================================

"""Stencils MDF e ciclo V/U-Net UNICO para MDF/MEF, sem treinamento."""
import numpy as np
import torch
import torch.nn as nn
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


# ============================================================================
# 6. FORMULACAO MEF P1 COM MASSA CONSISTENTE
# ============================================================================

"""MEF/Galerkin P1 para difusao de neutrons em uma dimensao.

1. Forma fraca: integral(D phi' v' + Sigma_a phi v) = integral(nuSigma_f phi v)/k.
2. Cada elemento tem dois nos. Absorcao e fissao usam massa consistente.
3. Os coeficientes sao integrados por material, inclusive quando uma interface
   corta um elemento. N continua sendo o numero de elementos, com N+1 nos.
4. As tres diagonais sao buffers PyTorch fixos: nao ha pesos treinaveis.
5. A e F sao escalonados por H^-1 e usam o MESMO banco Conv1d e ciclo do MDF.
   A massa de absorcao e fissao continua consistente, sem lumping.
   Thomas e usado somente pelo caminho externo de referencia do solver.
"""

import numpy as np
import torch



class OperadorMEF1D(OperadorStencil1D):
    """A = K + M_a e F = M_f, com contornos homogeneos.

    Em um elemento de comprimento h e material constante:
      K_e = D/h * [[1, -1], [-1, 1]];
      M_e(c) = c*h/6 * [[2, 1], [1, 2]].
    A integracao de Gauss em cada trecho material tambem atende elementos
    cortados por interfaces. Fluxo nulo e imposto por eliminacao simetrica;
    corrente nula (reflexiva) e uma condicao natural da forma fraca.
    """

    def __init__(self, x, materiais, cond_esquerda='reflexiva', cond_direita='vácuo', *, dtype=torch.float64):
        x = np.asarray(x, dtype=float)
        if x.ndim != 1 or len(x) < 2 or not np.all(np.diff(x) > 0):
            raise ValueError('A malha MEF deve conter pelo menos dois nos crescentes.')
        super().__init__(x, cond_esquerda, cond_direita, dtype)
        self.cond_esquerda = cond_esquerda
        self.cond_direita = cond_direita
        self.materiais = tuple(dict(m) for m in materiais)
        for cond in (cond_esquerda, cond_direita):
            if cond not in ('reflexiva', 'vácuo'):
                raise ValueError('MEF aceita contorno reflexivo ou vacuo (fluxo nulo).')
        diag = np.zeros(len(x))
        off = np.zeros(len(x) - 1)
        fdiag = np.zeros(len(x))
        foff = np.zeros(len(x) - 1)
        for e, (left, right) in enumerate(zip(x[:-1], x[1:])):
            h = right - left
            cuts = sorted(set([left, right] + [float(m[key]) for m in materiais
                         for key in ('inicio', 'fim') if left < m[key] < right]))
            ae = np.zeros((2, 2))
            fe = np.zeros((2, 2))
            for a, b in zip(cuts[:-1], cuts[1:]):
                midpoint = (a + b) / 2
                matches = [m for m in materiais if m['inicio'] < midpoint < m['fim']]
                if len(matches) != 1:
                    raise ValueError('Os materiais MEF devem cobrir o dominio sem lacunas ou sobreposicoes.')
                m = matches[0]
                if m['D'] <= 0 or m['Sigma_a'] < 0 or m['nuSigma_f'] < 0:
                    raise ValueError('MEF requer D > 0 e secoes de choque nao negativas.')
                grad = np.array([-1., 1.]) / h
                ae += m['D'] * (b-a) * np.outer(grad, grad)
                for sign in (-1., 1.):
                    point = midpoint + sign * (b-a) / (2*np.sqrt(3))
                    basis = np.array([(right-point)/h, (point-left)/h])
                    mass = (b-a) / 2 * np.outer(basis, basis)
                    ae += m['Sigma_a'] * mass
                    fe += m['nuSigma_f'] * mass
            diag[e:e+2] += np.diag(ae)
            off[e] = ae[0, 1]
            fdiag[e:e+2] += np.diag(fe)
            foff[e] = fe[0, 1]
        # Guardamos F completo: sua soma integra a producao em todo o dominio.
        # A fonte e zerada nos graus Dirichlet apenas para a resolucao linear.
        if cond_esquerda == 'vácuo':
            diag[0], off[0] = 1., 0.
        if cond_direita == 'vácuo':
            diag[-1], off[-1] = 1., 0.
        if np.any(diag <= 0):
            raise ValueError('Diagonal MEF nao positiva; verifique materiais e contornos.')
        # Matrizes integradas permanecem disponiveis para auditoria independente.
        for name, value in [('a_diag', diag), ('a_off', off),
                            ('f_diag', fdiag), ('f_off', foff)]:
            self.register_buffer(name, torch.tensor(value, dtype=dtype))
        H = medidas_nodais(x)
        self.definir_stencils(off/H[1:], diag/H, off/H[:-1],
                             foff/H[1:], fdiag/H, foff/H[:-1])

    def criar_nivel_grosso(self, x):
        return OperadorMEF1D(x, self.materiais, self.cond_esquerda,
                            self.cond_direita, dtype=self.x.dtype)


# Compatibilidade de importacao: NAO existe mais um segundo ciclo multigrid.
SolverFonteFixaMEF1D = SolverFonteFixaUNet1D


# ============================================================================
# 7. ITERACAO DE POTENCIA E NORMALIZACAO
# ============================================================================

"""Iteracao de potencia e diagnosticos; base v4.1 preservada."""
import platform
import time
import numpy as np
import torch

class SolverDifusaoAI4PDEs:
    REFERENCIAS = {
        'homogeneo': {'k_eff': None, 'fonte': 'Analítica'},
        'heterogeneo': {'k_eff': 1.09506, 'fonte': 'Couto (2003)'}
    }

    def __init__(self, L, N, materiais, cond_esquerda='reflexiva', cond_direita='vácuo',
                 tol_k=1e-6, tol_phi=1e-5, max_iter=1000, potencia_nominal=100.0,
                 pontos_interesse=None, progress_callback=None, dispositivo_preferido='auto',
                 omega_fonte=0.75, amortecimento_unet=0.20,
                 tol_fonte=None, max_iter_fonte=5000,
                 metodo_fonte="unet_multigrid", guardar_historicos_fonte=False,
                 permitir_thomas_comparacao=False, discretizacao='mdf', precisao='float64'):

        self.discretizacao = str(discretizacao).lower()
        if self.discretizacao not in ('mdf', 'mef'):
            raise ValueError('Discretizacao deve ser mdf ou mef.')
        if L <= 0 or N < 1 or int(N) != N or max_iter < 1 or max_iter_fonte < 1:
            raise ValueError('Comprimento, malha e limites de iteracoes devem ser positivos.')
        if min(tol_k, tol_phi, tol_phi if tol_fonte is None else tol_fonte) <= 0:
            raise ValueError('As tolerancias devem ser positivas.')
        if N == 1 and cond_esquerda == cond_direita == 'vácuo':
            raise ValueError('Dois contornos de fluxo nulo requerem N >= 2.')
        if precisao not in ('float32', 'float64'):
            raise ValueError('Precisao deve ser float32 ou float64.')
        self.precisao = precisao
        self.dtype = getattr(torch, precisao)
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
        self.permitir_thomas_comparacao = bool(permitir_thomas_comparacao)

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
            "Neural Physics 1D adaptado com PyTorch CUDA"
            if self.device.type == 'cuda'
            else "Neural Physics 1D adaptado com PyTorch CPU"
        )
        self.modelo_A = None
        self.solver_fonte_fixa = None
        self.iteracoes_fonte_fixa = []
        self.residuos_fonte_fixa = []
        self.convergiu_fonte_fixa = []
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
        self.tempos_detalhados = {
            "tempo_montagem_operador_s": 0.0,
            "tempo_montagem_hierarquia_s": 0.0,
            "tempo_montagem_total_s": 0.0,
            "tempo_transferencia_numpy_torch_s": 0.0,
            "tempo_transferencia_torch_numpy_s": 0.0,
            "tempo_transferencia_total_s": 0.0,
            "tempo_loop_iterativo_s": 0.0,
            "tempo_iteracao_numerica_s": 0.0,
            "tempo_fonte_fixa_s": 0.0,
            "tempo_atualizacao_potencia_s": 0.0,
            "tempo_total_s": 0.0,
            "tempo_medio_iteracao_s": 0.0,
        }

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

    def _sincronizar(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    def _tensor(self, value):
        """Mede conversoes e copias separadas do custo numerico."""
        self._sincronizar()
        start = time.perf_counter()
        result = torch.as_tensor(value, dtype=self.dtype, device=self.device)
        self._sincronizar()
        self.tempos_detalhados['tempo_transferencia_numpy_torch_s'] += time.perf_counter()-start
        return result

    def _numpy(self, value):
        self._sincronizar()
        start = time.perf_counter()
        result = value.detach().cpu().numpy()
        self._sincronizar()
        self.tempos_detalhados['tempo_transferencia_torch_numpy_s'] += time.perf_counter()-start
        return result

    def fonte_fissao(self, phi):
        """Mesmo caminho Conv1d fixo para F_MDF diagonal e F_MEF consistente."""
        if torch.is_tensor(phi):
            return self.modelo_A.aplicar_fissao(phi)
        return self._numpy(self.modelo_A.aplicar_fissao(self._tensor(phi)))

    def integral_fissao(self, phi):
        return float(np.dot(self.modelo_A.pesos_producao_numpy, phi))

    def descricao_discretizacao(self):
        if self.discretizacao == 'mef':
            formulation = 'MEF P1: A=H^-1(K+M_a), F=H^-1 M_f, massa consistente (sem lumping). '
        else:
            formulation = 'MDF conservativo: difusao com media harmonica e fissao diagonal. '
        return (formulation + 'Banco Conv1d fixo comum para A/F; mesmo ciclo V, '
                'Jacobi, P linear e R=Hc^-1 P^T Hf. Precisao: ' + self.precisao + '.')

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
        inicio_operador = time.perf_counter()
        if self.discretizacao == 'mef':
            self.modelo_A = OperadorMEF1D(self.x, self.materiais,
                                          self.cond_esquerda, self.cond_direita, dtype=self.dtype)
        else:
            self.modelo_A = OperadorDifusao1D(self.D_arr, self.Sigma_a_arr, self.h,
                                              self.cond_esquerda, self.cond_direita,
                                              x=self.x, materiais=self.materiais,
                                              nuSigma_f_arr=self.nuSigma_f_arr, dtype=self.dtype)
        self.tempos_detalhados["tempo_montagem_operador_s"] += time.perf_counter() - inicio_operador

        inicio_hierarquia = time.perf_counter()
        self.solver_fonte_fixa = None
        if self.metodo_fonte == 'unet_multigrid':
            self.solver_fonte_fixa = SolverFonteFixaUNet1D(self.modelo_A, omega=self.omega_fonte,
                                            amortecimento_unet=self.amortecimento_unet)
        self.tempos_detalhados["tempo_montagem_hierarquia_s"] += time.perf_counter() - inicio_hierarquia
        self._sincronizar()
        inicio_transferencia = time.perf_counter()
        if self.solver_fonte_fixa is not None:
            self.solver_fonte_fixa.to(self.device)
        else:
            self.modelo_A.to(self.device)
        self._sincronizar()
        self.tempos_detalhados['tempo_transferencia_numpy_torch_s'] += time.perf_counter()-inicio_transferencia

        self.metodo_executado = (
            "Adaptação 1D Neural Physics/AI4PDEs: operadores convolucionais fixos "
            "e ciclo U-Net/multigrid geométrico sem treinamento"
        )

    def resolver_fonte_fixa_unet(self, S, chute=None):
        if self.solver_fonte_fixa is None:
            raise RuntimeError("O resolvedor de fonte fixa ainda não foi criado.")
        S_tensor = S if torch.is_tensor(S) else self._tensor(S)
        chute_tensor = None
        if chute is not None:
            chute_tensor = chute if torch.is_tensor(chute) else self._tensor(chute)
        resultado = self.solver_fonte_fixa.resolver(
            S_tensor,
            chute=chute_tensor,
            tol=self.tol_fonte,
            max_iter=self.max_iter_fonte,
        )
        phi = resultado.phi
        n_iter = resultado.iteracoes
        residuo = resultado.residuo
        historico = resultado.historico_residuo
        self.iteracoes_fonte_fixa.append(int(n_iter))
        self.residuos_fonte_fixa.append(float(residuo))
        self.convergiu_fonte_fixa.append(bool(resultado.convergiu))
        self.historico_residuo_fonte_ultima_chamada = list(historico)
        if self.guardar_historicos_fonte:
            self.historicos_residuo_fonte.append(list(historico))
        return phi

    def resolver_fonte_fixa_thomas(self, S):
        if not self.permitir_thomas_comparacao:
            raise RuntimeError(
                "Thomas está disponível apenas como referência numérica clássica. "
                "O método principal deve ser unet_multigrid."
            )
        if self.modelo_A is None:
            raise RuntimeError("O operador A ainda não foi criado.")
        S_np = self._numpy(S) if torch.is_tensor(S) else np.asarray(S, dtype=float)
        lower, diag, upper, rhs = self.modelo_A.matriz_tridiagonal_numpy(S_np)
        phi_np = resolver_tridiagonal_thomas(lower, diag, upper, rhs)
        phi_tensor = self._tensor(phi_np)
        residuo_tensor = self._tensor(rhs) - self.modelo_A(phi_tensor)
        residuo = float(torch.linalg.vector_norm(residuo_tensor, ord=float('inf')).item())
        norma_rhs = max(float(np.linalg.norm(rhs, ord=np.inf)), torch.finfo(self.dtype).tiny)
        self.iteracoes_fonte_fixa.append(1)
        self.residuos_fonte_fixa.append(residuo)
        self.convergiu_fonte_fixa.append(True)
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
        P_atual = 3.2e-11 * self.integral_fissao(phi)
        if P_atual > 0:
            return phi * (self.potencia_nominal / P_atual)
        return phi

    def fluxo_analitico_pontos(self, pontos, potencia=False):
        """Avalia a formula no x solicitado, mesmo quando x nao e um no."""
        grid = fluxo_analitico_homogeneo(self.x, self.L, self.cond_esquerda, self.cond_direita)
        scale = np.max(grid)
        values = fluxo_analitico_homogeneo(np.asarray(pontos), self.L,
                                          self.cond_esquerda, self.cond_direita) / scale
        if potencia:
            normalized = self.normalizar_potencia(self.phi_ref)
            index = np.argmax(self.phi_ref)
            values = values * normalized[index] / self.phi_ref[index]
        values[np.abs(values) <= 1e-12 * max(np.max(np.abs(values)), 1.)] = 0.
        return values

    def resolver(self, metodo_fonte=None):
        if metodo_fonte is not None:
            self.metodo_fonte = metodo_fonte
        if self.metodo_fonte == 'thomas' and not self.permitir_thomas_comparacao:
            raise RuntimeError('Thomas disponivel apenas como referencia numerica classica.')
        if self.metodo_fonte not in ('unet_multigrid', 'thomas'):
            raise ValueError('Metodo de fonte desconhecido.')
        self.tempos_detalhados = {
            "tempo_montagem_operador_s": 0.0,
            "tempo_montagem_hierarquia_s": 0.0,
            "tempo_montagem_total_s": 0.0,
            "tempo_transferencia_numpy_torch_s": 0.0,
            "tempo_transferencia_torch_numpy_s": 0.0,
            "tempo_transferencia_total_s": 0.0,
            "tempo_loop_iterativo_s": 0.0,
            "tempo_iteracao_numerica_s": 0.0,
            "tempo_fonte_fixa_s": 0.0,
            "tempo_atualizacao_potencia_s": 0.0,
            "tempo_preparacao_fonte_s": 0.0,
            "tempo_total_s": 0.0,
            "tempo_medio_iteracao_s": 0.0,
        }
        tempo_total_inicio_global = time.perf_counter()
        self.criar_operador_A()
        self.calcular_referencias()
        n = self.N + 1
        phi = np.ones(n)
        if self.cond_esquerda == 'vácuo':
            phi[0] = 0.
        if self.cond_direita == 'vácuo':
            phi[-1] = 0.
        k_eff = 1.0
        self.historico_k = []
        self.historico_phi = []
        self.tempos_iteracao = []
        self.iteracoes_fonte_fixa = []
        self.residuos_fonte_fixa = []
        self.convergiu_fonte_fixa = []
        self.historico_residuo_fonte_ultima_chamada = []
        self.historicos_residuo_fonte = []
        self.convergiu = False
        self.iteracoes_totais = 0
        self.erro_phi_pontos = {}
        self.metodo_executado = (
            f"{self.nome_metodo(self.metodo_fonte)} / {self.discretizacao.upper()} aplicado ao problema de fonte fixa"
        )

        print("\n" + "="*60)
        print("SOLVER DE DIFUSÃO DE NÊUTRONS COM AI4PDEs 1D ADAPTADO")
        print("="*60)
        print(f"Malha: {self.N} intervalos, h = {self.h:.4f} cm")
        print(f"Tipo: {'Homogêneo' if self.is_homogeneo else 'Heterogêneo'}")
        print(f"Método: {self.metodo_executado}")
        print(f"Precisao comum: {self.precisao}")
        print(f"Dispositivo: {self.nome_dispositivo}")
        if self.k_ref:
            print(f"k_ref: {self.k_ref:.8f} ({self.fonte_ref})")
        print("="*60)

        tempo_inicio_solver = time.perf_counter()

        for iteracao in range(1, self.max_iter + 1):
            inicio_iter = time.perf_counter()
            inicio_preparacao = time.perf_counter()
            transf_antes = self.tempos_detalhados['tempo_transferencia_numpy_torch_s']
            S_tensor = self.fonte_fissao(self._tensor(phi)) / k_eff
            self._sincronizar()
            transf_preparacao = self.tempos_detalhados['tempo_transferencia_numpy_torch_s'] - transf_antes
            self.tempos_detalhados['tempo_preparacao_fonte_s'] += max(
                0., time.perf_counter()-inicio_preparacao-transf_preparacao)

            inicio_fonte = time.perf_counter()
            transf_antes = (self.tempos_detalhados['tempo_transferencia_numpy_torch_s']
                           + self.tempos_detalhados['tempo_transferencia_torch_numpy_s'])
            phi_novo_tensor = self.resolver_fonte_fixa(S_tensor, chute=phi, metodo=self.metodo_fonte)
            self._sincronizar()
            transf_dentro = (self.tempos_detalhados['tempo_transferencia_numpy_torch_s']
                            + self.tempos_detalhados['tempo_transferencia_torch_numpy_s'] - transf_antes)
            self.tempos_detalhados["tempo_fonte_fixa_s"] += max(0., time.perf_counter()-inicio_fonte-transf_dentro)
            if self.convergiu_fonte_fixa and not self.convergiu_fonte_fixa[-1]:
                self.iteracoes_totais = iteracao
                self.convergiu = False
                self.tempos_iteracao.append(time.perf_counter()-inicio_iter)
                print(
                    "\nFONTE FIXA NÃO CONVERGIU: "
                    f"atingiu max_iter_fonte={self.max_iter_fonte} na iteração externa {iteracao}. "
                    "Ajuste tol_fonte, max_fonte, omega ou amortecimento antes de continuar."
                )
                break
            phi_novo = self._numpy(phi_novo_tensor)

            inicio_atualizacao = time.perf_counter()
            integral_novo = self.integral_fissao(phi_novo)
            integral_antigo = self.integral_fissao(phi)
            if abs(integral_antigo) < 1e-30:
                raise ValueError("Integral da fonte muito pequena")
            k_eff_novo = k_eff * (integral_novo / integral_antigo)

            max_phi = np.max(phi_novo)
            if max_phi > 0:
                phi_novo = phi_novo / max_phi

            erro_k = abs(k_eff_novo - k_eff) / abs(k_eff_novo) if abs(k_eff_novo) > 0 else 1.0
            erro_phi = np.max(np.abs(phi_novo - phi)) / (np.max(np.abs(phi_novo)) + 1e-15)
            self.erro_phi_iterativo = float(erro_phi)
            self.tempos_detalhados["tempo_atualizacao_potencia_s"] += time.perf_counter() - inicio_atualizacao

            self.historico_k.append(k_eff_novo)
            if iteracao % 10 == 0 or iteracao == 1:
                self.historico_phi.append(phi_novo.copy())
            self.tempos_iteracao.append(time.perf_counter() - inicio_iter)

            phi = phi_novo
            k_eff = k_eff_novo

            if self.progress_callback:
                self.progress_callback(iteracao, self.max_iter, k_eff, erro_k, erro_phi)

            if iteracao % 10 == 0 or iteracao == 1:
                status_fonte = "ok" if self.convergiu_fonte_fixa[-1] else "max_iter_fonte"
                print(f"Iteração {iteracao:4d}: k_eff = {k_eff:.8f}, "
                      f"erro_k = {erro_k:.2e}, erro_phi = {erro_phi:.2e},"
                      f"Iterações phi: {self.iteracoes_fonte_fixa[-1]}, fonte: {status_fonte}")

            if erro_k < self.tol_k and erro_phi < self.tol_phi:
                self.convergiu = True
                self.iteracoes_totais = iteracao
                print(f"\nCONVERGÊNCIA ALCANÇADA em {iteracao} iterações!")
                print(f"k_eff = {k_eff:.8f}")
                break

        if not self.convergiu:
            if self.iteracoes_totais <= 0:
                self.iteracoes_totais = self.max_iter
            print(f"\nNão convergiu após {self.iteracoes_totais} iterações")

        self.tempos_detalhados["tempo_loop_iterativo_s"] = time.perf_counter() - tempo_inicio_solver
        self.tempos_detalhados["tempo_montagem_total_s"] = (
            self.tempos_detalhados["tempo_montagem_operador_s"]
            + self.tempos_detalhados["tempo_montagem_hierarquia_s"]
        )
        self.tempos_detalhados["tempo_transferencia_total_s"] = (
            self.tempos_detalhados["tempo_transferencia_numpy_torch_s"]
            + self.tempos_detalhados["tempo_transferencia_torch_numpy_s"]
        )
        self.tempos_detalhados["tempo_iteracao_numerica_s"] = max(
            0.0,
            self.tempos_detalhados['tempo_preparacao_fonte_s']
            + self.tempos_detalhados['tempo_fonte_fixa_s']
            + self.tempos_detalhados['tempo_atualizacao_potencia_s'],
        )
        self.tempo_total = time.perf_counter() - tempo_total_inicio_global
        self.tempo_medio_iteracao = (
            self.tempos_detalhados['tempo_iteracao_numerica_s'] / max(self.iteracoes_totais, 1)
            if self.iteracoes_totais else 0.0
        )
        self.tempos_detalhados["tempo_total_s"] = self.tempo_total
        self.tempos_detalhados["tempo_medio_iteracao_s"] = self.tempo_medio_iteracao
        self.k_eff = k_eff
        self.phi = phi

        if self.k_ref:
            self.erro_k = abs(self.k_eff - self.k_ref) / abs(self.k_ref)

        if self.phi_ref is not None:
            for ponto, val_ref in zip(self.pontos_interesse, self.fluxo_analitico_pontos(self.pontos_interesse)):
                val_calc = np.interp(ponto, self.x, self.phi)
                self.erro_phi_pontos[ponto] = (abs(val_calc-val_ref)/abs(val_ref)
                                              if abs(val_ref) > 1e-12 else None)

        erros_validos = [v for v in self.erro_phi_pontos.values() if v is not None]
        self.erro_phi_max = (
            float(np.max(erros_validos))
            if erros_validos else 0.0
        )
        self.phi_normalizado = self.normalizar_potencia(phi)
        self.tempo_total = time.perf_counter() - tempo_total_inicio_global
        self.tempos_detalhados['tempo_total_s'] = self.tempo_total
        return k_eff, self.phi_normalizado

    def resumo_resultado(self, caso=None):
        return {
            "Caso": caso or ("Homogêneo" if self.is_homogeneo else "Heterogêneo"),
            "Discretizacao": self.discretizacao.upper(),
            "Nos": self.N + 1,
            "Precisao": str(self.dtype),
            "Ciclo": "V" if self.solver_fonte_fixa is not None else "nao se aplica",
            "Niveis": len(self.solver_fonte_fixa.operadores) if self.solver_fonte_fixa is not None else 0,
            "Pre_suavizacoes": self.solver_fonte_fixa.pre_suavizacoes if self.solver_fonte_fixa is not None else 0,
            "Post_suavizacoes": self.solver_fonte_fixa.post_suavizacoes if self.solver_fonte_fixa is not None else 0,
            "Suavizacoes_grosso": self.solver_fonte_fixa.suavizacoes_grosso if self.solver_fonte_fixa is not None else 0,
            "Método": self.nome_metodo(self.metodo_fonte),
            "N": self.N,
            "k_eff": self.k_eff,
            "Referência": self.k_ref,
            "Erro k (%)": self.erro_k * 100.0 if self.erro_k is not None else None,
            "Iter. externas": self.iteracoes_totais,
            "Iter. fonte média": float(np.mean(self.iteracoes_fonte_fixa)) if self.iteracoes_fonte_fixa else 0.0,
            "Resíduo final": float(self.residuos_fonte_fixa[-1]) if self.residuos_fonte_fixa else None,
            "Fonte fixa convergiu (todas as chamadas)": bool(self.convergiu_fonte_fixa) and all(self.convergiu_fonte_fixa),
            "Chamadas fonte fixa não convergidas": int(sum(1 for ok in self.convergiu_fonte_fixa if not ok)),
            "Bateu max_iter externo": not self.convergiu,
            "Tempo (s)": self.tempo_total,
            "tempo_montagem_operador_s": self.tempos_detalhados.get("tempo_montagem_operador_s", 0.0),
            "tempo_montagem_hierarquia_s": self.tempos_detalhados.get("tempo_montagem_hierarquia_s", 0.0),
            "tempo_montagem_total_s": self.tempos_detalhados.get("tempo_montagem_total_s", 0.0),
            "tempo_transferencia_numpy_torch_s": self.tempos_detalhados.get("tempo_transferencia_numpy_torch_s", 0.0),
            "tempo_transferencia_torch_numpy_s": self.tempos_detalhados.get("tempo_transferencia_torch_numpy_s", 0.0),
            "tempo_transferencia_total_s": self.tempos_detalhados.get("tempo_transferencia_total_s", 0.0),
            "tempo_loop_iterativo_s": self.tempos_detalhados.get("tempo_loop_iterativo_s", 0.0),
            "tempo_iteracao_numerica_s": self.tempos_detalhados.get("tempo_iteracao_numerica_s", 0.0),
            "tempo_fonte_fixa_s": self.tempos_detalhados.get("tempo_fonte_fixa_s", 0.0),
            "tempo_atualizacao_potencia_s": self.tempos_detalhados.get("tempo_atualizacao_potencia_s", 0.0),
            "tempo_medio_iteracao_s": self.tempos_detalhados.get("tempo_medio_iteracao_s", self.tempo_medio_iteracao),
            "tol_fonte": self.tol_fonte,
            "max_iter_fonte": self.max_iter_fonte,
            "omega_fonte": self.omega_fonte,
            "amortecimento_unet": self.amortecimento_unet,
        }


def executar_comparacao_resolvedores(config_base, metodos=("unet_multigrid", "thomas")):
    resultados = []
    solvers = []
    for metodo in metodos:
        cfg = dict(config_base)
        cfg["metodo_fonte"] = metodo
        if metodo == "thomas":
            cfg["permitir_thomas_comparacao"] = True
        solver = SolverDifusaoAI4PDEs(**cfg)
        solver.resolver()
        resultados.append(solver.resumo_resultado())
        solvers.append(solver)
    return resultados, solvers


# ============================================================================
# 8. CAMPANHAS MDF/MEF SEQUENCIAIS OU PARALELAS
# ============================================================================

"""Execucao de MDF/MEF com configuracao comum e resultados independentes.

Nao altera stencils, ciclos V ou tolerancias. No modo paralelo, cada thread
possui seu proprio solver e materiais. Tkinter e Matplotlib nao sao chamados
pelos trabalhadores. Eventos sao entregues a uma fila da interface.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass, field
import time



AVISO_TEMPOS_PARALELOS = (
    'Execucao paralela: MDF e MEF compartilham recursos de CPU/GPU. '
    'Os tempos medidos sob concorrencia nao representam desempenho isolado. '
    'Para comparar desempenho, use o modo sequencial e repeticoes controladas.'
)


@dataclass
class ResultadoCampanha:
    """Snapshot da campanha; chaves dos resultados: (discretizacao, N, metodo)."""

    configuracao: dict
    discretizacoes: tuple
    malhas: tuple
    incluir_thomas: bool
    modo_solicitado: str
    modo_efetivo: str
    solvers: dict = field(default_factory=dict)
    erros: dict = field(default_factory=dict)
    tempo_parede: float = 0.

    @property
    def aviso_tempos(self):
        return AVISO_TEMPOS_PARALELOS if self.modo_efetivo == 'paralelo' else (
            'Execucao sequencial: sem sobreposicao dos solvers desta campanha. '
            'Tempos individuais nao incluem geracao de graficos ou relatorios.'
        )

    def chaves(self):
        methods = ('unet_multigrid', 'thomas') if self.incluir_thomas else ('unet_multigrid',)
        return [(d, n, m) for d in self.discretizacoes for n in self.malhas for m in methods]

    def resumos(self):
        rows = []
        for key in self.chaves():
            solver = self.solvers.get(key)
            if solver is None:
                continue
            row = solver.resumo_resultado()
            row.update(metodo_id=key[2], modo_execucao=self.modo_efetivo,
                       modo_solicitado=self.modo_solicitado,
                       tempo_parede_campanha_s=self.tempo_parede,
                       aviso_tempos=self.aviso_tempos,
                       erro_fluxo_global_inf=None, erro_fluxo_global_L2=None)
            main = self.solvers.get((key[0], key[1], 'unet_multigrid'))
            ref = self.solvers.get((key[0], key[1], 'thomas'))
            if main is not None and ref is not None and main.convergiu and ref.convergiu:
                inf, l2 = erro_forma_fluxo(main.phi, ref.phi)
                row.update(erro_fluxo_global_inf=inf, erro_fluxo_global_L2=l2)
            rows.append(row)
        return rows

    def metadados(self):
        return dict(configuracao=deepcopy(self.configuracao),
                    discretizacoes=self.discretizacoes, malhas=self.malhas,
                    incluir_thomas=self.incluir_thomas,
                    modo_solicitado=self.modo_solicitado, modo_efetivo=self.modo_efetivo,
                    tempo_parede_campanha_s=self.tempo_parede, aviso_tempos=self.aviso_tempos,
                    erros=[dict(discretizacao=d, N=n, metodo=m, mensagem=msg)
                           for (d, n, m), msg in self.erros.items()])


def executar_campanha(configuracao, discretizacoes, modo='sequencial',
                      incluir_thomas=False, malhas=None, evento=None):
    """Executa cada perfil MDF/MEF, sem alterar parametros comuns/global threads.

    No paralelo, sao no maximo dois trabalhadores, um por discretizacao.
    As malhas de um mesmo perfil e sua referencia Thomas sao sequenciais.
    Falha de um caso e registrada, sem descartar os demais resultados.
    """
    discs = tuple(dict.fromkeys(str(d).lower() for d in discretizacoes))
    if not discs or any(d not in ('mdf', 'mef') for d in discs):
        raise ValueError('Selecione MDF, MEF ou ambos.')
    if modo not in ('sequencial', 'paralelo'):
        raise ValueError('Modo deve ser sequencial ou paralelo.')
    cfg = deepcopy(configuracao)
    for key in ('discretizacao', 'metodo_fonte', 'progress_callback', 'permitir_thomas_comparacao'):
        cfg.pop(key, None)
    values = tuple(sorted(set(malhas if malhas is not None else [cfg['N']])))
    if not values or any(n < 1 or int(n) != n for n in values):
        raise ValueError('Informe malhas inteiras positivas.')
    effective = 'paralelo' if modo == 'paralelo' and len(discs) > 1 else 'sequencial'
    result = ResultadoCampanha(cfg, discs, values, bool(incluir_thomas), modo, effective)
    emit = evento or (lambda event: None)

    def profile(disc):
        found, errors = {}, {}
        methods = ('unet_multigrid', 'thomas') if incluir_thomas else ('unet_multigrid',)
        for n in values:
            for method in methods:
                key = (disc, n, method)
                emit(dict(tipo='inicio', chave=key, inicio=time.perf_counter()))
                try:
                    local = deepcopy(cfg)
                    local.update(N=n, discretizacao=disc, metodo_fonte=method,
                                 permitir_thomas_comparacao=(method == 'thomas'))
                    def progress(it, maximum, k, ek, ephi, key=key):
                        emit(dict(tipo='progresso', chave=key, iteracao=it,
                                  max_iter=maximum, k_eff=k, erro_k=ek, erro_phi=ephi))
                    local['progress_callback'] = progress
                    solver = SolverDifusaoAI4PDEs(**local)
                    solver.resolver()
                    found[key] = solver
                    emit(dict(tipo='resultado', chave=key, solver=solver))
                except Exception as exc:
                    errors[key] = str(exc)
                    emit(dict(tipo='erro', chave=key, mensagem=str(exc)))
        return found, errors

    start = time.perf_counter()
    if effective == 'paralelo':
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix='MDF-MEF') as pool:
            futures = [pool.submit(profile, d) for d in discs]
            for future in as_completed(futures):
                found, errors = future.result()
                result.solvers.update(found)
                result.erros.update(errors)
    else:
        for disc in discs:
            found, errors = profile(disc)
            result.solvers.update(found)
            result.erros.update(errors)
    result.tempo_parede = time.perf_counter()-start
    return result


# ============================================================================
# 9. GRAFICOS COMPARATIVOS COMUNS A GUI/PDF
# ============================================================================

"""Graficos compartilhados pela GUI e PDF: mesmos resultados, mesmos rotulos."""

import numpy as np
from matplotlib.ticker import MaxNLocator


CORES_DISCRETIZACAO = {'mdf': '#0072b2', 'mef': '#d55e00'}


def desenhar_resultados(ax, solvers, referencias, modo, aviso_tempos='', refinamento=()):
    """Sobrepoe perfis do mesmo problema; Thomas permanece referencia por perfil."""
    if not solvers:
        return
    first = next(iter(solvers.values()))
    labels = ' + '.join(d.upper() for d in solvers)
    extra_axes = []
    if modo in ('fluxo', 'comparacao'):
        for disc, solver in solvers.items():
            color = CORES_DISCRETIZACAO[disc]
            values = solver.phi if modo == 'comparacao' else solver.phi_normalizado
            status = '' if solver.convergiu else ' (NAO convergiu)'
            ax.plot(solver.x, values, color=color, lw=2,
                    marker='o' if disc=='mdf' else '^', markersize=4,
                    markerfacecolor='none', markevery=max(1,len(solver.x)//10),
                    label='Neural Physics / '+disc.upper()+f' / N={solver.N}'+status)
            ref = referencias.get(disc)
            if ref is not None:
                values_ref = ref.phi if modo == 'comparacao' else ref.phi_normalizado
                ax.plot(ref.x, values_ref, '--', color=color, lw=1.5,
                        label='Thomas / '+disc.upper()+f' / N={ref.N}'+('' if ref.convergiu else ' (NAO convergiu)'))
        if modo == 'comparacao' and first.phi_ref is not None:
            ax.plot(first.x, first.phi_ref, ':', color='#202020', lw=2, label='Analitica')
        if not first.is_homogeneo:
            xs = sorted({0., first.L} | {float(m['fim']) for m in first.materiais})
            for x in xs:
                ax.axvline(x, ls=':', color='#555555', lw=.7, alpha=.6)
                ax.text(x, .98, f'x={x:g}', transform=ax.get_xaxis_transform(),
                        rotation=90, va='top', ha='right', fontsize=7)
        ax.set(xlabel='x [cm]', ylabel='Fluxo / maximo' if modo == 'comparacao' else 'Fluxo normalizado por potencia',
               title='Comparacao da forma do fluxo' if modo == 'comparacao' else 'Distribuicao do fluxo')
    elif modo == 'convergencia':
        for disc, solver in solvers.items():
            ax.plot(range(1,len(solver.historico_k)+1), solver.historico_k,
                    color=CORES_DISCRETIZACAO[disc],
                    label=disc.upper()+('' if solver.convergiu else ' (NAO convergiu)'))
        if first.k_ref is not None:
            ax.axhline(first.k_ref, ls=':', color='#202020', label=first.fonte_ref)
        ax.set(xlabel='Iteracao externa', ylabel='k_eff', title='Convergencia de k_eff')
        ax.ticklabel_format(axis='y', useOffset=False, style='plain')
    elif modo == 'residuo_fonte':
        for disc, solver in solvers.items():
            hist = solver.historico_residuo_fonte_ultima_chamada
            if hist:
                ax.semilogy(range(1,len(hist)+1), np.maximum(hist, np.finfo(float).tiny),
                            color=CORES_DISCRETIZACAO[disc], label=disc.upper())
        ax.set(xlabel='Iteracao interna (ultima chamada de cada perfil)', ylabel='Residuo relativo',
               title='Residuo da ultima fonte fixa')
    elif modo == 'tempo_metodo':
        data = [(d, s, 'NP') for d,s in solvers.items()]
        data += [(d, s, 'Thomas') for d,s in referencias.items()]
        x = np.arange(len(data))
        phases = [('Montagem','tempo_montagem_total_s','#0072b2'),
                  ('Transferencia','tempo_transferencia_total_s','#009e73'),
                  ('Iteracao','tempo_iteracao_numerica_s','#d55e00')]
        for i,(name,key,color) in enumerate(phases):
            values = [max(s.tempos_detalhados[key],np.finfo(float).tiny) for _,s,_ in data]
            ax.bar(x+(i-1)*.24, values, width=.24, label=name, color=color)
        ax.set_xticks(x, [d.upper()+' / '+name+f'\nN={s.N}' for d,s,name in data])
        ax.set_yscale('log')
        ax.set(ylabel='Tempo [s]', title='Tempos separados por configuracao')
        if 'paralela' in aviso_tempos.lower():
            ax.text(.01,.99,'Tempos sob concorrencia\nNao representam desempenho isolado',
                    transform=ax.transAxes, va='top', fontsize=7)
    elif modo in ('tabela_N', 'escalabilidade'):
        error_axis = ax.twinx() if modo == 'tabela_N' and refinamento else None
        if error_axis is not None:
            error_axis.set_ylabel('Erro k (%)')
            extra_axes.append(error_axis)
        for disc in ('mdf','mef'):
            rows = sorted((r for r in refinamento if r['discretizacao'].lower() == disc), key=lambda r:r['N'])
            if not rows:
                continue
            good = [r for r in rows if r.get('convergiu',True)]
            bad = [r for r in rows if not r.get('convergiu',True)]
            key = 'k_eff' if modo == 'tabela_N' else 'tempo_total'
            ax.plot([r['N'] for r in good], [r[key] for r in good], 'o-',
                    color=CORES_DISCRETIZACAO[disc], label=disc.upper())
            if bad:
                ax.plot([r['N'] for r in bad], [r[key] for r in bad], 'x',
                        color=CORES_DISCRETIZACAO[disc], label=disc.upper()+' (NAO convergiu)')
            if error_axis is not None:
                known = [r for r in good if r.get('erro_k') is not None]
                error_axis.plot([r['N'] for r in known], [100*r['erro_k'] for r in known],
                                's--', color=CORES_DISCRETIZACAO[disc], alpha=.7,
                                label='Erro k / '+disc.upper())
        ax.set(xlabel='N', ylabel='k_eff' if modo == 'tabela_N' else 'Tempo total [s]',
               title='Refinamento de malha' if modo == 'tabela_N' else 'Escalabilidade: N vs tempo')
        ax.set_xscale('linear')
        meshes = sorted({r['N'] for r in refinamento})
        if len(meshes) <= 8:
            ax.set_xticks(meshes, [str(n) for n in meshes], rotation=35, ha='right', fontsize=8)
        else:
            ax.xaxis.set_major_locator(MaxNLocator(nbins=7, integer=True))
        if modo == 'escalabilidade':
            ax.set_yscale('log')
            if 'paralela' in aviso_tempos.lower():
                ax.text(.01,.99,'Tempos sob concorrencia', transform=ax.transAxes, va='top', fontsize=7)
    elif modo in ('todos','materiais'):
        for values, name, color in [(first.D_arr,'D','#0072b2'),(first.Sigma_a_arr,'Sigma_a','#009e73'),
                                     (first.nuSigma_f_arr,'nuSigma_f','#d55e00')]:
            ax.plot(first.x, values, label=name, color=color)
        for x in sorted({0.,first.L} | {float(m['fim']) for m in first.materiais}):
            ax.axvline(x,ls=':',color='#555555',lw=.8,alpha=.6)
            ax.text(x,.98,f'x={x:g}',transform=ax.get_xaxis_transform(),
                    rotation=90,va='top',ha='right',fontsize=7)
        ax.set(xlabel='x [cm]', ylabel='Coeficientes materiais', title='Materiais comuns aos perfis')
    ax.set_title(ax.get_title()+' / '+labels)
    ax.grid(True, alpha=.25)
    handles, names = ax.get_legend_handles_labels()
    for other in extra_axes:
        h, n = other.get_legend_handles_labels()
        handles += h
        names += n
    if handles:
        ax.legend(handles,names,fontsize=8)


# ============================================================================
# 10. INTERFACE TKINTER E RELATORIOS
# ============================================================================

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


# ============================================================================
# 11. EXPERIMENTOS E EXPORTACAO PARA ARTIGO
# ============================================================================

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
        precisao=args.precisao,
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
        'Precisao comum: ' + caminhos.get('precisao', 'float64') + '. Mesmo banco Conv1d, ciclo V, malhas e transferencias para MDF/MEF.',
        'MEF: A=H^-1(K+M_a), F=H^-1 M_f; R=Hc^-1 P^T Hf para ambos. A massa permanece consistente.',
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
        story.append(Paragraph('Precisao comum: ' + caminhos.get('precisao', 'float64') + '. Mesmo banco Conv1d, ciclo V e transferencias. '
                               'Consulte nos CSVs os indicadores de convergencia antes de usar os resultados.', normal))
        story.append(Paragraph('MEF: A=H^-1(K+M_a), F=H^-1 M_f, sem lumping. Para ambos, R=Hc^-1 P^T Hf. '
                               'Ciclo V: ate 6 niveis, 2 passos pre/pos e 20 passos Jacobi no grosso; sem solver direto.', normal))
        story.append(Spacer(1, 0.3*cm))
        story.append(Paragraph("Comparações principais", h2))
        data = [["Caso", "N", "k NP", "k Thomas", "erro k", "phi inf", "phi L2", "Conv."]]
        for r in comparacoes:
            data.append([r['caso_id'], r['N'], f"{r['k_np']:.8f}", f"{r['k_thomas']:.8f}", f"{r['erro_k_np_thomas']:.2e}", f"{r['erro_fluxo_inf_np_thomas']:.2e}", f"{r['erro_fluxo_l2_np_thomas']:.2e}", 'sim' if r['convergiu_np'] and r['convergiu_thomas'] else 'NAO'])
        largura = A4[0] - 3.2*cm
        tbl = Table(data, repeatRows=1,
                    colWidths=[largura*w for w in (.16,.04,.12,.12,.12,.12,.12,.08)])
        tbl.setStyle(TableStyle([
            ('GRID', (0,0), (-1,-1), 0.25, colors.grey),
            ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
            ('FONTSIZE', (0,0), (-1,-1), 7),
            ('LEFTPADDING', (0,0), (-1,-1), 3),
            ('RIGHTPADDING', (0,0), (-1,-1), 3),
        ]))
        story.append(tbl)
        story.append(PageBreak())
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
        tbl_tempo = Table(data_tempo, repeatRows=1,
                          colWidths=[largura*w for w in (.16,.04,.14,.132,.132,.132,.132,.132)])
        tbl_tempo.setStyle(TableStyle([
            ('GRID', (0,0), (-1,-1), 0.25, colors.grey),
            ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
            ('FONTSIZE', (0,0), (-1,-1), 6),
            ('LEFTPADDING', (0,0), (-1,-1), 3),
            ('RIGHTPADDING', (0,0), (-1,-1), 3),
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
    caminhos['precisao'] = args.precisao
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
                tr = {"caso_id": caso_id, "discretizacao": args.discretizacao,
                      "precisao": args.precisao, "N": N, "metodo": metodo}
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
            "precisao": np_solver.precisao,
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
    caminhos['precisao'] = args.precisao
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
    parser.add_argument('--precisao', choices=['float64', 'float32'], default='float64',
                        help='mesma precisao para MDF e MEF (padrao float64)')
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
    if args.experimento_artigo:
        executar_experimento_artigo(args)
        return

    # Padrão: abrir GUI, preservando o comportamento da versão anterior.
    root = tk.Tk()
    app = DifusaoGUI_AI4PDEs(root)
    app.selecionar_discretizacoes(('mdf','mef') if args.discretizacao == 'ambos' else (args.discretizacao,))
    app.precisao_var.set(args.precisao)
    root.mainloop()


if __name__ == '__main__':
    main()
