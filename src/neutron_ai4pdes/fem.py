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
from .stencils import OperadorStencil1D, medidas_nodais

from .models import SolverFonteFixaUNet1D


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
