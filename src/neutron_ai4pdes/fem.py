"""MEF/Galerkin P1 para difusao de neutrons em uma dimensao.

1. Forma fraca: integral(D phi' v' + Sigma_a phi v) = integral(nuSigma_f phi v)/k.
2. Cada elemento tem dois nos. Absorcao e fissao usam massa consistente.
3. Os coeficientes sao integrados por material, inclusive quando uma interface
   corta um elemento. N continua sendo o numero de elementos, com N+1 nos.
4. As tres diagonais sao buffers PyTorch fixos: nao ha pesos treinaveis.
5. O ciclo multigrid aplica Jacobi, restricao adjunta e interpolacao linear.
   Thomas e usado somente pelo caminho externo de referencia do solver.
"""

import numpy as np
import torch
import torch.nn as nn

from .models import SolverFonteFixaUNet1D


class OperadorMEF1D(nn.Module):
    """A = K + M_a e F = M_f, com contornos homogeneos.

    Em um elemento de comprimento h e material constante:
      K_e = D/h * [[1, -1], [-1, 1]];
      M_e(c) = c*h/6 * [[2, 1], [1, 2]].
    A integracao de Gauss em cada trecho material tambem atende elementos
    cortados por interfaces. Fluxo nulo e imposto por eliminacao simetrica;
    corrente nula (reflexiva) e uma condicao natural da forma fraca.
    """

    def __init__(self, x, materiais, cond_esquerda='reflexiva', cond_direita='vácuo'):
        super().__init__()
        x = np.asarray(x, dtype=float)
        if x.ndim != 1 or len(x) < 2 or not np.all(np.diff(x) > 0):
            raise ValueError('A malha MEF deve conter pelo menos dois nos crescentes.')
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
        for name, value in [('x', x), ('a_diag', diag), ('a_off', off),
                            ('f_diag', fdiag), ('f_off', foff)]:
            self.register_buffer(name, torch.tensor(value, dtype=torch.float64))
        self._fdiag_numpy = fdiag.copy()
        self._foff_numpy = foff.copy()

    @staticmethod
    def _aplicar(diag, off, phi):
        phi = phi.reshape(-1).to(device=diag.device, dtype=diag.dtype)
        result = diag * phi
        result[:-1] += off * phi[1:]
        result[1:] += off * phi[:-1]
        return result

    def forward(self, phi):
        return self._aplicar(self.a_diag, self.a_off, phi)

    def diagonal(self):
        return self.a_diag

    def aplicar_contorno_fluxo(self, phi):
        phi = phi.clone()
        if self.cond_esquerda == 'vácuo':
            phi[0] = 0.
        if self.cond_direita == 'vácuo':
            phi[-1] = 0.
        return phi

    def aplicar_fissao_numpy(self, phi):
        """Produto M_f phi na CPU para a iteracao externa de potencia."""
        diag = self._fdiag_numpy
        off = self._foff_numpy
        result = diag * np.asarray(phi)
        result[:-1] += off * phi[1:]
        result[1:] += off * phi[:-1]
        return result

    def matriz_tridiagonal_numpy(self, rhs=None):
        diag = self.a_diag.detach().cpu().numpy().copy()
        off = self.a_off.detach().cpu().numpy().copy()
        rhs = np.zeros(len(diag)) if rhs is None else np.asarray(rhs, dtype=float).copy()
        if self.cond_esquerda == 'vácuo':
            rhs[0] = 0.
        if self.cond_direita == 'vácuo':
            rhs[-1] = 0.
        return off.copy(), diag, off.copy(), rhs


class TransferenciaMEF1D(nn.Module):
    """P interpola os valores nodais; P^T restringe os residuos integrados.

    A restricao e a adjunta da interpolacao, sem dividir por dois: o residuo
    MEF e uma carga integrada. Nenhuma matriz densa precisa ser armazenada.
    """

    def __init__(self, fino, grosso):
        super().__init__()
        right = torch.searchsorted(grosso, fino).clamp(1, grosso.numel()-1)
        left = right - 1
        weight = (fino - grosso[left]) / (grosso[right] - grosso[left])
        self.n_grosso = grosso.numel()
        self.register_buffer('left', left)
        self.register_buffer('right', right)
        self.register_buffer('weight', weight)

    def prolongar(self, phi):
        return (1-self.weight) * phi[self.left] + self.weight * phi[self.right]

    def restringir(self, residual):
        result = residual.new_zeros(self.n_grosso)
        result.index_add_(0, self.left, (1-self.weight) * residual)
        result.index_add_(0, self.right, self.weight * residual)
        return result


class SolverFonteFixaMEF1D(SolverFonteFixaUNet1D):
    """Mesmo ciclo V/U-Net; operadores MEF rediscretizados em malhas aninhadas.

    As coordenadas reais dos nos sao preservadas no engrossamento, inclusive
    com N impar. Interfaces que coincidem com nos continuam em todos os niveis.
    O nivel mais grosso usa somente Jacobi ponderado.
    """

    norma_rhs_minima = 1e-30

    def _resolver_grosso(self, operador, rhs, chute=None):
        # Se a malha inteira e grossa (N=1 ou 3), os ciclos sucessivos devem
        # continuar a suavizacao; reiniciar em zero impediria refinar o residuo.
        phi = torch.zeros_like(rhs) if chute is None else chute.clone()
        phi = operador.aplicar_contorno_fluxo(phi)
        return self._suavizar_jacobi(operador, phi, rhs, operador.diagonal(), self.suavizacoes_grosso)

    def _construir_hierarquia(self):
        self.transferencias = nn.ModuleList()
        atual = self.operador_A
        for _ in range(1, self.max_niveis):
            x = atual.x.detach().cpu().numpy()
            if len(x) < 2*self.min_pontos_grosso-1:
                break
            selected = set(range(0, len(x), 2)) | {len(x)-1}
            for m in atual.materiais:
                for key in ('inicio', 'fim'):
                    selected.update(np.flatnonzero(np.isclose(x, m[key], atol=1e-12, rtol=0)).tolist())
            indices = sorted(selected)
            if len(indices) >= len(x):
                break
            coarse = OperadorMEF1D(x[indices], atual.materiais, atual.cond_esquerda,
                                   atual.cond_direita).to(atual.x.device)
            self.transferencias.append(TransferenciaMEF1D(atual.x, coarse.x))
            self.operadores.append(coarse)
            atual = coarse

    def _restringir(self, fine, nivel=0):
        residual = self.transferencias[nivel].restringir(fine)
        return self.operadores[nivel+1].aplicar_contorno_fluxo(residual)

    def _prolongar(self, coarse, tamanho_fino, nivel=0):
        return self.transferencias[nivel].prolongar(coarse)
