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
