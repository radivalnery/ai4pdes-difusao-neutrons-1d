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
