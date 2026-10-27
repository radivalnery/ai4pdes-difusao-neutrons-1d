import sys
from pathlib import Path

import pytest
import torch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neutron_ai4pdes.solver import SolverDifusaoAI4PDEs


@pytest.fixture(scope='module', autouse=True)
def uma_thread_para_validacao():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def cfg_homogeneo(N=100, max_iter=1000, max_iter_fonte=5000):
    return dict(
        L=26.0,
        N=N,
        materiais=[{"inicio": 0.0, "fim": 26.0, "D": 0.9, "Sigma_a": 0.065, "nuSigma_f": 0.0681}],
        cond_esquerda="reflexiva",
        cond_direita="vácuo",
        tol_k=1e-6,
        tol_phi=1e-5,
        max_iter=max_iter,
        potencia_nominal=100.0,
        pontos_interesse=[0.0, 13.0, 26.0],
        omega_fonte=0.75,
        amortecimento_unet=0.20,
        tol_fonte=1e-5,
        max_iter_fonte=max_iter_fonte,
    )


def cfg_heterogeneo(N=300, max_iter=1000, max_iter_fonte=5000):
    return dict(
        L=150.0,
        N=N,
        materiais=[
            {"inicio": 0.0, "fim": 50.0, "D": 1.333333, "Sigma_a": 0.200000, "nuSigma_f": 0.220000},
            {"inicio": 50.0, "fim": 100.0, "D": 1.333333, "Sigma_a": 0.240000, "nuSigma_f": 0.250000},
            {"inicio": 100.0, "fim": 150.0, "D": 2.777777, "Sigma_a": 0.110000, "nuSigma_f": 0.080000},
        ],
        cond_esquerda="reflexiva",
        cond_direita="vácuo",
        tol_k=1e-6,
        tol_phi=1e-5,
        max_iter=max_iter,
        potencia_nominal=100.0,
        pontos_interesse=[0.0, 50.0, 100.0, 150.0],
        omega_fonte=0.75,
        amortecimento_unet=0.20,
        tol_fonte=1e-5,
        max_iter_fonte=max_iter_fonte,
    )


@pytest.mark.slow
def test_homogeneo_regressao_keff_unet_multigrid():
    cfg = cfg_homogeneo()
    cfg.update(tol_k=1e-9, tol_phi=1e-8, tol_fonte=1e-10)
    solver = SolverDifusaoAI4PDEs(**cfg, metodo_fonte="unet_multigrid")
    solver.resolver()
    # Autovalor EXATO da matriz MDF uniforme, nao um numero do solver anterior.
    expected = .0681/(.065 + .9*4/solver.h**2*np.sin(np.pi/(4*solver.N))**2)
    assert solver.convergiu and all(solver.convergiu_fonte_fixa)
    assert solver.k_eff == pytest.approx(expected, abs=2e-8)


@pytest.mark.slow
def test_heterogeneo_regressao_keff_unet_multigrid():
    cfg = cfg_heterogeneo()
    cfg.update(tol_k=1e-9, tol_phi=1e-8, tol_fonte=1e-10)
    solver = SolverDifusaoAI4PDEs(**cfg, metodo_fonte="unet_multigrid")
    solver.resolver()
    classical = SolverDifusaoAI4PDEs(**cfg, metodo_fonte='thomas', permitir_thomas_comparacao=True)
    classical.resolver()
    assert solver.convergiu and classical.convergiu
    assert solver.k_eff == pytest.approx(classical.k_eff, abs=2e-8)
    assert solver.k_eff == pytest.approx(1.09506, rel=1e-4)


def test_unet_multigrid_e_thomas_keff_proximos():
    cfg = cfg_homogeneo(N=30, max_iter=80, max_iter_fonte=800)
    unet = SolverDifusaoAI4PDEs(**cfg, metodo_fonte="unet_multigrid")
    thomas = SolverDifusaoAI4PDEs(
        **cfg,
        metodo_fonte="thomas",
        permitir_thomas_comparacao=True,
    )
    unet.resolver()
    thomas.resolver()
    rel = abs(unet.k_eff - thomas.k_eff) / abs(thomas.k_eff)
    assert rel < 1e-4


def test_fonte_fixa_reporta_nao_convergencia_com_limite_baixo():
    cfg = cfg_homogeneo(N=30, max_iter=2, max_iter_fonte=5)
    solver = SolverDifusaoAI4PDEs(**cfg, metodo_fonte="unet_multigrid")
    solver.resolver()
    assert solver.convergiu_fonte_fixa
    assert any(not ok for ok in solver.convergiu_fonte_fixa)
    resumo = solver.resumo_resultado()
    assert resumo["Fonte fixa convergiu (todas as chamadas)"] is False
    assert resumo["Chamadas fonte fixa não convergidas"] > 0
