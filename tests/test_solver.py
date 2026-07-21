import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neutron_ai4pdes.solver import SolverDifusaoAI4PDEs


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
    solver = SolverDifusaoAI4PDEs(**cfg_homogeneo(), metodo_fonte="unet_multigrid")
    solver.resolver()
    assert abs(solver.k_eff - 0.99724257) < 1e-5


@pytest.mark.slow
def test_heterogeneo_regressao_keff_unet_multigrid():
    solver = SolverDifusaoAI4PDEs(**cfg_heterogeneo(), metodo_fonte="unet_multigrid")
    solver.resolver()
    assert abs(solver.k_eff - 1.09508219) < 1e-5


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
