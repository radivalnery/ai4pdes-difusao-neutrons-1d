"""Contrato experimental: muda o stencil, nao o algoritmo multigrid."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from neutron_ai4pdes.experiments import problemas_artigo
from neutron_ai4pdes.models import SolverFonteFixaUNet1D
from neutron_ai4pdes.solver import SolverDifusaoAI4PDEs


@pytest.fixture(scope='module', autouse=True)
def threads_para_testes():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def criar(discretizacao, caso='homogeneo', N=13, precisao='float64', **kwargs):
    cfg = dict(problemas_artigo()[caso])
    cfg.pop('nome')
    solver = SolverDifusaoAI4PDEs(**cfg, N=N, discretizacao=discretizacao,
                                  precisao=precisao, dispositivo_preferido='cpu', **kwargs)
    solver.criar_operador_A()
    return solver


@pytest.mark.parametrize('precisao', ['float32', 'float64'])
@pytest.mark.parametrize('caso,N', [('homogeneo', 13), ('heterogeneo', 48)])
def test_mesmo_ciclo_precisao_malhas_e_transferencias(precisao, caso, N):
    mdf, mef = [criar(d, caso, N, precisao) for d in ('mdf', 'mef')]
    a, b = mdf.solver_fonte_fixa, mef.solver_fonte_fixa
    assert type(a) is type(b) is SolverFonteFixaUNet1D
    for name in ('omega', 'amortecimento_unet', 'max_niveis', 'min_pontos_grosso',
                 'pre_suavizacoes', 'post_suavizacoes', 'suavizacoes_grosso'):
        assert getattr(a, name) == getattr(b, name)
    assert len(a.operadores) == len(b.operadores)
    for op1, op2 in zip(a.operadores, b.operadores):
        assert op1.x.dtype == op2.x.dtype == getattr(torch, precisao)
        torch.testing.assert_close(op1.x, op2.x, rtol=0, atol=0)
        torch.testing.assert_close(op1.convolucao.weight, op2.convolucao.weight, rtol=0, atol=0)
        assert all(not p.requires_grad for op in (op1, op2) for p in op.parameters())
    for t1, t2 in zip(a.transferencias, b.transferencias):
        for name in ('left', 'right', 'weight', 'hf', 'hc'):
            torch.testing.assert_close(getattr(t1, name), getattr(t2, name), rtol=0, atol=0)


@pytest.mark.parametrize('discretizacao', ['mdf', 'mef'])
@pytest.mark.parametrize('precisao', ['float32', 'float64'])
@pytest.mark.parametrize('caso', ['homogeneo', 'heterogeneo'])
def test_convolucao_aplica_tres_bandas_de_A_e_F(discretizacao, precisao, caso):
    solver = criar(discretizacao, caso, 12, precisao)
    op = solver.modelo_A
    phi = torch.linspace(.2, 1., 13, dtype=solver.dtype)
    calls = []
    hook = op.convolucao.register_forward_hook(lambda *args: calls.append(1))
    a_phi, f_phi = op(phi), solver.fonte_fissao(phi)
    hook.remove()
    assert len(calls) == 2
    for coef, actual in ((op.a_coef, a_phi), (op.f_coef, f_phi)):
        coef = coef.numpy()
        matrix = np.diag(coef[:, 1])+np.diag(coef[1:, 0], -1)+np.diag(coef[:-1, 2], 1)
        np.testing.assert_allclose(actual.numpy(), matrix@phi.numpy(),
                                   atol=3e-6 if precisao == 'float32' else 2e-14)


def test_mef_escalonado_preserva_matrizes_e_autovalor():
    solver = criar('mef', N=12)
    op = solver.modelo_A
    H = op.medida.numpy()
    a, f = op.a_coef.numpy(), op.f_coef.numpy()
    A = np.diag(a[:, 1])+np.diag(a[1:, 0], -1)+np.diag(a[:-1, 2], 1)
    F = np.diag(f[:, 1])+np.diag(f[1:, 0], -1)+np.diag(f[:-1, 2], 1)
    rawA = np.diag(op.a_diag.numpy())+np.diag(op.a_off.numpy(), -1)+np.diag(op.a_off.numpy(), 1)
    rawF = np.diag(op.f_diag.numpy())+np.diag(op.f_off.numpy(), -1)+np.diag(op.f_off.numpy(), 1)
    np.testing.assert_allclose((H[:, None]*A)[:-1, :-1], rawA[:-1, :-1])
    np.testing.assert_allclose(H[:, None]*F, rawF)
    assert np.all(op.f_off.numpy() > 0)  # F permanece consistente, nao diagonal.
    eig_scaled = np.linalg.eigvals(np.linalg.solve(A[:-1, :-1], F[:-1, :-1])).real.max()
    eig_raw = np.linalg.eigvals(np.linalg.solve(rawA[:-1, :-1], rawF[:-1, :-1])).real.max()
    assert eig_scaled == pytest.approx(eig_raw, abs=1e-14)


@pytest.mark.parametrize('discretizacao', ['mdf', 'mef'])
def test_fonte_fixa_e_potencia_na_mesma_escala(discretizacao):
    solver = criar(discretizacao, N=26, tol_fonte=1e-10, tol_k=1e-9, tol_phi=1e-8)
    solver.resolver()
    assert solver.convergiu and all(solver.convergiu_fonte_fixa)
    assert 3.2e-11*solver.integral_fissao(solver.phi_normalizado) == pytest.approx(100.)
    mg = solver.solver_fonte_fixa
    exact = solver.modelo_A.aplicar_contorno_fluxo(torch.linspace(1., 0., 27, dtype=solver.dtype))
    rhs = solver.modelo_A(exact)
    result1 = mg.resolver(rhs, tol=1e-9)
    result2 = mg.resolver(rhs*1e-8, tol=1e-9)
    assert result1.convergiu and result2.convergiu
    assert result1.iteracoes == result2.iteracoes
    np.testing.assert_allclose(result2.phi.numpy()/1e-8, result1.phi.numpy(), atol=1e-12)
