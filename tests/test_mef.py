"""Verificacoes independentes da formulacao P1 e da integracao com a aplicacao."""

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from neutron_ai4pdes.fem import OperadorMEF1D, SolverFonteFixaMEF1D
from neutron_ai4pdes.solver import SolverDifusaoAI4PDEs
from neutron_ai4pdes.experiments import problemas_artigo, construir_parser


@pytest.fixture(scope='module', autouse=True)
def uma_thread_nos_testes_pequenos():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def config(caso='homogeneo', N=13):
    cfg = dict(problemas_artigo()[caso])
    cfg.pop('nome')
    cfg.update(N=N, discretizacao='mef', dispositivo_preferido='cpu',
               tol_k=1e-8, tol_phi=1e-7, tol_fonte=1e-9, max_iter=900)
    return cfg


def dense(op):
    n = len(op.x)
    return np.column_stack([op(torch.eye(n, dtype=torch.float64)[:, i]).numpy()
                            for i in range(n)])


def test_matrizes_elementares_consistentes():
    mats = [dict(inicio=0., fim=2., D=3., Sigma_a=.2, nuSigma_f=.3)]
    op = OperadorMEF1D([0., 2.], mats, 'reflexiva', 'reflexiva')
    stiffness = 1.5 * np.array([[1., -1.], [-1., 1.]])
    mass = 2/6 * np.array([[2., 1.], [1., 2.]])
    np.testing.assert_allclose(dense(op), stiffness+.2*mass, atol=1e-14)
    np.testing.assert_allclose(op.aplicar_fissao_numpy(np.array([.4, .7])), .3*mass@np.array([.4, .7]))
    assert list(op.parameters()) == []


def test_material_cortando_elemento_e_integrado_sem_media_nodal():
    mats = [dict(inicio=0., fim=.4, D=1., Sigma_a=0., nuSigma_f=.2),
            dict(inicio=.4, fim=1., D=3., Sigma_a=0., nuSigma_f=.5)]
    op = OperadorMEF1D([0., 1.], mats, 'reflexiva', 'reflexiva')
    np.testing.assert_allclose(dense(op), 2.2*np.array([[1., -1.], [-1., 1.]]))
    assert np.sum(op.aplicar_fissao_numpy(np.ones(2))) == pytest.approx(.4*.2+.6*.5)


def test_transferencias_adjuntas_e_geometria_impar():
    cfg = config(N=13)
    op = OperadorMEF1D(np.linspace(0., 26., 14), cfg['materiais'])
    mg = SolverFonteFixaMEF1D(op)
    assert len(mg.operadores) > 1
    for level, transfer in enumerate(mg.transferencias):
        fine, coarse = mg.operadores[level], mg.operadores[level+1]
        assert coarse.x[-1] == fine.x[-1] == 26.
        np.testing.assert_allclose(transfer.prolongar(coarse.x).numpy(), fine.x.numpy())
        a = torch.linspace(.2, .9, len(fine.x), dtype=torch.float64)
        b = torch.linspace(.1, .7, len(coarse.x), dtype=torch.float64)
        assert torch.dot(a, transfer.prolongar(b)).item() == pytest.approx(torch.dot(transfer.restringir(a), b).item())


@pytest.mark.parametrize('caso,N', [('homogeneo', 104), ('heterogeneo', 384)])
def test_fonte_fixa_mef_nas_maiores_malhas(caso, N):
    cfg = config(caso, N)
    x = np.linspace(0., cfg['L'], N+1)
    op = OperadorMEF1D(x, cfg['materiais'])
    mg = SolverFonteFixaMEF1D(op)
    expected = torch.cos(torch.tensor(x, dtype=torch.float64) * np.pi/(2*cfg['L']))
    expected[-1] = 0.
    result = mg.resolver(op(expected), tol=1e-9, max_iter=5000)
    assert result.convergiu
    np.testing.assert_allclose(result.phi.numpy(), expected.numpy(), atol=3e-7)


@pytest.mark.parametrize('contornos', [('reflexiva','vácuo'), ('vácuo','reflexiva'),
                                     ('vácuo','vácuo'), ('reflexiva','reflexiva')])
def test_autovalor_mef_com_oraculo_denso_e_contornos(contornos):
    cfg = config(N=12)
    cfg.update(cond_esquerda=contornos[0], cond_direita=contornos[1])
    solver = SolverDifusaoAI4PDEs(**cfg)
    solver.resolver()
    assert solver.convergiu
    op = solver.modelo_A
    A = dense(op)
    eye = np.eye(len(op.x))
    F = np.column_stack([op.aplicar_fissao_numpy(eye[:, i]) for i in range(len(op.x))])
    free = np.arange(len(op.x))
    if contornos[0] == 'vácuo':
        free = free[1:]
        assert solver.phi[0] == 0.
    if contornos[1] == 'vácuo':
        free = free[:-1]
        assert solver.phi[-1] == 0.
    expected = max(np.linalg.eigvals(np.linalg.solve(A[np.ix_(free,free)], F[np.ix_(free,free)])).real)
    assert solver.k_eff == pytest.approx(expected, rel=2e-7)
    assert solver.k_eff == pytest.approx(solver.k_ref, rel=2e-3)
    power = 3.2e-11 * solver.integral_fissao(solver.phi_normalizado)
    assert power == pytest.approx(100., rel=1e-12)


@pytest.mark.parametrize('N', [3, 12, 48])
def test_heterogeneo_mef_multigrid_vs_thomas(N):
    cfg = config('heterogeneo', N)
    unet = SolverDifusaoAI4PDEs(**cfg)
    thomas = SolverDifusaoAI4PDEs(**cfg, metodo_fonte='thomas', permitir_thomas_comparacao=True)
    unet.resolver()
    thomas.resolver()
    assert unet.convergiu and thomas.convergiu
    assert unet.k_eff == pytest.approx(thomas.k_eff, rel=3e-7)
    np.testing.assert_allclose(unet.phi, thomas.phi, atol=3e-6)
    assert all(unet.convergiu_fonte_fixa)
    assert 3.2e-11*unet.integral_fissao(unet.phi_normalizado) == pytest.approx(100.)


def test_refinamento_mef_aproxima_analitico_com_ordem_dois():
    errors = []
    for N in (4, 8, 16):
        solver = SolverDifusaoAI4PDEs(**config(N=N), metodo_fonte='thomas', permitir_thomas_comparacao=True)
        solver.resolver()
        errors.append(abs(solver.k_eff-solver.k_ref))
    assert errors[0]/errors[1] > 3.8
    assert errors[1]/errors[2] > 3.8


def test_thomas_proibido_no_ciclo_multigrid(monkeypatch):
    def proibido(*args, **kwargs):
        raise AssertionError('Thomas foi chamado dentro do caminho principal')
    monkeypatch.setattr('neutron_ai4pdes.solver.resolver_tridiagonal_thomas', proibido)
    solver = SolverDifusaoAI4PDEs(**config(N=26))
    solver.resolver()
    assert solver.convergiu and len(solver.solver_fonte_fixa.operadores) > 1
    with pytest.raises(RuntimeError, match='referencia'):
        SolverDifusaoAI4PDEs(**config(N=3), metodo_fonte='thomas').resolver()


def test_mef_nao_convergido_e_tempos_separados():
    cfg = config(N=26)
    cfg.update(max_iter_fonte=1, max_iter=2)
    solver = SolverDifusaoAI4PDEs(**cfg)
    solver.resolver()
    assert not solver.convergiu
    assert not all(solver.convergiu_fonte_fixa)
    t = solver.tempos_detalhados
    assert t['tempo_montagem_total_s'] > 0 and t['tempo_transferencia_total_s'] > 0
    assert t['tempo_iteracao_numerica_s'] > 0
    assert (t['tempo_montagem_total_s'] + t['tempo_transferencia_total_s']
            + t['tempo_iteracao_numerica_s']) <= t['tempo_total_s']*1.01


def test_arquivo_unico_tem_mesmo_resultado_e_cli():
    path = ROOT / 'difusao_neutrons_neuralphysics_1d_mdf_mef_v4_2.py'
    spec = importlib.util.spec_from_file_location('standalone_v42_test', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    cfg = config(N=13)
    modular = SolverDifusaoAI4PDEs(**cfg)
    unico = module.SolverDifusaoAI4PDEs(**cfg)
    modular.resolver()
    unico.resolver()
    assert modular.k_eff == unico.k_eff
    np.testing.assert_array_equal(modular.phi, unico.phi)
    parser = construir_parser()
    assert parser.parse_args([]).malhas_homogeneo == '1,13,26,52,100,104'
    assert parser.parse_args([]).malhas_heterogeneo == '3,12,48,96,192,300,384'
