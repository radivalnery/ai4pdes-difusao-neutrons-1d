"""Controle de concorrencia e isolamento; nenhum teste pressupoe ganho de tempo."""
from pathlib import Path
import sys
import threading
import importlib.util

import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from neutron_ai4pdes.campaigns import executar_campanha
from neutron_ai4pdes.experiments import problemas_artigo


@pytest.fixture(scope='module',autouse=True)
def uma_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def config(case='homogeneo'):
    cfg = dict(problemas_artigo()[case])
    cfg.pop('nome')
    return dict(cfg,N=13 if case=='homogeneo' else 12,dispositivo_preferido='cpu')


@pytest.mark.parametrize('case',['homogeneo','heterogeneo'])
def test_sequencial_e_paralelo_produzem_mesmos_resultados(case):
    cfg = config(case)
    threads = torch.get_num_threads()
    a = executar_campanha(cfg,('mdf','mef'),incluir_thomas=True)
    b = executar_campanha(cfg,('mdf','mef'),modo='paralelo',incluir_thomas=True)
    assert torch.get_num_threads() == threads
    assert a.modo_efetivo == 'sequencial' and b.modo_efetivo == 'paralelo'
    assert not a.erros and not b.erros
    assert len(a.solvers) == len(b.solvers) == 4
    for key in a.chaves():
        left,right = a.solvers[key],b.solvers[key]
        assert left.convergiu and right.convergiu
        assert left.k_eff == right.k_eff
        np.testing.assert_array_equal(left.phi,right.phi)
        assert left.materiais is not right.materiais
    for row in b.resumos():
        assert row['modo_execucao'] == 'paralelo'
        assert 'nao representam desempenho isolado' in row['aviso_tempos']
        assert row['erro_fluxo_global_inf'] is not None
        assert row['tempo_montagem_total_s'] > 0 and row['tempo_iteracao_numerica_s'] > 0


def test_duas_discretizacoes_realmente_sobrepoem_trabalho(monkeypatch):
    rendezvous = threading.Barrier(2)
    ids = set()
    lock = threading.Lock()
    class Probe:
        def __init__(self,**cfg):
            self.cfg = cfg
        def resolver(self):
            with lock:
                ids.add(threading.get_ident())
            rendezvous.wait(timeout=5)
            self.cfg['materiais'][0]['D'] = 999.
    monkeypatch.setattr('neutron_ai4pdes.campaigns.SolverDifusaoAI4PDEs',Probe)
    cfg = config()
    result = executar_campanha(cfg,('mdf','mef'),modo='paralelo')
    assert not result.erros and len(ids) == 2
    assert cfg['materiais'][0]['D'] == .9
    assert result.configuracao['materiais'][0]['D'] == .9
    one,two = result.solvers.values()
    assert one.cfg['materiais'] is not two.cfg['materiais']


def test_sequencial_respeita_ordem_e_registra_erro_sem_descartar_outro(monkeypatch):
    calls = []
    class Probe:
        def __init__(self,**cfg):
            self.cfg = cfg
        def resolver(self):
            calls.append((self.cfg['discretizacao'],self.cfg['N']))
            if self.cfg['discretizacao']=='mef':
                raise RuntimeError('falha simulada')
    monkeypatch.setattr('neutron_ai4pdes.campaigns.SolverDifusaoAI4PDEs',Probe)
    events = []
    result = executar_campanha(config(),('mdf','mef'),malhas=[1,13],evento=events.append)
    assert calls == [('mdf',1),('mdf',13),('mef',1),('mef',13)]
    assert len(result.solvers) == len(result.erros) == 2
    assert all('falha simulada' in msg for msg in result.erros.values())
    assert sum(e['tipo']=='resultado' for e in events) == 2
    assert sum(e['tipo']=='erro' for e in events) == 2
    assert len(result.metadados()['erros']) == 2


def test_um_perfil_e_falha_de_convergencia_explicitos():
    cfg = config()
    cfg.update(max_iter=2,max_iter_fonte=1)
    result = executar_campanha(cfg,('mef',),modo='paralelo')
    assert result.modo_solicitado == 'paralelo' and result.modo_efetivo == 'sequencial'
    row = result.resumos()[0]
    assert row['Bateu max_iter externo']
    assert not row['Fonte fixa convergiu (todas as chamadas)']
    assert row['erro_fluxo_global_inf'] is None  # Nao inventar erro sem referencia.


@pytest.mark.parametrize('discs,mode',[((), 'sequencial'),(('invalido',),'sequencial'),(('mdf',),'errado')])
def test_validacao_de_selecao(discs,mode):
    with pytest.raises(ValueError):
        executar_campanha(config(),discs,modo=mode)


def test_arquivo_unico_executa_campanha_paralela_identica():
    path = Path(__file__).resolve().parents[1]/'difusao_neutrons_neuralphysics_1d_mdf_mef_v4_2.py'
    spec = importlib.util.spec_from_file_location('standalone_campaign_test',path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    cfg = config()
    modular = executar_campanha(cfg,('mdf','mef'),modo='paralelo')
    unico = module.executar_campanha(cfg,('mdf','mef'),modo='paralelo')
    for key in modular.chaves():
        assert modular.solvers[key].k_eff == unico.solvers[key].k_eff
        np.testing.assert_array_equal(modular.solvers[key].phi,unico.solvers[key].phi)
