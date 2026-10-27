"""Exportacao da campanha dupla com identificacao e flags de convergencia."""
import csv
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from neutron_ai4pdes.experiments import construir_parser, executar_experimento_artigo


def test_exportacao_dupla_preserva_discretizacao_e_nao_convergencia(tmp_path):
    args = construir_parser().parse_args([
        '--experimento-artigo', '--discretizacao', 'ambos', '--saida', str(tmp_path),
        '--malhas-homogeneo', '1', '--malhas-heterogeneo', '3',
        '--max-iter', '2', '--max-fonte', '5',
    ])
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        executar_experimento_artigo(args)
    finally:
        torch.set_num_threads(threads)
    with (tmp_path/'csv'/'resultados_consolidados.csv').open(encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 8
    assert {r['Discretizacao'] for r in rows} == {'MDF', 'MEF'}
    assert {r['Precisao'] for r in rows} == {'torch.float64'}
    assert {r['Ciclo'] for r in rows if r['metodo_id'] == 'unet_multigrid'} == {'V'}
    # N=1 com MEF tem um unico grau livre: pode convergir em duas iteracoes.
    # O heterogeneo deliberadamente limitado deve permanecer sinalizado.
    assert all(r['Bateu max_iter externo'] == 'True' for r in rows if 'Heterog' in r['Caso'])
    assert all(r['Bateu max_iter externo'] == 'False' for r in rows
               if r['Discretizacao'] == 'MEF' and r['N'] == '1')
    required = ['resultados_consolidados.csv', 'comparacao_neuralphysics_thomas.csv',
                'comparacao_couto_2003.csv', 'refinamento_malha.csv', 'tempos_execucao.csv']
    for name in required:
        assert (tmp_path/'csv'/name).exists()
    for folder in (tmp_path, tmp_path/'mdf', tmp_path/'mef'):
        assert (folder/'relatorios'/'relatorio_experimental.pdf').stat().st_size > 1000
    assert list((tmp_path/'mef'/'figuras_artigo').glob('*.png'))
    assert (tmp_path/'tabelas_latex'/'tab_refinamento_malha.tex').exists()
