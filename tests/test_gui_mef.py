"""Smoke da GUI: selecao, parametros e geracao real da tabela, sem janela visivel."""

from pathlib import Path
import sys
import time

import pytest
import torch
import tkinter as tk

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from neutron_ai4pdes.gui import DifusaoGUI_AI4PDEs


@pytest.mark.gui
def test_tkinter_mef_parametros_e_tabela(tmp_path, monkeypatch):
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f'Tk indisponivel neste ambiente: {exc}')
    root.withdraw()
    monkeypatch.setattr(root, 'state', lambda *args: 'withdrawn')
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    errors = []
    monkeypatch.setattr('neutron_ai4pdes.gui.messagebox.showerror', lambda *a, **k: errors.append(a))
    monkeypatch.setattr('neutron_ai4pdes.gui.messagebox.showinfo', lambda *a, **k: None)
    monkeypatch.chdir(tmp_path)
    try:
        app = DifusaoGUI_AI4PDEs(root)
        app.discretizacao_var.set('MEF')
        assert app.obter_configuracao_solver()['discretizacao'] == 'mef'
        assert app.obter_configuracao_solver()['precisao'] == 'float64'
        app.carregar_problema('Heterogêneo (Couto, 2003)')
        assert app.obter_lista_N_tabela() == [3,12,48,96,192,300,384]
        assert app.pontos_entry.get() == '0.0, 50.0, 100.0, 150.0'
        app.carregar_problema('Homogêneo (Lamarsh)')
        assert app.obter_lista_N_tabela() == [1,13,26,52,100,104]
        app.N_tabela_entry.delete(0, tk.END)
        app.N_tabela_entry.insert(0, '1,13')
        deadline = time.monotonic()+40
        def check():
            if len(app.tree.get_children()) == 6 or errors or time.monotonic() >= deadline:
                root.after(50, root.quit)
            else:
                root.after(20, check)
        root.after(0, app.gerar_tabela_variando_N_thread)
        root.after(20, check)
        root.mainloop()
        assert not errors
        assert len(app.tree.get_children()) == 6  # Uma linha por par (N,x).
        assert all(r['discretizacao'] == 'MEF' for r in app.resultados_tabela)
        assert all(r['precisao'] == 'float64' for r in app.resultados_tabela)
        assert all(r['erro_fluxo_pontos'][26.] is None for r in app.resultados_tabela)
        coarse = app.resultados_tabela[0]
        assert coarse['fluxos_pontos'][13.] == pytest.approx(coarse['fluxos_pontos'][0.]/2)
        assert (tmp_path/'resultados_refinamento.csv').exists()
        # Executa o mesmo caminho do botao 'ambos', mantendo a GUI invisivel.
        app.N_entry.delete(0, tk.END)
        app.N_entry.insert(0, '13')
        app.metodo_fonte_var.set('ambos')
        deadline = time.monotonic()+40
        def check_solver():
            if errors or (app.solver is not None and not app.executando) or time.monotonic() >= deadline:
                root.after(50, root.quit)
            else:
                root.after(20, check_solver)
        root.after(0, app.iniciar_execucao)
        root.after(20, check_solver)
        root.mainloop()
        assert not errors
        assert not app.executando and app.solver.convergiu and app.solver_thomas.convergiu
        assert app.solver.discretizacao == app.solver_thomas.discretizacao == 'mef'
        for mode in ('fluxo', 'comparacao', 'tempo_metodo'):
            app.modo_grafico.set(mode)
            app.atualizar_grafico()
            assert 'MEF' in app.ax.get_title()
            if mode == 'tempo_metodo':
                assert app.ax.get_yscale() == 'log'
            else:
                assert 'Thomas' in app.ax.get_legend_handles_labels()[1]
        # O gerador PDF da GUI deve consumir os resultados capturados do solver.
        target = tmp_path/'relatorio_tkinter_mef.pdf'
        monkeypatch.setattr('neutron_ai4pdes.gui.filedialog.asksaveasfilename', lambda **kw: str(target))
        app.gerar_relatorio()
        assert not errors
        assert target.exists() and target.stat().st_size > 1000
    finally:
        root.destroy()
        torch.set_num_threads(threads)
