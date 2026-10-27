"""Fluxo completo MDF/MEF com fila Tkinter e relatorio dos dados capturados."""
import csv
import json
from pathlib import Path
import sys
import threading
import time
import tkinter as tk

import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from neutron_ai4pdes.gui import DifusaoGUI_AI4PDEs


def esperar(root, app, seconds=40):
    deadline = time.monotonic()+seconds
    def check():
        if not app.executando or time.monotonic()>deadline:
            root.quit()
        else:
            root.after(30,check)
    root.after(40,check)
    root.mainloop()
    assert not app.executando, 'A campanha nao concluiu dentro do prazo do teste.'


@pytest.mark.gui
@pytest.mark.parametrize('mode',['Sequencial','Paralelo'])
def test_gui_selecao_dupla_e_relatorio_snapshot(tmp_path,monkeypatch,mode):
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(str(exc))
    root.withdraw()
    monkeypatch.setattr(root,'state',lambda *a:'withdrawn')
    errors, confirmations = [], []
    monkeypatch.setattr('neutron_ai4pdes.gui.messagebox.showerror',lambda *a,**k:errors.append(a))
    monkeypatch.setattr('neutron_ai4pdes.gui.messagebox.showwarning',lambda *a,**k:None)
    monkeypatch.setattr('neutron_ai4pdes.gui.messagebox.showinfo',lambda *a,**k:None)
    monkeypatch.setattr('neutron_ai4pdes.gui.messagebox.askyesno',
                        lambda *a,**k: confirmations.append(a) or True)
    monkeypatch.chdir(tmp_path)
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    main_thread = threading.get_ident()
    original_after = root.after
    def main_after(*args,**kwargs):
        assert threading.get_ident() == main_thread, 'Tkinter chamado por trabalhador!'
        return original_after(*args,**kwargs)
    monkeypatch.setattr(root,'after',main_after)
    try:
        app = DifusaoGUI_AI4PDEs(root)
        app.selecionar_discretizacoes(('mdf','mef'))
        app.modo_execucao_var.set(mode)
        app.atualizar_aviso_execucao()
        app.N_entry.delete(0,tk.END)
        app.N_entry.insert(0,'13')
        app.thomas_var.set(True)
        app.iniciar_execucao()
        assert app.executando and len(app.exec_tree.get_children()) == 4
        assert str(app.N_entry.cget('state')) == 'disabled'
        esperar(root,app)
        assert not errors
        assert bool(confirmations) == (mode=='Paralelo')
        assert set(app.solvers) == set(app.referencias) == {'mdf','mef'}
        assert all(s.convergiu for s in app.solvers.values())
        assert len(app.resultados_comparacao) == 4
        assert len(app.tree.get_children()) == 6
        for iid in app.exec_tree.get_children():
            assert app.exec_tree.item(iid,'values')[3] == 'Convergiu'
        meta = json.loads((tmp_path/'configuracao_execucao.json').read_text(encoding='utf-8'))
        assert meta['modo_efetivo'] == mode.lower() and meta['configuracao']['N'] == 13
        with (tmp_path/'resultados_refinamento.csv').open(encoding='utf-8') as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 6 and {r['Discr.'] for r in rows} == {'MDF','MEF'}
        for view in ('fluxo','comparacao','convergencia','residuo_fonte','tempo_metodo',
                     'tabela_N','escalabilidade','todos'):
            app.modo_grafico.set(view)
            app.atualizar_grafico()
            assert 'MDF + MEF' in app.ax.get_title()
            assert len(app.fig.axes) == (2 if view=='tabela_N' else 1)
            if view in ('fluxo','comparacao'):
                labels = app.ax.get_legend_handles_labels()[1]
                assert any('Neural Physics / MDF' in v for v in labels)
                assert any('Neural Physics / MEF' in v for v in labels)
                assert any('Thomas / MEF' in v for v in labels)
        # Alterar o formulario depois da execucao nao altera o relatorio.
        app.N_entry.delete(0,tk.END)
        app.N_entry.insert(0,'999')
        app.selecionar_discretizacoes(('mdf',))
        app.modo_execucao_var.set('Sequencial')
        md = tmp_path/'comparacao.md'
        app.gerar_relatorio_markdown(str(md))
        text = md.read_text(encoding='utf-8')
        assert '999' not in text and 'MDF, MEF' in text
        assert f'{mode.lower()}/{mode.lower()}' in text
        target = tmp_path/'comparacao.pdf'
        monkeypatch.setattr('neutron_ai4pdes.gui.filedialog.asksaveasfilename',lambda **k:str(target))
        app.gerar_relatorio()
        assert not errors and target.stat().st_size > 1000
        # Refinamento tambem respeita ambos os perfis, um registro por (discr,N,x).
        app.selecionar_discretizacoes(('mdf','mef'))
        app.modo_execucao_var.set(mode)
        app.N_tabela_entry.delete(0,tk.END)
        app.N_tabela_entry.insert(0,'1,13')
        app.thomas_var.set(False)
        app.gerar_tabela_variando_N_thread()
        esperar(root,app)
        assert not errors and len(app.tree.get_children()) == 12
        assert len(app.campanha.solvers) == 4
        assert all(r['erro_fluxo_pontos'][26.] is None for r in app.resultados_tabela)
        assert app.referencias == {}  # Nao reaproveitar Thomas da execucao anterior.
        # Cancelar o aviso nao inicia a campanha nem apaga os dados anteriores.
        previous = app.campanha
        app.modo_execucao_var.set('Paralelo')
        monkeypatch.setattr('neutron_ai4pdes.gui.messagebox.askyesno',lambda *a,**k:False)
        app.iniciar_execucao()
        assert not app.executando and app.campanha is previous
        app.selecionar_discretizacoes(())
        app.iniciar_execucao()
        assert not app.executando and app.campanha is previous
        assert len(errors) == 1 and 'Marque pelo menos' in errors[0][1]
    finally:
        root.destroy()
        torch.set_num_threads(threads)


def test_referencia_heterogenea_e_separada_por_discretizacao():
    app = object.__new__(DifusaoGUI_AI4PDEs)
    rows = []
    for disc, scale in [('MDF',1.),('MEF',10.)]:
        for n,value in [(3,scale),(12,2*scale)]:
            rows.append(dict(discretizacao=disc,N=n,is_homogeneo=False,convergiu=True,
                             fluxos_pontos={0.:value,150.:0.}))
    app.calcular_erros_fluxo_tabela(rows,[0.,150.])
    assert rows[0]['erro_fluxo_pontos'][0.] == rows[2]['erro_fluxo_pontos'][0.] == 50.
    assert rows[1]['erro_fluxo_pontos'][0.] == rows[3]['erro_fluxo_pontos'][0.] == 0.
    assert all(r['erro_fluxo_pontos'][150.] is None for r in rows)
    assert app.referencias_fluxo_tabela == {'MDF':12,'MEF':12}


@pytest.mark.gui
def test_botoes_acessiveis_em_janela_menor():
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(str(exc))
    root.withdraw()
    root.state = lambda *args:'withdrawn'
    root.attributes('-alpha',0.)
    try:
        app = DifusaoGUI_AI4PDEs(root)
        root.geometry('1280x720')
        root.deiconify()
        root.update_idletasks()
        for button in (app.btn_executar, app.btn_relatorio):
            top = button.winfo_rooty()-root.winfo_rooty()
            assert 0 <= top < top+button.winfo_height() <= root.winfo_height()
        bbox = app.settings_canvas.bbox('all')
        assert bbox[3] > app.settings_canvas.winfo_height()  # Dados rolaveis, acoes fixas.
    finally:
        root.destroy()
