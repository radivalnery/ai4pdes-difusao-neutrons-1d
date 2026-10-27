"""Graficos compartilhados pela GUI e PDF: mesmos resultados, mesmos rotulos."""

import numpy as np
from matplotlib.ticker import MaxNLocator


CORES_DISCRETIZACAO = {'mdf': '#0072b2', 'mef': '#d55e00'}


def desenhar_resultados(ax, solvers, referencias, modo, aviso_tempos='', refinamento=()):
    """Sobrepoe perfis do mesmo problema; Thomas permanece referencia por perfil."""
    if not solvers:
        return
    first = next(iter(solvers.values()))
    labels = ' + '.join(d.upper() for d in solvers)
    extra_axes = []
    if modo in ('fluxo', 'comparacao'):
        for disc, solver in solvers.items():
            color = CORES_DISCRETIZACAO[disc]
            values = solver.phi if modo == 'comparacao' else solver.phi_normalizado
            status = '' if solver.convergiu else ' (NAO convergiu)'
            ax.plot(solver.x, values, color=color, lw=2,
                    marker='o' if disc=='mdf' else '^', markersize=4,
                    markerfacecolor='none', markevery=max(1,len(solver.x)//10),
                    label='Neural Physics / '+disc.upper()+f' / N={solver.N}'+status)
            ref = referencias.get(disc)
            if ref is not None:
                values_ref = ref.phi if modo == 'comparacao' else ref.phi_normalizado
                ax.plot(ref.x, values_ref, '--', color=color, lw=1.5,
                        label='Thomas / '+disc.upper()+f' / N={ref.N}'+('' if ref.convergiu else ' (NAO convergiu)'))
        if modo == 'comparacao' and first.phi_ref is not None:
            ax.plot(first.x, first.phi_ref, ':', color='#202020', lw=2, label='Analitica')
        if not first.is_homogeneo:
            xs = sorted({0., first.L} | {float(m['fim']) for m in first.materiais})
            for x in xs:
                ax.axvline(x, ls=':', color='#555555', lw=.7, alpha=.6)
                ax.text(x, .98, f'x={x:g}', transform=ax.get_xaxis_transform(),
                        rotation=90, va='top', ha='right', fontsize=7)
        ax.set(xlabel='x [cm]', ylabel='Fluxo / maximo' if modo == 'comparacao' else 'Fluxo normalizado por potencia',
               title='Comparacao da forma do fluxo' if modo == 'comparacao' else 'Distribuicao do fluxo')
    elif modo == 'convergencia':
        for disc, solver in solvers.items():
            ax.plot(range(1,len(solver.historico_k)+1), solver.historico_k,
                    color=CORES_DISCRETIZACAO[disc],
                    label=disc.upper()+('' if solver.convergiu else ' (NAO convergiu)'))
        if first.k_ref is not None:
            ax.axhline(first.k_ref, ls=':', color='#202020', label=first.fonte_ref)
        ax.set(xlabel='Iteracao externa', ylabel='k_eff', title='Convergencia de k_eff')
        ax.ticklabel_format(axis='y', useOffset=False, style='plain')
    elif modo == 'residuo_fonte':
        for disc, solver in solvers.items():
            hist = solver.historico_residuo_fonte_ultima_chamada
            if hist:
                ax.semilogy(range(1,len(hist)+1), np.maximum(hist, np.finfo(float).tiny),
                            color=CORES_DISCRETIZACAO[disc], label=disc.upper())
        ax.set(xlabel='Iteracao interna (ultima chamada de cada perfil)', ylabel='Residuo relativo',
               title='Residuo da ultima fonte fixa')
    elif modo == 'tempo_metodo':
        data = [(d, s, 'NP') for d,s in solvers.items()]
        data += [(d, s, 'Thomas') for d,s in referencias.items()]
        x = np.arange(len(data))
        phases = [('Montagem','tempo_montagem_total_s','#0072b2'),
                  ('Transferencia','tempo_transferencia_total_s','#009e73'),
                  ('Iteracao','tempo_iteracao_numerica_s','#d55e00')]
        for i,(name,key,color) in enumerate(phases):
            values = [max(s.tempos_detalhados[key],np.finfo(float).tiny) for _,s,_ in data]
            ax.bar(x+(i-1)*.24, values, width=.24, label=name, color=color)
        ax.set_xticks(x, [d.upper()+' / '+name+f'\nN={s.N}' for d,s,name in data])
        ax.set_yscale('log')
        ax.set(ylabel='Tempo [s]', title='Tempos separados por configuracao')
        if 'paralela' in aviso_tempos.lower():
            ax.text(.01,.99,'Tempos sob concorrencia\nNao representam desempenho isolado',
                    transform=ax.transAxes, va='top', fontsize=7)
    elif modo in ('tabela_N', 'escalabilidade'):
        error_axis = ax.twinx() if modo == 'tabela_N' and refinamento else None
        if error_axis is not None:
            error_axis.set_ylabel('Erro k (%)')
            extra_axes.append(error_axis)
        for disc in ('mdf','mef'):
            rows = sorted((r for r in refinamento if r['discretizacao'].lower() == disc), key=lambda r:r['N'])
            if not rows:
                continue
            good = [r for r in rows if r.get('convergiu',True)]
            bad = [r for r in rows if not r.get('convergiu',True)]
            key = 'k_eff' if modo == 'tabela_N' else 'tempo_total'
            ax.plot([r['N'] for r in good], [r[key] for r in good], 'o-',
                    color=CORES_DISCRETIZACAO[disc], label=disc.upper())
            if bad:
                ax.plot([r['N'] for r in bad], [r[key] for r in bad], 'x',
                        color=CORES_DISCRETIZACAO[disc], label=disc.upper()+' (NAO convergiu)')
            if error_axis is not None:
                known = [r for r in good if r.get('erro_k') is not None]
                error_axis.plot([r['N'] for r in known], [100*r['erro_k'] for r in known],
                                's--', color=CORES_DISCRETIZACAO[disc], alpha=.7,
                                label='Erro k / '+disc.upper())
        ax.set(xlabel='N', ylabel='k_eff' if modo == 'tabela_N' else 'Tempo total [s]',
               title='Refinamento de malha' if modo == 'tabela_N' else 'Escalabilidade: N vs tempo')
        ax.set_xscale('linear')
        meshes = sorted({r['N'] for r in refinamento})
        if len(meshes) <= 8:
            ax.set_xticks(meshes, [str(n) for n in meshes], rotation=35, ha='right', fontsize=8)
        else:
            ax.xaxis.set_major_locator(MaxNLocator(nbins=7, integer=True))
        if modo == 'escalabilidade':
            ax.set_yscale('log')
            if 'paralela' in aviso_tempos.lower():
                ax.text(.01,.99,'Tempos sob concorrencia', transform=ax.transAxes, va='top', fontsize=7)
    elif modo in ('todos','materiais'):
        for values, name, color in [(first.D_arr,'D','#0072b2'),(first.Sigma_a_arr,'Sigma_a','#009e73'),
                                     (first.nuSigma_f_arr,'nuSigma_f','#d55e00')]:
            ax.plot(first.x, values, label=name, color=color)
        for x in sorted({0.,first.L} | {float(m['fim']) for m in first.materiais}):
            ax.axvline(x,ls=':',color='#555555',lw=.8,alpha=.6)
            ax.text(x,.98,f'x={x:g}',transform=ax.get_xaxis_transform(),
                    rotation=90,va='top',ha='right',fontsize=7)
        ax.set(xlabel='x [cm]', ylabel='Coeficientes materiais', title='Materiais comuns aos perfis')
    ax.set_title(ax.get_title()+' / '+labels)
    ax.grid(True, alpha=.25)
    handles, names = ax.get_legend_handles_labels()
    for other in extra_axes:
        h, n = other.get_legend_handles_labels()
        handles += h
        names += n
    if handles:
        ax.legend(handles,names,fontsize=8)
