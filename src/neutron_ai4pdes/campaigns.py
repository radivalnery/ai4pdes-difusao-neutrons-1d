"""Execucao de MDF/MEF com configuracao comum e resultados independentes.

Nao altera stencils, ciclos V ou tolerancias. No modo paralelo, cada thread
possui seu proprio solver e materiais. Tkinter e Matplotlib nao sao chamados
pelos trabalhadores. Eventos sao entregues a uma fila da interface.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass, field
import time

from .solver import SolverDifusaoAI4PDEs
from .metrics import erro_forma_fluxo


AVISO_TEMPOS_PARALELOS = (
    'Execucao paralela: MDF e MEF compartilham recursos de CPU/GPU. '
    'Os tempos medidos sob concorrencia nao representam desempenho isolado. '
    'Para comparar desempenho, use o modo sequencial e repeticoes controladas.'
)


@dataclass
class ResultadoCampanha:
    """Snapshot da campanha; chaves dos resultados: (discretizacao, N, metodo)."""

    configuracao: dict
    discretizacoes: tuple
    malhas: tuple
    incluir_thomas: bool
    modo_solicitado: str
    modo_efetivo: str
    solvers: dict = field(default_factory=dict)
    erros: dict = field(default_factory=dict)
    tempo_parede: float = 0.

    @property
    def aviso_tempos(self):
        return AVISO_TEMPOS_PARALELOS if self.modo_efetivo == 'paralelo' else (
            'Execucao sequencial: sem sobreposicao dos solvers desta campanha. '
            'Tempos individuais nao incluem geracao de graficos ou relatorios.'
        )

    def chaves(self):
        methods = ('unet_multigrid', 'thomas') if self.incluir_thomas else ('unet_multigrid',)
        return [(d, n, m) for d in self.discretizacoes for n in self.malhas for m in methods]

    def resumos(self):
        rows = []
        for key in self.chaves():
            solver = self.solvers.get(key)
            if solver is None:
                continue
            row = solver.resumo_resultado()
            row.update(metodo_id=key[2], modo_execucao=self.modo_efetivo,
                       modo_solicitado=self.modo_solicitado,
                       tempo_parede_campanha_s=self.tempo_parede,
                       aviso_tempos=self.aviso_tempos,
                       erro_fluxo_global_inf=None, erro_fluxo_global_L2=None)
            main = self.solvers.get((key[0], key[1], 'unet_multigrid'))
            ref = self.solvers.get((key[0], key[1], 'thomas'))
            if main is not None and ref is not None and main.convergiu and ref.convergiu:
                inf, l2 = erro_forma_fluxo(main.phi, ref.phi)
                row.update(erro_fluxo_global_inf=inf, erro_fluxo_global_L2=l2)
            rows.append(row)
        return rows

    def metadados(self):
        return dict(configuracao=deepcopy(self.configuracao),
                    discretizacoes=self.discretizacoes, malhas=self.malhas,
                    incluir_thomas=self.incluir_thomas,
                    modo_solicitado=self.modo_solicitado, modo_efetivo=self.modo_efetivo,
                    tempo_parede_campanha_s=self.tempo_parede, aviso_tempos=self.aviso_tempos,
                    erros=[dict(discretizacao=d, N=n, metodo=m, mensagem=msg)
                           for (d, n, m), msg in self.erros.items()])


def executar_campanha(configuracao, discretizacoes, modo='sequencial',
                      incluir_thomas=False, malhas=None, evento=None):
    """Executa cada perfil MDF/MEF, sem alterar parametros comuns/global threads.

    No paralelo, sao no maximo dois trabalhadores, um por discretizacao.
    As malhas de um mesmo perfil e sua referencia Thomas sao sequenciais.
    Falha de um caso e registrada, sem descartar os demais resultados.
    """
    discs = tuple(dict.fromkeys(str(d).lower() for d in discretizacoes))
    if not discs or any(d not in ('mdf', 'mef') for d in discs):
        raise ValueError('Selecione MDF, MEF ou ambos.')
    if modo not in ('sequencial', 'paralelo'):
        raise ValueError('Modo deve ser sequencial ou paralelo.')
    cfg = deepcopy(configuracao)
    for key in ('discretizacao', 'metodo_fonte', 'progress_callback', 'permitir_thomas_comparacao'):
        cfg.pop(key, None)
    values = tuple(sorted(set(malhas if malhas is not None else [cfg['N']])))
    if not values or any(n < 1 or int(n) != n for n in values):
        raise ValueError('Informe malhas inteiras positivas.')
    effective = 'paralelo' if modo == 'paralelo' and len(discs) > 1 else 'sequencial'
    result = ResultadoCampanha(cfg, discs, values, bool(incluir_thomas), modo, effective)
    emit = evento or (lambda event: None)

    def profile(disc):
        found, errors = {}, {}
        methods = ('unet_multigrid', 'thomas') if incluir_thomas else ('unet_multigrid',)
        for n in values:
            for method in methods:
                key = (disc, n, method)
                emit(dict(tipo='inicio', chave=key, inicio=time.perf_counter()))
                try:
                    local = deepcopy(cfg)
                    local.update(N=n, discretizacao=disc, metodo_fonte=method,
                                 permitir_thomas_comparacao=(method == 'thomas'))
                    def progress(it, maximum, k, ek, ephi, key=key):
                        emit(dict(tipo='progresso', chave=key, iteracao=it,
                                  max_iter=maximum, k_eff=k, erro_k=ek, erro_phi=ephi))
                    local['progress_callback'] = progress
                    solver = SolverDifusaoAI4PDEs(**local)
                    solver.resolver()
                    found[key] = solver
                    emit(dict(tipo='resultado', chave=key, solver=solver))
                except Exception as exc:
                    errors[key] = str(exc)
                    emit(dict(tipo='erro', chave=key, mensagem=str(exc)))
        return found, errors

    start = time.perf_counter()
    if effective == 'paralelo':
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix='MDF-MEF') as pool:
            futures = [pool.submit(profile, d) for d in discs]
            for future in as_completed(futures):
                found, errors = future.result()
                result.solvers.update(found)
                result.erros.update(errors)
    else:
        for disc in discs:
            found, errors = profile(disc)
            result.solvers.update(found)
            result.erros.update(errors)
    result.tempo_parede = time.perf_counter()-start
    return result
