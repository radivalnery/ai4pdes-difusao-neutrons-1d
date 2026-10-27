"""Gera o arquivo unico detalhado a partir das mesmas fontes da GUI modular.

Uso: python tools/build_single_v4_2.py
Nao sobrescreve as versoes MDF v4.0/v4.1 nem copia para outras unidades.
Somente imports relativos e blocos de entrada sao removidos; comentarios,
docstrings e formulas das fontes permanecem no arquivo para avaliacao.
"""

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / 'src' / 'neutron_ai4pdes'
OUTPUT = ROOT / 'difusao_neutrons_neuralphysics_1d_mdf_mef_v4_2.py'
MODULES = [
    ('method.py', '1. DESCRICAO DO METODO'),
    ('references.py', '2. SOLUCOES ANALITICAS'),
    ('metrics.py', '3. METRICAS DO FLUXO'),
    ('models.py', '4. OPERADOR MDF E CICLO MULTIGRID'),
    ('fem.py', '5. FORMULACAO MEF P1 E TRANSFERENCIAS'),
    ('solver.py', '6. ITERACAO DE POTENCIA E NORMALIZACAO'),
    ('gui.py', '7. INTERFACE TKINTER E RELATORIOS'),
    ('experiments.py', '8. EXPERIMENTOS E EXPORTACAO PARA ARTIGO'),
]


def standalone_source(path):
    source = path.read_text(encoding='utf-8')
    lines = source.splitlines(keepends=True)
    removed = set()
    for node in ast.walk(ast.parse(source)):
        relative_import = isinstance(node, ast.ImportFrom) and node.level > 0
        entry = (isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
                 and isinstance(node.test.left, ast.Name) and node.test.left.id == '__name__')
        if relative_import or entry:
            removed.update(range(node.lineno-1, node.end_lineno))
    return '\n'.join(line.rstrip() for i, line in enumerate(lines) if i not in removed).strip()


def main():
    header = '''"""Difusao de neutrons 1D - v4.2 MDF/MEF.

Arquivo unico gerado de src/neutron_ai4pdes; usa as MESMAS funcoes dos modulos.
GUI: python difusao_neutrons_neuralphysics_1d_mdf_mef_v4_2.py
Experimento: acrescente --experimento-artigo --discretizacao mef
MDF e MEF escolhem a discretizacao; U-Net/multigrid e o resolvedor principal.
Thomas e referencia opcional e nao participa de nenhum nivel do multigrid.
As versoes antigas e o backup continuam disponiveis separadamente.
"""'''
    parts = [header]
    for filename, title in MODULES:
        parts.append('# ' + '='*76 + '\n# ' + title + '\n# ' + '='*76
                     + '\n\n' + standalone_source(PACKAGE / filename))
    parts.append("if __name__ == '__main__':\n    main()\n")
    result = '\n\n\n'.join(parts)
    compile(result, str(OUTPUT), 'exec')
    OUTPUT.write_text(result, encoding='utf-8')
    print(OUTPUT)


if __name__ == '__main__':
    main()
