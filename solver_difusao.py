"""Interface pública do solver numérico sem dependência de GUI/relatórios."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neutron_ai4pdes.models import (
    OperadorDifusao1D,
    ResultadoFonteFixa,
    SolverFonteFixaMultiescala1D,
    SolverFonteFixaUNet1D,
    resolver_tridiagonal_thomas,
)
from neutron_ai4pdes.references import (
    calcular_k_eff_analitico,
    fluxo_analitico_homogeneo,
    normalizar_condicao_contorno,
)
from neutron_ai4pdes.solver import (
    SolverDifusaoAI4PDEs,
    executar_comparacao_resolvedores,
    executar_sensibilidade,
)

__all__ = [
    "OperadorDifusao1D",
    "ResultadoFonteFixa",
    "SolverFonteFixaMultiescala1D",
    "SolverFonteFixaUNet1D",
    "SolverDifusaoAI4PDEs",
    "resolver_tridiagonal_thomas",
    "calcular_k_eff_analitico",
    "fluxo_analitico_homogeneo",
    "normalizar_condicao_contorno",
    "executar_comparacao_resolvedores",
    "executar_sensibilidade",
]
