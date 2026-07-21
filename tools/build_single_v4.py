from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "neutron_ai4pdes"
OUT_LOCAL = ROOT / "difusao_neutrons_neuralphysics_1d_comparativo_v4_0.py"
OUT_GDRIVE = Path(
    r"G:\Meu Drive\Doutorado UESC\Artigos\AI4PDEs\Cógigos\difusao_neutrons_neuralphysics_1d_comparativo_v4_0.py"
)


def strip_module(path, skip_prefixes):
    lines = path.read_text(encoding="utf-8").splitlines()
    out = []
    i = 0
    while i < len(lines) and not lines[i].strip():
        i += 1
    if i < len(lines) and lines[i].lstrip().startswith('"""'):
        if lines[i].count('"""') >= 2:
            i += 1
        else:
            i += 1
            while i < len(lines) and '"""' not in lines[i]:
                i += 1
            if i < len(lines):
                i += 1
    lines = lines[i:]
    for line in lines:
        if any(line.startswith(prefix) for prefix in skip_prefixes):
            continue
        out.append(line)
    return "\n".join(out).strip()


def remove_triple_quoted_blocks(text):
    return re.sub(r'""".*?"""', '', text, flags=re.DOTALL)


def main():
    header = '''"""
Solver de difusão de nêutrons 1D para comparação ENMC 2026.

Versão única gerada a partir do pacote modular neutron_ai4pdes.
Implementa:
- operadores convolucionais fixos sem treinamento;
- arquitetura algorítmica inspirada em U-Net/multigrid;
- método clássico de Thomas para sistemas tridiagonais;
- comparação entre resolvedores;
- interface Tkinter e relatórios PDF/Markdown.
"""

import csv
import os
import platform
import tempfile
import threading
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm, inch
    from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

'''
    parts = [
        header,
        strip_module(SRC / "method.py", []),
        strip_module(SRC / "references.py", ["import numpy"]),
        strip_module(SRC / "models.py", ["import numpy", "import torch", "import torch.nn"]),
        strip_module(
            SRC / "solver.py",
            ["import platform", "import time", "import numpy", "import torch", "from ."],
        ),
        strip_module(
            SRC / "gui.py",
            [
                "import csv", "import os", "import platform", "import tempfile", "import threading",
                "import time", "from datetime", "import numpy", "import torch", "import tkinter",
                "from tkinter", "from matplotlib", "from .", "try:", "    from reportlab",
                "    REPORTLAB_AVAILABLE", "except ImportError:", "    REPORTLAB_AVAILABLE",
            ],
        ),
        '''
def main():
    print("=" * 60)
    print("SOLVER DE DIFUSÃO DE NÊUTRONS 1D")
    print("Neural Physics: operadores convolucionais fixos sem treinamento")
    print("=" * 60)
    print(f"CUDA disponível: {torch.cuda.is_available()}")
    print("=" * 60)

    root = tk.Tk()
    DifusaoGUI_AI4PDEs(root)
    root.mainloop()


if __name__ == "__main__":
    main()
''',
    ]
    text = "\n\n\n".join(part for part in parts if part)
    OUT_LOCAL.write_text(text, encoding="utf-8")
    OUT_GDRIVE.write_text(text, encoding="utf-8")
    print(OUT_LOCAL)
    print(OUT_GDRIVE)


if __name__ == "__main__":
    main()
