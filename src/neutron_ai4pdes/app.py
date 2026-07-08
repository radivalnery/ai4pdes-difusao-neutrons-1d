"""Ponto de entrada da aplicação Tkinter."""

import tkinter as tk

import torch

from .gui import DifusaoGUI_AI4PDEs


def main():
    print("=" * 60)
    print("SOLVER DE DIFUSÃO DE NÊUTRONS 1D")
    print("Neural Physics/AI4PDEs: stencil + Conv1d + ciclo U-Net/multigrid")
    print("=" * 60)
    print("Pacote ai4pdes 2D/3D não é chamado diretamente neste caso 1D")
    print(f"CUDA disponível: {torch.cuda.is_available()}")
    print("=" * 60)

    root = tk.Tk()
    DifusaoGUI_AI4PDEs(root)
    root.mainloop()


if __name__ == "__main__":
    main()
