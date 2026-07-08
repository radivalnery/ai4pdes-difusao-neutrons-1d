# AI4PDEs - Difusao de Neutrons 1D

Este repositorio organiza um solver 1D para difusao de neutrons inspirado na filosofia AI4PDEs/Neural Physics. O codigo usa operadores discretos por stencil/convolucao fixa em PyTorch e iteracoes multiescala, sem treinamento de pesos.

## Objetivo

Resolver problemas de autovalor da difusao de neutrons em geometria slab 1D, com interface grafica para configurar:

- problema homogeneo;
- problema heterogeneo baseado em Nozimar;
- condicoes de contorno reflexiva ou vacuo;
- potencia nominal;
- tolerancias de fluxo e de `k_eff`;
- materiais por regiao;
- tabela por refinamento de malha.

## Estrutura do Codigo

```text
main.py                         ponto de entrada da interface
src/neutron_ai4pdes/
  app.py                        inicializacao da aplicacao Tkinter
  gui.py                        frontend Tkinter, graficos e relatorio
  solver.py                     backend numerico do problema de autovalor
  models.py                     operador discreto e solver de fonte fixa
  references.py                 formulas e solucoes de referencia
  method.py                     descricao do metodo AI4PDEs 1D adaptado
```

## Instalar Dependencias

```bash
pip install -r requirements.txt
```

## Executar

```bash
python main.py
```

## Observacao Metodologica

O pacote AI4PDEs original possui recursos voltados principalmente a operadores 2D/3D. Para este problema 1D, o codigo implementa uma adaptacao local da mesma filosofia: operadores discretos fixos, tensores PyTorch e iteracoes sem treinamento.

Para comparar com Nozimar, use o mesmo problema fisico, materiais, condicoes de contorno, potencia nominal e pontos de avaliacao da Tabela III.3.2. A comparacao com os valores END1D1G e DF publicados deve ser feita no texto/relatorio, mantendo claro que o metodo numerico aqui e o AI4PDEs 1D adaptado.
