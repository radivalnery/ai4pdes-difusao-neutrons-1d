"""Funções compartilhadas para tabelas de relatórios PDF/Markdown."""


def formatar_valor(valor, fmt="{:.6e}", vazio="N/A"):
    if valor is None:
        return vazio
    try:
        return fmt.format(valor)
    except Exception:
        return str(valor)


def linha_resultado_comparacao(resultado):
    """Formata uma linha de comparação para PDF ou Markdown."""
    return [
        str(resultado.get("Caso", "")),
        str(resultado.get("Método", "")),
        str(resultado.get("N", "")),
        formatar_valor(resultado.get("k_eff"), "{:.8f}"),
        formatar_valor(resultado.get("Referência"), "{:.8f}"),
        formatar_valor(resultado.get("Erro k (%)"), "{:.4e}"),
        str(resultado.get("Iter. externas", "")),
        formatar_valor(resultado.get("Iter. fonte média"), "{:.2f}"),
        formatar_valor(resultado.get("Resíduo final"), "{:.3e}"),
        "sim" if resultado.get("Fonte fixa convergiu (todas as chamadas)") else "não",
        str(resultado.get("Chamadas fonte fixa não convergidas", 0)),
        formatar_valor(resultado.get("Tempo (s)"), "{:.4f}"),
    ]


def cabecalho_comparacao():
    return [
        "Caso",
        "Método",
        "N",
        "k_eff",
        "Referência",
        "Erro k (%)",
        "Iter. externas",
        "Iter. fonte média",
        "Resíduo final",
        "Fonte convergiu",
        "Falhas fonte",
        "Tempo (s)",
    ]


def tabela_comparacao(resultados):
    return [cabecalho_comparacao()] + [linha_resultado_comparacao(r) for r in resultados]


def markdown_tabela(linhas):
    if not linhas:
        return ""
    header = "| " + " | ".join(linhas[0]) + " |"
    sep = "| " + " | ".join(["---"] * len(linhas[0])) + " |"
    body = ["| " + " | ".join(map(str, row)) + " |" for row in linhas[1:]]
    return "\n".join([header, sep] + body)
