"""Tabelas (CSV e LaTeX/abnTeX2), figuras e relatorio da ablacao, gerados a partir das execucoes reais."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from .estatistica import holm
from .utils import escrever_csv_atomico, escrever_texto_atomico

# cores: slots 1 e 2 da paleta categorica de referencia (validada) e um cinza neutro para o baseline
AZUL, LARANJA, CINZA, TINTA, TINTA2 = "#2a78d6", "#eb6834", "#8a8984", "#0b0b0b", "#52514e"
FONTE_TABELA = "Elaborado pelos autores (2026)."
ROT_DEG = {
    "oclusao_mao": "Oclusão de mão",
    "ruido": "Ruído",
    "queda_fps": "Queda de FPS",
    "sem_face": "Sem face",
    "escuro": "Vídeo escuro",
    "claro": "Vídeo claro",
    "desfoque": "Desfoque",
    "ruido_imagem": "Ruído de imagem",
    "rotacao_zoom": "Rotação e zoom",
}


def br(x, casas=3) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "-"
    return f"{x:.{casas}f}".replace(".", ",")


def p_txt(p) -> str:
    if p is None or not np.isfinite(p):
        return "-"
    return "< 0,001" if p < 0.001 else br(p, 3)


def metricas_de_matriz(cm, classes) -> dict:
    cm = np.asarray(cm)
    out = {}
    for i, c in enumerate(classes):
        tp, prev, sup = cm[i, i], cm[:, i].sum(), cm[i, :].sum()
        pr = tp / prev if prev else 0.0
        rc = tp / sup if sup else 0.0
        out[c] = {"precisao": pr, "recall": rc, "f1": 2 * pr * rc / (pr + rc) if pr + rc else 0.0, "suporte": int(sup)}
    return out


def _latex_tabela(
    titulo: str, rotulo: str, cab: list[str], linhas: list[list[str]], alinh: str, nota: str | None = None
) -> str:
    esc = lambda t: (  # noqa: E731
        str(t)
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("<", "$<$")  # noqa: E731
        .replace("±", "$\\pm$")
    )
    corpo = "\n".join("    " + " & ".join(esc(c) for c in l) + " \\\\" for l in linhas)
    nota_tex = f"\n  \\nota{{{esc(nota)}}}" if nota else ""
    return (
        "% Gerado automaticamente por scripts/ablacao.py a partir de execucoes reais. Nao editar os numeros a mao.\n"
        "\\begin{table}[htb]\n"
        f"  \\caption{{{esc(titulo)}}}\n  \\label{{{rotulo}}}\n  \\centering\n  \\small\n"
        f"  \\begin{{tabular}}{{{alinh}}}\n    \\hline\n"
        f"    {' & '.join(esc(c) for c in cab)} \\\\\n    \\hline\n{corpo}\n    \\hline\n  \\end{{tabular}}\n"
        f"  \\fonte{{{FONTE_TABELA}}}{nota_tex}\n\\end{{table}}\n"
    )


def _fig_barras(
    conds: list[tuple[str, float, float, bool]], base_media: float, titulo: str, eixo: str, destino: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(conds)
    fig, ax = plt.subplots(figsize=(7.5, 0.55 * n + 1.4))
    y = np.arange(n)[::-1]
    for yi, (rot, m, dp, eh_base) in zip(y, conds):
        cor = CINZA if eh_base else AZUL
        ax.barh(yi, m, height=0.55, color=cor, xerr=dp, error_kw={"ecolor": TINTA2, "elinewidth": 1, "capsize": 3})
        ax.text(min(m + (dp or 0) + 0.01, 1.08), yi, br(m), va="center", fontsize=8, color=TINTA)
    ax.axvline(base_media, color=TINTA2, lw=1, ls="--")
    ax.set_yticks(y, [c[0] for c in conds], fontsize=8, color=TINTA)
    ax.set_xlim(0, 1.15)
    ax.set_xlabel(eixo, color=TINTA2)
    ax.set_title(titulo, fontsize=10, color=TINTA, loc="left")
    ax.grid(axis="x", alpha=0.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(destino, dpi=200)
    plt.close(fig)


def _fig_robustez(cols: list[str], series: list[tuple[str, list[float], str]], eixo: str, destino: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.arange(len(cols))
    w = 0.8 / max(1, len(series))
    fig, ax = plt.subplots(figsize=(1.3 * len(cols) + 2.5, 3.6))
    for i, (nome, vals, cor) in enumerate(series):
        v = [np.nan if x_ is None else x_ for x_ in vals]
        ax.bar(x + (i - (len(series) - 1) / 2) * w, v, width=w * 0.9, color=cor, label=nome)
    ax.set_xticks(x, cols, fontsize=8, color=TINTA)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(eixo, color=TINTA2)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(destino, dpi=200)
    plt.close(fig)


def gerar_saidas(
    cfg, pasta: Path, res: dict, chave: str, k: int, r: int, melhor: str | None, avisos: list[str]
) -> None:
    from .ablacao import NOMES_METRICA, comparar

    alfa = float(cfg["ablacao"]["alfa"])
    nome_met = NOMES_METRICA[chave]
    met, nivel = chave.rsplit("_", 1)
    classes = res["baseline"].resumo["classes"]
    base = res["baseline"]
    principais = [n for n, x in res.items() if x.condicao.etapa in (1, 2, 3, 4)]
    comps = {n: comparar(base, res[n], chave, k, r) for n in principais if n != "baseline"}
    ajust = holm([comps[n]["p_t_corrigido"] for n in comps])
    for n, pa in zip(comps, ajust):
        comps[n]["p_holm"] = pa
        d = comps[n]["diferenca_media"]
        if np.isfinite(pa) and pa < alfa:
            comps[n]["conclusao"] = "ganho significativo" if d > 0 else "perda significativa"
        else:
            comps[n]["conclusao"] = "sem evidencia de diferenca"

    # ---------------------------------------------------------- tabela principal
    linhas_csv, linhas_tex = [], []
    for n in principais:
        x = res[n]
        rs = x.resumo
        c = comps.get(n, {})
        l = {
            "condicao": n,
            "descricao": x.condicao.rotulo,
            "familias": " ".join(x.condicao.familias) or "-",
            "intensidade": x.condicao.intensidade if x.condicao.familias else "-",
            "acuracia_video": rs["acuracia_video_media"],
            "acuracia_video_dp": rs["acuracia_video_dp"],
            "f1_macro_video": rs["f1_macro_video_media"],
            "f1_macro_video_dp": rs["f1_macro_video_dp"],
            "acuracia_janela": rs["acuracia_janela_media"],
            "acuracia_janela_dp": rs["acuracia_janela_dp"],
            "f1_macro_janela": rs["f1_macro_janela_media"],
            "f1_macro_janela_dp": rs["f1_macro_janela_dp"],
            f"val_{chave}": rs.get(f"val_{chave}_media"),
            "delta_principal": c.get("diferenca_media"),
            "ic95_inf": (c.get("ic95") or [None, None])[0],
            "ic95_sup": (c.get("ic95") or [None, None])[1],
            "p_t_corrigido": c.get("p_t_corrigido"),
            "p_holm": c.get("p_holm"),
            "p_wilcoxon": c.get("p_wilcoxon"),
            "conclusao": c.get("conclusao", "referencia"),
            "segundos_treino_total": rs["segundos_treino_total"],
            "latencia_modelo_ms": rs["latencia_modelo_ms_mediana"],
        }
        linhas_csv.append(l)
        linhas_tex.append(
            [
                x.condicao.rotulo,
                l["intensidade"],
                f"{br(rs[f'{met}_{nivel}_media'])} ± {br(rs[f'{met}_{nivel}_dp'])}",
                "-"
                if n == "baseline"
                else f"{'+' if (c['diferenca_media'] or 0) >= 0 else ''}{br(c['diferenca_media'])}",
                "-" if n == "baseline" else f"[{br(c['ic95'][0])}; {br(c['ic95'][1])}]",
                "-" if n == "baseline" else p_txt(c["p_holm"]),
            ]
        )
    escrever_csv_atomico(
        pasta / "tabela_ablacao.csv",
        list(linhas_csv[0].keys()),
        [{k2: (f"{v:.6f}" if isinstance(v, float) else v) for k2, v in l.items()} for l in linhas_csv],
    )
    escrever_texto_atomico(
        pasta / "tabela_ablacao.tex",
        _latex_tabela(
            f"Ablação da data augmentation ({nome_met}, média ± desvio-padrão)",
            "tab:ablacao",
            ["Condição", "Intensidade", nome_met, "Diferença", "IC 95%", "p (Holm)"],
            linhas_tex,
            "llcccc",
            nota=(
                f"{k} partições por sinalizador × {r} seed(s). Diferença em relação ao baseline; teste t corrigido "
                "para validação cruzada repetida (Nadeau; Bengio, 2003) com correção de Holm."
            ),
        ),
    )

    # ---------------------------------------------------------- por classe
    destaque = (
        ["baseline"]
        + ([melhor] if melhor else [])
        + (["combinacao"] if "combinacao" in res and "combinacao" != melhor else [])
    )
    for n in destaque:
        cm = res[n].resumo[f"matriz_{nivel}_soma"]
        pc = metricas_de_matriz(cm, classes)
        rows = [
            {"classe": c, **{k2: (f"{v:.4f}" if isinstance(v, float) else v) for k2, v in pc[c].items()}}
            for c in classes
        ]
        escrever_csv_atomico(pasta / f"tabela_por_classe_{n}.csv", list(rows[0].keys()), rows)
        escrever_texto_atomico(
            pasta / f"tabela_por_classe_{n}.tex",
            _latex_tabela(
                f"Desempenho por classe: {res[n].condicao.rotulo} ({'vídeos' if nivel == 'video' else 'janelas'} de teste, soma das partições)",
                f"tab:classes_{n}",
                ["Classe", "Precisão", "Revocação", "F1", "Suporte"],
                [
                    [c, br(pc[c]["precisao"]), br(pc[c]["recall"]), br(pc[c]["f1"]), str(pc[c]["suporte"])]
                    for c in classes
                ],
                "lcccc",
                nota=f"Suporte somado sobre {r} seed(s): cada amostra de teste conta uma vez por seed.",
            ),
        )
        origem = res[n].pasta / f"matriz_confusao_{nivel}s_total.png"
        if origem.exists():
            shutil.copy(origem, pasta / f"fig_matriz_{n}.png")

    # ---------------------------------------------------------- robustez
    rob_conds = [n for n in destaque if res[n].resumo.get("robustez")]
    if rob_conds:
        degs = list(res["baseline"].resumo["robustez"].keys())
        cab = ["Condição", "Teste original"] + [ROT_DEG.get(d, d) for d in degs]
        rows_csv, rows_tex, series = [], [], []
        for i, n in enumerate(rob_conds):
            rs = res[n].resumo
            vals = [rs[f"{met}_{nivel}_media"]]
            for d in degs:
                m = rs["robustez"].get(d, {})
                vals.append(m.get(f"{met}_{nivel}", {}).get("media") if "indisponivel" not in m else None)
            rows_csv.append({"condicao": n, **{c: ("" if v is None else f"{v:.6f}") for c, v in zip(cab[1:], vals)}})
            rows_tex.append([res[n].condicao.rotulo] + [br(v) for v in vals])
            series.append((res[n].condicao.rotulo, vals, [CINZA, AZUL, LARANJA][i % 3]))
        escrever_csv_atomico(pasta / "tabela_robustez.csv", ["condicao"] + cab[1:], rows_csv)
        escrever_texto_atomico(
            pasta / "tabela_robustez.tex",
            _latex_tabela(
                f"Robustez a degradações do conjunto de teste ({nome_met})",
                "tab:robustez",
                cab,
                rows_tex,
                "l" + "c" * (len(cab) - 1),
                nota="Degradações aplicadas apenas aos dados de teste; o modelo nunca foi treinado nessas cópias.",
            ),
        )
        _fig_robustez(["original"] + [ROT_DEG.get(d, d) for d in degs], series, nome_met, pasta / "fig_robustez.png")

    # ---------------------------------------------------------- protocolos
    if "baseline_aleatorio_por_video" in res:
        sec = res["baseline_aleatorio_por_video"].resumo
        rows = [
            [
                "Por sinalizador (principal)",
                br(base.resumo["acuracia_video_media"]),
                br(base.resumo["f1_macro_video_media"]),
                br(base.resumo["acuracia_janela_media"]),
            ],
            [
                "Aleatório por vídeo (secundário)",
                br(sec["acuracia_video_media"]),
                br(sec["f1_macro_video_media"]),
                br(sec["acuracia_janela_media"]),
            ],
        ]
        escrever_texto_atomico(
            pasta / "tabela_protocolos.tex",
            _latex_tabela(
                "Baseline sob dois protocolos de avaliação",
                "tab:protocolos",
                ["Protocolo", "Acurácia (vídeo)", "F1 macro (vídeo)", "Acurácia (janela)"],
                rows,
                "lccc",
                nota="No protocolo secundário o mesmo sinalizador aparece em treino e teste; o resultado tende a ser "
                "otimista e não mede generalização para pessoas novas.",
            ),
        )
        escrever_csv_atomico(
            pasta / "tabela_protocolos.csv",
            ["protocolo", "acuracia_video", "f1_macro_video", "acuracia_janela"],
            [dict(zip(["protocolo", "acuracia_video", "f1_macro_video", "acuracia_janela"], l)) for l in rows],
        )

    # ---------------------------------------------------------- figura principal
    barras = [
        (
            res[n].condicao.rotulo,
            res[n].resumo[f"{met}_{nivel}_media"],
            res[n].resumo[f"{met}_{nivel}_dp"],
            n == "baseline",
        )
        for n in principais
    ]
    _fig_barras(
        barras,
        base.resumo[f"{met}_{nivel}_media"],
        f"{nome_met} por condição (média ± dp; {k} partições × {r} seed(s))",
        nome_met,
        pasta / "fig_ablacao.png",
    )

    # ---------------------------------------------------------- relatorio
    resultados = {
        "metrica_principal": chave,
        "alfa": alfa,
        "particoes": k,
        "seeds": r,
        "melhor_pela_validacao": melhor,
        "avisos": avisos,
        "condicoes": {
            n: {
                "descricao": x.condicao.rotulo,
                "familias": x.condicao.familias,
                "intensidade": x.condicao.intensidade,
                "etapa": x.condicao.etapa,
                "pasta": x.pasta.name,
                "resumo": {k2: v for k2, v in x.resumo.items() if k2 not in ("versoes",)},
                "comparacao_baseline": comps.get(n),
            }
            for n, x in res.items()
        },
        "hardware": base.resumo.get("hardware"),
    }
    escrever_texto_atomico(
        pasta / "resultados.json", json.dumps(resultados, indent=2, ensure_ascii=False, default=float)
    )
    escrever_texto_atomico(
        pasta / "RELATORIO.md",
        _relatorio_md(res, comps, chave, nome_met, k, r, melhor, avisos, alfa, base.resumo.get("hardware") or {}),
    )


def _relatorio_md(res, comps, chave, nome_met, k, r, melhor, avisos, alfa, hw) -> str:
    met, nivel = chave.rsplit("_", 1)
    b = res["baseline"].resumo
    L = [
        "# Relatorio da ablacao de data augmentation",
        "",
        "Gerado automaticamente a partir das execucoes desta pasta. Todos os numeros vem de treino e teste reais.",
        "",
        f"- Metrica principal: {nome_met}",
        f"- Protocolo: {b['protocolo']} | {k} particoes x {r} seeds | {b['n_videos']} videos, "
        f"{b['n_sinalizadores']} sinalizadores",
        f"- Hardware: {hw.get('processador')} | {hw.get('nucleos_logicos')} nucleos | {hw.get('ram_gb')} GB",
        f"- Baseline: {br(b[f'{met}_{nivel}_media'])} ± {br(b[f'{met}_{nivel}_dp'])}",
        f"- Melhor condicao pela validacao (usada nas intensidades): {melhor or '-'}",
        "",
        "## Comparacoes com o baseline",
        "",
        "| Condicao | Diferenca | IC 95% | p (t corrigido) | p (Holm) | p (Wilcoxon) | Conclusao |",
        "|---|---|---|---|---|---|---|",
    ]
    for n, c in comps.items():
        L.append(
            f"| {res[n].condicao.rotulo} | {br(c['diferenca_media'])} | [{br(c['ic95'][0])}; {br(c['ic95'][1])}] "
            f"| {p_txt(c['p_t_corrigido'])} | {p_txt(c['p_holm'])} | {p_txt(c['p_wilcoxon'])} | {c['conclusao']} |"
        )
    ganhos = [n for n, c in comps.items() if c["conclusao"] == "ganho significativo"]
    L += ["", "## Leitura", ""]
    if ganhos:
        L.append(
            f"Com alfa = {br(alfa, 2)} e correcao de Holm, houve ganho estatisticamente significativo em: "
            f"{', '.join(res[n].condicao.rotulo for n in ganhos)}."
        )
    else:
        L.append(
            f"Com alfa = {br(alfa, 2)} e correcao de Holm, nenhuma condicao mostrou diferenca estatisticamente "
            "significativa em relacao ao baseline. Diferencas numericas nas medias nao devem ser apresentadas "
            "como ganho."
        )
    pmin = next(iter(comps.values()), {}).get("p_wilcoxon_minimo_possivel")
    if pmin and pmin > alfa:
        L.append(
            f"Observacao: com {k} particoes, o menor p possivel no Wilcoxon e {br(pmin, 4)}; ele nao pode "
            "indicar significancia neste desenho e serve apenas como conferencia."
        )
    if "baseline_aleatorio_por_video" in res:
        s = res["baseline_aleatorio_por_video"].resumo
        vs, vb = s[f"{met}_{nivel}_media"], b[f"{met}_{nivel}_media"]
        txt = (
            f"Protocolo secundario (aleatorio por video): {nome_met} {br(vs)}, contra {br(vb)} no protocolo por "
            "sinalizador. "
        )
        if vs > vb:
            txt += (
                f"A diferenca ({br(vs - vb)}) sugere o quanto avaliar com sinalizadores ja vistos superestima o "
                "desempenho (comparacao descritiva: os conjuntos de teste dos dois protocolos sao diferentes)."
            )
        else:
            txt += "Neste caso o protocolo secundario nao deu resultado maior que o principal."
        L += ["", txt]
    if avisos:
        L += ["", "## Avisos", ""] + [f"- {a}" for a in avisos]
    L += [
        "",
        "## Arquivos",
        "",
        "- tabela_ablacao.csv/.tex, tabela_por_classe_*.csv/.tex, tabela_robustez.csv/.tex, tabela_protocolos.csv/.tex",
        "- fig_ablacao.png, fig_robustez.png, fig_matriz_*.png",
        "- uma pasta por condicao, com as execucoes completas (curvas por particao em seed_*/fold_*/curvas.png)",
        "- resultados.json (tudo em formato legivel por maquina)",
    ]
    return "\n".join(L) + "\n"
