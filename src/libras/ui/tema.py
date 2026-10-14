"""Tema visual: le os tokens de tokens.json (unica fonte de cores, tipografia, espacos, raios,
elevacao, movimento e alvos de toque) e gera a folha de estilo Qt e o estilo do matplotlib.

Nenhum outro arquivo da interface define cor: todos pedem aqui (teste_tokens garante).
Estado global de apresentacao: escala de fonte (100% a 150%), reduzir movimento e faixa de
largura (compacta, media, expandida). Mudar qualquer um deles exige reaplicar folha_de_estilo().
"""

from __future__ import annotations

import json
from pathlib import Path

ARQ_TOKENS = Path(__file__).with_name("tokens.json")
TOKENS: dict = json.loads(ARQ_TOKENS.read_text(encoding="utf-8"))
COR: dict = TOKENS["cor"]
RAMPA: list[str] = TOKENS["rampa"]["pontos"]
TIPO: dict = TOKENS["tipografia"]
ESPACO: dict = {int(k): v for k, v in TOKENS["espaco"].items()}
RAIO: dict = TOKENS["raio"]
ALVO_TOQUE = int(TOKENS["alvo_toque_px"])
FAIXAS = TOKENS["faixas"]
FONTES_ALTERNATIVAS = TIPO["familias"]
FONTE = FONTES_ALTERNATIVAS[0]

# Nomes antigos usados pelas paginas, mapeados para os tokens (compatibilidade).
CORES = {
    "fundo": COR["fundo"],
    "painel": COR["superficie_1"],
    "painel2": COR["superficie_2"],
    "borda": COR["borda"],
    "borda_forte": COR["borda_forte"],
    "borda_controle": COR["borda_controle"],
    "texto": COR["texto"],
    "texto2": COR["texto_secundario"],
    "texto3": COR["texto_terciario"],
    "desabilitado": COR["texto_desabilitado"],
    "sobre_acento": COR["texto_sobre_acento"],
    "acento1": COR["acento"],
    "acento2": COR["acento_2"],
    "selecao": COR["selecao"],
    "sucesso": COR["sucesso"],
    "aviso": COR["alerta"],
    "erro": COR["erro"],
    "erro_borda": COR["erro_borda"],
    "info": COR["info"],
    "hover": COR["superficie_3"],
    "pressionado": COR["pressionado"],
    "foco": COR["foco"],
    "video": COR["video_fundo"],
    "matriz_escuro": COR["matriz_texto_escuro"],
    # series de graficos vem da rampa (treino: neutro; validacao: rampa)
    "serie1": RAMPA[1],
    "serie2": RAMPA[3],
}

ESTADO = {"escala_fonte": 1.0, "reduzir_movimento": False, "faixa": "expandida"}


def configurar(
    escala_fonte: float | None = None, reduzir_movimento: bool | None = None, faixa: str | None = None
) -> None:
    if escala_fonte is not None:
        ESTADO["escala_fonte"] = float(min(1.5, max(1.0, escala_fonte)))
    if reduzir_movimento is not None:
        ESTADO["reduzir_movimento"] = bool(reduzir_movimento)
    if faixa is not None:
        ESTADO["faixa"] = faixa


def px(nome: str) -> int:
    """Tamanho de fonte em px ja com a escala escolhida em Configuracoes."""
    return int(round(TIPO["tamanho_px"][nome] * ESTADO["escala_fonte"]))


def duracao_ms() -> int:
    return 0 if ESTADO["reduzir_movimento"] else int(TOKENS["movimento"]["duracao_ms"])


def faixa_para_largura(largura: int) -> str:
    if largura < FAIXAS["compacta_max_px"]:
        return "compacta"
    if largura < FAIXAS["expandida_min_px"]:
        return "media"
    return "expandida"


# ----------------------------------------------------------------------
# Cores auxiliares
# ----------------------------------------------------------------------
def _rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def hex_de(rgb) -> str:
    return "#%02X%02X%02X" % tuple(int(round(c)) for c in rgb[:3])


def cor_rampa(t: float) -> str:
    """Cor na rampa de 4 pontos (t de 0 a 1), interpolada em RGB."""
    t = min(1.0, max(0.0, float(t)))
    pos = t * (len(RAMPA) - 1)
    i = min(int(pos), len(RAMPA) - 2)
    f = pos - i
    a, b = _rgb(RAMPA[i]), _rgb(RAMPA[i + 1])
    return hex_de([x + (y - x) * f for x, y in zip(a, b)])


def rampa(n: int) -> list[str]:
    return [cor_rampa(i / max(1, n - 1)) for i in range(n)]


def qcolor(nome_ou_hex: str):
    """QColor de um token (aceita #RRGGBBAA, que o Qt le como #AARRGGBB)."""
    from PySide6.QtGui import QColor

    h = COR.get(nome_ou_hex, CORES.get(nome_ou_hex, nome_ou_hex))
    if len(h) == 9:
        r, g, b = _rgb(h)
        return QColor(r, g, b, int(h[7:9], 16))
    return QColor(h)


def gradiente(nome: str, vertical: bool = False) -> str:
    a, b = TOKENS["gradiente"][nome]
    x2, y2 = (0, 1) if vertical else (1, 0)
    return f"qlineargradient(x1:0, y1:0, x2:{x2}, y2:{y2}, stop:0 {a}, stop:1 {b})"


def gradiente_rampa() -> str:
    paradas = ", ".join(f"stop:{i / (len(RAMPA) - 1):.3f} {c}" for i, c in enumerate(RAMPA))
    return f"qlineargradient(x1:0, y1:0, x2:1, y2:0, {paradas})"


# ----------------------------------------------------------------------
# Contraste (WCAG 2.x)
# ----------------------------------------------------------------------
def luminancia(h: str) -> float:
    def canal(c):
        c = c / 255
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    r, g, b = _rgb(h)
    return 0.2126 * canal(r) + 0.7152 * canal(g) + 0.0722 * canal(b)


def contraste(a: str, b: str) -> float:
    la, lb = luminancia(a), luminancia(b)
    return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)


def composto(fundo: str, opacidade: float, base: str) -> str:
    """Cor resultante de um fundo translucido sobre uma base (canal alfa simples)."""
    a = min(1.0, max(0.0, float(opacidade)))
    return hex_de([a * f + (1 - a) * b for f, b in zip(_rgb(fundo), _rgb(base))])


def contraste_pior_caso(texto: str, fundo: str, opacidade: float) -> float:
    """Contraste do texto sobre um fundo translucido em cima de QUALQUER imagem: o pior entre o fundo
    composto sobre branco e sobre preto (regra de D18 para texto sobre video)."""
    return min(
        contraste(texto, composto(fundo, opacidade, COR["imagem_pior_caso_clara"])),
        contraste(texto, composto(fundo, opacidade, COR["imagem_pior_caso_escura"])),
    )


# ----------------------------------------------------------------------
# matplotlib
# ----------------------------------------------------------------------
def estilo_matplotlib() -> dict:
    """rcParams vindos dos tokens: fundo igual ao painel, textos e grade neutros."""
    c = CORES
    tam = px("grafico") * 0.75  # o matplotlib mede fonte em pontos (1 pt = 4/3 px a 96 dpi)
    return {
        "figure.facecolor": c["painel"],
        "axes.facecolor": c["painel"],
        "savefig.facecolor": c["painel"],
        "axes.edgecolor": c["borda_forte"],
        "axes.labelcolor": c["texto2"],
        "axes.titlecolor": c["texto"],
        "xtick.color": c["texto2"],
        "ytick.color": c["texto2"],
        "grid.color": c["borda"],
        "text.color": c["texto"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.linewidth": 0.6,
        "font.family": "sans-serif",
        "font.sans-serif": FONTES_ALTERNATIVAS,
        "font.size": tam,
        "axes.titlesize": px("corpo") * 0.75,
        "axes.labelsize": tam,
        "xtick.labelsize": tam,
        "ytick.labelsize": tam,
        "legend.fontsize": tam,
        "legend.frameon": False,
        "legend.labelcolor": c["texto"],
    }


# ----------------------------------------------------------------------
# Folha de estilo
# ----------------------------------------------------------------------
def _recurso(nome: str) -> str:
    return (Path(__file__).parent / "recursos" / nome).as_posix()


def _estados(seletor: str, regras: dict[str, str]) -> str:
    """Gera cada estado como pseudo-estado do Qt e tambem como propriedade [estado="..."], para a
    galeria de componentes mostrar hover, pressionado e foco sem precisar do mouse."""
    pseudo = {"hover": ":hover", "pressionado": ":pressed", "foco": ":focus", "desabilitado": ":disabled"}
    out = []
    for estado, corpo in regras.items():
        if estado == "normal":
            out.append(f"{seletor} {{ {corpo} }}")
            continue
        sels = [f"{s.strip()}{pseudo[estado]}" for s in seletor.split(",")]
        sels += [f'{s.strip()}[estado="{estado}"]' for s in seletor.split(",")]
        out.append(f"{', '.join(sels)} {{ {corpo} }}")
    return "\n    ".join(out)


def folha_de_estilo(escala: float | None = None, faixa: str | None = None) -> str:
    if escala is not None or faixa is not None:
        configurar(escala_fonte=escala, faixa=faixa)
    c = CORES
    toque = ESTADO["faixa"] in ("compacta", "media")
    # altura minima do conteudo (sem padding e borda) para chegar ao alvo de 44 px
    alt_botao = ALVO_TOQUE - 2 * 9 - 2 if toque else 20
    alt_campo = ALVO_TOQUE - 2 * 8 - 2 if toque else 22
    alt_check = ALVO_TOQUE if toque else 24
    r_p, r_m, r_g = RAIO["p"], RAIO["m"], RAIO["g"]
    e = ESPACO
    baixo, cima, marca = _recurso("seta_baixo.svg"), _recurso("seta_cima.svg"), _recurso("marca.svg")
    grad, grad_h, grad_p, grad_nav = (
        gradiente("primario"),
        gradiente("primario_hover"),
        gradiente("primario_pressionado"),
        gradiente("navegacao"),
    )
    botoes = "\n    ".join(
        [
            _estados(
                "QPushButton, QToolButton#acao",
                {
                    "normal": f"background: {c['painel2']}; color: {c['texto']}; border: 1px solid {c['borda_controle']}; "
                    f"border-radius: {r_m}px; padding: {e[2] + 1}px {e[4]}px; min-height: {alt_botao}px; "
                    f"font-size: {px('corpo')}px; font-weight: 500;",
                    "hover": f"background: {c['hover']}; border-color: {c['acento1']};",
                    "pressionado": f"background: {c['pressionado']}; border-color: {c['acento1']};",
                    "foco": f"border: 2px solid {c['foco']};",
                    "desabilitado": f"color: {c['desabilitado']}; border-color: {c['borda']}; background: {c['painel']};",
                },
            ),
            # primario: rotulo em texto grande e negrito (>= 18,66 px), exigencia de contraste 3:1 (WCAG 1.4.3)
            _estados(
                "QPushButton#primario",
                {
                    "normal": f"background: {grad}; border: 2px solid transparent; color: {c['sobre_acento']}; "
                    f"font-weight: 700; font-size: {max(19, px('corpo'))}px;",
                    "hover": f"background: {grad_h};",
                    "pressionado": f"background: {grad_p};",
                    "foco": f"border: 2px solid {c['foco']};",
                    "desabilitado": f"background: {c['borda']}; color: {c['desabilitado']};",
                },
            ),
            _estados(
                "QPushButton#fantasma",
                {
                    "normal": f"background: transparent; border: 1px solid transparent; color: {c['texto2']};",
                    "hover": f"background: {c['hover']}; color: {c['texto']};",
                    "pressionado": f"background: {c['pressionado']};",
                    "foco": f"border: 2px solid {c['foco']};",
                    "desabilitado": f"color: {c['desabilitado']}; background: transparent;",
                },
            ),
            _estados(
                "QPushButton#perigo",
                {
                    "normal": f"border-color: {c['erro_borda']}; color: {c['erro']};",
                    "hover": f"background: {COR['erro_fundo']}; border-color: {c['erro_borda']};",
                    "pressionado": f"background: {COR['erro_fundo_pressionado']};",
                    "foco": f"border: 2px solid {c['foco']};",
                    "desabilitado": f"border-color: {c['borda']}; color: {c['desabilitado']}; background: {c['painel']};",
                },
            ),
        ]
    )
    campos = _estados(
        "QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QDateEdit, QTextEdit, QPlainTextEdit",
        {
            "normal": f"background: {c['fundo']}; color: {c['texto']}; border: 1px solid {c['borda_controle']}; "
            f"border-radius: {r_p + 2}px; padding: {e[1] + 1}px {e[2]}px; min-height: {alt_campo}px; "
            f"font-size: {px('corpo')}px; selection-background-color: {c['selecao']}; "
            f"selection-color: {c['sobre_acento']};",
            "hover": f"border-color: {c['texto3']};",
            "foco": f"border: 2px solid {c['foco']};",
            "desabilitado": f"color: {c['desabilitado']}; background: {c['painel']}; border-color: {c['borda']};",
        },
    )
    return f"""
    * {{ color: {c["texto"]}; font-size: {px("corpo")}px; }}
    QMainWindow, QWidget#raiz, QStackedWidget, QScrollArea, QScrollArea > QWidget > QWidget, QWidget#conteudo {{
        background: {c["fundo"]}; }}
    QToolTip {{ background: {c["hover"]}; color: {c["texto"]}; border: 1px solid {c["borda_controle"]};
        padding: {e[1]}px {e[2]}px; }}
    QMessageBox, QDialog {{ background: {c["painel"]}; }}

    /* ---- navegacao ---- */
    QFrame#barra_lateral {{ background: {c["painel"]}; border-right: 1px solid {c["borda"]}; }}
    QFrame#barra_inferior {{ background: {c["painel"]}; border-top: 1px solid {c["borda"]}; }}
    QLabel#marca {{ font-size: {px("marca")}px; font-weight: 700; padding: {e[4]}px {e[4]}px 0 {e[4]}px; }}
    QLabel#marca_sub {{ color: {c["texto2"]}; font-size: {px("legenda")}px; padding: {e[1]}px {e[4]}px {e[4]}px; }}
    QFrame#assinatura {{ background: {gradiente_rampa()}; border: none; min-height: 3px; max-height: 3px; }}
    {
        _estados(
            "QToolButton#navitem",
            {
                "normal": f"text-align: left; padding: {e[2] + 2}px {e[3]}px; margin: 2px {e[2]}px; border: 2px solid "
                f"transparent; border-radius: {r_m}px; color: {c['texto2']}; background: transparent; "
                f"font-size: {px('corpo')}px; min-height: {ALVO_TOQUE - 2 * (e[2] + 2) - 4}px;",
                "hover": f"background: {c['painel2']}; color: {c['texto']};",
                "pressionado": f"background: {c['pressionado']};",
                "foco": f"border: 2px solid {c['foco']};",
            },
        )
    }
    QToolButton#navitem:checked, QToolButton#navitem[estado="ativo"] {{ background: {grad_nav};
        color: {c["sobre_acento"]}; font-weight: 700; }}
    QToolButton#navitem[modo="trilho"] {{ padding: {e[2]}px {e[1]}px; margin: 2px {e[1]}px; font-size: {
        px("legenda")
    }px; }}
    QToolButton#navitem[modo="inferior"] {{ padding: {e[1]}px 0; margin: {e[1]}px 2px; font-size: {px("legenda")}px; }}

    /* ---- superficies (elevacao por luminosidade) ---- */
    QFrame#painel, QFrame#cartao {{ background: {c["painel"]}; border: 1px solid {c["borda"]}; border-radius: {
        r_g
    }px; }}
    QFrame#painel2 {{ background: {c["painel2"]}; border: 1px solid {c["borda"]}; border-radius: {r_m}px; }}
    QFrame#estado_vazio {{ background: {c["painel"]}; border: 1px dashed {c["borda_controle"]}; border-radius: {
        r_g
    }px; }}
    QFrame#aviso_desktop {{ background: {c["painel2"]}; border: 1px solid {c["borda_controle"]}; border-radius: {
        r_g
    }px; }}
    QFrame#separador {{ background: {c["borda"]}; border: none; min-height: 1px; max-height: 1px; }}
    QLabel#tela_video {{ background: {c["video"]}; border-radius: {r_g}px; color: {c["texto3"]}; }}

    /* ---- texto ---- */
    QLabel#titulo_pagina {{ font-size: {px("titulo_pagina")}px; font-weight: 700; }}
    QLabel#subtitulo, QLabel#secundario, QLabel#dica {{ color: {c["texto2"]}; }}
    QLabel#dica {{ padding-left: {e[1]}px; border-left: 2px solid {c["borda_controle"]}; }}
    QLabel#titulo_painel {{ font-size: {px("titulo_secao")}px; font-weight: 600; }}
    QLabel#numero {{ font-size: {px("numero")}px; font-weight: 700; }}
    QLabel#sinal_grande {{ font-size: {px("sinal")}px; font-weight: 700; }}
    QLabel#texto_ao_vivo {{ font-size: {px("texto_ao_vivo")}px; font-weight: 600; color: {c["texto"]}; }}
    QLabel#parcial {{ font-size: {px("corpo_grande")}px; color: {c["texto2"]}; }}
    QLabel#mono {{ font-family: {", ".join(repr(f) for f in TIPO["mono"])}; color: {c["texto2"]}; }}
    QLabel#alerta_erro {{ color: {c["erro"]}; }}
    QLabel#alerta_aviso {{ color: {c["aviso"]}; }}
    QLabel#ok {{ color: {c["sucesso"]}; }}

    /* ---- etiquetas: cor + texto + borda (nunca so cor) ---- */
    QLabel#etiqueta {{ border-radius: {r_p}px; padding: 2px {e[2]}px; font-size: {px("legenda")}px;
        font-weight: 700; border: 1px solid {c["borda_controle"]}; color: {c["texto2"]}; background: transparent; }}
    QLabel#etiqueta[tipo="ok"] {{ color: {c["sucesso"]}; border-color: {c["sucesso"]}; }}
    QLabel#etiqueta[tipo="pendente"] {{ color: {c["aviso"]}; border-color: {c["aviso"]}; border-style: dashed; }}
    QLabel#etiqueta[tipo="ensaio"] {{ color: {c["aviso"]}; border: 2px solid {c["aviso"]}; }}
    QLabel#etiqueta[tipo="erro"] {{ color: {c["erro"]}; border-color: {c["erro_borda"]}; }}
    QLabel#etiqueta[tipo="info"] {{ color: {c["info"]}; border-color: {c["info"]}; }}
    QLabel#etiqueta[tipo="captando"] {{ color: {c["sucesso"]}; border-color: {c["sucesso"]}; }}

    /* ---- toast ---- */
    QFrame#toast {{ background: {c["hover"]}; border: 1px solid {c["borda_controle"]}; border-left: 4px solid {
        c["info"]
    };
        border-radius: {r_m}px; }}
    QFrame#toast[tipo="ok"] {{ border-left-color: {c["sucesso"]}; }}
    QFrame#toast[tipo="aviso"] {{ border-left-color: {c["aviso"]}; }}
    QFrame#toast[tipo="erro"] {{ border-left-color: {c["erro_borda"]}; }}

    /* ---- botoes ---- */
    {botoes}

    /* ---- campos ---- */
    {campos}
    QSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
        width: {22 if toque else 16}px; background: {c["painel2"]}; border: none; }}
    QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{ image: url({cima}); width: 10px; height: 10px; }}
    QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{ image: url({baixo}); width: 10px; height: 10px; }}
    QComboBox::drop-down, QDateEdit::drop-down {{ border: none; width: 28px; }}
    QComboBox::down-arrow, QDateEdit::down-arrow {{ image: url({baixo}); width: 12px; height: 12px; }}
    QComboBox QAbstractItemView {{ background: {c["painel2"]}; border: 1px solid {c["borda_controle"]};
        selection-background-color: {c["selecao"]}; selection-color: {c["sobre_acento"]}; }}
    QCheckBox, QRadioButton {{ min-height: {alt_check}px; spacing: {e[2]}px; }}
    QCheckBox::indicator, QTableWidget::indicator {{ border: 1px solid {c["borda_controle"]}; border-radius: 4px;
        background: {c["fundo"]}; width: 18px; height: 18px; }}
    QCheckBox::indicator:checked, QTableWidget::indicator:checked {{ background: {c["selecao"]};
        border-color: {c["foco"]}; image: url({marca}); }}
    QRadioButton::indicator {{ border: 1px solid {c["borda_controle"]}; border-radius: 9px; background: {c["fundo"]};
        width: 18px; height: 18px; }}
    QRadioButton::indicator:checked {{ background: qradialgradient(cx:0.5, cy:0.5, radius:0.5, fx:0.5, fy:0.5,
        stop:0 {c["foco"]}, stop:0.45 {c["foco"]}, stop:0.55 {c["fundo"]}, stop:1 {c["fundo"]});
        border-color: {c["foco"]}; }}
    QCheckBox:hover, QRadioButton:hover {{ color: {c["sobre_acento"]}; }}
    QCheckBox::indicator:hover, QRadioButton::indicator:hover {{ border-color: {c["foco"]}; }}
    QCheckBox:focus, QRadioButton:focus {{ color: {c["foco"]}; }}
    QCheckBox:disabled, QRadioButton:disabled {{ color: {c["desabilitado"]}; }}
    QCheckBox::indicator:disabled {{ background: {c["painel"]}; border-color: {c["borda"]}; }}

    /* ---- tabelas ---- */
    QTableWidget, QListWidget {{ background: {c["painel"]}; border: 1px solid {c["borda"]}; border-radius: {r_p + 2}px;
        gridline-color: {c["borda"]}; alternate-background-color: {c["painel2"]}; }}
    QTableWidget:focus {{ border: 2px solid {c["foco"]}; }}
    QTableWidget::item {{ padding: {e[1]}px; min-height: {ALVO_TOQUE if toque else 28}px; }}
    QTableWidget::item:selected, QListWidget::item:selected {{ background: {c["selecao"]}; color: {
        c["sobre_acento"]
    }; }}
    QTableWidget::item:hover {{ background: {c["hover"]}; }}
    QTableWidget::item:disabled {{ color: {c["desabilitado"]}; }}
    QHeaderView::section {{ background: {c["painel2"]}; color: {c["texto2"]}; border: none;
        border-bottom: 1px solid {c["borda"]}; padding: {e[2] - 2}px; font-weight: 600; }}
    QHeaderView {{ background: {c["painel2"]}; border: none; }}
    QTableCornerButton::section {{ background: {c["painel2"]}; border: none; }}

    /* ---- progresso, abas, controles deslizantes, rolagem ---- */
    QProgressBar {{ background: {c["fundo"]}; border: 1px solid {c["borda_controle"]}; border-radius: {r_p}px;
        min-height: 12px; text-align: center; color: {c["texto"]}; font-weight: 600; }}
    QProgressBar::chunk {{ background: {grad}; border-radius: {r_p}px; }}
    QProgressBar#ocupado {{ border: none; background: {c["borda"]}; border-radius: 2px; min-height: 4px; }}
    QProgressBar#ocupado::chunk {{ background: {gradiente_rampa()}; border-radius: 2px; }}
    QTabWidget::pane {{ border: 1px solid {c["borda"]}; border-radius: {r_m}px; background: {c["painel"]}; }}
    QTabBar::tab {{ background: transparent; color: {c["texto2"]}; padding: {e[2] + 2}px {e[4]}px; border: none;
        border-bottom: 2px solid transparent; min-height: {ALVO_TOQUE - 24 if toque else 18}px; }}
    QTabBar::tab:selected {{ color: {c["texto"]}; border-bottom: 3px solid {c["foco"]}; font-weight: 600; }}
    QTabBar::tab:hover {{ color: {c["texto"]}; background: {c["hover"]}; border-radius: {r_p}px; }}
    QTabBar::tab:focus {{ color: {c["foco"]}; }}
    QSlider::groove:horizontal {{ height: 6px; background: {c["borda"]}; border-radius: 3px; }}
    QSlider::sub-page:horizontal {{ background: {grad}; border-radius: 3px; }}
    QSlider::handle:horizontal {{ background: {c["texto"]}; width: 18px; margin: -6px 0; border-radius: 9px; }}
    QSlider::handle:horizontal:hover {{ background: {c["sobre_acento"]}; border: 2px solid {c["acento1"]}; }}
    QSlider::handle:horizontal:pressed {{ background: {c["acento1"]}; }}
    QSlider::handle:horizontal:disabled {{ background: {c["desabilitado"]}; }}
    QSlider:focus {{ border: 1px solid {c["foco"]}; border-radius: {r_p}px; }}
    QScrollBar:vertical {{ background: {c["fundo"]}; width: {12 if toque else 10}px; }}
    QScrollBar::handle:vertical {{ background: {c["borda_forte"]}; border-radius: 5px; min-height: 30px; }}
    QScrollBar:horizontal {{ background: {c["fundo"]}; height: {12 if toque else 10}px; }}
    QScrollBar::handle:horizontal {{ background: {c["borda_forte"]}; border-radius: 5px; min-width: 30px; }}
    QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover {{ background: {c["texto3"]}; }}
    QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
    QScrollBar::add-page, QScrollBar::sub-page {{ background: none; }}
    QSplitter::handle {{ background: transparent; }}
    QSplitter::handle:hover {{ background: {c["borda"]}; border-radius: 3px; }}
    """
