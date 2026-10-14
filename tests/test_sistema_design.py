"""Item 12.2/12.3/12.6: sistema de design. Tokens, contraste WCAG AA medido a partir dos tokens,
ausencia de cores fora dos tokens no codigo da interface e galeria de componentes."""

import os
import re
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from libras.ui import tema  # noqa: E402

UI = Path(tema.__file__).parent
PASTAS_INTERFACE = [UI, UI.parent / "app", UI.parent / "app_real"]  # bancada, comum e app real
SUPERFICIES = ["fundo", "superficie_1", "superficie_2"]


def test_tokens_completos():
    t = tema.TOKENS
    for chave in (
        "cor",
        "gradiente",
        "rampa",
        "tipografia",
        "espaco",
        "raio",
        "elevacao",
        "movimento",
        "alvo_toque_px",
        "faixas",
    ):
        assert chave in t, chave
    assert t["alvo_toque_px"] >= 44 and t["movimento"]["duracao_ms"] == 150
    assert all(v % 4 == 0 for v in tema.ESPACO.values())  # grade 4/8
    assert len(tema.RAMPA) == 4
    # corpo minimo de 14 px (o tamanho "grafico" e so rotulo de eixo de grafico)
    assert all(v >= 14 for k, v in tema.TIPO["tamanho_px"].items() if k != "grafico")


@pytest.mark.parametrize(
    "nome", ["texto", "texto_secundario", "texto_terciario", "sucesso", "alerta", "erro", "info", "foco"]
)
def test_contraste_de_texto(nome):
    for s in SUPERFICIES:
        assert tema.contraste(tema.COR[nome], tema.COR[s]) >= 4.5, (nome, s)


@pytest.mark.parametrize("nome", ["borda_controle", "acento", "erro_borda", "curva_dados"])
def test_contraste_de_componentes(nome):
    for s in SUPERFICIES:
        assert tema.contraste(tema.COR[nome], tema.COR[s]) >= 3.0, (nome, s)


def test_rampa_e_graficos_3_para_1():
    for cor in tema.RAMPA:
        for s in ("superficie_1", "superficie_2"):  # graficos e barras ficam sobre os paineis
            assert tema.contraste(cor, tema.COR[s]) >= 3.0, (cor, s)


def test_texto_sobre_acentos():
    branco = tema.COR["texto_sobre_acento"]
    assert tema.contraste(branco, tema.COR["selecao"]) >= 4.5  # linha selecionada da tabela
    for cor in tema.TOKENS["gradiente"]["navegacao"]:  # item ativo da barra: texto normal
        assert tema.contraste(branco, cor) >= 4.5
    # botao primario (gradiente da marca, mantido): rotulo em texto grande e negrito -> 3:1 (WCAG 1.4.3)
    for cor in tema.TOKENS["gradiente"]["primario"]:
        assert tema.contraste(branco, cor) >= 3.0
    qss = tema.folha_de_estilo()
    regra = re.search(r"QPushButton#primario \{([^}]*)\}", qss).group(1)
    tam = int(re.search(r"font-size: (\d+)px", regra).group(1))
    assert "font-weight: 700" in regra and tam >= 18.66


def test_legenda_sobre_video_legivel_em_qualquer_imagem():
    """A legenda do Tempo real fica sobre o video: o fundo translucido e composto sobre o pior caso
    (imagem branca e preta) e o texto precisa de 4.5:1 nos dois."""
    h = tema.COR["legenda_fundo"]
    a = int(h[7:9], 16) / 255
    r, g, b = tema._rgb(h)
    for base in (255, 0):
        comp = tema.hex_de([a * c + (1 - a) * base for c in (r, g, b)])
        for nome in ("legenda_texto", "legenda_parcial"):
            assert tema.contraste(tema.COR[nome], comp) >= 4.5, (nome, comp)


def test_sem_cor_fora_dos_tokens():
    proibido = re.compile(r"#[0-9A-Fa-f]{3,8}\b|[\"'](white|black|red|green|blue|gray|grey)[\"']|QColor\(\s*\d")
    achados = []
    for arq in (a for pasta in PASTAS_INTERFACE if pasta.exists() for a in pasta.rglob("*.py")):
        for n, linha in enumerate(arq.read_text(encoding="utf-8").splitlines(), 1):
            if proibido.search(linha.split("#  ")[0]) and not linha.strip().startswith("#"):
                achados.append(f"{arq.relative_to(UI.parent)}:{n}: {linha.strip()}")
    assert not achados, "Cores fora de tokens.json:\n" + "\n".join(achados)


def test_folha_de_estilo_por_faixa_e_escala():
    try:
        a = tema.folha_de_estilo(escala=1.0, faixa="expandida")
        b = tema.folha_de_estilo(escala=1.5, faixa="compacta")
        assert "font-size: 14px" in a and "font-size: 21px" in b
        # alvo de toque de 44 px nas faixas compacta e media: botao = conteudo + 2x9 de padding + 2 de borda
        assert re.search(r"min-height: 24px", b) and not re.search(
            r"QPushButton, QToolButton#acao \{[^}]*min-height: 24px", a
        )
    finally:
        tema.configurar(escala_fonte=1.0, faixa="expandida")


def test_galeria_abre(cfg):
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    from libras.app.janela import JanelaPrincipal

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    w = JanelaPrincipal(cfg, transicao_ms=0)
    try:
        g = w.abrir_galeria()
        app.processEvents()
        assert g.isVisible() and len(g.secoes) >= 6
        botoes = g.findChildren(QtWidgets.QPushButton)
        assert any(b.property("estado") == "pressionado" for b in botoes)
        assert any(not b.isEnabled() for b in botoes)
        g.close()
    finally:
        w.close()
        w.deleteLater()
