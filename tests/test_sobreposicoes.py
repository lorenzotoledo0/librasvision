"""13.4, 13.8 e 13.9: gerenciador unico de posicoes. Nenhum retangulo (legenda, barra tecnica, etiqueta
de enquadramento, botao de Configuracoes, barra minima com o selo) cruza outro, nas tres posicoes da
legenda e nas tres faixas, com tudo ligado e com a cabeca simulada (inclusive no topo)."""

import itertools

import pytest

from libras.ui.sobreposicoes import ALVO, MARGEM, Ret, SuavizadorCabeca, cabeca_na_tela, lado_na_tela, posicionar

# (janela, altura da barra minima, linhas da barra tecnica) nas tres faixas (390, 900 e 1366 px)
FAIXAS = {
    "compacta": (Ret(0, 0, 390, 844), 127, 2),
    "media": (Ret(0, 0, 900, 700), 99, 1),
    "expandida": (Ret(0, 0, 1366, 800), 99, 1),
}
CABECAS = {  # caixas normalizadas da imagem original (SINTETICAS)
    "centro": [0.42, 0.35, 0.58, 0.55],
    "no_topo": [0.40, 0.0, 0.60, 0.15],
    "na_borda": [0.0, 0.2, 0.15, 0.4],
    "perdida": None,
}


def imagem_de(jan: Ret, bm: float, faixa: str) -> Ret:
    h_disp = jan.h - (bm if faixa == "compacta" else 0)
    w = min(jan.w, h_disp * 4 / 3)
    h = w / (4 / 3)
    return Ret((jan.w - w) / 2, (h_disp - h) / 2, w, h)


def tamanho_legenda(faixa: str, linhas: int = 3):
    return ({"compacta": 300, "media": 520, "expandida": 760}[faixa], 32 + linhas * 44 + 26)


CHAVES = ("legenda", "barra_tecnica", "etiqueta", "botao_config", "barra_minima")


@pytest.mark.parametrize(
    "faixa,posicao,cabeca", list(itertools.product(FAIXAS, ("embaixo", "pessoa", "topo"), CABECAS))
)
def test_nada_se_cruza_com_tudo_ligado(faixa, posicao, cabeca):
    jan, bm, linhas_tec = FAIXAS[faixa]
    img = imagem_de(jan, bm, faixa)
    out = posicionar(
        jan,
        img,
        legenda=tamanho_legenda(faixa),
        posicao=posicao,
        cabeca=cabeca_na_tela(CABECAS[cabeca], img),
        botao_config=True,
        barra_tecnica=max(ALVO, linhas_tec * 20 + 16),
        etiqueta=(520, 40 if faixa != "compacta" else 66),
        barra_minima=bm,
        bordas=["esquerda", "direita", "topo", "base"],
    )
    rets = {k: out[k] for k in CHAVES if out[k] is not None}
    assert set(rets) == set(CHAVES), f"faltou posicionar: {set(CHAVES) - set(rets)}"
    for (a, ra), (b, rb) in itertools.combinations(rets.items(), 2):
        assert not ra.cruza(rb), (faixa, posicao, cabeca, a, ra, b, rb)
    # a barra tecnica termina antes do botao (nunca passa por baixo dele)
    assert out["barra_tecnica"].direita <= out["botao_config"].x
    # legenda e barra tecnica ficam dentro da imagem; o botao, no canto superior direito da janela
    for k in ("legenda", "barra_tecnica"):
        assert out[k].dentro(img), (k, out[k], img)
    assert out["etiqueta"].dentro(jan) and out["botao_config"].dentro(jan)
    assert jan.direita - out["botao_config"].direita == MARGEM and out["botao_config"].y - jan.y == MARGEM
    for lado, r in out["bordas"].items():  # barras de mao cortada nao cobrem nada
        for k in CHAVES:
            assert not r.cruza(out[k]), (lado, k)


def test_posicoes_da_legenda():
    jan, bm, _ = FAIXAS["expandida"]
    img = imagem_de(jan, bm, "expandida")
    tam = tamanho_legenda("expandida", 2)
    cab = cabeca_na_tela(CABECAS["centro"], img)
    em = posicionar(jan, img, legenda=tam, posicao="embaixo", barra_minima=bm)
    assert em["legenda"].base == pytest.approx(min(img.base, jan.base - bm) - 8)
    pe = posicionar(jan, img, legenda=tam, posicao="pessoa", cabeca=cab, barra_minima=bm)
    assert pe["posicao_efetiva"] == "pessoa"
    assert pe["legenda"].base <= cab.y - 8 + 1e-6  # acima da cabeca, sem descer sobre o rosto
    assert pe["legenda"].centro_x == pytest.approx(cab.centro_x, abs=1)
    to = posicionar(jan, img, legenda=tam, posicao="topo", barra_minima=bm)
    assert to["legenda"].y == pytest.approx(img.y + 8 + ALVO + 8)  # logo abaixo da fila do botao
    # pessoa perdida (o chamador ja aplicou o 1 s): faixa fixa no alto
    assert posicionar(jan, img, legenda=tam, posicao="pessoa", cabeca=None)["posicao_efetiva"] == "topo"
    # cabeca no topo: sem espaco acima, a legenda nunca desce sobre o rosto (vai para baixo)
    alto = posicionar(jan, img, legenda=tam, posicao="pessoa", cabeca=cabeca_na_tela(CABECAS["no_topo"], img))
    assert alto["posicao_efetiva"] == "embaixo" and not alto["legenda"].cruza(cabeca_na_tela(CABECAS["no_topo"], img))


def test_barra_tecnica_reserva_o_topo_e_desligada_nao_existe():
    jan, bm, _ = FAIXAS["media"]
    img = imagem_de(jan, bm, "media")
    tam = tamanho_legenda("media", 2)
    sem = posicionar(jan, img, legenda=tam, posicao="topo", barra_minima=bm)
    com = posicionar(jan, img, legenda=tam, posicao="topo", barra_minima=bm, barra_tecnica=60)
    assert sem["barra_tecnica"] is None
    assert com["legenda"].y == pytest.approx(com["barra_tecnica"].base + 8)  # comeca logo abaixo da barra
    # pessoa: a margem do topo conta a partir da base da barra
    cab = cabeca_na_tela([0.4, 0.2, 0.6, 0.4], img)
    p = posicionar(jan, img, legenda=tam, posicao="pessoa", cabeca=cab, barra_tecnica=60, barra_minima=bm)
    assert p["legenda"].y >= p["barra_tecnica"].base + 8 or p["posicao_efetiva"] == "embaixo"
    # embaixo: nada muda com a barra
    a = posicionar(jan, img, legenda=tam, posicao="embaixo", barra_minima=bm)
    b = posicionar(jan, img, legenda=tam, posicao="embaixo", barra_minima=bm, barra_tecnica=60)
    assert a["legenda"] == b["legenda"]


def test_etiqueta_nunca_move_a_legenda():
    jan, bm, _ = FAIXAS["expandida"]
    img = imagem_de(jan, bm, "expandida")
    tam = tamanho_legenda("expandida", 2)
    for pos in ("embaixo", "topo"):
        a = posicionar(jan, img, legenda=tam, posicao=pos, barra_minima=bm)
        b = posicionar(jan, img, legenda=tam, posicao=pos, barra_minima=bm, etiqueta=(500, 40))
        assert a["legenda"] == b["legenda"] and b["etiqueta"] is not None


def test_espelhamento_da_cabeca_e_dos_lados():
    img = Ret(0, 0, 400, 300)
    c = cabeca_na_tela([0.1, 0.2, 0.3, 0.4], img)  # na imagem original a pessoa esta a esquerda
    assert c.x == pytest.approx(0.7 * 400) and c.direita == pytest.approx(0.9 * 400)  # na previa espelhada, a direita
    assert lado_na_tela("esquerda") == "direita" and lado_na_tela("direita") == "esquerda"
    assert lado_na_tela("topo") == "topo" and lado_na_tela("esquerda", espelhada=False) == "esquerda"


def test_suavizacao_e_perda_da_pessoa():
    s = SuavizadorCabeca(alfa=0.25, perda_s=1.0)
    assert s.atualizar([0.4, 0.2, 0.6, 0.4], 0.0) == [0.4, 0.2, 0.6, 0.4]
    v = s.atualizar([0.5, 0.2, 0.7, 0.4], 0.05)
    assert v[0] == pytest.approx(0.425)  # anda so 25% do salto: nao treme
    assert s.atualizar(None, 0.9) is not None  # sumiu ha menos de 1 s: mantem
    assert s.atualizar(None, 1.2) is None  # mais de 1 s: volta a faixa fixa
