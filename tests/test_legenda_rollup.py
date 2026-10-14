"""13.4, 13.9-C e 13.9-D: legenda roll-up, estilo e contraste, destaque da palavra e pausa.
Logica pura com relogio injetado; palavras SINTETICAS (nao vem de modelo)."""

import pytest

from libras.ui import tema
from libras.ui.legenda import TOK, RollUp, estilo_de


def rollup(linhas=2, reduzir=False, destacar=True, largura=20):
    r = RollUp(linhas, sumir_s=4.0, reduzir_movimento=reduzir, destacar=destacar, relogio=lambda: 0.0)
    r.cabe = lambda s: len(s) <= largura
    return r


def textos(r, t):
    return [" ".join(w for w, _ in l) for l in r.estado(t).linhas]


def test_encher_subir_e_a_mais_antiga_sai():
    r = rollup(linhas=2, largura=12)
    for i, p in enumerate(["um", "dois", "tres", "quatro", "cinco", "seis"]):
        r.adicionar(p, t=i * 0.5)
    # "um dois tres" (12) cabe; "quatro" abre linha; "cinco" ainda cabe; "seis" abre a 3a e a 1a sai
    assert textos(r, 2.6) == ["quatro cinco", "seis"]
    assert r.estado(2.5).saindo is not None and r.estado(2.5).progresso_rolagem == 0.0  # rolagem comecando
    assert r.estado(2.5 + 0.2).saindo is None  # depois de 150 ms acabou
    assert r.texto_completo() == "um dois tres quatro cinco seis"


def test_tres_linhas():
    r = rollup(linhas=3, largura=5)
    for i, p in enumerate(["aa", "bb", "cc", "dd", "ee"]):
        r.adicionar(p, t=i)
    assert textos(r, 4.0) == ["aa bb", "cc dd", "ee"]
    r.adicionar("ff", t=5)
    r.adicionar("gg", t=6)
    assert textos(r, 6.5) == ["cc dd", "ee ff", "gg"]


def test_esmaece_depois_de_T_segundos():
    r = rollup()
    r.adicionar("oi", t=0.0)
    assert r.estado(3.9).opacidade == 1.0
    assert 0.0 < r.estado(4.15).opacidade < 1.0  # esmaecendo (300 ms)
    r.atualizar(4.4)
    assert r.estado(4.4).linhas == [] and r.texto_visivel() == ""
    r.adicionar("nova", t=10.0)  # recomeca numa linha nova
    assert textos(r, 10.0) == ["nova"]


def test_reduzir_movimento_troca_instantanea():
    r = rollup(linhas=2, largura=5, reduzir=True)
    for i, p in enumerate(["aa", "bb", "cc", "dd", "ee"]):
        r.adicionar(p, t=i * 0.1)
    est = r.estado(0.41)
    assert est.saindo is None and est.progresso_rolagem == 1.0  # sem animacao de rolagem
    assert r.estado(4.39).opacidade == 1.0 and r.estado(4.41).opacidade == 0.0  # some de uma vez


def test_refluir_quando_a_largura_muda():
    r = rollup(linhas=3, largura=30)
    for p in ["um", "dois", "tres", "quatro"]:
        r.adicionar(p, t=0)
    assert textos(r, 0) == ["um dois tres quatro"]
    r.cabe = lambda s: len(s) <= 9
    r.refluir()
    assert textos(r, 0) == ["um dois", "tres", "quatro"]


# ---------------------------------------------------------------------- destaque (13.9-C)
def test_um_destaque_por_palavra_e_volta_ao_normal():
    r = rollup()
    r.adicionar("oi", t=0.0)
    est = r.estado(0.1)
    assert est.destaque == "pulso" and est.linhas[-1][-1] == ("oi", True)
    assert r.estado(0.31).destaque is None and r.estado(0.31).linhas[-1][-1] == ("oi", False)
    r.adicionar("tudo", t=2.0)
    est = r.estado(2.1)
    assert [d for _, d in est.linhas[-1]] == [False, True]  # so a palavra nova


def test_reduzir_movimento_sublinhado_por_1_2_s():
    r = rollup(reduzir=True)
    r.adicionar("oi", t=0.0)
    assert r.estado(1.1).destaque == "reduzido" and r.estado(1.25).destaque is None


@pytest.mark.parametrize("reduzir", [False, True])
@pytest.mark.parametrize("intervalo", [0.2, 0.4, 0.8])
def test_no_maximo_3_mudancas_por_segundo(reduzir, intervalo):
    """Confirmacoes em sequencia: o destaque (liga, desliga ou passa de palavra) muda no maximo 3 vezes
    em qualquer janela de 1 s (WCAG 2.3.1)."""
    r = rollup(reduzir=reduzir, largura=500)
    chegadas = [round(i * intervalo, 2) for i in range(int(6 / intervalo))]
    eventos, ant, t = [], None, 0.0
    while t < 8.0:
        while chegadas and chegadas[0] <= t:
            r.adicionar(f"p{len(r.historico)}", t=chegadas.pop(0))
        est = r.estado(t)
        chave = tuple(w for l in est.linhas for w, d in l if d)
        if ant is not None and chave != ant:
            eventos.append(t)
        ant = chave
        t = round(t + 0.01, 2)
    assert len(eventos) >= 2  # houve destaque
    for a in eventos:
        assert sum(1 for e in eventos if a <= e < a + 1.0) <= 3, eventos


def test_sem_destaque_quando_desligado():
    r = rollup(destacar=False)
    r.adicionar("oi", t=0.0)
    assert r.estado(0.1).destaque is None


# ---------------------------------------------------------------------- pausa (13.9-D)
def test_pausa_congela_buffer_contador_e_ordem():
    r = rollup(linhas=2, largura=40)
    r.adicionar("um", t=0.0)
    r.adicionar("dois", t=0.5)
    r.pausar()
    congelado = r.texto_visivel()
    for i, p in enumerate(["tres", "quatro", "cinco"]):
        r.adicionar(p, t=1.0 + i)
    est = r.estado(30.0)  # pausada: nao esmaece nem rola
    assert est.pausado and est.novas == 3 and est.opacidade == 1.0
    assert r.texto_visivel() == congelado == "um dois"
    r.retomar(t=31.0)
    assert r.texto_visivel() == "um dois tres quatro cinco"
    assert r.estado(31.0).destaque is None  # muitas de uma vez: sem animacao nem destaque
    assert r.estado(31.0).saindo is None


def test_pausa_depois_de_esmaecer_volta_o_ultimo_texto():
    r = rollup()
    r.adicionar("oi", t=0.0)
    r.atualizar(10.0)
    assert r.texto_visivel() == ""
    r.pausar()
    assert r.texto_visivel() == "oi" and r.estado(10.0).opacidade == 1.0


def test_limpar_limpa_tela_buffer_e_retoma():
    r = rollup()
    r.adicionar("um", t=0)
    r.pausar()
    r.adicionar("dois", t=1)
    r.limpar()
    assert not r.pausado and r.buffer == [] and r.texto_visivel() == "" and r.texto_completo() == ""


def test_sequencia_identica_com_e_sem_pausa():
    seq = ["biblioteca", "prova", "secretaria", "prova", "biblioteca", "secretaria"]
    a, b = rollup(largura=200), rollup(largura=200)
    for i, p in enumerate(seq):
        a.adicionar(p, t=i)
        if i == 2:
            b.pausar()
        b.adicionar(p, t=i)
        if i == 4:
            b.retomar(t=i)
    assert a.texto_completo() == b.texto_completo() == " ".join(seq)
    assert a.texto_visivel() == b.texto_visivel()


# ---------------------------------------------------------------------- estilo e contraste
@pytest.mark.parametrize("pred", list(TOK["predefinicoes"]))
def test_contraste_das_predefinicoes(pred):
    e = estilo_de({"predefinicao": pred})
    assert tema.contraste_pior_caso(e.texto, e.fundo, e.opacidade) >= 4.5
    assert tema.contraste_pior_caso(e.parcial, e.fundo, e.opacidade) >= 4.5
    assert e.destaque and tema.contraste(e.texto, e.destaque) >= 4.5  # palavra destacada continua legivel
    assert not e.avisos


def test_personalizada_com_pouco_contraste_avisa_mas_nao_bloqueia():
    e = estilo_de({"predefinicao": "personalizada", "cor_texto": "#777777", "cor_fundo": "#888888", "opacidade": 0.3})
    assert e.avisos and "4,5:1" in e.avisos[0]
    assert e.texto == "#777777"  # a escolha e respeitada
    e2 = estilo_de({"predefinicao": "personalizada", "cor_texto": "#FFFFFF", "cor_fundo": "#000000", "opacidade": 1.0})
    assert not e2.avisos and e2.destaque == tema.COR["selecao"]


def test_preferencias_padrao():
    e = estilo_de(None)
    assert e.linhas == 2 and e.sumir_s == 4.0 and e.posicao == "embaixo" and e.destacar
    assert e.tamanho_px == TOK["tamanhos_px"][TOK["tamanho_padrao"]]


def test_transcritor_da_a_mesma_sequencia_com_e_sem_pausa():
    """A pausa e so da tela: o Transcritor (D15) recebe as mesmas predicoes e confirma as mesmas palavras;
    a legenda pausada entrega todas, na ordem, ao retomar. Predicoes SINTETICAS."""
    from libras.inferencia import Predicao, Transcritor

    seq = (["a"] * 3 + [None] * 2 + ["b"] * 4 + [None] + ["a"] * 3 + ["c"] * 3 + [None] * 2 + ["b"] * 3) * 2

    def rodar(pausar_em=None, retomar_em=None):
        tr, r = Transcritor(3, 100), rollup(largura=500)
        aceitos = []
        for i, c in enumerate(seq):
            if i == pausar_em:
                r.pausar()
            if i == retomar_em:
                r.retomar(t=i * 0.1)
            p = Predicao(i * 0.1, c, 0.9 if c else 0.0, None, "" if c else "nenhum_sinal")
            a = tr.atualizar(p)
            if a:
                aceitos.append(a)
                r.adicionar(a, t=i * 0.1)
        return aceitos, tr.texto, r

    a1, t1, r1 = rodar()
    a2, t2, r2 = rodar(pausar_em=5, retomar_em=30)
    assert a1 == a2 and t1 == t2 and len(a1) >= 6
    assert r1.texto_completo() == r2.texto_completo() == " ".join(a1)
    assert r1.texto_visivel() == r2.texto_visivel()
