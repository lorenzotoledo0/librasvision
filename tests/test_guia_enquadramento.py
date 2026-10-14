"""13.9-A: medidas de enquadramento no servico (ombros, mao cortada, brilho, cabeca) e avisos com
histerese. Landmarks e quadros SINTETICOS (gerados por codigo)."""

import re
import time
from pathlib import Path

import numpy as np
import pytest

from libras import decisao
from libras.app_real.guia import TEXTOS, GuiaAvisos, Histerese, condicoes, itens_primeira_abertura
from libras.enquadramento import MedidorEnquadramento, brilho_medio, caixa_cabeca, lados_mao_cortada, situacao_ombros
from libras.landmarks import N_FACE, N_MAO, N_POSE, LandmarksFrame

PARAMS = {"borda_mao": 0.03, "brilho_minimo": 60, "visibilidade_ombro": 0.5, "janela_s": 1.0}


def pose_sintetica(ombros=((0.4, 0.6), (0.6, 0.6)), vis=0.99):
    p = np.full((N_POSE, 4), np.nan, np.float32)
    for i in range(11):  # rosto ao redor de (0.5, 0.3)
        p[i] = [0.47 + 0.006 * i, 0.28 + 0.004 * (i % 3), 0, vis]
    p[7] = [0.45, 0.3, 0, vis]
    p[8] = [0.55, 0.3, 0, vis]
    for k, (x, y) in zip((11, 12), ombros):
        p[k] = [x, y, 0, vis]
    return p


def mao_em(cx, cy, raio=0.04):
    a = np.linspace(0, 2 * np.pi, N_MAO)
    return np.stack([cx + raio * np.cos(a), cy + raio * np.sin(a), np.zeros(N_MAO)], 1).astype(np.float32)


def frame(pose=None, me=None, md=None):
    nan = lambda n, c: np.full((n, c), np.nan, np.float32)  # noqa: E731
    return LandmarksFrame(
        pose if pose is not None else nan(N_POSE, 4),
        me if me is not None else nan(N_MAO, 3),
        md if md is not None else nan(N_MAO, 3),
        nan(N_FACE, 3),
        np.array([pose is not None, me is not None, md is not None, False]),
    )


CLARO = np.full((480, 640, 3), 150, np.uint8)
ESCURO = np.full((480, 640, 3), 20, np.uint8)


def test_ombros_ok_parcial_e_fora():
    assert situacao_ombros(pose_sintetica(), True, 0.5) == "ok"
    assert situacao_ombros(pose_sintetica(ombros=((0.4, 0.6), (0.6, 1.08))), True, 0.5) == "parcial"  # um fora
    assert situacao_ombros(pose_sintetica(ombros=((0.4, 1.1), (0.6, 1.2))), True, 0.5) == "fora"
    assert situacao_ombros(pose_sintetica(vis=0.1), True, 0.5) == "fora"
    assert situacao_ombros(pose_sintetica(), False, 0.5) == "fora"


def test_mao_cortada_so_com_mao_detectada():
    assert lados_mao_cortada([mao_em(0.5, 0.6)], 0.03) == []
    assert lados_mao_cortada([mao_em(0.02, 0.6)], 0.03) == ["esquerda"]  # lado da imagem ORIGINAL
    assert lados_mao_cortada([mao_em(0.97, 0.98)], 0.03) == ["direita", "base"]
    assert lados_mao_cortada([], 0.03) == []  # mao ausente (repouso fora da imagem) nao gera aviso


def test_brilho_e_custo_abaixo_de_1_ms():
    assert brilho_medio(ESCURO) < 60 < brilho_medio(CLARO)
    m = MedidorEnquadramento(PARAMS, {"normalizacao": {"fracao_minima_pose": 0.5}}, {"min_fracao_mao": 0.3})
    f = frame(pose_sintetica(), md=mao_em(0.5, 0.6))
    tempos = []
    for i in range(200):
        t0 = time.perf_counter()
        r = m.medir(CLARO, f, i / 30)
        tempos.append((time.perf_counter() - t0) * 1000)
    assert np.median(tempos) < 1.0, f"medida custa {np.median(tempos):.3f} ms por frame"
    assert r["ombros"] == "ok" and r["mao_cortada"] == [] and not r["pouca_luz"] and r["pessoa"]
    assert r["cabeca"] is not None and r["cabeca"][1] < 0.28  # a caixa sobe acima dos olhos (testa)


def test_fracoes_usam_os_minimos_do_modelo():
    meta = {"normalizacao": {"fracao_minima_pose": 0.5}}
    m = MedidorEnquadramento(PARAMS, meta, {"min_fracao_mao": 0.3})
    assert m.min_pose == decisao.fracao_minima_pose(meta) and m.min_mao == decisao.min_fracao_mao(
        {"min_fracao_mao": 0.3}
    )
    # 4 de 10 frames com pose (40% < 50%): sem pessoa
    for i in range(10):
        r = m.medir(CLARO, frame(pose_sintetica() if i < 4 else None), i / 10)
    assert not r["pessoa"] and r["fracao_pose"] == pytest.approx(0.4)
    m2 = MedidorEnquadramento(PARAMS, None, None)  # sem metadados: padroes do modulo comum
    assert (
        m2.min_pose == decisao.FRACAO_MINIMA_POSE_PADRAO and m2.min_mao == decisao.PARAMETROS_PADRAO["min_fracao_mao"]
    )


def test_constantes_de_50_e_30_por_cento_nao_sao_copiadas():
    """As fracoes minimas de pose (50%) e de maos (30%) so existem em libras.decisao."""
    raiz = Path(__file__).resolve().parents[1] / "src" / "libras"
    proibido = re.compile(r"(?<![\w.])0?\.(3|5)0*(?![\d])|(?<![\w.])(30|50)\s*%")
    for arq in [raiz / "enquadramento.py", raiz / "app_real" / "guia.py"]:
        for n, linha in enumerate(arq.read_text(encoding="utf-8").splitlines(), 1):
            codigo = linha.split("#")[0]
            assert not proibido.search(codigo), f"{arq.name}:{n}: {linha.strip()}"


def test_cabeca_sem_pontos_suficientes():
    assert caixa_cabeca(np.full((N_POSE, 4), np.nan, np.float32)) is None


# ---------------------------------------------------------------------- avisos
def enq(pessoa=True, ombros="ok", lados=(), pouca_luz=False, maos=True):
    return {
        "pessoa": pessoa,
        "ombros": ombros,
        "mao_cortada": list(lados),
        "pouca_luz": pouca_luz,
        "maos_suficientes": maos,
    }


def test_condicoes_e_mao_ausente():
    assert condicoes(enq()) == {"sem_pessoa": False, "ombros": False, "mao_cortada": False, "pouca_luz": False}
    assert condicoes(enq(pessoa=False, ombros="fora"))["sem_pessoa"]
    assert not condicoes(enq(pessoa=False, ombros="fora"))["ombros"]  # sem pessoa cobre os ombros
    assert condicoes(enq(ombros="parcial"))["ombros"]
    assert not any(condicoes(enq(maos=False)).values())  # sem maos visiveis nao e aviso


def test_histerese_de_1_s_nos_dois_sentidos():
    h = Histerese(1.0)
    assert not h.atualizar(True, 0.0) and not h.atualizar(True, 0.9)
    assert h.atualizar(True, 1.0)  # durou 1 s: aparece
    assert h.atualizar(False, 1.5)  # sumiu ha 0,5 s: continua
    assert h.atualizar(True, 1.6)  # voltou: contagem zera
    assert h.atualizar(False, 2.0) and h.atualizar(False, 2.99)
    assert not h.atualizar(False, 3.0)  # 1 s sem ocorrer: some
    h2 = Histerese(1.0)
    for t in np.arange(0, 3, 0.1):  # piscando a cada 0,1 s nunca aparece
        assert not h2.atualizar(int(t * 10) % 2 == 0, float(t))


def test_um_aviso_por_vez_na_ordem_de_prioridade():
    g = GuiaAvisos(1.0)
    tudo = enq(pessoa=True, ombros="parcial", lados=["esquerda"], pouca_luz=True)
    assert g.atualizar(tudo, 0.0) is None
    assert g.atualizar(tudo, 1.0) == "ombros"  # prioridade: sem pessoa > ombros > mao cortada > pouca luz
    sem = enq(pessoa=False, ombros="fora", pouca_luz=True)
    g.atualizar(sem, 1.1)
    assert g.atualizar(sem, 2.2) == "sem_pessoa"
    ok_luz = enq(lados=["direita"], pouca_luz=True)
    assert g.atualizar(ok_luz, 2.3) == "sem_pessoa"  # histerese para sumir
    assert g.atualizar(ok_luz, 3.4) == "mao_cortada" and g.lados == ["direita"]
    assert g.texto("pouca_luz") == TEXTOS["pouca_luz"] and "luz" in TEXTOS["pouca_luz"]
    for t in TEXTOS.values():  # o texto nao usa esquerda/direita (rotulo do Holistic nao conferido, D3)
        assert "esquerda" not in t and "direita" not in t
    g.habilitado = False
    assert g.atualizar(ok_luz, 3.5) is None  # chave que desliga os avisos


def test_lista_da_primeira_abertura():
    assert [ok for _, ok in itens_primeira_abertura(enq())] == [True, True, True]
    assert [ok for _, ok in itens_primeira_abertura(enq(ombros="fora", maos=False, pouca_luz=True))] == [False] * 3
    assert [ok for _, ok in itens_primeira_abertura(None)] == [False] * 3


def test_fracoes_minimas_sem_copia_em_todo_o_codigo():
    """Nenhum modulo (alem de libras.decisao) define as fracoes minimas de pose e de maos com um numero."""
    raiz = Path(__file__).resolve().parents[1] / "src" / "libras"
    nome = r"(fracao_minima_pose|min_fracao_mao|fracao_pose|fracao_mao)"
    padroes = [
        re.compile(nome + r"\s*(:\s*float)?\s*=\s*\(?\s*0?\.\d"),
        re.compile(r"get\(\s*[\"']" + nome + r"[\"']\s*,\s*0?\.\d"),
    ]
    for arq in raiz.rglob("*.py"):
        if arq.name == "decisao.py":
            continue
        for n, linha in enumerate(arq.read_text(encoding="utf-8").splitlines(), 1):
            codigo = linha.split("#")[0]
            assert not any(p.search(codigo) for p in padroes), f"{arq.relative_to(raiz)}:{n}: {linha.strip()}"
