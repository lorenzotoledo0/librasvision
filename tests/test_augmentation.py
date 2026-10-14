"""Propriedades da data augmentation. Janelas SINTETICAS (esqueletos geometricos, nao Libras)."""

import numpy as np
import pytest

from conftest import partes_sinteticas
from libras.augmentation import Aumentador, gerar_copias_offline
from libras.augmentation.transformacoes import (
    ParamEspacial,
    ParamRuido,
    deslocar,
    espacial,
    espelhar,
    remover_frames,
    ruido,
    sortear_espacial,
    time_warp,
)
from libras.erros import ErroConfig
from libras.normalizacao import OMBRO_D, OMBRO_E, desmontar, espec_de_config, montar, preparar_janela


@pytest.fixture
def espec(cfg):
    return espec_de_config(cfg)


def janela(cfg, espec, seed=0, mao_esq=True):
    p = partes_sinteticas(30, mover_mao=(0, 30), seed=seed)
    if mao_esq:  # adiciona mao esquerda espelhando a direita
        p.mao_esq[:] = p.mao_dir
        p.mao_esq[..., 0] = 1.0 - p.mao_dir[..., 0]
        p.presenca[:, 1] = True
    r = preparar_janela(p, 640, 480, espec, cfg["normalizacao"])
    assert r.valida
    return r.X


def dist_mao(d, nome):
    m = d[nome][..., :2]
    return np.linalg.norm(m[:, :, None, :] - m[:, None, :, :], axis=-1)


def test_desmontar_montar_identidade(cfg, espec):
    X = janela(cfg, espec)
    np.testing.assert_array_equal(montar(desmontar(X, espec), espec), X)


@pytest.mark.parametrize("fam", ["espacial", "temporal", "ruido"])
def test_forma_sem_nan_e_mascara_coerente(cfg, espec, fam):
    cfg["augmentation"][fam]["prob"] = 1.0
    aum = Aumentador(cfg, espec, familias=[fam], intensidade="forte")
    x = np.stack([janela(cfg, espec, seed=s) for s in range(6)])
    x[2, 5:20, espec.blocos["mao_dir"][0] : espec.blocos["mao_dir"][1]] = 0  # mao ausente num trecho
    x[2, 5:20, espec.blocos["presenca"][0] + 2] = 0
    y = aum(x, np.random.default_rng(1))
    assert y.shape == x.shape and y.dtype == np.float32 and np.isfinite(y).all()
    pi = espec.blocos["presenca"][0]
    for nome, k in (("pose", 0), ("mao_esq", 1), ("mao_dir", 2), ("face", 3)):
        i, f = espec.blocos[nome]
        ausente = y[:, :, pi + k] < 0.5
        assert np.all(y[ausente][:, i:f] == 0), f"{nome} ausente com coordenadas != 0"
    assert set(np.unique(y[:, :, pi:])) <= {0.0, 1.0}
    assert aum.contagem[fam] == 6


def test_determinismo_por_seed(cfg, espec):
    for f in ("espacial", "temporal", "ruido"):
        cfg["augmentation"][f]["prob"] = 1.0
    aum = Aumentador(cfg, espec, familias=["espacial", "temporal", "ruido"])
    x = np.stack([janela(cfg, espec, seed=s) for s in range(4)])
    a = aum(x, np.random.default_rng(123))
    b = aum(x, np.random.default_rng(123))
    c = aum(x, np.random.default_rng(124))
    np.testing.assert_array_equal(a, b)
    assert not np.allclose(a, c)


def test_desligado_e_identidade(cfg, espec):
    aum = Aumentador(cfg, espec, familias=[])
    x = np.stack([janela(cfg, espec)])
    assert aum(x, np.random.default_rng(0)) is x


def test_rotacao_preserva_geometria_e_angulo(cfg, espec):
    d = desmontar(janela(cfg, espec), espec)
    ang = np.deg2rad(12)
    e = espacial(d, ParamEspacial(angulo_rad=ang))
    # distancias dentro de cada mao: inalteradas (rotacao preserva a largura dos ombros)
    for nome in ("mao_esq", "mao_dir"):
        np.testing.assert_allclose(dist_mao(e, nome), dist_mao(d, nome), atol=1e-4)
    v0 = d["pose"][:, OMBRO_E, :2] - d["pose"][:, OMBRO_D, :2]
    v1 = e["pose"][:, OMBRO_E, :2] - e["pose"][:, OMBRO_D, :2]
    giro = np.arctan2(v1[:, 1], v1[:, 0]) - np.arctan2(v0[:, 1], v0[:, 0])
    np.testing.assert_allclose(giro, ang, atol=1e-4)


def test_escala_da_mao_muda_distancias_por_um_unico_fator(cfg, espec):
    d = desmontar(janela(cfg, espec), espec)
    e = espacial(d, ParamEspacial(escala_mao_dir=1.08, angulo_rad=np.deg2rad(-7)))
    a, b = dist_mao(d, "mao_dir"), dist_mao(e, "mao_dir")
    razao = b[a > 1e-6] / a[a > 1e-6]
    np.testing.assert_allclose(razao, 1.08, atol=1e-3)
    np.testing.assert_allclose(dist_mao(e, "mao_esq"), dist_mao(d, "mao_esq"), atol=1e-4)


def test_escala_e_translacao_globais_sao_absorvidas_pela_normalizacao(cfg, espec):
    """Zoom e deslocamento globais (Islam et al., 2019, em imagem) nao tem efeito em landmarks
    normalizados pelo corpo: a renormalizacao os desfaz. Este teste documenta esse fato."""
    d = desmontar(janela(cfg, espec), espec)
    e = espacial(d, ParamEspacial(sx=1.1, sy=1.1))
    np.testing.assert_allclose(montar(e, espec), montar(d, espec), atol=1e-5)


def test_referencia_dos_ombros_mantida(cfg, espec):
    d = desmontar(janela(cfg, espec), espec)
    e = espacial(d, ParamEspacial(sx=1.08, sy=0.93, cisalhamento=0.1, angulo_rad=0.2))
    oe, od = e["pose"][:, OMBRO_E, :2], e["pose"][:, OMBRO_D, :2]
    np.testing.assert_allclose(np.median((oe + od) / 2, axis=0), 0, atol=1e-5)
    np.testing.assert_allclose(np.median(np.linalg.norm(oe - od, axis=1)), 1, atol=1e-5)


def test_magnitudes_dentro_da_faixa(cfg):
    c = cfg["augmentation"]["espacial"]
    rng = np.random.default_rng(0)
    for m in (0.5, 1.0, 1.5):
        for _ in range(300):
            p = sortear_espacial(rng, c, m)
            assert abs(np.rad2deg(p.angulo_rad)) <= c["rotacao_graus"] * m + 1e-9
            assert abs(p.sx - 1) <= c["escala_xy"] * m + 1e-9 and abs(p.sy - 1) <= c["escala_xy"] * m + 1e-9
            assert abs(p.escala_mao_dir - 1) <= c["escala_maos"] * m + 1e-9
            assert not p.espelhar


def test_time_warp_e_deslocamento(cfg, espec):
    d = desmontar(janela(cfg, espec), espec)
    np.testing.assert_allclose(time_warp(d, 1.0)["mao_dir"], d["mao_dir"], atol=1e-6)
    w = time_warp(d, 1.2)
    assert w["mao_dir"].shape == d["mao_dir"].shape
    # o frame central nao se move; as bordas puxam frames mais distantes (sinal mais rapido)
    np.testing.assert_allclose(w["mao_dir"][14:16].mean(0), d["mao_dir"][14:16].mean(0), atol=0.05)
    s = deslocar(d, 2)
    np.testing.assert_allclose(s["mao_dir"][:28], d["mao_dir"][2:], atol=1e-6)
    np.testing.assert_allclose(s["mao_dir"][28:], np.repeat(d["mao_dir"][29:], 2, 0), atol=1e-6)


def test_remocao_de_frames_reinterpola(cfg, espec):
    d = desmontar(janela(cfg, espec), espec)
    r = remover_frames(d, np.array([10]))
    np.testing.assert_allclose(r["mao_dir"][10], (d["mao_dir"][9] + d["mao_dir"][11]) / 2, atol=1e-5)
    np.testing.assert_array_equal(r["mao_dir"][9], d["mao_dir"][9])
    assert remover_frames(d, np.array([0, 29]))["mao_dir"].tolist() == d["mao_dir"].tolist()


def test_oclusao_e_perda_de_face_atualizam_mascara(cfg, espec):
    d = desmontar(janela(cfg, espec), espec)
    r = ruido(d, ParamRuido(oclusao=("mao_esq", 5, 12), perda_face=(0, 3)))
    assert (r["presenca"][5:12, 1] == 0).all() and (r["presenca"][:5, 1] == 1).all()
    assert (r["mao_esq"][5:12] == 0).all() and (r["face"][:3] == 0).all()
    assert (r["presenca"][:, 0] == 1).all()


def test_jitter_proporcional_ao_sigma(cfg, espec):
    d = desmontar(janela(cfg, espec), espec)
    r = ruido(d, ParamRuido(sigma=0.01, semente_jitter=3))
    dif = (r["mao_dir"] - d["mao_dir"])[..., :2]
    assert 0.005 < dif.std() < 0.02


def test_espelhamento_bloqueado_com_face(cfg, espec):
    cfg["augmentation"]["espacial"]["espelhamento"] = True
    with pytest.raises(ErroConfig):
        Aumentador(cfg, espec, familias=["espacial"])


def test_espelhamento_troca_lados(cfg):
    cfg["features"]["face"]["modo"] = "nenhuma"
    espec = espec_de_config(cfg)
    d = desmontar(janela(cfg, espec), espec)
    d["presenca"][:, 1] = 0
    d["mao_esq"][:] = 0
    e = espelhar(d)
    assert (e["presenca"][:, 2] == 0).all() and (e["presenca"][:, 1] == 1).all()
    np.testing.assert_allclose(e["mao_esq"][..., 0], -d["mao_dir"][..., 0])
    np.testing.assert_allclose(e["pose"][:, 11, 0], -d["pose"][:, 12, 0])


def test_rotulo_nao_e_tocado(cfg, espec):
    """O aumentador so recebe x; o rotulo segue intacto pelo gerador do dataset."""
    import inspect

    assert list(inspect.signature(Aumentador.__call__).parameters) == ["self", "x", "rng"]


def test_copias_offline_reprodutiveis(cfg, espec):
    for f in ("espacial", "temporal", "ruido"):
        cfg["augmentation"][f]["prob"] = 1.0
    aum = Aumentador(cfg, espec, familias=["espacial", "ruido"])
    x = np.stack([janela(cfg, espec, seed=s) for s in range(3)]).astype(np.float16)
    a = gerar_copias_offline(x, aum, 2, seed=5)
    b = gerar_copias_offline(x, aum, 2, seed=5)
    assert a.shape == (6,) + x.shape[1:]
    np.testing.assert_array_equal(a, b)
