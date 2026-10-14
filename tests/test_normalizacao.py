"""Normalizacao e janelas. Todos os esqueletos sao SINTETICOS."""

import numpy as np

from conftest import partes_sinteticas
from libras.normalizacao import (
    FACE_REGIOES,
    energia_maos,
    espec_de_config,
    inicios_janelas,
    interpolar_lacunas,
    janelas_do_video,
    preparar_janela,
    reamostrar,
    tempos_uniformes,
)

CN = {"interpolar_lacunas_max_frames": 5, "fracao_minima_pose": 0.5}


def test_dimensao_padrao(cfg):
    e = espec_de_config(cfg)
    assert e.dim == 132 + 126 + 96 * 3 + 4 == 550
    assert e.blocos["presenca"] == (546, 550)


def test_indices_face_iguais_ao_mediapipe():
    from libras.landmarks import indices_face

    for regiao, idx in FACE_REGIOES.items():
        assert indices_face([regiao]) == idx


def _como_pixels(p, larg, alt, dx=0.0, dy=0.0, s=1.0):
    """Reposiciona o esqueleto na imagem: (pixel*s + d) e reconverte para coordenadas normalizadas."""
    q = p.copia()
    for arr in q.lista():
        px = arr[..., 0] * 640 * s + dx
        py = arr[..., 1] * 480 * s + dy
        arr[..., 0] = px / larg
        arr[..., 1] = py / alt
        arr[..., 2] = arr[..., 2] * 640 * s / larg
    return q


def test_invariante_a_translacao_escala_e_proporcao(cfg):
    espec = espec_de_config(cfg)
    p = partes_sinteticas(30)
    a = preparar_janela(p, 640, 480, espec, CN)
    # mesma pessoa, mais longe/perto e deslocada, gravada em 1280x720 (16:9)
    q = _como_pixels(p, 1280, 720, dx=200, dy=50, s=1.3)
    b = preparar_janela(q, 1280, 720, espec, CN)
    assert a.valida and b.valida
    np.testing.assert_allclose(a.X, b.X, atol=1e-4)


def test_ombros_na_origem_e_largura_unitaria(cfg):
    espec = espec_de_config(cfg)
    r = preparar_janela(partes_sinteticas(30), 640, 480, espec, CN)
    pose = r.X[:, :132].reshape(30, 33, 4)
    centro = (pose[:, 11, :2] + pose[:, 12, :2]) / 2
    np.testing.assert_allclose(centro, 0, atol=1e-5)
    np.testing.assert_allclose(np.linalg.norm(pose[:, 11, :2] - pose[:, 12, :2], axis=1), 1, atol=1e-5)


def test_lacuna_curta_interpolada_longa_zerada(cfg):
    espec = espec_de_config(cfg)
    p = partes_sinteticas(30)
    p.presenca[10:13, 2] = False  # 3 frames: interpola
    p.mao_dir[10:13] = np.nan
    p.presenca[20:30, 2] = False  # borda final: nao interpola
    p.mao_dir[20:30] = np.nan
    r = preparar_janela(p, 640, 480, espec, CN)
    pres = r.X[:, espec.blocos["presenca"][0] + 2]
    assert pres[10:13].tolist() == [1, 1, 1]
    assert pres[20:].sum() == 0
    i, f = espec.blocos["mao_dir"]
    assert np.all(r.X[20:, i:f] == 0)
    assert not np.isnan(r.X).any()


def test_interpolacao_linear():
    p = partes_sinteticas(10)
    p.mao_dir[3:5] = np.nan
    p.presenca[3:5, 2] = False
    p.mao_dir[2] = 0.0
    p.mao_dir[5] = 0.3
    q = interpolar_lacunas(p, 5)
    np.testing.assert_allclose(q.mao_dir[3, 0, 0], 0.1, atol=1e-6)
    np.testing.assert_allclose(q.mao_dir[4, 0, 0], 0.2, atol=1e-6)


def test_sem_pose_janela_invalida(cfg):
    p = partes_sinteticas(30)
    p.presenca[:20, 0] = False
    r = preparar_janela(p, 640, 480, espec_de_config(cfg), CN)
    assert not r.valida and "pose" in r.motivo


def test_reamostrar_15_para_30fps():
    p = partes_sinteticas(16)
    p.mao_dir[:, 0, 0] = np.arange(16, dtype=np.float32)
    r = reamostrar(p, tempos_uniformes(16, 15.0), 30.0)
    assert r.n == 31
    np.testing.assert_allclose(r.mao_dir[:, 0, 0], np.arange(31) / 2.0, atol=1e-5)


def test_reamostrar_presenca_pelo_vizinho_mais_proximo():
    p = partes_sinteticas(4)
    p.presenca[2, 2] = False
    p.mao_dir[2] = np.nan
    r = reamostrar(p, tempos_uniformes(4, 10.0), 20.0)
    # instantes 0, .05, .10, ..., .30; o frame 2 (t=.20) esta ausente. Em .15 e .25 ha empate de
    # distancia (nao testado); nos demais pontos vale o vizinho mais proximo.
    assert r.n == 7
    assert r.presenca[[0, 1, 2, 6], 2].all() and not r.presenca[4, 2]
    assert np.isnan(r.mao_dir[4]).all() and not np.isnan(r.mao_dir[2]).any()


def test_inicios_janelas():
    assert inicios_janelas(75, 30, 5) == list(range(0, 46, 5))
    assert inicios_janelas(32, 30, 5) == [0, 2]
    assert inicios_janelas(20, 30, 5) == [0]


def test_janelas_selecao_por_atividade(cfg):
    espec = espec_de_config(cfg)
    p = partes_sinteticas(90, mover_mao=(30, 60))
    t = tempos_uniformes(90, 30.0)
    jv = janelas_do_video(p, t, 640, 480, espec, cfg, eh_nenhum_sinal=False)
    todas = janelas_do_video(p, t, 640, 480, espec, cfg, eh_nenhum_sinal=True)
    assert len(todas.X) == len(inicios_janelas(90, 30, 5)) == 13
    assert 0 < len(jv.X) < len(todas.X)
    # as janelas mantidas cobrem o trecho com movimento
    assert all(i + 30 > 30 and i < 60 for i in jv.inicios)
    assert jv.descartadas.get("pouco movimento (selecao por atividade)") == len(todas.X) - len(jv.X)


def test_video_curto_preenchido(cfg):
    p = partes_sinteticas(20, mover_mao=(0, 20))
    jv = janelas_do_video(p, tempos_uniformes(20, 30.0), 640, 480, espec_de_config(cfg), cfg, False)
    assert jv.X.shape == (1, 30, 550)
    assert any(a.startswith("video_curto") for a in jv.avisos)


def test_tempo_real_igual_ao_treino(cfg):
    """O buffer do tempo real (instantes irregulares) passa pelo mesmo caminho do treino."""
    espec = espec_de_config(cfg)
    p = partes_sinteticas(60, mover_mao=(0, 60))
    t = tempos_uniformes(60, 30.0)
    treino = janelas_do_video(p, t, 640, 480, espec, cfg, eh_nenhum_sinal=True, passo=30)
    # tempo real: ultimos 30 frames do buffer, reamostrados e preparados
    buf = reamostrar(p, t, 30.0)
    ao_vivo = preparar_janela(buf.fatia(buf.n - 30, buf.n), 640, 480, espec, cfg["normalizacao"])
    np.testing.assert_allclose(treino.X[-1], ao_vivo.X, atol=1e-6)


def test_energia_zero_sem_maos(cfg):
    espec = espec_de_config(cfg)
    p = partes_sinteticas(30)
    p.presenca[:, 1:3] = False
    r = preparar_janela(p, 640, 480, espec, CN)
    assert energia_maos(r.X, espec) == 0.0
