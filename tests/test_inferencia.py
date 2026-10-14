"""Inferencia em tempo real com landmarks SINTETICOS e um modelo treinado em dados sinteticos
(so para testar o encanamento; nenhuma predicao daqui tem significado)."""

import numpy as np
import pytest

from conftest import criar_videos_com_landmarks, partes_sinteticas
from libras.erros import ErroDadosInsuficientes
from libras.inferencia import Predicao, Reconhecedor, Transcritor, carregar_modelo, localizar_modelo
from libras.landmarks import LandmarksFrame
from libras.normalizacao import janelas_do_video, tempos_uniformes


def test_sem_modelo_mensagem_clara(cfg):
    with pytest.raises(ErroDadosInsuficientes, match="--final"):
        localizar_modelo(cfg)


@pytest.fixture(scope="module")
def modelo_sintetico(tmp_path_factory):
    import shutil

    from conftest import RAIZ
    from libras.config import carregar_config
    from libras.treino import OpcoesTreino, treinar_final

    base = tmp_path_factory.mktemp("proj")
    (base / "configs").mkdir()
    shutil.copy(RAIZ / "configs" / "config.yaml", base / "configs" / "config.yaml")
    cfg = carregar_config(base / "configs" / "config.yaml")
    criar_videos_com_landmarks(cfg, n_sinalizadores=3, por_classe=1)
    treinar_final(
        cfg,
        OpcoesTreino(nome="final", verbose=0, sobrescrever={"treino": {"epocas": 1}, "modelo": {"camadas_lstm": [6]}}),
    )
    return cfg, carregar_modelo(localizar_modelo(cfg))


def _frames(p):
    return [LandmarksFrame(p.pose[i], p.mao_esq[i], p.mao_dir[i], p.face[i], p.presenca[i]) for i in range(p.n)]


def test_tempo_real_usa_a_mesma_janela_do_treino(modelo_sintetico):
    cfg, modelo = modelo_sintetico
    entradas = []

    def preditor(x):
        entradas.append(x.copy())
        return modelo.keras.predict_on_batch(x)

    rec = Reconhecedor(modelo, {"predizer_a_cada": 1, "min_fracao_mao": 0.0, "suavizacao": 1}, preditor=preditor)
    p = partes_sinteticas(60, mover_mao=(0, 60))
    t = tempos_uniformes(60, 30.0)
    preds = [rec.adicionar(f, ti, 640, 480) for f, ti in zip(_frames(p), t)]
    assert all(pr.motivo == "aquecendo" for pr in preds[:29])
    assert preds[29].probs is not None
    treino = janelas_do_video(p, t, 640, 480, modelo.espec, cfg, eh_nenhum_sinal=True, passo=1)
    np.testing.assert_allclose(entradas[-1][0], treino.X[-1], atol=1e-5)
    np.testing.assert_allclose(preds[-1].probs.sum(), 1, atol=1e-5)


def test_fps_irregular_e_reamostrado(modelo_sintetico):
    _, modelo = modelo_sintetico
    entradas = []
    rec = Reconhecedor(
        modelo,
        {"predizer_a_cada": 1, "min_fracao_mao": 0.0},
        preditor=lambda x: (entradas.append(x), modelo.keras.predict_on_batch(x))[1],
    )
    p = partes_sinteticas(40, mover_mao=(0, 40))
    t = np.cumsum(np.random.default_rng(0).uniform(0.03, 0.07, 40))  # ~20 FPS irregular
    for f, ti in zip(_frames(p), t):
        rec.adicionar(f, ti, 640, 480)
    assert entradas and entradas[-1].shape == (1, 30, modelo.espec.dim)


def test_sem_maos_nao_roda_o_modelo(modelo_sintetico):
    _, modelo = modelo_sintetico
    chamadas = []
    rec = Reconhecedor(
        modelo,
        {"predizer_a_cada": 1, "min_fracao_mao": 0.3},
        preditor=lambda x: chamadas.append(1) or modelo.keras.predict_on_batch(x),
    )
    p = partes_sinteticas(40)
    p.presenca[:, 1:3] = False
    for f, ti in zip(_frames(p), tempos_uniformes(40, 30.0)):
        pr = rec.adicionar(f, ti, 640, 480)
    assert pr.motivo == "sem_maos" and pr.classe is None and not chamadas


def test_limiar_impede_texto(modelo_sintetico):
    _, modelo = modelo_sintetico
    rec = Reconhecedor(modelo, {"predizer_a_cada": 1, "min_fracao_mao": 0.0, "limiar_confianca": 1.01})
    p = partes_sinteticas(40, mover_mao=(0, 40))
    for f, ti in zip(_frames(p), tempos_uniformes(40, 30.0)):
        pr = rec.adicionar(f, ti, 640, 480)
    assert pr.classe is None and pr.motivo in ("baixa_confianca", "nenhum_sinal")


def test_transcritor():
    tr = Transcritor(frames_estaveis=3)
    P = lambda c, m="": Predicao(0, c, 0.9, None, m)  # noqa: E731
    seq = [P("prova")] * 2 + [P("biblioteca")] + [P("prova")] * 3 + [P("prova")] * 5
    aceitos = [tr.atualizar(x) for x in seq]
    assert [a for a in aceitos if a] == ["prova"]
    tr.atualizar(P(None, "sem_maos"))
    for _ in range(3):
        tr.atualizar(P("prova"))
    assert tr.texto == ["prova", "prova"]
