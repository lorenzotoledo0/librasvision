"""Item 5: gravacao salva video + landmarks; cortes e divisao de takes.
Clipes SINTETICOS (quadros coloridos) e MediaPipe substituido por ExtratorFalso."""

import cv2
import numpy as np
import pytest

from conftest import ExtratorFalso, cadastrar, frame_sintetico_assimetrico, gerar_video_sintetico
from libras.coleta.cortes import cortar_take, dividir_take
from libras.coleta.gravacao import MetaTake, registrar_take
from libras.coleta.manifesto import Manifesto
from libras.coleta.qualidade import avaliar_clipe
from libras.coleta.sinalizadores import CadastroSinalizadores
from libras.erros import ErroLibras
from libras.extracao import cache_valido, caminho_npz, carregar_npz, extrair_lote, meta_esperada


def _take(cfg, sid, n=60, desloc=0, irregular=False):
    frames = [frame_sintetico_assimetrico(i=i + desloc) for i in range(n)]
    ts = list(100.0 + np.arange(n) / 30.0)
    if irregular:
        ts = list(100.0 + np.cumsum(np.r_[0, np.random.default_rng(1).uniform(0.025, 0.05, n - 1)]))
    progresso = []
    info = avaliar_clipe(frames, ts, 30, cfg["coleta"], ExtratorFalso(), progresso=lambda i, t: progresso.append(i))
    assert progresso[-1] == n  # todos os frames passaram pelo extrator
    r = registrar_take(
        cfg,
        Manifesto(cfg),
        CadastroSinalizadores(cfg),
        frames,
        info,
        MetaTake("prova", sid, "natural", "media", "neutro", "direita"),
    )
    return r, ts


def test_gravacao_salva_video_e_npz(cfg):
    (sid,) = cadastrar(cfg)
    r, ts = _take(cfg, sid, irregular=True)
    npz = caminho_npz(cfg, r)
    assert r["landmarks_path"] == "data/landmarks/prova/S01_prova_001.npz" and r["sha256_npz"]
    assert cache_valido(npz, meta_esperada(cfg, r))
    d = carregar_npz(npz)
    assert d.partes.n == 60 and d.meta["origem_landmarks"] == "gravacao"
    np.testing.assert_allclose(d.tempos, np.array(ts) - ts[0])  # instantes reais, a partir de 0
    assert Manifesto(cfg).verificar_integridade().ok
    assert len(extrair_lote(cfg).em_cache) == 1  # a extracao em lote reaproveita


def test_importacao_gera_npz(cfg, tmp_path, com_extrator_falso):
    from libras.coleta.importacao import MetaPadrao, importar

    cadastrar(cfg)
    v = gerar_video_sintetico(tmp_path / "o" / "x.mp4")
    itens, _ = importar(cfg, [v], MetaPadrao(classe="prova", signer_id="S01"))
    l = Manifesto(cfg).linhas[0]
    assert itens[0].situacao == "importado" and l["landmarks_path"] and l["sha256_npz"]
    assert carregar_npz(caminho_npz(cfg, l)).meta["origem_landmarks"] == "arquivo"


def test_cortar_take(cfg):
    (sid,) = cadastrar(cfg)
    r, _ = _take(cfg, sid, n=60)
    cap = cv2.VideoCapture(str(cfg.raiz / r["arquivo"]))
    frames_orig = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames_orig.append(f)
    novo = cortar_take(cfg, r["video_id"], 10, 50)
    assert novo["video_id"] == r["video_id"] and novo["n_frames"] == "40" and novo["frames_originais"] == "60"
    assert "frames 10-50 de 60" in novo["corte"]
    d = carregar_npz(caminho_npz(cfg, novo))
    assert d.partes.n == 40 and d.tempos[0] == 0.0 and d.meta["origem_landmarks"] == "corte"
    assert cache_valido(caminho_npz(cfg, novo), meta_esperada(cfg, novo))
    cap = cv2.VideoCapture(str(cfg.raiz / novo["arquivo"]))
    ok, f0 = cap.read()
    assert ok and np.abs(f0.astype(int) - frames_orig[10].astype(int)).mean() < 3
    lix = list((cfg.caminho("lixeira") / "cortes").iterdir())
    assert len(lix) == 1 and {p.suffix for p in lix[0].iterdir()} == {".mp4", ".npz"}
    assert Manifesto(cfg).verificar_integridade().ok


def test_corte_curto_recusado_sem_alterar(cfg):
    (sid,) = cadastrar(cfg)
    r, _ = _take(cfg, sid, n=60)
    with pytest.raises(ErroLibras, match="30 frames"):
        cortar_take(cfg, r["video_id"], 0, 29)
    assert Manifesto(cfg).obter(r["video_id"])["sha256"] == r["sha256"]
    assert not (cfg.caminho("lixeira") / "cortes").exists()


def test_dividir_take(cfg):
    (sid,) = cadastrar(cfg)
    r, _ = _take(cfg, sid, n=70)
    a, b = dividir_take(cfg, r["video_id"], 30)
    assert a["video_id"] == "S01_prova_001" and a["n_frames"] == "30"
    assert b["video_id"] == "S01_prova_002" and b["n_frames"] == "40" and "dividido de" in b["observacoes"]
    for x in (a, b):
        assert cache_valido(caminho_npz(cfg, x), meta_esperada(cfg, x))
        assert carregar_npz(caminho_npz(cfg, x)).tempos[0] == 0.0
    assert Manifesto(cfg).verificar_integridade().ok
    with pytest.raises(ErroLibras, match="30 frames"):
        dividir_take(cfg, b["video_id"], 20)


def test_integridade_detecta_npz_alterado(cfg):
    (sid,) = cadastrar(cfg)
    r, _ = _take(cfg, sid)
    npz = caminho_npz(cfg, r)
    npz.write_bytes(npz.read_bytes() + b"x")
    rel = Manifesto(cfg).verificar_integridade()
    assert rel.landmarks_alterados and not rel.ok
    npz.unlink()
    assert Manifesto(cfg).verificar_integridade().landmarks_ausentes


def test_desenho_do_esqueleto():
    from conftest import partes_sinteticas
    from libras.landmarks import desenhar_esqueleto

    p = partes_sinteticas(1)
    f = np.zeros((480, 640, 3), np.uint8)
    img = desenhar_esqueleto(f, p.pose[0], p.mao_esq[0], p.mao_dir[0], p.presenca[0])
    assert img.sum() > 0 and f.sum() == 0
