"""Extracao (cache), splits e dataset. Landmarks e videos SINTETICOS; MediaPipe substituido."""

import numpy as np
import pytest

from conftest import cadastrar, criar_videos_com_landmarks, gerar_video_sintetico, partes_sinteticas
from libras.coleta.manifesto import Manifesto
from libras.dataset import criar_tf_dataset, gerador_lotes, montar_conjunto, pesos_classes
from libras.erros import ErroDadosInsuficientes, ErroLibras
from libras.extracao import cache_valido, caminho_npz, carregar_npz, extrair_lote, meta_esperada
from libras.splits import carregar_split, gerar_splits, salvar_split, verificar_vazamento, videos_elegiveis


# --------------------------- extracao ------------------------------------
@pytest.fixture
def extracao_falsa(monkeypatch):
    """Substitui o MediaPipe por um esqueleto SINTETICO e conta chamadas."""
    from libras import landmarks as lm

    chamadas = []

    def falso(caminho, cfg_mp):
        chamadas.append(str(caminho))
        p = partes_sinteticas(60)
        return lm.SequenciaLandmarks(p.pose, p.mao_esq, p.mao_dir, p.face, p.presenca, 640, 480, 30.0)

    monkeypatch.setattr(lm, "extrair_video", falso)
    return chamadas


def _registrar_videos(cfg, n=3):
    from libras.utils import sha256_arquivo

    (sid,) = cadastrar(cfg)
    m = Manifesto(cfg)
    for t in range(1, n + 1):
        arq = gerar_video_sintetico(cfg.caminho("videos") / "prova" / f"{sid}_{t:03d}.mp4", variante=t)
        m.adicionar(
            {
                "video_id": f"{sid}_prova_{t:03d}",
                "sha256": sha256_arquivo(arq),
                "classe": "prova",
                "signer_id": sid,
                "take": t,
                "arquivo": m.relativo(arq),
                "status": "ok",
            }
        )
    return m


def test_extracao_incremental_e_cache(cfg, extracao_falsa):
    _registrar_videos(cfg, 3)
    r1 = extrair_lote(cfg)
    assert len(r1.processados) == 3 and not r1.falhas
    assert all(l["landmarks_path"].endswith(".npz") for l in Manifesto(cfg).linhas)
    r2 = extrair_lote(cfg)
    assert len(r2.processados) == 0 and len(r2.em_cache) == 3 and len(extracao_falsa) == 3
    d = carregar_npz(caminho_npz(cfg, Manifesto(cfg).linhas[0]))
    assert d.partes.pose.shape == (60, 33, 4) and d.largura == 640 and d.fps == 30.0
    assert d.tempos[1] == pytest.approx(1 / 30)


def test_cache_invalida_se_mudar_mediapipe_ou_video(cfg, extracao_falsa):
    _registrar_videos(cfg, 1)
    extrair_lote(cfg)
    l = Manifesto(cfg).linhas[0]
    assert cache_valido(caminho_npz(cfg, l), meta_esperada(cfg, l))
    cfg["mediapipe"]["model_complexity"] = 2
    assert not cache_valido(caminho_npz(cfg, l), meta_esperada(cfg, l))
    assert len(extrair_lote(cfg).processados) == 1
    l2 = dict(l, sha256="outro")
    assert not cache_valido(caminho_npz(cfg, l2), meta_esperada(cfg, l2))


def test_extrair_sem_videos_mensagem_clara(cfg):
    with pytest.raises(ErroDadosInsuficientes, match="Nenhum video"):
        extrair_lote(cfg)


def test_excluir_video_remove_landmarks(cfg, extracao_falsa):
    _registrar_videos(cfg, 2)
    extrair_lote(cfg)
    m = Manifesto(cfg)
    l = m.linhas[0]
    m.excluir(l["video_id"])
    assert not caminho_npz(cfg, l).exists()
    assert m.verificar_integridade().landmarks_orfaos == []


# --------------------------- splits --------------------------------------
def test_splits_por_sinalizador_sem_vazamento(cfg):
    criar_videos_com_landmarks(cfg, n_sinalizadores=6, por_classe=2)
    linhas, excl = videos_elegiveis(cfg)
    assert len(linhas) == 6 * 4 * 2 and not excl
    sp = gerar_splits(cfg, linhas, "grupos_sinalizador", n_folds=3)
    assert len(sp["folds"]) == 3
    por_id = {l["video_id"]: l for l in linhas}
    assert verificar_vazamento(sp, por_id) == []
    for f in sp["folds"]:
        assert f["validacao_por"] == "sinalizadores"
        assert not set(f["sinalizadores_treino"]) & set(f["sinalizadores_teste"])
        assert not set(f["sinalizadores_validacao"]) & set(f["sinalizadores_teste"])
        assert not set(f["sinalizadores_treino"]) & set(f["sinalizadores_validacao"])
    testes = sorted(v for f in sp["folds"] for v in f["teste"])
    assert testes == sorted(por_id)


def test_loso_e_determinismo(cfg):
    criar_videos_com_landmarks(cfg, n_sinalizadores=4, por_classe=1)
    linhas, _ = videos_elegiveis(cfg)
    a = gerar_splits(cfg, linhas, "loso")
    b = gerar_splits(cfg, linhas, "loso")
    assert len(a["folds"]) == 4
    assert [f["sinalizadores_teste"] for f in a["folds"]] == [["S01"], ["S02"], ["S03"], ["S04"]]
    assert [f["validacao"] for f in a["folds"]] == [f["validacao"] for f in b["folds"]]


def test_recusa_um_sinalizador_e_avisa_com_dois(cfg):
    criar_videos_com_landmarks(cfg, n_sinalizadores=1, por_classe=2)
    linhas, _ = videos_elegiveis(cfg)
    with pytest.raises(ErroDadosInsuficientes, match="pelo menos 2"):
        gerar_splits(cfg, linhas)


def test_avisa_com_dois_sinalizadores(cfg):
    criar_videos_com_landmarks(cfg, n_sinalizadores=2, por_classe=2)
    linhas, _ = videos_elegiveis(cfg)
    sp = gerar_splits(cfg, linhas)
    assert any("pouco informativa" in a for a in sp["avisos"])
    assert len(sp["folds"]) == 2 and all(f["validacao_por"] == "videos" for f in sp["folds"])


def test_aleatorio_por_video_marcado_secundario(cfg):
    criar_videos_com_landmarks(cfg, n_sinalizadores=2, por_classe=3)
    linhas, _ = videos_elegiveis(cfg)
    sp = gerar_splits(cfg, linhas, "aleatorio_por_video")
    assert sp["secundario"] is True
    assert verificar_vazamento(sp, {l["video_id"]: l for l in linhas}) == []


def test_split_desatualizado_e_detectado(cfg):
    criar_videos_com_landmarks(cfg, n_sinalizadores=3, por_classe=1)
    linhas, _ = videos_elegiveis(cfg)
    salvar_split(cfg, gerar_splits(cfg, linhas))
    assert carregar_split(cfg)["n_videos"] == 12
    Manifesto(cfg).definir_status(linhas[0]["video_id"], "rejeitado")
    with pytest.raises(ErroLibras, match="outro conjunto"):
        carregar_split(cfg)


def test_vazamento_manual_e_detectado(cfg):
    criar_videos_com_landmarks(cfg, n_sinalizadores=3, por_classe=1)
    linhas, _ = videos_elegiveis(cfg)
    sp = gerar_splits(cfg, linhas)
    f = sp["folds"][0]
    f["treino"].append(f["teste"][0])
    prob = verificar_vazamento(sp, {l["video_id"]: l for l in linhas})
    assert any("treino/teste" in p for p in prob)
    assert any("sinalizador" in p for p in prob)


# --------------------------- dataset -------------------------------------
def test_conjunto_e_tf_dataset(cfg):
    criar_videos_com_landmarks(cfg, n_sinalizadores=3, por_classe=2)
    linhas, _ = videos_elegiveis(cfg)
    conj = montar_conjunto(cfg, linhas)
    assert conj.X.dtype == np.float16 and conj.X.shape[1:] == (30, 550)
    assert set(conj.y.tolist()) == {0, 1, 2, 3}
    sp = gerar_splits(cfg, linhas)
    f = sp["folds"][0]
    idx_tr, idx_te = conj.indices(f["treino"]), conj.indices(f["teste"])
    assert not set(conj.video[idx_tr]) & set(conj.video[idx_te])
    ds = criar_tf_dataset(conj, idx_tr, 8, True, 42)
    n = 0
    for x, y in ds:
        assert x.shape[1:] == (30, 550) and x.dtype.name == "float32"
        n += int(x.shape[0])
    assert n == len(idx_tr)


def test_embaralhamento_reprodutivel_e_transformacao_so_quando_pedida(cfg):
    criar_videos_com_landmarks(cfg, n_sinalizadores=2, por_classe=2)
    linhas, _ = videos_elegiveis(cfg)
    conj = montar_conjunto(cfg, linhas)
    idx = np.arange(conj.n)
    a = [y.tolist() for _, y in gerador_lotes(conj, idx, 4, True, 7)()]
    b = [y.tolist() for _, y in gerador_lotes(conj, idx, 4, True, 7)()]
    assert a == b
    chamadas = []

    def marcar(x, rng):
        chamadas.append(len(x))
        return x + 1.0

    x0, _ = next(gerador_lotes(conj, idx, 4, False, 7)())
    x1, _ = next(gerador_lotes(conj, idx, 4, False, 7, transformar=marcar)())
    assert chamadas == [4] and np.allclose(x1, x0 + 1.0)


def test_limite_de_memoria(cfg):
    criar_videos_com_landmarks(cfg, n_sinalizadores=2, por_classe=2)
    linhas, _ = videos_elegiveis(cfg)
    cfg["dataset"]["limite_memoria_mb"] = 0.01
    with pytest.raises(ErroLibras, match="limite"):
        montar_conjunto(cfg, linhas)


def test_pesos_classes():
    p = pesos_classes(np.array([0, 0, 0, 1]), 3)
    assert p[0] == pytest.approx(4 / 9) and p[1] == pytest.approx(4 / 3) and p[2] == 0.0
