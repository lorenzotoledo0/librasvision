"""Testes de manifesto e gravacao. Clipes SINTETICOS (sem pessoas)."""

import cv2
import numpy as np
import pytest

from conftest import cadastrar, frame_sintetico_assimetrico
from libras.camera import frame_para_exibicao
from libras.coleta.gravacao import MetaTake, registrar_take, salvar_clipe
from libras.coleta.manifesto import Manifesto
from libras.coleta.qualidade import InfoVideo, avaliar_clipe, fps_por_timestamps
from libras.coleta.sinalizadores import CadastroSinalizadores
from libras.erros import ErroConsentimento, ErroLibras, ErroManifesto


def _clipe(n=60, desloc=0):
    return [frame_sintetico_assimetrico(i=i + desloc) for i in range(n)]


def _meta(sid, classe="prova"):
    return MetaTake(classe, sid, "natural", "media", "neutro", "direita")


def _info(frames, fps=30.0):
    ts = [i / fps for i in range(len(frames))]
    return avaliar_clipe(
        frames,
        ts,
        fps,
        {
            "qualidade": {
                "fps_minimo": 24,
                "brilho_minimo": 60,
                "brilho_maximo": 210,
                "taxa_minima_mao": 0.6,
                "taxa_minima_pose": 0.8,
            }
        },
        None,
    )


def test_arquivo_salvo_nao_e_espelhado(tmp_path):
    frames = _clipe(20)
    destino = tmp_path / "x.mp4"
    salvar_clipe(frames, 30.0, destino)
    cap = cv2.VideoCapture(str(destino))
    ok, lido = cap.read()
    cap.release()
    assert ok
    esq, dir_ = lido[:, :320].mean(), lido[:, 320:].mean()
    assert esq > 150 and dir_ < 50, "o lado claro precisa continuar a esquerda (sem espelhamento)"


def test_exibicao_espelha_so_a_copia():
    f = frame_sintetico_assimetrico()
    original = f.copy()
    tela = frame_para_exibicao(f)
    assert tela[:, :320].mean() < 50 and tela[:, 320:].mean() > 150
    assert np.array_equal(f, original)


def test_salvar_nao_sobrescreve(tmp_path):
    d = tmp_path / "a.mp4"
    salvar_clipe(_clipe(5), 30, d)
    with pytest.raises(ErroLibras):
        salvar_clipe(_clipe(5), 30, d)


def test_registrar_take_cria_arquivo_e_registro(cfg):
    (sid,) = cadastrar(cfg)
    m, cad = Manifesto(cfg), CadastroSinalizadores(cfg)
    frames, outros = _clipe(), _clipe(desloc=5)
    r1 = registrar_take(cfg, m, cad, frames, _info(frames), _meta(sid))
    r2 = registrar_take(cfg, m, cad, outros, _info(outros), _meta(sid))
    assert r1["arquivo"] == f"data/raw_videos/prova/{sid}_001.mp4"
    assert r2["take"] == "2"
    assert (cfg.raiz / r1["arquivo"]).exists()
    assert Manifesto(cfg).obter(r1["video_id"])["origem"] == "gravado"
    assert m.verificar_integridade(cad.ids(), cad.ids_com_tcle()).ok


def test_registrar_take_sem_tcle_nao_grava_nada(cfg):
    (sid,) = cadastrar(cfg, tcle=False)
    m, cad = Manifesto(cfg), CadastroSinalizadores(cfg)
    frames = _clipe()
    with pytest.raises(ErroConsentimento):
        registrar_take(cfg, m, cad, frames, _info(frames), _meta(sid))
    assert len(m) == 0
    assert not (cfg.caminho("videos") / "prova").exists()


def test_classe_fora_do_config(cfg):
    (sid,) = cadastrar(cfg)
    frames = _clipe()
    with pytest.raises(ErroLibras):
        registrar_take(
            cfg, Manifesto(cfg), CadastroSinalizadores(cfg), frames, _info(frames), _meta(sid, classe="cantina")
        )


def test_excluir_vai_para_lixeira_e_numero_nao_e_reusado(cfg):
    (sid,) = cadastrar(cfg)
    m, cad = Manifesto(cfg), CadastroSinalizadores(cfg)
    frames = _clipe()
    r = registrar_take(cfg, m, cad, frames, _info(frames), _meta(sid))
    destino = m.excluir(r["video_id"])
    assert destino.exists() and "lixeira" in destino.parts
    assert m.obter(r["video_id"]) is None
    assert m.proximo_take(sid, "prova") == 2


def test_integridade_detecta_orfaos(cfg):
    (sid,) = cadastrar(cfg)
    m, cad = Manifesto(cfg), CadastroSinalizadores(cfg)
    frames = _clipe()
    r = registrar_take(cfg, m, cad, frames, _info(frames), _meta(sid))
    solto = cfg.caminho("videos") / "prova" / "solto.mp4"
    salvar_clipe(frames[:5], 30, solto)
    (cfg.raiz / r["arquivo"]).unlink()
    rel = m.verificar_integridade(cad.ids(), cad.ids_com_tcle())
    assert not rel.ok
    assert any("solto.mp4" in x for x in rel.arquivos_sem_linha)
    assert any(r["video_id"] in x for x in rel.linhas_sem_arquivo)


def test_adicionar_exige_arquivo_existente(cfg):
    with pytest.raises(ErroManifesto):
        Manifesto(cfg).adicionar(
            {
                "video_id": "x",
                "sha256": "a",
                "classe": "prova",
                "signer_id": "S01",
                "take": 1,
                "arquivo": "data/raw_videos/prova/nao_existe.mp4",
                "status": "ok",
            }
        )


def test_qualidade_fps_e_brilho():
    assert fps_por_timestamps([0, 0.05, 0.1, 0.15]) == pytest.approx(20.0)
    escuros = [np.full((48, 64, 3), 10, np.uint8)] * 30
    info = avaliar_clipe(
        escuros,
        [i / 15 for i in range(30)],
        30,
        {
            "qualidade": {
                "fps_minimo": 24,
                "brilho_minimo": 60,
                "brilho_maximo": 210,
                "taxa_minima_mao": 0.6,
                "taxa_minima_pose": 0.8,
            }
        },
        None,
    )
    assert any(a.startswith("fps_baixo") for a in info.avisos)
    assert any(a.startswith("escuro") for a in info.avisos)
    assert info.duracao_s == pytest.approx(30 / 15, rel=0.05)


def test_take_com_erro_nao_salva(cfg):
    (sid,) = cadastrar(cfg)
    info = InfoVideo(erros=["nenhum frame capturado"])
    with pytest.raises(ErroLibras):
        registrar_take(cfg, Manifesto(cfg), CadastroSinalizadores(cfg), _clipe(3), info, _meta(sid))


def test_clipe_identico_nao_duplica(cfg):
    (sid,) = cadastrar(cfg)
    m, cad = Manifesto(cfg), CadastroSinalizadores(cfg)
    frames = _clipe()
    registrar_take(cfg, m, cad, frames, _info(frames), _meta(sid))
    with pytest.raises(ErroLibras, match="identico"):
        registrar_take(cfg, m, cad, frames, _info(frames), _meta(sid))
    assert len(m) == 1 and len(list((cfg.caminho("videos") / "prova").iterdir())) == 1
