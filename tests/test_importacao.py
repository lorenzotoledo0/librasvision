"""Testes de importacao com videos SINTETICOS e MediaPipe substituido por ExtratorFalso."""

import pytest

from conftest import cadastrar, gerar_video_sintetico
from libras.coleta.importacao import MetaPadrao, importar, parse_mapa
from libras.coleta.manifesto import Manifesto
from libras.erros import ErroLibras
from libras.utils import ler_csv, sha256_arquivo


def _estrutura_tcc1(base):
    for k, classe in enumerate(("biblioteca", "prova")):
        for u in (1, 2):
            gerar_video_sintetico(
                base / "dataset" / "sinais" / classe / f"usuario{u}_01.mp4", n_frames=60 + u, variante=k
            )
    return base / "dataset" / "sinais"


def test_importa_estrutura_tcc1_com_mapa(cfg, tmp_path, com_extrator_falso):
    cadastrar(cfg, 2)
    pasta = _estrutura_tcc1(tmp_path / "origem")
    hashes = {p: sha256_arquivo(p) for p in pasta.rglob("*.mp4")}
    itens, rel = importar(
        cfg,
        [pasta],
        MetaPadrao(iluminacao="natural"),
        padrao_nome="tcc1",
        mapa=parse_mapa(["usuario1=S01", "usuario2=S02"]),
    )
    assert [i.situacao for i in itens] == ["importado"] * 4, [i.motivo for i in itens]
    m = Manifesto(cfg)
    assert len(m) == 4
    assert {l["classe"] for l in m.linhas} == {"biblioteca", "prova"}
    assert {l["signer_id"] for l in m.linhas} == {"S01", "S02"}
    assert all(
        l["origem"] == "importado"
        and l["arquivo_original"] == "usuario1_01.mp4"
        or l["arquivo_original"] == "usuario2_01.mp4"
        for l in m.linhas
    )
    # originais intactos
    assert all(sha256_arquivo(p) == h for p, h in hashes.items())
    assert rel.exists() and len(ler_csv(rel)) == 4


def test_reimportar_e_idempotente(cfg, tmp_path, com_extrator_falso):
    cadastrar(cfg, 2)
    pasta = _estrutura_tcc1(tmp_path / "origem")
    kw = dict(padrao_nome="tcc1", mapa=parse_mapa(["usuario1=S01", "usuario2=S02"]))
    importar(cfg, [pasta], MetaPadrao(), **kw)
    itens, _ = importar(cfg, [pasta], MetaPadrao(), **kw)
    assert [i.situacao for i in itens] == ["ignorado"] * 4
    assert len(Manifesto(cfg)) == 4


def test_duplicata_no_mesmo_lote(cfg, tmp_path, com_extrator_falso):
    cadastrar(cfg)
    a = gerar_video_sintetico(tmp_path / "o" / "a.mp4")
    b = tmp_path / "o" / "b.mp4"
    b.write_bytes(a.read_bytes())
    itens, _ = importar(cfg, [a, b], MetaPadrao(classe="prova", signer_id="S01"))
    assert sorted(i.situacao for i in itens) == ["ignorado", "importado"]


def test_rotulo_sem_mapa_e_rejeitado(cfg, tmp_path, com_extrator_falso):
    cadastrar(cfg, 2)
    pasta = _estrutura_tcc1(tmp_path / "origem")
    itens, _ = importar(cfg, [pasta], MetaPadrao(), padrao_nome="tcc1", mapa=parse_mapa(["usuario1=S01"]))
    rej = [i for i in itens if i.situacao == "rejeitado"]
    assert len(rej) == 2 and all("usuario2" in i.motivo for i in rej)


def test_sem_tcle_rejeita(cfg, tmp_path, com_extrator_falso):
    cadastrar(cfg, 1, tcle=False)
    v = gerar_video_sintetico(tmp_path / "o" / "x.mp4")
    itens, _ = importar(cfg, [v], MetaPadrao(classe="prova", signer_id="S01"))
    assert itens[0].situacao == "rejeitado" and "TCLE" in itens[0].motivo
    assert len(Manifesto(cfg)) == 0


def test_pessoa_ausente_rejeita(cfg, tmp_path, com_extrator_falso):
    cadastrar(cfg)
    com_extrator_falso["pose"] = False
    v = gerar_video_sintetico(tmp_path / "o" / "x.mp4")
    itens, _ = importar(cfg, [v], MetaPadrao(classe="prova", signer_id="S01"))
    assert itens[0].situacao == "rejeitado" and "pessoa" in itens[0].motivo


def test_mao_ausente_fica_pendente(cfg, tmp_path, com_extrator_falso):
    cadastrar(cfg)
    com_extrator_falso["mao"] = False
    v = gerar_video_sintetico(tmp_path / "o" / "x.mp4")
    itens, _ = importar(cfg, [v], MetaPadrao(classe="prova", signer_id="S01"))
    assert itens[0].situacao == "importado"
    assert Manifesto(cfg).linhas[0]["status"] == "pendente"


def test_video_curto_e_longo(cfg, tmp_path, com_extrator_falso):
    cadastrar(cfg)
    curto = gerar_video_sintetico(tmp_path / "o" / "curto.mp4", n_frames=10)
    longo = gerar_video_sintetico(tmp_path / "o" / "longo.mp4", n_frames=30 * 20, larg=160, alt=120)
    itens, _ = importar(cfg, [curto, longo], MetaPadrao(classe="prova", signer_id="S01"))
    assert [i.situacao for i in itens] == ["rejeitado", "rejeitado"]
    assert "curto" in itens[0].motivo and "longo" in itens[1].motivo


def test_simular_nao_escreve(cfg, tmp_path, com_extrator_falso):
    cadastrar(cfg)
    v = gerar_video_sintetico(tmp_path / "o" / "x.mp4")
    itens, rel = importar(cfg, [v], MetaPadrao(classe="prova", signer_id="S01"), simular=True)
    assert rel is None and itens[0].situacao == "importado"
    assert len(Manifesto(cfg)) == 0 and not cfg.caminho("videos").exists()


def test_padronizar_preserva_original(cfg, tmp_path, com_extrator_falso):
    cadastrar(cfg)
    v = gerar_video_sintetico(tmp_path / "o" / "x.avi", larg=1280, alt=720)
    h = sha256_arquivo(v)
    itens, _ = importar(cfg, [v], MetaPadrao(classe="prova", signer_id="S01"), padronizar=True)
    assert itens[0].situacao == "importado", itens[0].motivo
    l = Manifesto(cfg).linhas[0]
    assert (l["largura"], l["altura"]) == ("640", "480") and l["arquivo"].endswith(".mp4")
    assert sha256_arquivo(v) == h and l["sha256_origem"] == h and l["sha256"] != h


def test_classe_pela_pasta(cfg, tmp_path, com_extrator_falso):
    cadastrar(cfg)
    gerar_video_sintetico(tmp_path / "o" / "secretaria" / "S01_7.mp4")
    itens, _ = importar(cfg, [tmp_path / "o"], MetaPadrao(), padrao_nome="projeto")
    assert itens[0].classe == "secretaria" and itens[0].signer_id == "S01"
    assert "take original 7" in Manifesto(cfg).linhas[0]["observacoes"]


def test_sem_videos_mensagem_clara(cfg, tmp_path):
    (tmp_path / "vazia").mkdir()
    with pytest.raises(ErroLibras, match="Nenhum video"):
        importar(cfg, [tmp_path / "vazia"], MetaPadrao())


def test_mapa_invalido():
    with pytest.raises(ErroLibras):
        parse_mapa(["usuario1=Joao"])
