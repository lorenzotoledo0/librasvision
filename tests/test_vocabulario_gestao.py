"""Item 3: vocabulario em JSON e remocao de sinalizadores/classes. Dados SINTETICOS."""

import json

import pytest

from conftest import cadastrar, criar_videos_com_landmarks, frame_sintetico_assimetrico
from libras.coleta.gestao import PALAVRA_EXCLUSAO, excluir_definitivamente, remover_classe, remover_sinalizador
from libras.coleta.manifesto import Manifesto
from libras.coleta.sinalizadores import CadastroSinalizadores
from libras.erros import ErroLibras
from libras.vocabulario import Vocabulario, slugificar


def test_slug():
    assert slugificar("Sala de Aula!") == "sala_de_aula"
    assert slugificar("  Ção  ") == "cao"
    assert slugificar("3a prova") == "sinal_3a_prova"
    assert slugificar("!!!") == ""


def test_presets_criados_na_primeira_execucao(cfg):
    assert not cfg.caminho("vocabulario").exists()
    v = Vocabulario(cfg)
    assert cfg.caminho("vocabulario").exists()
    assert [c.slug for c in v.todas()] == ["biblioteca", "prova", "secretaria", "nenhum_sinal"]
    assert v.obter("nenhum_sinal").tipo == "nenhum_sinal"
    assert cfg.classes == ["biblioteca", "prova", "secretaria", "nenhum_sinal"]


def test_nova_palavra_duplicata_e_meta(cfg):
    v = Vocabulario(cfg)
    c = v.adicionar("Sala de Aula", meta=12)
    assert c.slug == "sala_de_aula" and cfg.classes[-1] == "sala_de_aula"
    for dup in ("sala de aula", "SALA_DE_AULA", "Sála de aula"):
        with pytest.raises(ErroLibras, match="ja existe"):
            v.adicionar(dup)
    v.definir_meta_sinalizador("prova", "S02", 15)
    v2 = Vocabulario(cfg)
    assert v2.meta("prova", "S02") == 15 and v2.meta("prova", "S01") == 10 and v2.meta("sala_de_aula") == 12


def test_arquivar_tira_da_gravacao_e_do_treino_mas_mantem_registros(cfg):
    criar_videos_com_landmarks(cfg, n_sinalizadores=2, por_classe=1)
    Vocabulario(cfg).arquivar("secretaria")
    assert "secretaria" not in cfg.classes and "secretaria" in cfg.classes_todas
    assert Manifesto(cfg).verificar_integridade().classes_desconhecidas == []
    from libras.selecao import classes_treino

    assert "secretaria" not in classes_treino(cfg)


def test_classe_nova_recebe_gravacao(cfg):
    from libras.coleta.gravacao import MetaTake, registrar_take
    from libras.coleta.qualidade import avaliar_clipe

    Vocabulario(cfg).adicionar("Professor")
    (sid,) = cadastrar(cfg)
    frames = [frame_sintetico_assimetrico(i=i) for i in range(40)]
    info = avaliar_clipe(frames, [i / 30 for i in range(40)], 30, cfg["coleta"], None)
    r = registrar_take(
        cfg,
        Manifesto(cfg),
        CadastroSinalizadores(cfg),
        frames,
        info,
        MetaTake("professor", sid, "natural", "media", "neutro", "direita"),
    )
    assert r["arquivo"].startswith("data/raw_videos/professor/")


def test_remover_e_excluir_sinalizador(cfg):
    from libras.splits import gerar_splits, salvar_split, videos_elegiveis

    criar_videos_com_landmarks(cfg, n_sinalizadores=3, por_classe=1)
    linhas, _ = videos_elegiveis(cfg)
    salvar_split(cfg, gerar_splits(cfg, linhas))
    v = Vocabulario(cfg)
    v.definir_meta_sinalizador("prova", "S02", 20)
    with pytest.raises(ErroLibras, match="Confirmacao"):
        remover_sinalizador(cfg, "S02", "S2")
    r = remover_sinalizador(cfg, "S02", "S02")
    assert r.videos == 4 and r.landmarks == 4 and r.splits_invalidados == ["grupos_sinalizador.json"]
    assert "S02" not in CadastroSinalizadores(cfg).ids()
    assert all(l["signer_id"] != "S02" for l in Manifesto(cfg).linhas)
    assert (r.pasta_lixeira / "manifesto_removido.csv").exists() and (r.pasta_lixeira / "sinalizador.json").exists()
    assert len(list(r.pasta_lixeira.rglob("*.mp4"))) == 4 and len(list(r.pasta_lixeira.rglob("*.npz"))) == 4
    assert "S02" not in Vocabulario(cfg).obter("prova").metas_sinalizador
    assert Manifesto(cfg).verificar_integridade().ok
    assert not list(cfg.caminho("splits").glob("*.json"))
    aud = [json.loads(l) for l in cfg.caminho("auditoria").read_text(encoding="utf-8").splitlines()]
    assert aud[-1]["acao"] == "remover_sinalizador" and aud[-1]["alvo"] == "S02"
    with pytest.raises(ErroLibras, match="Confirmacao"):
        excluir_definitivamente(cfg, "S02", "S02", "excluir")
    n = excluir_definitivamente(cfg, "S02", "S02", PALAVRA_EXCLUSAO)
    assert n >= 10 and not r.pasta_lixeira.exists()
    assert not list(cfg.caminho("lixeira").rglob("S02_*"))
    assert (
        json.loads(cfg.caminho("auditoria").read_text(encoding="utf-8").splitlines()[-1])["acao"]
        == "excluir_definitivamente"
    )


def test_excluir_exige_remocao_antes(cfg):
    cadastrar(cfg)
    with pytest.raises(ErroLibras, match="Remova-o primeiro"):
        excluir_definitivamente(cfg, "S01", "S01", PALAVRA_EXCLUSAO)


def test_remover_take_individual_tambem_sai_na_exclusao(cfg):
    criar_videos_com_landmarks(cfg, n_sinalizadores=2, por_classe=2)
    m = Manifesto(cfg)
    alvo = [l for l in m.linhas if l["signer_id"] == "S01"][0]
    m.excluir(alvo["video_id"])  # vai para data/lixeira/<classe>/
    r = remover_sinalizador(cfg, "S01", "S01")
    assert r.outros_arquivos == 1
    excluir_definitivamente(cfg, "S01", "S01", PALAVRA_EXCLUSAO)
    assert not list(cfg.caminho("lixeira").rglob("S01_*"))


def test_remover_classe_vazia_e_com_dados(cfg):
    v = Vocabulario(cfg)
    v.adicionar("Cantina")
    remover_classe(cfg, "cantina", "cantina")
    assert "cantina" not in Vocabulario(cfg).slugs()
    criar_videos_com_landmarks(cfg, n_sinalizadores=2, por_classe=1)
    r = remover_classe(cfg, "prova", "prova")
    assert r.videos == 2 and r.pasta_lixeira.name.startswith("classe_prova_")
    assert "prova" not in cfg.classes_todas
    assert all(l["classe"] != "prova" for l in Manifesto(cfg).linhas)
    assert Manifesto(cfg).verificar_integridade().ok
