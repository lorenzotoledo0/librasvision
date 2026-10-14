"""Item 2: treino com selecao de classes e minimos. Landmarks SINTETICOS; numeros sem significado."""

import json

import pytest

from conftest import criar_videos_com_landmarks
from libras.coleta.manifesto import Manifesto
from libras.erros import ErroDadosInsuficientes, ErroLibras
from libras.selecao import TEXTO_ENSAIO, avaliar_selecao, classes_treino, fixar_classes, tabela_classes
from libras.splits import carregar_split, gerar_splits, salvar_split, videos_elegiveis

RAPIDO = {"treino": {"epocas": 1, "paciencia": 1}, "modelo": {"camadas_lstm": [4]}}


@pytest.fixture
def dados(cfg):
    """3 sinalizadores; a classe secretaria fica so com S01 (os outros videos sao rejeitados)."""
    criar_videos_com_landmarks(cfg, n_sinalizadores=3, por_classe=1)
    m = Manifesto(cfg)
    for l in m.linhas:
        if l["classe"] == "secretaria" and l["signer_id"] != "S01":
            m.definir_status(l["video_id"], "rejeitado")
    return cfg


def test_tabela_por_classe(dados):
    t = {l.classe: l for l in tabela_classes(dados)}
    assert t["biblioteca"].videos_ok == 3 and t["biblioteca"].sinalizadores == 3 and t["biblioteca"].janelas > 0
    assert t["secretaria"].videos_ok == 1 and t["secretaria"].sinalizadores == 1


def test_classe_abaixo_do_minimo_bloqueada(dados):
    # sem selecao, o padrao deixa de fora a classe abaixo do minimo (item 6)
    assert "secretaria" not in classes_treino(dados)
    fixar_classes(dados, ["biblioteca", "prova", "secretaria", "nenhum_sinal"])
    av = avaliar_selecao(dados, tabela=tabela_classes(dados, contar_janelas=False))
    assert "secretaria" in av.bloqueios and not av.pode_treinar
    linhas, _ = videos_elegiveis(dados)
    with pytest.raises(ErroDadosInsuficientes, match="secretaria"):
        gerar_splits(dados, linhas)


def test_selecao_parcial_gera_split_so_com_as_classes(dados):
    fixar_classes(dados, ["biblioteca", "prova", "nenhum_sinal"])
    linhas, excl = videos_elegiveis(dados)
    assert {l["classe"] for l in linhas} == {"biblioteca", "prova", "nenhum_sinal"}
    assert any("fora da selecao" in m for m in excl.values())
    sp = gerar_splits(dados, linhas)
    assert sp["classes"] == ["biblioteca", "prova", "nenhum_sinal"] and not sp["ensaio"]
    salvar_split(dados, sp)
    assert carregar_split(dados)["classes"] == sp["classes"]
    fixar_classes(dados, ["biblioteca", "prova"])  # mudou a selecao: split antigo nao serve
    with pytest.raises(ErroLibras, match="classes"):
        carregar_split(dados)


def test_assinatura_inclui_classes(dados):
    from libras.splits import assinatura

    linhas, _ = videos_elegiveis(dados)
    assert assinatura(linhas, ["a", "b"]) != assinatura(linhas, ["a"])


def test_sem_nenhum_sinal_avisa(dados):
    fixar_classes(dados, ["biblioteca", "prova"])
    linhas, _ = videos_elegiveis(dados)
    sp = gerar_splits(dados, linhas)
    assert any("nenhum_sinal" in a and "D7" in a for a in sp["avisos"])


def test_minimo_de_classes(dados):
    fixar_classes(dados, ["biblioteca"])
    linhas, _ = videos_elegiveis(dados)
    with pytest.raises(ErroDadosInsuficientes, match="pelo menos 2 classes"):
        gerar_splits(dados, linhas)


def test_classe_inexistente(dados):
    from libras.erros import ErroConfig

    with pytest.raises(ErroConfig, match="cantina"):
        fixar_classes(dados, ["biblioteca", "cantina"])
    # selecao SALVA com classe inexistente e ignorada em silencio (item 6)
    dados["treino"]["classes"] = ["biblioteca", "cantina", "prova"]
    assert classes_treino(dados) == ["biblioteca", "prova"]


def test_ensaio_marca_todas_as_saidas(dados, tmp_path):
    from libras.treino import OpcoesTreino, executar, treinar_final

    dados["treino"]["ensaio"] = True
    assert "secretaria" in classes_treino(dados)  # em ENSAIO o padrao inclui as classes abaixo do minimo
    linhas, _ = videos_elegiveis(dados)
    sp = gerar_splits(dados, linhas)
    assert sp["ensaio"] and TEXTO_ENSAIO in sp["avisos"]
    salvar_split(dados, sp)
    pasta, r = executar(
        dados,
        OpcoesTreino(
            nome="e", familias=[], verbose=0, sobrescrever=RAPIDO, pasta_runs=tmp_path, folds=[0], salvar_modelos=False
        ),
    )
    assert r["ensaio"] is True
    assert (pasta / "LEIA-ME.txt").read_text(encoding="utf-8").startswith("ENSAIO")
    final = treinar_final(dados, OpcoesTreino(nome="final", verbose=0, sobrescrever=RAPIDO, pasta_runs=tmp_path))
    meta = json.loads((final / "metadados.json").read_text(encoding="utf-8"))
    assert meta["ensaio"] is True


def test_metadados_com_classes_selecionadas(dados, tmp_path):
    from libras.treino import OpcoesTreino, treinar_final

    fixar_classes(dados, ["prova", "biblioteca"])  # ordem do vocabulario prevalece
    final = treinar_final(dados, OpcoesTreino(nome="final", verbose=0, sobrescrever=RAPIDO, pasta_runs=tmp_path))
    meta = json.loads((final / "metadados.json").read_text(encoding="utf-8"))
    assert meta["classes"] == ["biblioteca", "prova"] and meta["ensaio"] is False
    assert any("nenhum_sinal" in a for a in meta["avisos_selecao"])
