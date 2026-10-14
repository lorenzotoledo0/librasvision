"""Treino de ponta a ponta com landmarks SINTETICOS e poucas epocas.

Verifica o encanamento (artefatos, metricas so no teste, ausencia de vazamento).
Os numeros produzidos aqui nao significam nada e nunca devem ser reportados.
"""

import json

import numpy as np
import pytest

from conftest import criar_videos_com_landmarks
from libras.erros import ErroDadosInsuficientes, ErroLibras
from libras.splits import gerar_splits, salvar_split, videos_elegiveis
from libras.treino import OpcoesTreino, executar, treinar_final, treinar_particao

RAPIDO = {"treino": {"epocas": 3, "paciencia": 5}, "modelo": {"camadas_lstm": [8, 4]}}


@pytest.fixture
def dados(cfg):
    criar_videos_com_landmarks(cfg, n_sinalizadores=3, por_classe=2)
    linhas, _ = videos_elegiveis(cfg)
    salvar_split(cfg, gerar_splits(cfg, linhas))
    return cfg


def test_baseline_gera_artefatos(dados, tmp_path):
    pasta, r = executar(dados, OpcoesTreino(nome="t", familias=[], verbose=0, sobrescrever=RAPIDO, pasta_runs=tmp_path))
    for arq in (
        "config.yaml",
        "ambiente.json",
        "split.json",
        "features.json",
        "resumo.json",
        "metricas_particoes.csv",
        "LEIA-ME.txt",
        "matriz_confusao_janelas_total.png",
    ):
        assert (pasta / arq).exists(), arq
    fold0 = pasta / "seed_42" / "fold_0"
    for arq in (
        "historico.csv",
        "curvas.png",
        "predicoes_janelas.csv",
        "predicoes_videos.csv",
        "metricas.json",
        "modelo.keras",
    ):
        assert (fold0 / arq).exists(), arq
    assert r["n_resultados"] == 3 and r["familias_augmentation"] == []
    # matriz somada = total de janelas de teste (cada janela testada exatamente uma vez)
    ms = [json.loads((pasta / "seed_42" / f"fold_{i}" / "metricas.json").read_text()) for i in range(3)]
    assert np.sum(r["matriz_janela_soma"]) == sum(m["janelas_teste"] for m in ms)
    assert np.sum(r["matriz_video_soma"]) == 24
    assert r["hardware"]["nucleos_logicos"] >= 1


def test_augmentation_so_no_treino_e_registrada(dados, tmp_path):
    from libras.dataset import montar_conjunto
    from libras.splits import carregar_split

    split = carregar_split(dados)
    linhas, _ = videos_elegiveis(dados)
    conj = montar_conjunto(dados, linhas)
    cfg = dados
    cfg["treino"].update(RAPIDO["treino"])
    cfg["modelo"].update(RAPIDO["modelo"])
    for f in ("espacial", "temporal", "ruido"):
        cfg["augmentation"][f]["prob"] = 1.0
    f0 = split["folds"][0]
    r = treinar_particao(cfg, conj, f0, 1, ["espacial", "temporal", "ruido"], None, True, None, 0, False)
    n_tr = len(conj.indices(f0["treino"]))
    # o aumentador so viu janelas de treino: (epocas) x (janelas de treino)
    assert r["augmentation"]["janelas_vistas"] == r["epocas_executadas"] * n_tr
    assert r["augmentation"]["fracao_alteradas"] == 1.0


def test_vazamento_e_bloqueado(dados):
    from libras.dataset import montar_conjunto
    from libras.splits import carregar_split

    split = carregar_split(dados)
    linhas, _ = videos_elegiveis(dados)
    conj = montar_conjunto(dados, linhas)
    f = dict(split["folds"][0])
    f["treino"] = f["treino"] + [f["teste"][0]]
    with pytest.raises(ErroLibras, match="Vazamento"):
        treinar_particao(dados, conj, f, 1, [], None, True, None)


def test_offline_multiplica_treino(dados, tmp_path):
    sob = {**RAPIDO, "augmentation": {"modo": "offline", "copias_offline": 2}}
    pasta, r = executar(
        dados,
        OpcoesTreino(
            nome="off",
            familias=["espacial"],
            verbose=0,
            sobrescrever=sob,
            pasta_runs=tmp_path,
            folds=[0],
            salvar_modelos=False,
        ),
    )
    m = json.loads((pasta / "seed_42" / "fold_0" / "metricas.json").read_text())
    assert m["augmentation"]["fator_offline"] == 3.0


def test_modelo_final_e_metadados(dados, tmp_path):
    pasta = treinar_final(dados, OpcoesTreino(nome="final", verbose=0, sobrescrever=RAPIDO, pasta_runs=tmp_path))
    meta = json.loads((pasta / "metadados.json").read_text())
    assert (pasta / "modelo.keras").exists()
    assert meta["classes"] == dados.classes and meta["features"]["dim"] == 550
    assert "nao possui avaliacao independente" in meta["aviso"]


def test_sem_split_mensagem_clara(cfg):
    with pytest.raises(ErroDadosInsuficientes, match="gerar_splits"):
        executar(cfg, OpcoesTreino(verbose=0))


def test_modelo_saida_softmax(cfg):
    import tensorflow as tf

    from libras.modelo import construir_modelo

    m = construir_modelo(cfg, 30, 550, 4)
    y = m(tf.random.normal((3, 30, 550)), training=False).numpy()
    assert y.shape == (3, 4)
    np.testing.assert_allclose(y.sum(1), 1, atol=1e-5)
    assert [l.name for l in m.layers if "lstm" in l.name] == ["lstm_1", "lstm_2"]
