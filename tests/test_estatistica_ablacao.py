"""Estatistica (valores conhecidos) e ablacao de ponta a ponta com landmarks SINTETICOS.
Os numeros da ablacao sintetica nao significam nada e nunca devem ser reportados."""

import json
import math

import numpy as np
import pytest

from conftest import criar_videos_com_landmarks
from libras.erros import ErroDadosInsuficientes
from libras.estatistica import holm, t_corrigido, wilcoxon_pareado


def test_t_corrigido_formula():
    d = [0.02, 0.05, 0.01, 0.04, 0.03]
    r = t_corrigido(d, n_treino=800, n_teste=200, k=5, r=1)
    var = np.var(d, ddof=1)
    t_esp = np.mean(d) / math.sqrt((1 / 5 + 200 / 800) * var)
    assert r["t"] == pytest.approx(t_esp) and r["gl"] == 4
    from scipy import stats

    assert r["p"] == pytest.approx(2 * stats.t.sf(t_esp, 4))
    assert r["ic95"][0] < r["media"] < r["ic95"][1]
    # a correcao torna o teste mais conservador que o t pareado comum
    assert r["p"] > stats.ttest_1samp(d, 0).pvalue


def test_t_corrigido_casos_limite():
    assert math.isnan(t_corrigido([0.1], 10, 2, 1, 1)["p"])
    assert t_corrigido([0.0, 0.0, 0.0], 10, 2, 3, 1)["p"] == 1.0


def test_holm_exemplo():
    assert holm([0.01, 0.04, 0.03, 0.005]) == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert holm([0.2, float("nan")])[1] != holm([0.2, float("nan")])[1]  # NaN preservado


def test_wilcoxon_p_minimo():
    w = wilcoxon_pareado([0.1, 0.2, 0.3, 0.4, 0.5], [0.2, 0.3, 0.4, 0.5, 0.6])
    assert w["p_minimo_possivel"] == pytest.approx(0.0625) and w["p"] == pytest.approx(0.0625)


@pytest.fixture
def dados_ablacao(cfg):
    from libras.splits import gerar_splits, salvar_split, videos_elegiveis

    criar_videos_com_landmarks(cfg, n_sinalizadores=3, por_classe=1)
    linhas, _ = videos_elegiveis(cfg)
    salvar_split(cfg, gerar_splits(cfg, linhas))
    cfg["ablacao"]["intensidades"] = ["fraca", "media"]
    return cfg


RAPIDO = {"treino": {"epocas": 2, "paciencia": 5}, "modelo": {"camadas_lstm": [6]}}


def test_ablacao_completa_sintetica(dados_ablacao, tmp_path):
    from libras.ablacao import executar_ablacao

    logs = []
    pasta = executar_ablacao(dados_ablacao, pasta=tmp_path / "abl", seeds=[1], sobrescrever=RAPIDO, log=logs.append)
    nomes = sorted(p.name for p in pasta.iterdir() if p.is_dir())
    assert {"baseline", "so_espacial", "so_temporal", "so_ruido", "combinacao", "baseline_aleatorio_por_video"} <= set(
        nomes
    )
    assert any(n.endswith("_fraca") for n in nomes)  # etapa de intensidades rodou
    for arq in (
        "RELATORIO.md",
        "tabela_ablacao.csv",
        "tabela_ablacao.tex",
        "tabela_robustez.tex",
        "tabela_protocolos.tex",
        "fig_ablacao.png",
        "fig_robustez.png",
        "resultados.json",
        "tabela_por_classe_baseline.tex",
    ):
        assert (pasta / arq).exists(), arq
    res = json.loads((pasta / "resultados.json").read_text())
    assert "video" not in res["condicoes"]["combinacao"]["familias"]
    assert any("video" in a for a in res["avisos"]) and any("ENSAIO REDUZIDO" in a for a in res["avisos"])
    c = res["condicoes"]["so_espacial"]["comparacao_baseline"]
    assert c["pares"] == 3 and "p_holm" in c
    tex = (pasta / "tabela_ablacao.tex").read_text()
    assert "\\fonte{Elaborado pelos autores" in tex and "\\begin{tabular}" in tex
    rob = res["condicoes"]["baseline"]["resumo"]["robustez"]
    assert {"oclusao_mao", "ruido", "queda_fps", "sem_face"} <= set(rob)
    assert "indisponivel" in rob["escuro"]
    # retomada: tudo reaproveitado
    logs.clear()
    executar_ablacao(dados_ablacao, pasta=pasta, seeds=[1], sobrescrever=RAPIDO, log=logs.append)
    assert sum("reaproveitando" in l for l in logs) == len(nomes)
    # plano diferente na mesma pasta: recusa
    with pytest.raises(ErroDadosInsuficientes, match="outro plano"):
        executar_ablacao(dados_ablacao, pasta=pasta, seeds=[2], sobrescrever=RAPIDO, log=logs.append)


def test_robustez_degradacoes(cfg):
    from conftest import partes_sinteticas
    from libras.normalizacao import espec_de_config, preparar_janela
    from libras.robustez import degradar_lote

    espec = espec_de_config(cfg)
    X = np.stack(
        [
            preparar_janela(partes_sinteticas(30, mover_mao=(0, 30), seed=s), 640, 480, espec, cfg["normalizacao"]).X
            for s in range(3)
        ]
    )
    pi = espec.blocos["presenca"][0]
    a = degradar_lote(X, espec, "oclusao_mao", {"frames": 10}, seed=5)
    assert ((a[:, :, pi + 2] == 0).sum(axis=1) == 10).all()
    b = degradar_lote(X, espec, "oclusao_mao", {"frames": 10}, seed=5)
    np.testing.assert_array_equal(a, b)
    s = degradar_lote(X, espec, "sem_face", {}, seed=1)
    assert (s[:, :, pi + 3] == 0).all()
    q = degradar_lote(X, espec, "queda_fps", {"fracao": 0.5}, seed=1)
    assert q.shape == X.shape and np.isfinite(q).all()
