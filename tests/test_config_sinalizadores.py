import pytest
import yaml

from conftest import cadastrar
from libras.coleta.sinalizadores import CadastroSinalizadores
from libras.config import carregar_config
from libras.erros import ErroConfig, ErroConsentimento, ErroLibras


def test_config_carrega_e_inclui_nenhum_sinal(cfg):
    assert cfg.classes == ["biblioteca", "prova", "secretaria", "nenhum_sinal"]
    assert cfg.nome_nenhum_sinal == "nenhum_sinal"
    assert cfg.caminho("manifesto") == cfg.raiz / "data" / "manifest.csv"


def test_config_rejeita_classe_com_acento(projeto):
    arq = projeto / "configs" / "config.yaml"
    d = yaml.safe_load(arq.read_text(encoding="utf-8"))
    d["classes"]["sinais"].append("aula prática")
    arq.write_text(yaml.safe_dump(d, allow_unicode=True), encoding="utf-8")
    with pytest.raises(ErroConfig):
        carregar_config(arq)


def test_ids_sequenciais_e_sem_nomes(cfg):
    ids = cadastrar(cfg, 3)
    assert ids == ["S01", "S02", "S03"]
    texto = cfg.caminho("sinalizadores").read_text(encoding="utf-8")
    assert texto.splitlines()[0].split(",") == [
        "signer_id",
        "tcle_confirmado",
        "data_tcle",
        "mao_dominante",
        "data_cadastro",
        "observacoes",
    ]


def test_sem_tcle_bloqueia(cfg):
    (sid,) = cadastrar(cfg, 1, tcle=False)
    cad = CadastroSinalizadores(cfg)
    with pytest.raises(ErroConsentimento):
        cad.exigir_tcle(sid)
    cad.confirmar_tcle(sid, "2026-10-05")
    assert CadastroSinalizadores(cfg).exigir_tcle(sid).tcle_confirmado
    cad.revogar_tcle(sid)
    with pytest.raises(ErroConsentimento):
        CadastroSinalizadores(cfg).exigir_tcle(sid)


def test_tcle_confirmado_exige_data(cfg):
    with pytest.raises(ErroLibras):
        CadastroSinalizadores(cfg).cadastrar("direita", True, "")


def test_sinalizador_inexistente(cfg):
    with pytest.raises(ErroLibras):
        CadastroSinalizadores(cfg).exigir_tcle("S99")
