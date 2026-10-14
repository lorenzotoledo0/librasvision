"""Os scripts devem funcionar com data/ vazio e falhar com mensagem clara (sem traceback)."""

import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]


def _rodar(projeto, *args):
    cfg = projeto / "configs" / "config.yaml"
    return subprocess.run(
        [sys.executable, str(RAIZ / "scripts" / args[0]), *args[1:], "--config", str(cfg)],
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_verificar_manifesto_vazio(projeto):
    r = _rodar(projeto, "verificar_manifesto.py")
    assert r.returncode == 0, r.stderr
    assert "Ainda nao ha videos" in r.stdout


def test_importar_pasta_vazia_mensagem_clara(projeto, tmp_path):
    (tmp_path / "vazia").mkdir()
    r = _rodar(projeto, "importar.py", str(tmp_path / "vazia"))
    assert r.returncode == 2
    assert "[ERRO] Nenhum video" in r.stderr and "Traceback" not in r.stderr


def test_sinalizadores_cli(projeto):
    r = _rodar(projeto, "sinalizadores.py", "cadastrar", "--mao", "direita")
    assert r.returncode == 0 and "S01" in r.stdout and "PENDENTE" in r.stdout
    r = _rodar(projeto, "sinalizadores.py", "confirmar-tcle", "S01", "--data", "2026-10-05")
    assert r.returncode == 0
    r = _rodar(projeto, "sinalizadores.py", "listar")
    assert "TCLE=sim" in r.stdout


def test_extrair_vazio_mensagem_clara(projeto):
    r = _rodar(projeto, "extrair.py")
    assert r.returncode == 2 and "[ERRO] Nenhum video" in r.stderr and "Traceback" not in r.stderr


def test_gerar_splits_vazio_mensagem_clara(projeto):
    r = _rodar(projeto, "gerar_splits.py")
    assert r.returncode == 2 and "Nenhum video elegivel" in r.stderr and "Traceback" not in r.stderr


def test_validar_dados_vazio(projeto):
    r = _rodar(projeto, "validar_dados.py")
    assert r.returncode == 0 and "Ainda nao ha videos" in r.stdout


def test_treinar_vazio_mensagem_clara(projeto):
    r = _rodar(projeto, "treinar.py", "--aug", "nenhuma")
    assert r.returncode == 2 and "[ERRO]" in r.stderr and "Traceback" not in r.stderr
