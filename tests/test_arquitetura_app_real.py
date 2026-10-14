"""13.6: o app real nao importa coleta, dataset, treino, ablacao, augmentation nem a interface da
bancada; a interface do app real nunca importa TensorFlow nem MediaPipe (D15)."""

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
SRC = RAIZ / "src"
PROIBIDOS = (
    "libras.coleta",
    "libras.dataset",
    "libras.treino",
    "libras.ablacao",
    "libras.augmentation",
    "libras.app",
    "libras.publicacao",
    "libras.extracao",
    "libras.splits",
    "libras.selecao",
    "libras.relatorio_ablacao",
    "libras.robustez",
    "libras.vocabulario",
    "libras.teste_ao_vivo",
    "libras.validacao",
    "libras.metricas",
    "libras.modelo",
    "libras.estatistica",
    "libras.tabelas",
    "libras.treino",
)
MODULOS_APP_REAL = sorted(
    f"libras.app_real.{p.stem}"
    for p in (SRC / "libras" / "app_real").glob("*.py")
    if p.stem not in ("__init__", "__main__")
)


def proibido(mod: str) -> bool:
    return any(mod == p or mod.startswith(p + ".") for p in PROIBIDOS)


def _modulos_importados(codigo: str) -> list[str]:
    env = {**os.environ, "PYTHONPATH": str(SRC), "QT_QPA_PLATFORM": "offscreen"}
    r = subprocess.run(
        [sys.executable, "-c", codigo + "\nimport sys, json; print(json.dumps(sorted(sys.modules)))"],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_interface_do_app_real_nao_importa_a_bancada_nem_tensorflow():
    assert MODULOS_APP_REAL, "app real sem modulos?"
    mods = _modulos_importados("\n".join(f"import {m}" for m in MODULOS_APP_REAL))
    errados = [m for m in mods if proibido(m)]
    assert not errados, f"o app real importa: {errados}"
    assert "tensorflow" not in mods and "mediapipe" not in mods and "keras" not in mods


def test_imports_declarados_no_codigo_do_app_real():
    """Tambem os imports dentro de funcoes (que so rodariam num clique)."""
    for arq in (SRC / "libras" / "app_real").glob("*.py"):
        for no in ast.walk(ast.parse(arq.read_text(encoding="utf-8"))):
            if isinstance(no, ast.ImportFrom):
                pacote = {2: "libras", 1: "libras.app_real", 0: ""}[no.level]
                base = ".".join(x for x in (pacote, no.module or "") if x)
                assert not proibido(base), f"{arq.name}: from {'.' * no.level}{no.module}"
                if no.module is None:  # from .. import treino
                    for a in no.names:
                        assert not proibido(f"{base}.{a.name}"), f"{arq.name}: from {'.' * no.level} import {a.name}"
            elif isinstance(no, ast.Import):
                for a in no.names:
                    assert not proibido(a.name) and a.name.split(".")[0] not in ("tensorflow", "mediapipe"), a.name


def test_modulos_compartilhados_nao_importam_a_bancada():
    """O que o app real usa da pasta comum (ui, servico, pacote, camera, decisao, enquadramento)."""
    comuns = [
        "libras.ui.tema",
        "libras.ui.componentes",
        "libras.ui.dinamica",
        "libras.ui.icones",
        "libras.ui.legenda",
        "libras.ui.sobreposicoes",
        "libras.tempo_real_servico",
        "libras.pacote",
        "libras.config_real",
        "libras.camera",
        "libras.decisao",
        "libras.enquadramento",
        "libras.inferencia",
    ]
    mods = _modulos_importados("\n".join(f"import {m}" for m in comuns))
    assert not [m for m in mods if proibido(m)]
    assert "tensorflow" not in mods  # o servico so importa TensorFlow ao carregar o modelo, no subprocesso
