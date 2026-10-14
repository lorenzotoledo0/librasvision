"""Configuracao do app real (configs/app_real.yaml) e do servico de tempo real quando ele roda para o
app real (item 13, D19). Compartilhado e leve: so le YAML, sem Qt, TensorFlow ou MediaPipe.

O servico recebe a mesma estrutura que recebe da bancada (secoes coleta.camera e tempo_real), com os
parametros de decisao vindos do pacote publicado, nunca do usuario.
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

from .config import RAIZ_PADRAO, Config
from .decisao import parametros
from .erros import ErroConfig

ARQUIVO_PADRAO = RAIZ_PADRAO / "configs" / "app_real.yaml"
ARQUIVO_PREFERENCIAS = "app_real_preferencias.json"  # em configs/, fora do Git

PADRAO = {
    "pasta_modelos": "publicados",
    "camera": {
        "indice": 0,
        "largura": 640,
        "altura": 480,
        "fps": 30,
        "backends_windows": ["DSHOW", "MSMF", "ANY"],
        "forcar_mjpg": True,
    },
    "enquadramento": {
        "borda_mao": 0.03,
        "brilho_minimo": 60,
        "visibilidade_ombro": 0.5,
        "janela_s": 1.0,
        "histerese_s": 1.0,
    },
    "informacoes_tecnicas": {"intervalo_ms": 500, "janela_latencia_s": 10},
    "legenda": {"tempo_para_sumir_s": 4},
}


def _mesclar(base: dict, extra: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (extra or {}).items():
        out[k] = _mesclar(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def carregar(arquivo: Path | str | None = None) -> Config:
    """Config do app real. A raiz e a pasta que contem configs/ (como na bancada)."""
    arq = Path(arquivo) if arquivo else ARQUIVO_PADRAO
    dados = {}
    if arq.exists():
        try:
            dados = yaml.safe_load(arq.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as e:
            raise ErroConfig(f"{arq.name} com erro de sintaxe: {e}") from e
    raiz = arq.resolve().parent.parent
    return Config(_mesclar(PADRAO, dados), raiz, arq)


def pasta_modelos(cfg: Config) -> Path:
    p = Path(cfg["pasta_modelos"])
    return p if p.is_absolute() else cfg.raiz / p


def config_servico(cfg_real: Config, pasta_pacote: Path, cartao: dict, indice_camera: int | None = None) -> Config:
    """Estrutura que o servico de tempo real espera, montada para o app real."""
    camera = dict(cfg_real["camera"])
    if indice_camera is not None:
        camera["indice"] = int(indice_camera)
    tr = parametros(cartao.get("decisao"))
    tr["modelo"] = str(Path(pasta_pacote).resolve())
    return Config(
        {
            "coleta": {"camera": camera},
            "tempo_real": tr,
            "enquadramento": dict(cfg_real["enquadramento"]),
            "informacoes_tecnicas": dict(cfg_real["informacoes_tecnicas"]),
        },
        cfg_real.raiz,
        cfg_real.arquivo,
    )
