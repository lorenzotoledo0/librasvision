"""Rotinas de manutencao executadas na abertura dos aplicativos."""

from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path

from ..utils import carimbo

log = logging.getLogger("libras.manutencao")

PADROES_TEMPORARIOS = ("*.parcial.*", "*.tmp")


def limpar_temporarios(cfg, idade_min_s: float = 600.0) -> list[Path]:
    """Move para data/lixeira/orfaos/<data>/ os arquivos temporarios deixados por um fechamento
    abrupto (take pela metade, escrita atomica interrompida). Nunca apaga."""
    dados = cfg.caminho("dados")
    if not dados.exists():
        return []
    lixeira = cfg.caminho("lixeira")
    achados = []
    for padrao in PADROES_TEMPORARIOS:
        for p in dados.rglob(padrao):
            # so arquivos parados ha algum tempo: outro processo pode estar escrevendo agora
            if p.is_file() and lixeira not in p.parents and time.time() - p.stat().st_mtime >= idade_min_s:
                achados.append(p)
    movidos = []
    if achados:
        destino_base = lixeira / "orfaos" / carimbo()
        for p in achados:
            destino = destino_base / p.relative_to(dados)
            destino.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.move(str(p), str(destino))
                movidos.append(destino)
                log.warning("Arquivo temporario orfao movido: %s -> %s", p, destino)
            except OSError as e:  # arquivo em uso por outro processo: deixa onde esta
                log.warning("Nao foi possivel mover %s: %s", p, e)
    return movidos
