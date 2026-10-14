"""Preferencias do app real (13.4, 13.8, 13.9): arquivo proprio, fora do Git, separado do
configs/local.yaml da bancada. As cores escolhidas pelo usuario so existem aqui; o codigo continua sem
cor fora de tokens.json.
"""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path

from ..config_real import ARQUIVO_PREFERENCIAS
from ..ui.legenda import PREFERENCIAS_PADRAO
from ..utils import escrever_texto_atomico

log = logging.getLogger("libras.app_real")

GRUPOS_TECNICOS = ("taxa", "quadros", "tempo", "recursos", "sistema")
NOMES_GRUPOS = {
    "taxa": "Taxa (FPS da câmera, processados e de exibição)",
    "quadros": "Quadros (capturados, processados e pulados)",
    "tempo": "Tempo (latência de ponta a ponta, MediaPipe e modelo)",
    "recursos": "Recursos (memória da interface e do reconhecimento)",
    "sistema": "Sistema (modelo, câmera e estado)",
}

PADRAO = {
    "legenda": dict(PREFERENCIAS_PADRAO),
    "tecnicas": {g: False for g in GRUPOS_TECNICOS},  # todas desligadas: a barra nao existe
    "camera": {"indice": None, "avisos_enquadramento": True},
    "modelo": {"ativo": None, "anterior": None, "recusados": []},
    "pasta_modelos": None,
    "guia_visto": False,
    "reduzir_movimento": False,
}


def _mesclar(base: dict, extra: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (extra or {}).items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _mesclar(out[k], v)
        elif k in out:
            out[k] = v
    return out


class Preferencias:
    def __init__(self, arquivo: Path):
        self.arquivo = Path(arquivo)
        self.dados = copy.deepcopy(PADRAO)
        self.carregar()

    @classmethod
    def da_raiz(cls, raiz: Path) -> "Preferencias":
        return cls(Path(raiz) / "configs" / ARQUIVO_PREFERENCIAS)

    def carregar(self) -> None:
        if not self.arquivo.exists():
            return
        try:
            self.dados = _mesclar(PADRAO, json.loads(self.arquivo.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            log.warning("Preferencias ilegiveis; usando o padrao")
            self.dados = copy.deepcopy(PADRAO)

    def salvar(self) -> None:
        try:
            escrever_texto_atomico(self.arquivo, json.dumps(self.dados, indent=2, ensure_ascii=False))
        except OSError:
            log.warning("Nao foi possivel salvar as preferencias em %s", self.arquivo.name)

    def __getitem__(self, chave):
        return self.dados[chave]

    def restaurar_legenda(self) -> None:
        self.dados["legenda"] = dict(PREFERENCIAS_PADRAO)
        self.salvar()

    def algum_grupo_tecnico(self) -> bool:
        return any(self.dados["tecnicas"].get(g) for g in GRUPOS_TECNICOS)
