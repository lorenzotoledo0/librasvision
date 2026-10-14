"""Carregamento e validacao da configuracao central (configs/config.yaml).

A raiz do projeto e a pasta que contem configs/. Todos os caminhos do YAML
sao relativos a ela, o que permite rodar os scripts de qualquer diretorio.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
from pathlib import Path

import yaml

from .erros import ErroConfig

RAIZ_PADRAO = Path(__file__).resolve().parents[2]
CONFIG_PADRAO = RAIZ_PADRAO / "configs" / "config.yaml"

# usados quando um config de versao anterior nao tem a chave
CAMINHOS_PADRAO = {
    "dados": "data",
    "videos": "data/raw_videos",
    "landmarks": "data/landmarks",
    "landmarks_aug": "data/landmarks_aug",
    "manifesto": "data/manifest.csv",
    "sinalizadores": "data/sinalizadores.csv",
    "vocabulario": "data/vocabulario.json",
    "splits": "data/splits",
    "lixeira": "data/lixeira",
    "backup": "data/backup",
    "relatorios_importacao": "data/relatorios_importacao",
    "relatorios": "data/relatorios",
    "auditoria": "data/auditoria.log",
    "runs": "runs",
    "logs": "runs/logs",
    "publicados": "publicados",
}

_NOME_CLASSE = re.compile(r"^[a-z][a-z0-9_]*$")

_SECOES_OBRIGATORIAS = [
    "projeto",
    "caminhos",
    "mediapipe",
    "features",
    "coleta",
    "importacao",
    "extracao",
    "normalizacao",
    "janela",
    "avaliacao",
    "dataset",
]


class Config(dict):
    """Dicionario da configuracao com alguns atalhos. Imutavel por convencao."""

    def __init__(self, dados: dict, raiz: Path, arquivo: Path | None = None):
        super().__init__(dados)
        self.raiz = Path(raiz)
        self.arquivo = arquivo

    # --- caminhos -----------------------------------------------------
    def caminho(self, chave: str) -> Path:
        """Caminho absoluto de uma entrada da secao 'caminhos'."""
        rel = (self.get("caminhos") or {}).get(chave) or CAMINHOS_PADRAO.get(chave)
        if rel is None:
            raise ErroConfig(f"Caminho '{chave}' nao definido na secao 'caminhos' do config.")
        p = Path(rel)
        return p if p.is_absolute() else self.raiz / p

    # --- classes (fonte: data/vocabulario.json; ver libras.vocabulario) ---
    @property
    def classes(self) -> list[str]:
        """Classes ATIVAS do vocabulario, na ordem do arquivo (ordem dos indices do modelo)."""
        from .vocabulario import vocabulario_em_cache

        return vocabulario_em_cache(self).slugs_ativos()

    @property
    def classes_todas(self) -> list[str]:
        """Inclui as arquivadas (validas para registros antigos do manifesto)."""
        from .vocabulario import vocabulario_em_cache

        return vocabulario_em_cache(self).slugs()

    @property
    def nome_nenhum_sinal(self) -> str | None:
        from .vocabulario import vocabulario_em_cache

        return vocabulario_em_cache(self).nome_nenhum_sinal()

    # --- reprodutibilidade --------------------------------------------
    def hash_secao(self, *secoes: str) -> str:
        """Hash estavel de secoes da config (usado para invalidar caches)."""
        parte = {s: self.get(s) for s in secoes}
        txt = json.dumps(parte, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(txt.encode()).hexdigest()[:16]

    def como_dict(self) -> dict:
        return copy.deepcopy(dict(self))


def _validar(d: dict) -> None:
    faltando = [s for s in _SECOES_OBRIGATORIAS if s not in d]
    if faltando:
        raise ErroConfig(f"Secoes ausentes no config: {', '.join(faltando)}.")
    # 'classes' agora so define os presets da primeira execucao (o vocabulario vive em data/vocabulario.json)
    cl = d.get("classes") or {}
    todas = list(cl.get("sinais") or [])
    if cl.get("usar_nenhum_sinal"):
        todas.append(cl.get("nome_nenhum_sinal", "nenhum_sinal"))
    for c in todas:
        if not isinstance(c, str) or not _NOME_CLASSE.match(c):
            raise ErroConfig(
                f"Nome de classe invalido nos presets: {c!r}. Use letras minusculas, numeros e '_', "
                "sem acento nem espaco (ex.: 'sala_de_aula')."
            )
    if len(set(todas)) != len(todas):
        raise ErroConfig("Ha classes repetidas em classes.sinais.")
    modo_face = d["features"]["face"]["modo"]
    if modo_face not in ("nenhuma", "subconjunto", "completa"):
        raise ErroConfig("features.face.modo deve ser 'nenhuma', 'subconjunto' ou 'completa'.")
    j = d["janela"]
    if int(j["comprimento"]) < 2 or int(j["passo"]) < 1 or float(j["fps_alvo"]) <= 0:
        raise ErroConfig("janela: comprimento >= 2, passo >= 1 e fps_alvo > 0.")
    if j.get("selecao", "todas") not in ("todas", "atividade"):
        raise ErroConfig("janela.selecao deve ser 'todas' ou 'atividade'.")
    if d["avaliacao"]["protocolo"] not in ("grupos_sinalizador", "loso", "aleatorio_por_video"):
        raise ErroConfig("avaliacao.protocolo deve ser grupos_sinalizador, loso ou aleatorio_por_video.")
    for nome, rx in (d["importacao"].get("padroes_nome") or {}).items():
        try:
            re.compile(rx)
        except re.error as e:
            raise ErroConfig(f"Regex invalida em importacao.padroes_nome.{nome}: {e}") from e


def mesclar(base: dict, extra: dict) -> dict:
    """Mescla recursiva: valores de extra sobrescrevem os de base."""
    for k, v in (extra or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            mesclar(base[k], v)
        else:
            base[k] = v
    return base


def salvar_local(cfg: "Config", ajustes: dict) -> Path:
    """Grava ajustes em configs/local.yaml (mesclados aos ja existentes)."""
    from .utils import escrever_texto_atomico

    assert cfg.arquivo is not None
    arq = Path(cfg.arquivo).with_name("local.yaml")
    atual = yaml.safe_load(arq.read_text(encoding="utf-8")) if arq.exists() else {}
    mesclar(atual := atual or {}, ajustes)
    escrever_texto_atomico(
        arq,
        "# Ajustes locais feitos pelo aplicativo (aba Configuracoes).\n"
        "# Prevalecem sobre config.yaml. Apague este arquivo para voltar ao padrao.\n"
        + yaml.safe_dump(atual, allow_unicode=True, sort_keys=False),
    )
    return arq


def carregar_config(arquivo: str | os.PathLike | None = None) -> Config:
    """Carrega o YAML. Ordem: argumento > variavel LIBRAS_CONFIG > configs/config.yaml."""
    arq = Path(arquivo or os.environ.get("LIBRAS_CONFIG") or CONFIG_PADRAO).resolve()
    if not arq.is_file():
        raise ErroConfig(f"Arquivo de configuracao nao encontrado: {arq}")
    try:
        dados = yaml.safe_load(arq.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as e:
        raise ErroConfig(f"Erro de sintaxe no YAML {arq}:\n{e}") from e
    local = arq.with_name("local.yaml")
    if local.exists():  # ajustes feitos pela aba Configuracoes (o config.yaml comentado nao e reescrito)
        try:
            extra = yaml.safe_load(local.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as e:
            raise ErroConfig(f"Erro de sintaxe em {local}:\n{e}\nApague o arquivo para voltar ao padrao.") from e
        mesclar(dados, extra)
    _validar(dados)
    raiz = arq.parent.parent if arq.parent.name == "configs" else arq.parent
    return Config(dados, raiz=raiz, arquivo=arq)
