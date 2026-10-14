"""Vocabulario (item 3): fonte unica das classes, em data/vocabulario.json.

Cada classe tem: slug (nome tecnico, sem acento), rotulo (como aparece na tela), tipo
(sinal | nenhum_sinal), meta de videos por sinalizador (com excecoes por sinalizador),
ativa/arquivada e data de criacao. Na primeira execucao o arquivo e criado com os presets
do TCC (biblioteca, prova, secretaria, nenhum_sinal), todos editaveis e removiveis.

Classes arquivadas continuam validas para os registros antigos do manifesto, mas nao
aparecem para gravacao nem para treino.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .erros import ErroConfig, ErroLibras
from .utils import agora_iso, escrever_texto_atomico

TIPOS = ("sinal", "nenhum_sinal")
_RX_SLUG = re.compile(r"^[a-z][a-z0-9_]{0,39}$")
VERSAO_ARQUIVO = 1


def slugificar(texto: str) -> str:
    """'Sala de Aula!' -> 'sala_de_aula'; 'Ção' -> 'cao'. Comeca com letra (prefixo 'sinal_' se preciso)."""
    t = unicodedata.normalize("NFKD", str(texto)).encode("ascii", "ignore").decode("ascii").lower()
    t = re.sub(r"[^a-z0-9]+", "_", t).strip("_")
    t = re.sub(r"_+", "_", t)
    if t and not t[0].isalpha():
        t = "sinal_" + t
    return t[:40].rstrip("_")


@dataclass
class Classe:
    slug: str
    rotulo: str
    tipo: str = "sinal"
    meta_por_sinalizador: int = 10
    ativa: bool = True
    metas_sinalizador: dict[str, int] = field(default_factory=dict)
    criada_em: str = ""

    @staticmethod
    def de_dict(d: dict) -> "Classe":
        return Classe(
            slug=str(d["slug"]),
            rotulo=str(d.get("rotulo") or d["slug"]),
            tipo=str(d.get("tipo", "sinal")),
            meta_por_sinalizador=int(d.get("meta_por_sinalizador", 10)),
            ativa=bool(d.get("ativa", True)),
            metas_sinalizador={str(k): int(v) for k, v in (d.get("metas_sinalizador") or {}).items()},
            criada_em=str(d.get("criada_em", "")),
        )


def presets(cfg) -> list[Classe]:
    """Classes iniciais a partir do config (secao 'classes'), usadas so na primeira execucao."""
    c = cfg.get("classes") or {}
    meta = int(((cfg.get("coleta") or {}).get("metas") or {}).get("takes_por_classe_por_sinalizador", 10))
    agora = agora_iso()
    out = [Classe(s, s, "sinal", meta, True, {}, agora) for s in (c.get("sinais") or [])]
    if c.get("usar_nenhum_sinal", True):
        out.append(
            Classe(c.get("nome_nenhum_sinal", "nenhum_sinal"), "nenhum sinal", "nenhum_sinal", meta, True, {}, agora)
        )
    return out


class Vocabulario:
    def __init__(self, cfg, criar: bool = True):
        self.cfg = cfg
        self.arquivo: Path = cfg.caminho("vocabulario")
        self._classes: dict[str, Classe] = {}
        self.carregar(criar)

    # ------------------------------------------------------------------
    def carregar(self, criar: bool = True) -> None:
        if not self.arquivo.exists():
            self._classes = {c.slug: c for c in presets(self.cfg)}
            if criar:
                self.salvar()
            return
        try:
            d = json.loads(self.arquivo.read_text(encoding="utf-8-sig"))
            itens = d["classes"] if isinstance(d, dict) else d
            classes = [Classe.de_dict(x) for x in itens]
        except Exception as e:
            raise ErroConfig(
                f"O arquivo {self.arquivo} esta corrompido ({type(e).__name__}: {e}). Restaure uma "
                "copia de data/backup/ ou apague o arquivo para recriar o vocabulario padrao."
            ) from e
        self._classes = {}
        for c in classes:
            if not _RX_SLUG.match(c.slug) or c.tipo not in TIPOS:
                raise ErroConfig(f"Classe invalida em {self.arquivo.name}: {c.slug!r} (tipo {c.tipo!r}).")
            self._classes[c.slug] = c

    def salvar(self) -> None:
        dados = {"versao": VERSAO_ARQUIVO, "classes": [asdict(c) for c in self._classes.values()]}
        escrever_texto_atomico(self.arquivo, json.dumps(dados, indent=2, ensure_ascii=False))
        _invalidar_cache(self.cfg)

    # ------------------------------------------------------------------
    def todas(self) -> list[Classe]:
        return list(self._classes.values())

    def ativas(self) -> list[Classe]:
        return [c for c in self._classes.values() if c.ativa]

    def slugs_ativos(self) -> list[str]:
        return [c.slug for c in self.ativas()]

    def slugs(self) -> list[str]:
        return list(self._classes)

    def obter(self, slug: str) -> Classe:
        if slug not in self._classes:
            raise ErroLibras(f"Classe {slug!r} nao existe no vocabulario.")
        return self._classes[slug]

    def nome_nenhum_sinal(self) -> str | None:
        for c in self.ativas():
            if c.tipo == "nenhum_sinal":
                return c.slug
        return None

    def meta(self, slug: str, signer_id: str | None = None) -> int:
        c = self.obter(slug)
        if signer_id and signer_id in c.metas_sinalizador:
            return c.metas_sinalizador[signer_id]
        return c.meta_por_sinalizador

    # ------------------------------------------------------------------
    def adicionar(self, rotulo: str, tipo: str = "sinal", meta: int | None = None) -> Classe:
        rotulo = " ".join(str(rotulo).split())
        slug = slugificar(rotulo)
        if not slug:
            raise ErroLibras("Digite uma palavra com pelo menos uma letra.")
        if tipo not in TIPOS:
            raise ErroLibras(f"Tipo invalido: {tipo}. Opcoes: {TIPOS}")
        for c in self._classes.values():
            if c.slug == slug or c.rotulo.casefold() == rotulo.casefold():
                estado = "" if c.ativa else " (arquivada; reative-a em Vocabulario)"
                raise ErroLibras(f"A palavra ja existe no vocabulario como {c.slug!r}{estado}.")
        if tipo == "nenhum_sinal" and any(c.tipo == "nenhum_sinal" for c in self._classes.values()):
            raise ErroLibras("Ja existe uma classe do tipo nenhum_sinal.")
        meta = (
            int(meta)
            if meta is not None
            else int(((self.cfg.get("coleta") or {}).get("metas") or {}).get("takes_por_classe_por_sinalizador", 10))
        )
        if meta < 1:
            raise ErroLibras("A meta precisa ser de pelo menos 1 video.")
        c = Classe(slug, rotulo, tipo, meta, True, {}, agora_iso())
        self._classes[slug] = c
        self.salvar()
        return c

    def editar(self, slug: str, rotulo: str | None = None, meta: int | None = None) -> Classe:
        c = self.obter(slug)
        if rotulo is not None:
            rotulo = " ".join(rotulo.split())
            if not rotulo:
                raise ErroLibras("O rotulo nao pode ficar vazio.")
            if any(o.slug != slug and o.rotulo.casefold() == rotulo.casefold() for o in self._classes.values()):
                raise ErroLibras(f"Ja existe outra classe com o rotulo {rotulo!r}.")
            c.rotulo = rotulo  # o slug nao muda: ele nomeia pastas e registros ja gravados
        if meta is not None:
            if int(meta) < 1:
                raise ErroLibras("A meta precisa ser de pelo menos 1 video.")
            c.meta_por_sinalizador = int(meta)
        self.salvar()
        return c

    def definir_meta_sinalizador(self, slug: str, signer_id: str, meta: int | None) -> None:
        """meta None remove a excecao (volta a valer a meta da classe)."""
        c = self.obter(slug)
        if meta is None:
            c.metas_sinalizador.pop(signer_id, None)
        else:
            if int(meta) < 1:
                raise ErroLibras("A meta precisa ser de pelo menos 1 video.")
            c.metas_sinalizador[signer_id] = int(meta)
        self.salvar()

    def arquivar(self, slug: str, ativa: bool = False) -> None:
        self.obter(slug).ativa = ativa
        self.salvar()

    def remover_entrada(self, slug: str) -> None:
        """Tira a classe do vocabulario. Quem chama garante que nao ha dados (ver gestao.remover_classe)."""
        self.obter(slug)
        del self._classes[slug]
        self.salvar()

    def remover_metas_de(self, signer_id: str) -> None:
        mudou = False
        for c in self._classes.values():
            if c.metas_sinalizador.pop(signer_id, None) is not None:
                mudou = True
        if mudou:
            self.salvar()


# ----------------------------------------------------------------------
# Cache usado por Config.classes (relido quando o arquivo muda)
# ----------------------------------------------------------------------
def _invalidar_cache(cfg) -> None:
    try:
        cfg._cache_vocab = None
    except AttributeError:
        pass


def vocabulario_em_cache(cfg) -> Vocabulario:
    arq = cfg.caminho("vocabulario")
    chave = (str(arq), arq.stat().st_mtime_ns if arq.exists() else None)
    cache = getattr(cfg, "_cache_vocab", None)
    if cache is not None and cache[0] == chave:
        return cache[1]
    v = Vocabulario(cfg, criar=False)
    cfg._cache_vocab = (chave, v)
    return v
