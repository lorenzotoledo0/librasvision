"""Remocao de sinalizadores e de classes, com lixeira e auditoria (item 3).

Sinalizador, em duas etapas:
  (a) remover_sinalizador: confirmacao digitando o ID. Move videos, landmarks (inclusive
      variantes de video) e takes antigos da lixeira para data/lixeira/<ID>_<data>/,
      guarda la as linhas do manifesto e o registro do cadastro, tira o sinalizador do
      manifesto e do cadastro e registra em data/auditoria.log. Reversivel a mao.
  (b) excluir_definitivamente: para retirada de consentimento. Dupla confirmacao (o ID e a
      palavra EXCLUIR). Apaga de fato tudo o que estiver na lixeira em nome do sinalizador.
Ambas invalidam os splits (movidos para data/splits/invalidos/<data>/) com aviso.

Classe: so pode ser apagada direto se nao tiver dados; com dados, os arquivos vao para
data/lixeira/classe_<slug>_<data>/ pelo mesmo fluxo, e a classe sai do vocabulario.

O registro de auditoria guarda acao, alvo (ID anonimo ou classe), quantidades e data;
nunca nomes de pessoas.
"""

from __future__ import annotations

import csv
import json
import logging
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..config import Config
from ..erros import ErroLibras
from ..utils import agora_iso, carimbo
from .manifesto import COLUNAS, Manifesto
from .sinalizadores import CadastroSinalizadores

log = logging.getLogger("libras.gestao")
PALAVRA_EXCLUSAO = "EXCLUIR"


@dataclass
class ResultadoRemocao:
    alvo: str
    pasta_lixeira: Path | None
    videos: int = 0
    landmarks: int = 0
    outros_arquivos: int = 0
    splits_invalidados: list[str] = field(default_factory=list)
    avisos: list[str] = field(default_factory=list)


def auditar(cfg: Config, acao: str, alvo: str, **detalhes) -> None:
    arq = cfg.caminho("auditoria")
    arq.parent.mkdir(parents=True, exist_ok=True)
    reg = {"quando": agora_iso(), "acao": acao, "alvo": alvo, **detalhes}
    with open(arq, "a", encoding="utf-8") as f:
        f.write(json.dumps(reg, ensure_ascii=False, default=str) + "\n")
    log.info("Auditoria: %s", reg)


def invalidar_splits(cfg: Config, motivo: str) -> list[str]:
    """Move os splits salvos para data/splits/invalidos/<data>/ (os dados mudaram)."""
    pasta = cfg.caminho("splits")
    if not pasta.exists():
        return []
    arquivos = sorted(p for p in pasta.glob("*.json"))
    if not arquivos:
        return []
    destino = pasta / "invalidos" / carimbo()
    destino.mkdir(parents=True, exist_ok=True)
    for p in arquivos:
        shutil.move(str(p), str(destino / p.name))
    (destino / "MOTIVO.txt").write_text(motivo + "\n", encoding="utf-8")
    return [p.name for p in arquivos]


def _mover(origem: Path, base_dados: Path, destino_base: Path) -> bool:
    if not origem.exists():
        return False
    try:
        rel = origem.resolve().relative_to(base_dados.resolve())
    except ValueError:
        rel = Path(origem.name)
    destino = destino_base / rel
    destino.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(origem), str(destino))
    return True


def _arquivos_derivados(cfg: Config, linha: dict) -> list[Path]:
    from ..extracao import caminho_npz

    out = [caminho_npz(cfg, linha)]
    aug = cfg.caminho("landmarks_aug")
    if aug.exists():
        out += list(aug.glob(f"*/{linha['classe']}/{linha['video_id']}.npz"))
    return out


def _salvar_linhas(destino: Path, linhas: list[dict]) -> None:
    destino.parent.mkdir(parents=True, exist_ok=True)
    with open(destino, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUNAS, extrasaction="ignore")
        w.writeheader()
        w.writerows(linhas)


def _mover_registros(cfg: Config, linhas: list[dict], destino: Path, res: ResultadoRemocao) -> None:
    man = Manifesto(cfg)
    dados = cfg.caminho("dados")
    for l in linhas:
        if _mover(man.caminho_absoluto(l), dados, destino):
            res.videos += 1
        for p in _arquivos_derivados(cfg, l):
            if _mover(p, dados, destino):
                res.landmarks += 1
    _salvar_linhas(destino / "manifesto_removido.csv", linhas)
    ids = {l["video_id"] for l in linhas}
    with man.trava():
        man.recarregar()
        man._linhas = [x for x in man._linhas if x["video_id"] not in ids]
        man.salvar()


# ======================================================================
# Sinalizador
# ======================================================================
def remover_sinalizador(cfg: Config, signer_id: str, confirmacao: str) -> ResultadoRemocao:
    if confirmacao.strip() != signer_id:
        raise ErroLibras(f"Confirmacao incorreta: digite exatamente {signer_id} para remover.")
    cad = CadastroSinalizadores(cfg)
    sinal = cad.obter(signer_id)
    man = Manifesto(cfg)
    linhas = [l for l in man.linhas if l["signer_id"] == signer_id]
    if sinal is None and not linhas:
        raise ErroLibras(f"Sinalizador {signer_id} nao encontrado no cadastro nem no manifesto.")
    destino = cfg.caminho("lixeira") / f"{signer_id}_{carimbo()}"
    res = ResultadoRemocao(signer_id, destino)
    destino.mkdir(parents=True, exist_ok=False)
    _mover_registros(cfg, linhas, destino, res)
    # takes que ja estavam na lixeira (exclusoes individuais, cortes, orfaos): vao junto
    lix = cfg.caminho("lixeira")
    for p in sorted(lix.rglob(f"{signer_id}_*")):
        if not p.is_file():
            continue
        if any(parte.startswith(f"{signer_id}_") for parte in p.relative_to(lix).parts[:-1]):
            continue  # ja esta numa pasta de remocao deste sinalizador (inclusive a atual)
        if _mover(p, lix, destino / "lixeira_anterior"):
            res.outros_arquivos += 1
    if sinal is not None:
        (destino / "sinalizador.json").write_text(
            json.dumps(asdict(sinal), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        cad.remover(signer_id)
    from ..vocabulario import Vocabulario

    Vocabulario(cfg).remover_metas_de(signer_id)
    res.splits_invalidados = invalidar_splits(cfg, f"Sinalizador {signer_id} removido em {agora_iso()}.")
    if res.splits_invalidados:
        res.avisos.append("Os splits salvos foram invalidados (os dados mudaram). Gere de novo antes de treinar.")
    res.avisos.append(
        "Execucoes e modelos ja treinados em runs/ nao mudam; se este sinalizador estava no treino, "
        "retreine antes de divulgar novos resultados."
    )
    auditar(
        cfg,
        "remover_sinalizador",
        signer_id,
        videos=res.videos,
        landmarks=res.landmarks,
        outros=res.outros_arquivos,
        lixeira=destino.name,
        splits_invalidados=res.splits_invalidados,
    )
    return res


def excluir_definitivamente(cfg: Config, signer_id: str, confirmacao_id: str, confirmacao_palavra: str) -> int:
    """Apaga de fato o conteudo da lixeira deste sinalizador. Retorna quantos arquivos foram apagados."""
    if confirmacao_id.strip() != signer_id or confirmacao_palavra.strip() != PALAVRA_EXCLUSAO:
        raise ErroLibras(f"Confirmacao incorreta: digite {signer_id} e depois {PALAVRA_EXCLUSAO}.")
    if any(l["signer_id"] == signer_id for l in Manifesto(cfg).linhas) or CadastroSinalizadores(cfg).obter(signer_id):
        raise ErroLibras(
            f"{signer_id} ainda esta no manifesto ou no cadastro. Remova-o primeiro (etapa 1); "
            "a exclusao definitiva apaga apenas o que ja esta na lixeira."
        )
    lix = cfg.caminho("lixeira")
    apagados = 0
    if lix.exists():
        for d in sorted(lix.glob(f"{signer_id}_*")):
            if d.is_dir():
                apagados += sum(1 for p in d.rglob("*") if p.is_file())
                shutil.rmtree(d)
        for p in sorted(lix.rglob(f"{signer_id}_*")):  # restos fora das pastas do sinalizador (cortes, orfaos)
            if p.is_file():
                p.unlink()
                apagados += 1
    invalidar_splits(cfg, f"Dados de {signer_id} excluidos definitivamente em {agora_iso()}.")
    auditar(cfg, "excluir_definitivamente", signer_id, arquivos_apagados=apagados)
    return apagados


# ======================================================================
# Classe
# ======================================================================
def remover_classe(cfg: Config, slug: str, confirmacao: str) -> ResultadoRemocao:
    from ..vocabulario import Vocabulario

    voc = Vocabulario(cfg)
    voc.obter(slug)
    if confirmacao.strip() != slug:
        raise ErroLibras(f"Confirmacao incorreta: digite exatamente {slug} para remover a classe.")
    linhas = [l for l in Manifesto(cfg).linhas if l["classe"] == slug]
    pasta_videos = cfg.caminho("videos") / slug
    sobra = [p for p in pasta_videos.rglob("*") if p.is_file()] if pasta_videos.exists() else []
    if not linhas and not sobra:
        voc.remover_entrada(slug)
        auditar(cfg, "remover_classe", slug, videos=0)
        return ResultadoRemocao(slug, None)
    destino = cfg.caminho("lixeira") / f"classe_{slug}_{carimbo()}"
    res = ResultadoRemocao(slug, destino)
    destino.mkdir(parents=True, exist_ok=False)
    _mover_registros(cfg, linhas, destino, res)
    dados = cfg.caminho("dados")
    for base in (cfg.caminho("videos") / slug, cfg.caminho("landmarks") / slug):
        if base.exists():
            for p in sorted(base.rglob("*")):
                if p.is_file() and _mover(p, dados, destino / "sem_registro"):
                    res.outros_arquivos += 1
            shutil.rmtree(base, ignore_errors=True)
    voc.remover_entrada(slug)
    res.splits_invalidados = invalidar_splits(cfg, f"Classe {slug} removida em {agora_iso()}.")
    if res.splits_invalidados:
        res.avisos.append("Os splits salvos foram invalidados. Gere de novo antes de treinar.")
    auditar(
        cfg,
        "remover_classe",
        slug,
        videos=res.videos,
        landmarks=res.landmarks,
        outros=res.outros_arquivos,
        lixeira=destino.name,
    )
    return res
