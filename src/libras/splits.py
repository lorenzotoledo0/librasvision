"""Divisao dos dados por sinalizador, salva em data/splits/ para reprodutibilidade.

Regras garantidas (e conferidas por verificar_vazamento):
- a unidade de divisao e o VIDEO: todas as janelas de um video ficam do mesmo lado;
- nos protocolos principais, a unidade e o SINALIZADOR: ninguem aparece em treino
  e teste (nem em validacao e teste) da mesma particao;
- todo video elegivel aparece no teste de exatamente uma particao.

Protocolos:
- grupos_sinalizador (principal): StratifiedGroupKFold, k particoes, grupos = sinalizador;
- loso: leave-one-signer-out, uma particao por sinalizador;
- aleatorio_por_video (SECUNDARIO): StratifiedKFold por video; o mesmo sinalizador
  aparece em treino e teste. Serve so para comparar com o protocolo do TCC I.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .config import Config
from .erros import ErroDadosInsuficientes, ErroLibras
from .utils import agora_iso

PROTOCOLOS = ("grupos_sinalizador", "loso", "aleatorio_por_video")


# ----------------------------------------------------------------------
# Videos elegiveis
# ----------------------------------------------------------------------
def videos_elegiveis(cfg: Config) -> tuple[list[dict], dict[str, str]]:
    """Videos 'ok', das classes SELECIONADAS para o treino, com landmarks atualizados.

    Retorna (linhas_elegiveis, {video_id: motivo_de_exclusao}).
    """
    from .coleta.manifesto import Manifesto
    from .extracao import cache_valido, caminho_npz, meta_esperada
    from .selecao import classes_treino

    sel = set(classes_treino(cfg))
    excluidos: dict[str, str] = {}
    ok = []
    for l in Manifesto(cfg).linhas:
        if l["status"] != "ok":
            continue
        if l["classe"] not in sel:
            excluidos[l["video_id"]] = f"classe {l['classe']} fora da selecao de treino"
            continue
        if not cache_valido(caminho_npz(cfg, l), meta_esperada(cfg, l)):
            excluidos[l["video_id"]] = "landmarks ausentes ou desatualizados (rode scripts/extrair.py)"
            continue
        ok.append(l)
    return ok, excluidos


def assinatura(linhas: list[dict], classes: list[str] | None = None) -> str:
    """Identifica o conjunto de dados E de classes usado para gerar os splits."""
    chave = sorted((l["video_id"], l["signer_id"], l["classe"], l["sha256"]) for l in linhas)
    base = {"videos": chave, "classes": list(classes) if classes is not None else None}
    return hashlib.sha256(json.dumps(base).encode()).hexdigest()[:16]


# ----------------------------------------------------------------------
# Geracao
# ----------------------------------------------------------------------
def _exigir_sinalizadores(linhas: list[dict], classes: list[str]) -> list[str]:
    avisos = []
    sinalizadores = sorted({l["signer_id"] for l in linhas})
    if len(sinalizadores) < 2:
        raise ErroDadosInsuficientes(
            f"A avaliacao independente de sinalizador precisa de pelo menos 2 sinalizadores com videos "
            f"'ok' e landmarks extraidos; ha {len(sinalizadores)} ({', '.join(sinalizadores) or 'nenhum'}). "
            "Colete videos de mais pessoas."
        )
    if len(sinalizadores) < 3:
        avisos.append(
            f"Apenas {len(sinalizadores)} sinalizadores: a avaliacao e pouco informativa (o ideal e 5 ou mais)."
        )
    for c in classes:
        quem = {l["signer_id"] for l in linhas if l["classe"] == c}
        if 0 < len(quem) < 2:
            avisos.append(
                f"A classe {c} tem videos de apenas {len(quem)} sinalizador(es); "
                "em algumas particoes ela nao tera exemplos no treino ou no teste."
            )
    return avisos


def _validacao(
    treino: list[dict], cfg_av: dict, rng: np.random.Generator, por_sinalizador: bool
) -> tuple[list[dict], list[dict], str]:
    """Separa validacao (early stopping) de dentro do treino."""
    sig = sorted({l["signer_id"] for l in treino})
    if por_sinalizador and len(sig) >= 3:
        n_val = max(1, int(round(cfg_av["fracao_validacao_sinalizadores"] * len(sig))))
        n_val = min(n_val, len(sig) - 2)  # deixa ao menos 2 pessoas no treino
        val_sig = set(rng.choice(sig, size=n_val, replace=False).tolist())
        return (
            [l for l in treino if l["signer_id"] not in val_sig],
            [l for l in treino if l["signer_id"] in val_sig],
            "sinalizadores",
        )
    # sem pessoas suficientes: validacao por video, estratificada por classe
    frac = float(cfg_av["fracao_validacao_videos"])
    por_classe: dict[str, list[dict]] = defaultdict(list)
    for l in treino:
        por_classe[l["classe"]].append(l)
    val_ids: set[str] = set()
    for c, ls in por_classe.items():
        if len(ls) < 2:
            continue
        idx = rng.permutation(len(ls))[: max(1, int(round(frac * len(ls))))]
        val_ids.update(ls[i]["video_id"] for i in idx)
    return (
        [l for l in treino if l["video_id"] not in val_ids],
        [l for l in treino if l["video_id"] in val_ids],
        "videos",
    )


def gerar_splits(
    cfg: Config, linhas: list[dict], protocolo: str | None = None, n_folds: int | None = None, seed: int | None = None
) -> dict:
    from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

    cfg_av = cfg["avaliacao"]
    protocolo = protocolo or cfg_av["protocolo"]
    if protocolo not in PROTOCOLOS:
        raise ErroLibras(f"Protocolo desconhecido: {protocolo}. Opcoes: {PROTOCOLOS}")
    seed = int(cfg["projeto"]["seed"] if seed is None else seed)
    n_folds = int(n_folds or cfg_av["n_folds"])
    from .selecao import classes_treino, exigir_selecao_valida

    classes = classes_treino(cfg)
    if protocolo != "aleatorio_por_video":
        _exigir_sinalizadores(linhas, [])  # primeiro o requisito global (mensagem mais util)
    if not classes:
        exigir_selecao_valida(cfg, linhas)  # explica o que falta (minimo de classes com videos)
    linhas = sorted([l for l in linhas if l["classe"] in classes], key=lambda l: l["video_id"])
    if not linhas:
        raise ErroDadosInsuficientes("Nenhum video elegivel (status 'ok' com landmarks extraidos).")
    avisos = _exigir_sinalizadores(linhas, classes) if protocolo != "aleatorio_por_video" else []
    sel = exigir_selecao_valida(cfg, linhas)  # minimos por classe; ensaio so se pedido
    avisos = list(sel.avisos) + avisos
    if protocolo == "aleatorio_por_video":
        faltando = [c for c in classes if sum(l["classe"] == c for l in linhas) < 2]
        if faltando:
            raise ErroDadosInsuficientes(f"Classes com menos de 2 videos: {', '.join(faltando)}.")
    y = np.array([classes.index(l["classe"]) for l in linhas])
    grupos = np.array([l["signer_id"] for l in linhas])
    sinalizadores = sorted(set(grupos))

    if protocolo == "loso":
        particoes = [(np.where(grupos != s)[0], np.where(grupos == s)[0]) for s in sinalizadores]
    elif protocolo == "grupos_sinalizador":
        k = min(n_folds, len(sinalizadores))
        if k < n_folds:
            avisos.append(
                f"{n_folds} particoes pedidas, mas so ha {len(sinalizadores)} sinalizadores: "
                f"usando {k} (equivale a leave-one-signer-out)."
            )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # aviso do sklearn sobre classes pequenas: tratado nos nossos avisos
            particoes = list(
                StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed).split(np.zeros(len(y)), y, grupos)
            )
    else:
        menor = min(Counter(y).values())
        k = min(n_folds, menor)
        if k < n_folds:
            avisos.append(f"Usando {k} particoes (a menor classe tem {menor} videos).")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            particoes = list(StratifiedKFold(n_splits=k, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))

    folds = []
    for i, (tr, te) in enumerate(particoes):
        rng = np.random.default_rng(seed + 1000 + i)
        treino_l = [linhas[j] for j in tr]
        teste_l = [linhas[j] for j in te]
        treino_l, val_l, modo_val = _validacao(treino_l, cfg_av, rng, protocolo != "aleatorio_por_video")
        sem_classe_teste = [c for c in classes if not any(l["classe"] == c for l in teste_l)]
        sem_classe_treino = [c for c in classes if not any(l["classe"] == c for l in treino_l)]
        if sem_classe_treino:
            avisos.append(f"Particao {i}: classes sem exemplo no treino: {', '.join(sem_classe_treino)}.")
        folds.append(
            {
                "indice": i,
                "treino": [l["video_id"] for l in treino_l],
                "validacao": [l["video_id"] for l in val_l],
                "teste": [l["video_id"] for l in teste_l],
                "sinalizadores_treino": sorted({l["signer_id"] for l in treino_l}),
                "sinalizadores_validacao": sorted({l["signer_id"] for l in val_l}),
                "sinalizadores_teste": sorted({l["signer_id"] for l in teste_l}),
                "validacao_por": modo_val,
                "classes_ausentes_no_teste": sem_classe_teste,
            }
        )
    split = {
        "protocolo": protocolo,
        "secundario": protocolo == "aleatorio_por_video",
        "seed": seed,
        "criado_em": agora_iso(),
        "assinatura_dados": assinatura(linhas, classes),
        "classes": classes,
        "ensaio": bool(sel.ensaio and sel.bloqueios),
        "n_videos": len(linhas),
        "n_sinalizadores": len(sinalizadores),
        "sinalizadores": sinalizadores,
        "avisos": avisos,
        "folds": folds,
    }
    problemas = verificar_vazamento(split, {l["video_id"]: l for l in linhas})
    if problemas:  # nunca deveria acontecer; se acontecer, e defeito de codigo
        raise ErroLibras("Defeito na geracao dos splits (vazamento):\n" + "\n".join(problemas))
    return split


# ----------------------------------------------------------------------
# Verificacao, persistencia
# ----------------------------------------------------------------------
def verificar_vazamento(split: dict, por_id: dict[str, dict]) -> list[str]:
    """Lista de problemas (vazia = sem vazamento)."""
    prob = []
    principal = not split.get("secundario", False)
    contagem_teste: Counter = Counter()
    for f in split["folds"]:
        i = f["indice"]
        tr, va, te = set(f["treino"]), set(f["validacao"]), set(f["teste"])
        for a, b, nome in ((tr, te, "treino/teste"), (va, te, "validacao/teste"), (tr, va, "treino/validacao")):
            comum = a & b
            if comum:
                prob.append(f"Particao {i}: {len(comum)} video(s) em {nome}: {sorted(comum)[:5]}")
        for vid in tr | va | te:
            if vid not in por_id:
                prob.append(f"Particao {i}: video {vid} nao esta entre os elegiveis atuais")
        contagem_teste.update(te)
        if principal:
            s_tr = {por_id[v]["signer_id"] for v in tr if v in por_id}
            s_va = {por_id[v]["signer_id"] for v in va if v in por_id}
            s_te = {por_id[v]["signer_id"] for v in te if v in por_id}
            if s_tr & s_te:
                prob.append(f"Particao {i}: sinalizador(es) em treino e teste: {sorted(s_tr & s_te)}")
            if s_va & s_te:
                prob.append(f"Particao {i}: sinalizador(es) em validacao e teste: {sorted(s_va & s_te)}")
    for vid in por_id:
        if contagem_teste[vid] != 1:
            prob.append(f"Video {vid} aparece no teste {contagem_teste[vid]} vez(es) (deveria ser 1)")
    return prob


def caminho_split(cfg: Config, protocolo: str) -> Path:
    return cfg.caminho("splits") / f"{protocolo}.json"


def salvar_split(cfg: Config, split: dict) -> Path:
    from .utils import escrever_texto_atomico

    p = caminho_split(cfg, split["protocolo"])
    escrever_texto_atomico(p, json.dumps(split, indent=2, ensure_ascii=False))
    return p


def carregar_split(cfg: Config, protocolo: str | None = None, conferir: bool = True) -> dict:
    protocolo = protocolo or cfg["avaliacao"]["protocolo"]
    p = caminho_split(cfg, protocolo)
    if not p.exists():
        raise ErroDadosInsuficientes(
            f"Nao ha splits salvos para o protocolo {protocolo}. Rode: scripts/gerar_splits.py --protocolo {protocolo}"
        )
    split = json.loads(p.read_text(encoding="utf-8"))
    if conferir:
        from .selecao import classes_treino

        linhas, _ = videos_elegiveis(cfg)
        classes = classes_treino(cfg)
        if split.get("classes") != classes:
            raise ErroLibras(
                f"Os splits em {p.name} foram gerados com as classes {split.get('classes')}, mas a selecao atual e "
                f"{classes}. Gere de novo: scripts/gerar_splits.py (com a mesma selecao de classes)."
            )
        if assinatura(linhas, classes) != split["assinatura_dados"]:
            raise ErroLibras(
                f"Os splits em {p.name} foram gerados com outro conjunto de videos (houve coleta, revisao ou "
                "exclusao depois). Gere de novo: scripts/gerar_splits.py. Resultados de execucoes com splits "
                "diferentes nao sao comparaveis."
            )
        prob = verificar_vazamento(split, {l["video_id"]: l for l in linhas})
        if prob:
            raise ErroLibras("Vazamento detectado nos splits salvos:\n" + "\n".join(prob))
    return split


def resumo_split(split: dict, por_id: dict[str, dict]) -> str:
    linhas = [
        f"Protocolo: {split['protocolo']}"
        + (" (SECUNDARIO: mesmo sinalizador em treino e teste)" if split.get("secundario") else ""),
        f"Videos: {split['n_videos']} | sinalizadores: {split['n_sinalizadores']} | "
        f"particoes: {len(split['folds'])} | seed: {split['seed']}",
    ]
    for f in split["folds"]:
        cont = Counter(por_id[v]["classe"] for v in f["teste"] if v in por_id)
        linhas.append(
            f"  Particao {f['indice']}: treino {len(f['treino'])} | validacao {len(f['validacao'])} "
            f"(por {f['validacao_por']}) | teste {len(f['teste'])} "
            f"[{', '.join(f['sinalizadores_teste']) if not split.get('secundario') else 'videos aleatorios'}] "
            f"| teste por classe: {dict(sorted(cont.items()))}"
        )
    for a in split.get("avisos", []):
        linhas.append(f"  AVISO: {a}")
    return "\n".join(linhas)
