"""Estudo de ablacao da data augmentation (Fase 5).

Etapas, todas com o MESMO split e as MESMAS seeds:
  1. baseline sem augmentation;
  2. cada familia isolada (espacial, temporal, ruido e, se extraida, video);
  3. combinacao de todas as familias disponiveis;
  4. intensidades fraca/media/forte da melhor condicao das etapas 2-3, escolhida pela
     metrica de VALIDACAO (nunca pelo teste, para nao contaminar a comparacao);
  5. opcional: baseline no protocolo secundario (aleatorio por video), para mostrar o
     quanto o resultado fica otimista quando o sinalizador do teste ja foi visto.

Cada condicao e uma execucao completa em <pasta>/<condicao>/ (mesmo formato de
scripts/treinar.py). A ablacao pode ser retomada: condicoes com resumo.json sao reaproveitadas.
Comparacoes contra o baseline: teste t corrigido para CV repetida, Wilcoxon por particao e Holm.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import Config
from .erros import ErroDadosInsuficientes
from .estatistica import t_corrigido, wilcoxon_pareado
from .utils import carimbo, escrever_texto_atomico

ONLINE = ("espacial", "temporal", "ruido")
NOMES_METRICA = {
    "f1_macro_video": "F1 macro (vídeo)",
    "f1_macro_janela": "F1 macro (janela)",
    "acuracia_video": "Acurácia (vídeo)",
    "acuracia_janela": "Acurácia (janela)",
}
ROTULO_FAMILIA = {"espacial": "espacial", "temporal": "temporal", "ruido": "ruído", "video": "vídeo"}


@dataclass
class Condicao:
    nome: str
    familias: list[str]
    intensidade: str = "media"
    protocolo: str | None = None
    rotulo: str = ""
    etapa: int = 1


@dataclass
class ResultadoCondicao:
    condicao: Condicao
    pasta: Path
    resumo: dict
    por_particao: dict = field(default_factory=dict)  # (seed, fold) -> metricas.json


def valor(m: dict, chave: str) -> float:
    met, nivel = chave.rsplit("_", 1)
    return float(m[nivel][met])


def carregar_particoes(pasta: Path) -> dict:
    out = {}
    for p in sorted(pasta.glob("seed_*/fold_*/metricas.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        out[(int(r["seed"]), int(r["fold"]))] = r
    return out


def condicoes_iniciais(familias_disp: list[str]) -> list[Condicao]:
    conds = [Condicao("baseline", [], rotulo="Sem augmentation (baseline)", etapa=1)]
    for f in familias_disp:
        conds.append(Condicao(f"so_{f}", [f], rotulo=f"Apenas {ROTULO_FAMILIA.get(f, f)}", etapa=2))
    if len(familias_disp) > 1:
        conds.append(Condicao("combinacao", list(familias_disp), rotulo="Todas as famílias", etapa=3))
    return conds


def _executar_condicao(
    cfg: Config,
    c: Condicao,
    pasta_abl: Path,
    dados,
    seeds,
    folds,
    verbose,
    sobrescrever: dict,
    log,
    arquivo_progresso=None,
) -> ResultadoCondicao:
    from .treino import OpcoesTreino, executar

    destino = pasta_abl / c.nome
    if (destino / "resumo.json").exists():
        log(f"[{c.nome}] ja concluida, reaproveitando.")
        resumo = json.loads((destino / "resumo.json").read_text(encoding="utf-8"))
    else:
        log(f"[{c.nome}] {c.rotulo} - familias {c.familias or 'nenhuma'}, intensidade {c.intensidade}")
        op = OpcoesTreino(
            nome=c.nome,
            protocolo=c.protocolo,
            familias=c.familias,
            intensidade=c.intensidade,
            seeds=seeds,
            folds=folds,
            verbose=verbose,
            salvar_modelos=False,
            pasta_runs=pasta_abl,
            pasta_fixa=c.nome,
            sobrescrever=sobrescrever,
            arquivo_progresso=arquivo_progresso,
        )
        _, resumo = executar(cfg, op, dados=dados)
    return ResultadoCondicao(c, destino, resumo, carregar_particoes(destino))


def comparar(base: ResultadoCondicao, outra: ResultadoCondicao, chave: str, k: int, r: int) -> dict:
    pares = sorted(set(base.por_particao) & set(outra.por_particao))
    d = [valor(outra.por_particao[p], chave) - valor(base.por_particao[p], chave) for p in pares]
    n_tr = np.mean([base.por_particao[p]["janelas_treino"] for p in pares]) if pares else 0
    n_te = np.mean([base.por_particao[p]["janelas_teste"] for p in pares]) if pares else 0
    t = t_corrigido(d, n_tr, n_te, k, r)
    folds = sorted({f for _, f in pares})
    a = [float(np.mean([valor(base.por_particao[(s, f2)], chave) for (s, f2) in pares if f2 == f])) for f in folds]
    b = [float(np.mean([valor(outra.por_particao[(s, f2)], chave) for (s, f2) in pares if f2 == f])) for f in folds]
    w = wilcoxon_pareado(a, b)
    return {
        "pares": len(pares),
        "diferenca_media": t["media"],
        "ic95": t["ic95"],
        "t": t.get("t"),
        "gl": t.get("gl"),
        "p_t_corrigido": t["p"],
        "p_wilcoxon": w["p"],
        "p_wilcoxon_minimo_possivel": w["p_minimo_possivel"],
        "obs": t.get("obs", ""),
    }


def executar_ablacao(
    cfg: Config,
    pasta: Path | None = None,
    seeds: list[int] | None = None,
    folds: list[int] | None = None,
    sobrescrever: dict | None = None,
    verbose: int = 0,
    log=print,
    arquivo_progresso=None,
) -> Path:
    from .splits import carregar_split, gerar_splits, salvar_split, videos_elegiveis
    from .treino import OpcoesTreino, config_efetiva, preparar_dados, variantes_disponiveis

    sobrescrever = sobrescrever or {}
    cfg = config_efetiva(cfg, OpcoesTreino(sobrescrever=sobrescrever))
    ca = cfg["ablacao"]
    seeds = list(seeds or ca["seeds"])
    chave = ca["metrica_principal"]
    if chave not in NOMES_METRICA:
        from .erros import ErroConfig

        raise ErroConfig(f"ablacao.metrica_principal deve ser uma de {list(NOMES_METRICA)}")

    # dados e familias disponiveis -------------------------------------------
    split = carregar_split(cfg)  # erro claro se faltar ou estiver desatualizado
    if split.get("secundario"):
        raise ErroDadosInsuficientes(
            "A ablacao principal precisa de um protocolo por sinalizador "
            "(avaliacao.protocolo: grupos_sinalizador ou loso)."
        )
    linhas, _ = videos_elegiveis(cfg)
    familias = [f for f in ca["familias"] if f in ONLINE]
    avisos = []
    if "video" in ca["familias"]:
        vv = list(cfg["augmentation"]["video"]["variantes"])
        if vv and variantes_disponiveis(cfg, linhas, vv) == vv:
            familias.append("video")
        else:
            avisos.append(
                "Familia 'video' fora da ablacao: variantes nao extraidas para todos os videos "
                f"({', '.join(vv)}). Rode scripts/aumentar_videos.py para inclui-la."
            )
    k = len([f for f in split["folds"] if folds is None or f["indice"] in folds])
    if k * len(seeds) < 2:
        raise ErroDadosInsuficientes("A ablacao precisa de pelo menos 2 pares (particao, seed) para comparar.")
    if k < 3:
        avisos.append(f"Apenas {k} particoes: os testes estatisticos terao pouco poder.")
    if split.get("ensaio"):
        from .selecao import TEXTO_ENSAIO

        avisos.insert(0, TEXTO_ENSAIO)
    if seeds != list(ca["seeds"]) or folds is not None or sobrescrever:
        avisos.insert(
            0,
            "ENSAIO REDUZIDO (seeds, particoes ou parametros diferentes do config). "
            "Estes numeros nao devem ser usados no TCC.",
        )
    log(f"Preparando janelas ({len(linhas)} videos)...")
    dados = preparar_dados(cfg, None, familias)

    pasta = Path(pasta) if pasta else cfg.caminho("runs") / f"{carimbo()}_ablacao"
    pasta.mkdir(parents=True, exist_ok=True)
    plano = {
        "seeds": seeds,
        "folds": folds,
        "metrica_principal": chave,
        "familias": familias,
        "protocolo": split["protocolo"],
        "assinatura_dados": split["assinatura_dados"],
        "sobrescrever": sobrescrever,
    }
    plano_arq = pasta / "plano.json"
    if plano_arq.exists():
        antigo = json.loads(plano_arq.read_text(encoding="utf-8"))
        if {k2: antigo.get(k2) for k2 in plano} != plano:
            raise ErroDadosInsuficientes(
                f"A pasta {pasta} contem uma ablacao com outro plano (seeds, dados ou parametros). "
                "Use outra pasta para nao misturar resultados incomparaveis."
            )
    escrever_texto_atomico(plano_arq, json.dumps(plano, indent=2, ensure_ascii=False))

    # plano para a interface mostrar o progresso da ablacao inteira e estimar o tempo restante. As
    # intensidades dependem da melhor condicao (escolhida pela validacao); o numero e o previsto.
    from .treino import EmissorProgresso

    iniciais = condicoes_iniciais(familias)
    previstas = len(iniciais)
    if any(f != "video" for f in familias):
        previstas += len([i for i in ca["intensidades"] if i != cfg["augmentation"].get("intensidade", "media")])
    if ca.get("comparar_protocolo_secundario"):
        previstas += 1
    EmissorProgresso(arquivo_progresso).emitir(
        "plano_ablacao", condicoes=previstas, particoes_por_condicao=k * len(seeds)
    )

    res: dict[str, ResultadoCondicao] = {}
    for c in iniciais:
        res[c.nome] = _executar_condicao(
            cfg, c, pasta, dados, seeds, folds, verbose, sobrescrever, log, arquivo_progresso
        )

    # etapa 4: intensidades da melhor condicao, escolhida pela VALIDACAO
    candidatos = [n for n in res if n != "baseline"]
    val_key = f"val_{chave}_media"
    melhor = None
    vals = {n: res[n].resumo.get(val_key) for n in candidatos}
    vals = {n: v for n, v in vals.items() if v is not None and np.isfinite(v)}
    if vals:
        melhor = max(vals, key=vals.get)
        cond_melhor = res[melhor].condicao
        online = [f for f in cond_melhor.familias if f != "video"]
        if not online:
            avisos.append(
                f"A melhor condicao na validacao ({melhor}) so usa a familia 'video', cujas "
                "variantes tem parametros fixos: etapa de intensidades nao se aplica."
            )
        else:
            for inten in ca["intensidades"]:
                if inten == cond_melhor.intensidade:
                    continue
                c = Condicao(
                    f"{melhor}_{inten}", cond_melhor.familias, inten, rotulo=f"{cond_melhor.rotulo} ({inten})", etapa=4
                )
                res[c.nome] = _executar_condicao(
                    cfg, c, pasta, dados, seeds, folds, verbose, sobrescrever, log, arquivo_progresso
                )
    else:
        avisos.append("Sem metricas de validacao: etapa de intensidades nao executada.")

    # etapa 5: protocolo secundario
    if ca.get("comparar_protocolo_secundario"):
        try:
            carregar_split(cfg, "aleatorio_por_video")
        except Exception:
            salvar_split(cfg, gerar_splits(cfg, linhas, "aleatorio_por_video"))
        sec_dados = preparar_dados(cfg, "aleatorio_por_video", [])
        c = Condicao(
            "baseline_aleatorio_por_video",
            [],
            protocolo="aleatorio_por_video",
            rotulo="Baseline, divisão aleatória por vídeo (secundário)",
            etapa=5,
        )
        res[c.nome] = _executar_condicao(
            cfg, c, pasta, sec_dados, seeds, None, verbose, sobrescrever, log, arquivo_progresso
        )

    from .relatorio_ablacao import gerar_saidas

    gerar_saidas(cfg, pasta, res, chave, k, len(seeds), melhor, avisos)
    return pasta
