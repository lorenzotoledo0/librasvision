"""Treino e avaliacao por particao, com registro completo em runs/<data>_<nome>/.

Regras garantidas aqui (alem das ja conferidas pelos splits):
- augmentation so nas janelas de treino; validacao e teste nunca sao aumentados;
- variantes de video (familia D) so entram pelos videos de treino;
- antes de treinar cada particao, confere que nenhuma janela de teste (original ou
  aumentada) e nenhum sinalizador de teste aparecem no treino;
- metricas calculadas apenas sobre o teste da particao.

Sem dados reais, as funcoes param com ErroDadosInsuficientes (nunca treinam com dados falsos).
"""

from __future__ import annotations

import copy
import csv
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

from .config import Config
from .erros import ErroDadosInsuficientes, ErroLibras
from .metricas import agregar_por_video, media_dp, metricas
from .utils import carimbo, escrever_csv_atomico, escrever_texto_atomico, fixar_seeds, hardware, versoes


class EmissorProgresso:
    """Escreve eventos de progresso (uma linha JSON por evento) para a interface acompanhar ao vivo.

    Eventos: inicio, inicio_particao, epoca (SO treino e validacao), fim_particao (aqui, e so aqui,
    entram as metricas de TESTE), fim. O teste nunca aparece nos eventos por epoca: ele nao pode
    orientar nada durante o treino.
    """

    def __init__(self, arquivo: Path | str | None):
        self.arquivo = Path(arquivo) if arquivo else None
        if self.arquivo:
            self.arquivo.parent.mkdir(parents=True, exist_ok=True)
        self.contexto: dict = {}

    def emitir(self, tipo: str, **dados) -> None:
        if not self.arquivo:
            return
        reg = {"tipo": tipo, "t": round(time.time(), 3), **self.contexto, **dados}
        with open(self.arquivo, "a", encoding="utf-8") as f:
            f.write(json.dumps(reg, ensure_ascii=False, default=float) + "\n")
            f.flush()


def _callback_epocas(emissor: EmissorProgresso, seed: int, fold: int):
    import tensorflow as tf

    class _Cb(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            permitidas = ("loss", "accuracy", "val_loss", "val_accuracy")
            emissor.emitir(
                "epoca",
                seed=seed,
                fold=fold,
                epoca=epoch + 1,
                **{k: float(v) for k, v in logs.items() if k in permitidas},
            )

    return _Cb()


@dataclass
class OpcoesTreino:
    nome: str = "treino"
    protocolo: str | None = None
    familias: list[str] | None = None  # None = segue augmentation.ativo do config
    intensidade: str | None = None
    seeds: list[int] | None = None
    folds: list[int] | None = None  # None = todas as particoes
    verbose: int = 2
    salvar_modelos: bool | None = None
    pasta_runs: Path | None = None
    pasta_fixa: str | None = None  # nome fixo da pasta (a ablacao usa para retomar)
    arquivo_progresso: Path | str | None = None  # eventos JSON por linha para a interface
    sobrescrever: dict = field(default_factory=dict)  # ajustes pontuais (ex.: {"treino": {"epocas": 5}})


def _mesclar(base: dict, extra: dict) -> dict:
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _mesclar(base[k], v)
        else:
            base[k] = v
    return base


def config_efetiva(cfg: Config, op: OpcoesTreino) -> Config:
    d = _mesclar(copy.deepcopy(dict(cfg)), copy.deepcopy(op.sobrescrever))
    return Config(d, raiz=cfg.raiz, arquivo=cfg.arquivo)


def familias_efetivas(cfg: Config, op: OpcoesTreino) -> list[str]:
    if op.familias is not None:
        return list(op.familias)
    ca = cfg["augmentation"]
    if not ca.get("ativo", False):
        return []
    fams = [f for f in ("espacial", "temporal", "ruido", "video") if ca[f].get("ativo", False)]
    return fams


def _preparar_tf(seed: int) -> None:
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:  # versoes antigas ou ja habilitado
        pass


def _latencia_modelo_ms(modelo, L: int, D: int, repeticoes: int = 30) -> float:
    """Mediana do tempo de UMA predicao (so o modelo, sem MediaPipe) nesta maquina.

    Usa predict_on_batch (grafo compilado), a mesma chamada do tempo real. A chamada
    direta modelo(x) roda em modo eager e e cerca de 30x mais lenta na CPU.
    """
    x = np.zeros((1, L, D), np.float32)
    for _ in range(3):
        modelo.predict_on_batch(x)
    t = []
    for _ in range(repeticoes):
        t0 = time.perf_counter()
        modelo.predict_on_batch(x)
        t.append((time.perf_counter() - t0) * 1000)
    return float(np.median(t))


def _figura_curvas(hist: dict, destino: Path, titulo: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for i, (m, rot) in enumerate((("loss", "Perda"), ("accuracy", "Acuracia"))):
        ax[i].plot(hist.get(m, []), label="treino")
        if f"val_{m}" in hist:
            ax[i].plot(hist[f"val_{m}"], label="validacao")
        ax[i].set_xlabel("Epoca")
        ax[i].set_title(rot)
        ax[i].legend()
        ax[i].grid(alpha=0.3)
    fig.suptitle(titulo)
    fig.tight_layout()
    fig.savefig(destino, dpi=150)
    plt.close(fig)


def figura_matriz(cm: np.ndarray, classes: list[str], destino: Path, titulo: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(1.2 * len(classes) + 2, 1.1 * len(classes) + 1.5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes, rotation=30, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predito")
    ax.set_ylabel("Verdadeiro")
    lim = cm.max() / 2 if cm.size and cm.max() else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > lim else "black")
    ax.set_title(titulo)
    fig.tight_layout()
    fig.savefig(destino, dpi=150)
    plt.close(fig)


def _salvar_predicoes(destino: Path, ids, probs, y, classes, extra_cols: dict | None = None) -> None:
    linhas = []
    for i, ident in enumerate(ids):
        l = {
            "id": ident,
            "verdadeiro": classes[int(y[i])],
            "predito": classes[int(np.argmax(probs[i]))],
            "confianca": f"{float(np.max(probs[i])):.4f}",
        }
        for k, c in enumerate(classes):
            l[f"p_{c}"] = f"{float(probs[i, k]):.4f}"
        for k, v in (extra_cols or {}).items():
            l[k] = v[i]
        linhas.append(l)
    if linhas:
        escrever_csv_atomico(destino, list(linhas[0].keys()), linhas)


# ======================================================================
# Uma particao
# ======================================================================
def treinar_particao(
    cfg: Config,
    conj,
    fold: dict,
    seed: int,
    familias: list[str],
    intensidade: str | None,
    principal: bool,
    pasta: Path | None,
    verbose: int = 0,
    salvar_modelo: bool = True,
    robustez: bool = True,
    emissor: EmissorProgresso | None = None,
) -> dict:
    import tensorflow as tf

    from .augmentation import Aumentador, gerar_copias_offline
    from .dataset import criar_tf_dataset, pesos_classes
    from .modelo import compilar, construir_modelo

    variantes_video = list(cfg["augmentation"]["video"]["variantes"]) if "video" in familias else []
    online = [f for f in familias if f != "video"]
    idx_tr = conj.indices(fold["treino"], variantes=("",) + tuple(variantes_video))
    idx_va = conj.indices(fold["validacao"])
    idx_te = conj.indices(fold["teste"])
    if len(idx_tr) == 0 or len(idx_te) == 0:
        return {"ignorado": True, "motivo": "particao sem janelas de treino ou de teste"}

    # salvaguardas contra vazamento (alem das checagens do split)
    if set(conj.video[idx_tr]) & set(conj.video[idx_te]):
        raise ErroLibras("Vazamento: o mesmo video aparece no treino e no teste.")
    if principal and set(conj.signer[idx_tr]) & set(conj.signer[idx_te]):
        raise ErroLibras("Vazamento: um sinalizador de teste aparece no treino.")
    if principal and len(idx_va) and set(conj.signer[idx_va]) & set(conj.signer[idx_te]):
        raise ErroLibras("Vazamento: um sinalizador de teste aparece na validacao.")

    _preparar_tf(seed)
    fixar_seeds(seed)
    ct = cfg["treino"]
    n_cls = len(conj.classes)
    L, D = conj.X.shape[1], conj.X.shape[2]
    bs = int(cfg["dataset"]["batch_size"])

    aum = Aumentador(cfg, conj.espec, familias=online, intensidade=intensidade) if online else None
    fonte_treino = conj
    fator = 1.0
    if aum and cfg["augmentation"].get("modo", "online") == "offline":
        copias = int(cfg["augmentation"].get("copias_offline", 2))
        Xa = gerar_copias_offline(conj.X[idx_tr], aum, copias, seed)
        fonte_treino = SimpleNamespace(
            X=np.concatenate([conj.X[idx_tr], Xa]), y=np.concatenate([conj.y[idx_tr]] * (copias + 1))
        )
        idx_tr_fonte = np.arange(len(fonte_treino.y))
        transformar = None
        fator = 1.0 + copias
    else:
        idx_tr_fonte = idx_tr
        transformar = aum
    y_tr = fonte_treino.y[idx_tr_fonte]
    pesos = pesos_classes(y_tr, n_cls) if ct.get("pesos_classes", True) else None

    ds_tr = criar_tf_dataset(fonte_treino, idx_tr_fonte, bs, True, seed, transformar)
    ds_va = criar_tf_dataset(conj, idx_va, bs, False, seed) if len(idx_va) else None
    modelo = construir_modelo(cfg, L, D, n_cls)
    compilar(modelo, float(ct["taxa_aprendizado"]))
    monitor = "val_loss" if ds_va is not None else "loss"
    parada = tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=int(ct["paciencia"]), restore_best_weights=True)
    callbacks = [parada]
    if emissor is not None and emissor.arquivo:
        emissor.emitir(
            "inicio_particao",
            seed=int(seed),
            fold=int(fold["indice"]),
            sinalizadores_teste=fold["sinalizadores_teste"],
            janelas_treino=int(len(idx_tr_fonte)),
            janelas_validacao=int(len(idx_va)),
            monitor=monitor,
        )
        callbacks.append(_callback_epocas(emissor, int(seed), int(fold["indice"])))
    t0 = time.perf_counter()
    h = modelo.fit(
        ds_tr,
        validation_data=ds_va,
        epochs=int(ct["epocas"]),
        callbacks=callbacks,
        class_weight=pesos,
        verbose=verbose,
        shuffle=False,
    )  # o embaralhamento e feito pelo gerador
    t_treino = time.perf_counter() - t0
    hist = {k: [float(x) for x in v] for k, v in h.history.items()}
    melhor = int(np.argmin(hist[monitor])) + 1 if hist.get(monitor) else len(hist.get("loss", []))

    # teste: so janelas originais dos videos de teste, sem augmentation
    Xte = conj.X[idx_te].astype(np.float32)
    probs = modelo.predict(Xte, batch_size=bs, verbose=0)
    y_te = conj.y[idx_te]
    m_jan = metricas(y_te, probs.argmax(1), conj.classes)
    ids_v, probs_v, y_v, pred_v = agregar_por_video(probs, conj.video[idx_te], y_te)
    m_vid = metricas(y_v, pred_v, conj.classes)
    lat = _latencia_modelo_ms(modelo, L, D)

    # validacao (usada pela ablacao para escolher a melhor condicao SEM olhar o teste)
    m_val = None
    if len(idx_va):
        p_va = modelo.predict(conj.X[idx_va].astype(np.float32), batch_size=bs, verbose=0)
        y_va = conj.y[idx_va]
        _, pv_v, yv_v, predv_v = agregar_por_video(p_va, conj.video[idx_va], y_va)
        m_val = {"janela": metricas(y_va, p_va.argmax(1), conj.classes), "video": metricas(yv_v, predv_v, conj.classes)}

    rob = avaliar_robustez(cfg, conj, modelo, idx_te, fold, seed, bs) if robustez else {}

    res = {
        "fold": int(fold["indice"]),
        "seed": int(seed),
        "sinalizadores_teste": fold["sinalizadores_teste"],
        "janelas_treino": int(len(idx_tr_fonte)),
        "janelas_validacao": int(len(idx_va)),
        "janelas_teste": int(len(idx_te)),
        "videos_teste": len(ids_v),
        "janelas_treino_variantes_video": int((conj.variante[idx_tr] != "").sum()),
        "epocas_executadas": len(hist.get("loss", [])),
        "melhor_epoca": melhor,
        "segundos_treino": round(t_treino, 1),
        "latencia_modelo_ms": round(lat, 2),
        "janela": m_jan,
        "video": m_vid,
        "validacao": m_val,
        "robustez": rob,
        "augmentation": {
            "familias": familias,
            "modo": cfg["augmentation"].get("modo", "online"),
            "fator_offline": fator,
            **(aum.resumo_contagem() if aum else {}),
        },
    }
    if emissor is not None:
        emissor.emitir(
            "fim_particao",
            seed=int(seed),
            fold=int(fold["indice"]),
            melhor_epoca=melhor,
            sinalizadores_teste=fold["sinalizadores_teste"],
            epocas=res["epocas_executadas"],
            teste_janela={k: m_jan[k] for k in ("acuracia", "f1_macro")},
            teste_video={k: m_vid[k] for k in ("acuracia", "f1_macro")},
            matriz_video=m_vid["matriz_confusao"],
            matriz_janela=m_jan["matriz_confusao"],
            classes=conj.classes,
        )
    if pasta is not None:
        pasta.mkdir(parents=True, exist_ok=True)
        with open(pasta / "historico.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            chaves = list(hist)
            w.writerow(["epoca"] + chaves)
            for i in range(len(hist.get("loss", []))):
                w.writerow([i + 1] + [hist[k][i] for k in chaves])
        _figura_curvas(hist, pasta / "curvas.png", f"Particao {fold['indice']} (seed {seed})")
        figura_matriz(
            np.array(m_jan["matriz_confusao"]),
            conj.classes,
            pasta / "matriz_confusao_janelas.png",
            f"Teste, janelas - particao {fold['indice']}",
        )
        figura_matriz(
            np.array(m_vid["matriz_confusao"]),
            conj.classes,
            pasta / "matriz_confusao_videos.png",
            f"Teste, videos - particao {fold['indice']}",
        )
        ini = [str(i) for i in idx_te]
        _salvar_predicoes(
            pasta / "predicoes_janelas.csv",
            [f"{v}#{i}" for v, i in zip(conj.video[idx_te], ini)],
            probs,
            y_te,
            conj.classes,
            {"signer_id": list(conj.signer[idx_te])},
        )
        _salvar_predicoes(pasta / "predicoes_videos.csv", ids_v, probs_v, y_v, conj.classes)
        escrever_texto_atomico(pasta / "metricas.json", json.dumps(res, indent=2, ensure_ascii=False))
        if salvar_modelo:
            modelo.save(pasta / "modelo.keras")
    tf.keras.backend.clear_session()
    return res


def avaliar_robustez(cfg: Config, conj, modelo, idx_te: np.ndarray, fold: dict, seed: int, bs: int) -> dict:
    """Metricas do mesmo modelo em copias degradadas do teste. Mesma degradacao para toda condicao."""
    from .robustez import degradar_lote

    cr = cfg.get("robustez") or {}
    if not cr.get("ativo", False):
        return {}
    out = {}
    fr = float(cfg["normalizacao"]["fracao_minima_pose"])
    y_te = conj.y[idx_te]
    for nome, params in (cr.get("landmarks") or {}).items():
        Xd = degradar_lote(
            conj.X[idx_te], conj.espec, nome, params or {}, seed=10_000 + int(fold["indice"]), fracao_pose=fr
        )
        p = modelo.predict(Xd, batch_size=bs, verbose=0)
        _, _, yv, pv = agregar_por_video(p, conj.video[idx_te], y_te)
        out[nome] = {"janela": metricas(y_te, p.argmax(1), conj.classes), "video": metricas(yv, pv, conj.classes)}
    for var in cr.get("video") or []:
        idx = conj.indices(fold["teste"], variantes=(var,))
        if len(idx) == 0:
            out[var] = {"indisponivel": "variante nao extraida (rode scripts/aumentar_videos.py)"}
            continue
        p = modelo.predict(conj.X[idx].astype(np.float32), batch_size=bs, verbose=0)
        _, _, yv, pv = agregar_por_video(p, conj.video[idx], conj.y[idx])
        out[var] = {"janela": metricas(conj.y[idx], p.argmax(1), conj.classes), "video": metricas(yv, pv, conj.classes)}
    return out


# ======================================================================
# Execucao completa (todas as particoes x seeds)
# ======================================================================
def _linha_resumo(r: dict) -> dict:
    return {
        "seed": r["seed"],
        "fold": r["fold"],
        "sinalizadores_teste": " ".join(r["sinalizadores_teste"]),
        "acc_janela": f"{r['janela']['acuracia']:.4f}",
        "f1_janela": f"{r['janela']['f1_macro']:.4f}",
        "acc_video": f"{r['video']['acuracia']:.4f}",
        "f1_video": f"{r['video']['f1_macro']:.4f}",
        "janelas_teste": r["janelas_teste"],
        "videos_teste": r["videos_teste"],
        "epocas": r["epocas_executadas"],
        "melhor_epoca": r["melhor_epoca"],
        "segundos_treino": r["segundos_treino"],
        "latencia_modelo_ms": r["latencia_modelo_ms"],
    }


def resumir(resultados: list[dict], classes: list[str]) -> dict:
    validos = [r for r in resultados if not r.get("ignorado")]
    out = {"n_resultados": len(validos)}
    for nivel in ("janela", "video"):
        for met in ("acuracia", "f1_macro"):
            m, dp = media_dp([r[nivel][met] for r in validos])
            out[f"{met}_{nivel}_media"], out[f"{met}_{nivel}_dp"] = m, dp
        # matriz somada sobre as particoes (cada amostra de teste conta uma vez por seed)
        cm = np.zeros((len(classes), len(classes)), int)
        for r in validos:
            cm += np.array(r[nivel]["matriz_confusao"])
        out[f"matriz_{nivel}_soma"] = cm.tolist()
    seeds = sorted({r["seed"] for r in validos})
    out["por_seed"] = {}
    for s in seeds:
        rs = [r for r in validos if r["seed"] == s]
        out["por_seed"][str(s)] = {
            "acuracia_janela": media_dp([r["janela"]["acuracia"] for r in rs])[0],
            "f1_macro_janela": media_dp([r["janela"]["f1_macro"] for r in rs])[0],
            "acuracia_video": media_dp([r["video"]["acuracia"] for r in rs])[0],
            "f1_macro_video": media_dp([r["video"]["f1_macro"] for r in rs])[0],
        }
    # validacao (para selecao de condicoes sem olhar o teste)
    com_val = [r for r in validos if r.get("validacao")]
    for nivel in ("janela", "video"):
        for met in ("acuracia", "f1_macro"):
            out[f"val_{met}_{nivel}_media"] = media_dp([r["validacao"][nivel][met] for r in com_val])[0]
    # robustez: media por degradacao
    rob: dict = {}
    for r in validos:
        for nome, m in (r.get("robustez") or {}).items():
            if "indisponivel" in m:
                rob.setdefault(nome, {"indisponivel": m["indisponivel"]})
                continue
            for nivel in ("janela", "video"):
                for met in ("acuracia", "f1_macro"):
                    rob.setdefault(nome, {}).setdefault(f"{met}_{nivel}", []).append(m[nivel][met])
    out["robustez"] = {
        n: (
            {"indisponivel": v["indisponivel"]}
            if "indisponivel" in v
            else {k: {"media": media_dp(x)[0], "dp": media_dp(x)[1]} for k, x in v.items()}
        )
        for n, v in rob.items()
    }
    out["latencia_modelo_ms_mediana"] = (
        float(np.median([r["latencia_modelo_ms"] for r in validos])) if validos else None
    )
    out["segundos_treino_total"] = float(sum(r["segundos_treino"] for r in validos))
    return out


def preparar_dados(cfg: Config, protocolo: str | None, familias: list[str]):
    """Carrega split (conferido) e monta as janelas. Separado para a ablacao reaproveitar."""
    from .dataset import montar_conjunto
    from .splits import carregar_split, videos_elegiveis

    split = carregar_split(cfg, protocolo)
    linhas, _ = videos_elegiveis(cfg)
    variantes = list(cfg["augmentation"]["video"]["variantes"]) if "video" in familias else []
    rob = cfg.get("robustez") or {}
    if rob.get("ativo"):
        variantes += [v for v in variantes_disponiveis(cfg, linhas, rob.get("video") or []) if v not in variantes]
    conj = montar_conjunto(cfg, linhas, variantes=variantes)
    return split, linhas, conj


def variantes_disponiveis(cfg: Config, linhas: list[dict], variantes: list[str]) -> list[str]:
    """Variantes de video ja extraidas para TODOS os videos elegiveis."""
    from .augmentation.video import caminho_variante, meta_variante
    from .extracao import cache_valido

    return [
        v for v in variantes if all(cache_valido(caminho_variante(cfg, l, v), meta_variante(cfg, l, v)) for l in linhas)
    ]


def executar(cfg: Config, op: OpcoesTreino, dados=None) -> tuple[Path, dict]:
    cfg = config_efetiva(cfg, op)
    familias = familias_efetivas(cfg, op)
    split, linhas, conj = dados or preparar_dados(cfg, op.protocolo, familias)
    if "video" in familias and not (conj.variante != "").any():
        split, linhas, conj = preparar_dados(cfg, op.protocolo, familias)
    seeds = list(op.seeds or cfg["treino"]["seeds"])
    folds = [f for f in split["folds"] if op.folds is None or f["indice"] in op.folds]
    if not folds:
        raise ErroDadosInsuficientes("Nenhuma particao selecionada.")
    salvar = cfg["treino"].get("salvar_modelos_folds", True) if op.salvar_modelos is None else op.salvar_modelos
    base = Path(op.pasta_runs or cfg.caminho("runs"))
    if op.pasta_fixa:
        pasta = base / op.pasta_fixa
        if pasta.exists():
            import shutil

            shutil.rmtree(pasta)  # execucao incompleta anterior (sem resumo.json): refaz
    else:
        pasta = base / f"{carimbo()}_{op.nome}"
        n = 1
        while pasta.exists():
            n += 1
            pasta = base / f"{carimbo()}_{op.nome}_{n}"
    pasta.mkdir(parents=True)

    from .normalizacao import espec_de_config

    ambiente = {
        "versoes": versoes(),
        "hardware": hardware(),
        "seeds": seeds,
        "protocolo": split["protocolo"],
        "secundario": split.get("secundario", False),
        "familias_augmentation": familias,
        "intensidade": op.intensidade or cfg["augmentation"]["intensidade"],
        "assinatura_dados": split["assinatura_dados"],
        "n_videos": split["n_videos"],
        "n_sinalizadores": split["n_sinalizadores"],
        "janelas_totais": conj.n,
        "features": espec_de_config(cfg).descricao(),
        "classes": conj.classes,
        "ensaio": bool(split.get("ensaio")),
        "avisos_split": split.get("avisos", []),
    }
    escrever_texto_atomico(pasta / "config.yaml", yaml.safe_dump(dict(cfg), allow_unicode=True, sort_keys=False))
    escrever_texto_atomico(pasta / "ambiente.json", json.dumps(ambiente, indent=2, ensure_ascii=False))
    escrever_texto_atomico(pasta / "split.json", json.dumps(split, indent=2, ensure_ascii=False))
    escrever_texto_atomico(pasta / "features.json", json.dumps(conj.espec.como_dict(), indent=2))

    principal = not split.get("secundario", False)
    emissor = EmissorProgresso(op.arquivo_progresso)
    emissor.contexto = {"execucao": pasta.name}
    emissor.emitir(
        "inicio",
        pasta=str(pasta),
        seeds=seeds,
        folds=[f["indice"] for f in folds],
        classes=conj.classes,
        protocolo=split["protocolo"],
        familias=familias,
        ensaio=bool(split.get("ensaio")),
    )
    resultados = []
    for s in seeds:
        for f in folds:
            sub = pasta / f"seed_{s}" / f"fold_{f['indice']}"
            r = treinar_particao(
                cfg,
                conj,
                f,
                s,
                familias,
                op.intensidade,
                principal,
                sub,
                op.verbose,
                salvar,
                robustez=True,
                emissor=emissor,
            )
            r.setdefault("fold", f["indice"])
            r.setdefault("seed", s)
            resultados.append(r)
            if op.verbose and not r.get("ignorado"):
                print(
                    f"[seed {s} | particao {f['indice']}] teste em {', '.join(f['sinalizadores_teste'])}: "
                    f"acuracia janela {r['janela']['acuracia']:.3f}, video {r['video']['acuracia']:.3f} "
                    f"({r['epocas_executadas']} epocas, {r['segundos_treino']:.0f} s)"
                )
    resumo = {
        **ambiente,
        **resumir(resultados, conj.classes),
        "classes": conj.classes,
        "particoes_ignoradas": [r for r in resultados if r.get("ignorado")],
    }
    escrever_texto_atomico(pasta / "resumo.json", json.dumps(resumo, indent=2, ensure_ascii=False))
    validos = [r for r in resultados if not r.get("ignorado")]
    if validos:
        linhas_csv = [_linha_resumo(r) for r in validos]
        escrever_csv_atomico(pasta / "metricas_particoes.csv", list(linhas_csv[0].keys()), linhas_csv)
        for nivel in ("janela", "video"):
            figura_matriz(
                np.array(resumo[f"matriz_{nivel}_soma"]),
                conj.classes,
                pasta / f"matriz_confusao_{nivel}s_total.png",
                f"Teste ({nivel}s), soma das particoes",
            )
    escrever_texto_atomico(pasta / "LEIA-ME.txt", _leia_me(resumo))
    emissor.emitir(
        "fim",
        pasta=str(pasta),
        acuracia_video=resumo.get("acuracia_video_media"),
        f1_macro_video=resumo.get("f1_macro_video_media"),
    )
    return pasta, resumo


def _leia_me(r: dict) -> str:
    hw = r["hardware"]
    sec = " (SECUNDARIO: o mesmo sinalizador aparece em treino e teste)" if r.get("secundario") else ""
    from .selecao import TEXTO_ENSAIO

    topo = (TEXTO_ENSAIO + "\n\n") if r.get("ensaio") else ""
    return topo + (
        "Execucao de treino e avaliacao do LibrasVision\n"
        f"Classes: {', '.join(r.get('classes') or [])}\n"
        f"Protocolo: {r['protocolo']}{sec}\n"
        f"Videos: {r['n_videos']} | sinalizadores: {r['n_sinalizadores']} | seeds: {r['seeds']}\n"
        f"Augmentation: {r['familias_augmentation'] or 'nenhuma (baseline)'}\n"
        f"Hardware: {hw.get('processador')} | {hw.get('nucleos_logicos')} nucleos | {hw.get('ram_gb')} GB\n\n"
        f"Acuracia (janelas): {r['acuracia_janela_media']:.4f} +- {r['acuracia_janela_dp']:.4f}\n"
        f"F1 macro (janelas): {r['f1_macro_janela_media']:.4f} +- {r['f1_macro_janela_dp']:.4f}\n"
        f"Acuracia (videos):  {r['acuracia_video_media']:.4f} +- {r['acuracia_video_dp']:.4f}\n"
        f"F1 macro (videos):  {r['f1_macro_video_media']:.4f} +- {r['f1_macro_video_dp']:.4f}\n"
        "(media e desvio-padrao entre particoes e seeds; matrizes de confusao somam apenas dados de teste)\n\n"
        "Arquivos: config.yaml (configuracao usada), ambiente.json (versoes, hardware, seeds), split.json,\n"
        "features.json, resumo.json, metricas_particoes.csv, matriz_confusao_*_total.png e\n"
        "seed_<s>/fold_<i>/ (historico, curvas, matrizes, predicoes, metricas e modelo de cada particao).\n"
    )


# ======================================================================
# Modelo final (para o tempo real)
# ======================================================================
def treinar_final(cfg: Config, op: OpcoesTreino) -> Path:
    """Treina com todos os sinalizadores (validacao por sinalizadores para early stopping)
    e salva modelo + metadados de normalizacao para a inferencia em tempo real.

    Este modelo NAO tem avaliacao independente (todos os sinalizadores estao no treino);
    os numeros do TCC vem de executar().
    """
    from .dataset import montar_conjunto
    from .normalizacao import espec_de_config
    from .splits import _validacao, videos_elegiveis

    cfg = config_efetiva(cfg, op)
    familias = familias_efetivas(cfg, op)
    linhas, _ = videos_elegiveis(cfg)
    sinal = {l["signer_id"] for l in linhas}
    if len(sinal) < 1 or not linhas:
        raise ErroDadosInsuficientes("Nenhum video elegivel para treinar o modelo final.")
    from .selecao import exigir_selecao_valida

    sel = exigir_selecao_valida(cfg, linhas)
    ensaio = bool(sel.ensaio and sel.bloqueios)
    seed = int((op.seeds or cfg["treino"]["seeds"])[0])
    rng = np.random.default_rng(seed)
    treino, val, modo_val = _validacao(linhas, cfg["avaliacao"], rng, por_sinalizador=True)
    variantes = list(cfg["augmentation"]["video"]["variantes"]) if "video" in familias else []
    conj = montar_conjunto(cfg, linhas, variantes=variantes)
    fold = {
        "indice": 0,
        "treino": [l["video_id"] for l in treino],
        "validacao": [l["video_id"] for l in val],
        "teste": [l["video_id"] for l in val],
        "sinalizadores_teste": sorted({l["signer_id"] for l in val}),
    }
    base = Path(op.pasta_runs or cfg.caminho("runs"))
    pasta = base / f"{carimbo()}_{op.nome}"
    pasta.mkdir(parents=True, exist_ok=False)
    emissor = EmissorProgresso(op.arquivo_progresso)
    emissor.contexto = {"execucao": pasta.name, "final": True}
    emissor.emitir(
        "inicio",
        pasta=str(pasta),
        seeds=[seed],
        folds=[0],
        classes=conj.classes,
        protocolo="final",
        familias=familias,
        ensaio=ensaio,
    )
    r = treinar_particao(
        cfg,
        conj,
        fold,
        seed,
        familias,
        op.intensidade,
        principal=False,
        pasta=pasta / "treino",
        verbose=op.verbose,
        salvar_modelo=True,
        robustez=False,
        emissor=emissor,
    )
    (pasta / "treino" / "modelo.keras").replace(pasta / "modelo.keras")
    meta = {
        "classes": conj.classes,
        "nome_nenhum_sinal": cfg.nome_nenhum_sinal,
        "features": conj.espec.como_dict(),
        "janela": cfg["janela"],
        "normalizacao": cfg["normalizacao"],
        "mediapipe": cfg["mediapipe"],
        "versoes": versoes(),
        "hardware_treino": hardware(),
        "seed": seed,
        "familias_augmentation": familias,
        "validacao_por": modo_val,
        "videos": len(linhas),
        "sinalizadores": sorted(sinal),
        "epocas": r.get("epocas_executadas"),
        "aviso": "Modelo treinado com todos os sinalizadores; nao possui avaliacao independente.",
        "ensaio": ensaio,
        "avisos_selecao": sel.avisos,
        "criado_em": datetime.now().isoformat(timespec="seconds"),
        "protocolo": f"final: todos os sinalizadores (validação para early stopping por {modo_val})",
    }
    escrever_texto_atomico(pasta / "metadados.json", json.dumps(meta, indent=2, ensure_ascii=False))
    assert espec_de_config(cfg).dim == conj.espec.dim
    emissor.emitir("fim", pasta=str(pasta))
    return pasta
