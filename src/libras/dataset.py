"""Montagem das janelas de treino/teste e pipeline tf.data com uso de memoria controlado.

As janelas de todos os videos elegiveis sao calculadas uma unica vez (a normalizacao
e por janela, entao nao depende da particao) e guardadas em float16. Cada particao
apenas seleciona indices. O tf.data le por gerador, em lotes, sem copiar o conjunto
inteiro para dentro do grafo do TensorFlow.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import Config
from .erros import ErroDadosInsuficientes, ErroLibras
from .normalizacao import EspecFeatures, espec_de_config, janelas_do_video


@dataclass
class InfoVideoJanelas:
    video_id: str
    classe: str
    signer_id: str
    n_janelas: int
    descartadas: dict
    avisos: list[str]
    taxa_mao: float
    taxa_pose: float
    n_frames: int


@dataclass
class ConjuntoJanelas:
    X: np.ndarray  # (N, L, D) float16
    y: np.ndarray  # (N,) int32, indice em `classes`
    video: np.ndarray  # (N,) video_id de cada janela
    signer: np.ndarray  # (N,) signer_id de cada janela
    variante: np.ndarray  # (N,) "" = video original; senao, nome da variante degradada (familia D)
    classes: list[str]
    espec: EspecFeatures
    infos: dict[str, InfoVideoJanelas] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(len(self.y))

    def indices(self, video_ids, variantes=("",)) -> np.ndarray:
        """Indices das janelas dos videos pedidos. Por padrao so as dos videos ORIGINAIS;
        variantes degradadas so entram quando pedidas explicitamente (treino ou robustez)."""
        alvo = list(set(video_ids))
        if not alvo:
            return np.zeros(0, int)
        return np.where(np.isin(self.video, alvo) & np.isin(self.variante, list(variantes)))[0]

    def memoria_mb(self) -> float:
        return self.X.nbytes / 2**20


def montar_conjunto(
    cfg: Config,
    linhas: list[dict],
    espec: EspecFeatures | None = None,
    progresso=None,
    variantes: list[str] | None = None,
) -> ConjuntoJanelas:
    """Janelas dos videos originais e, se pedido, das variantes degradadas ja extraidas
    (scripts/aumentar_videos.py). Variante ausente no cache gera erro claro."""
    from .augmentation.video import caminho_variante, meta_variante
    from .extracao import cache_valido, caminho_npz, carregar_npz

    if not linhas:
        raise ErroDadosInsuficientes("Nenhum video elegivel para montar o dataset.")
    espec = espec or espec_de_config(cfg)
    from .selecao import classes_treino

    classes = classes_treino(cfg)
    L = int(cfg["janela"]["comprimento"])
    limite = float(cfg["dataset"]["limite_memoria_mb"])
    bytes_janela = L * espec.dim * 2
    Xs, ys, vs, ss, vars_ = [], [], [], [], []
    fontes = [(l, "") for l in linhas]
    faltando = []
    for v in variantes or []:
        for l in linhas:
            if cache_valido(caminho_variante(cfg, l, v), meta_variante(cfg, l, v)):
                fontes.append((l, v))
            else:
                faltando.append(f"{l['video_id']}/{v}")
    if faltando:
        raise ErroDadosInsuficientes(
            f"{len(faltando)} variante(s) de video sem landmarks (ex.: {', '.join(faltando[:3])}). "
            f"Rode: scripts/aumentar_videos.py --variantes {' '.join(variantes)}"
        )
    infos: dict[str, InfoVideoJanelas] = {}
    total_bytes = 0
    nome_nada = cfg.nome_nenhum_sinal
    for i, (l, var) in enumerate(fontes, 1):
        d = carregar_npz(caminho_variante(cfg, l, var) if var else caminho_npz(cfg, l))
        jv = janelas_do_video(
            d.partes, d.tempos, d.largura, d.altura, espec, cfg, eh_nenhum_sinal=(l["classe"] == nome_nada)
        )
        if not var:
            infos[l["video_id"]] = InfoVideoJanelas(
                l["video_id"],
                l["classe"],
                l["signer_id"],
                len(jv.X),
                jv.descartadas,
                jv.avisos,
                d.taxa_alguma_mao,
                d.taxa(0),
                d.partes.n,
            )
        if len(jv.X):
            total_bytes += len(jv.X) * bytes_janela
            if total_bytes / 2**20 > limite:
                raise ErroLibras(
                    f"As janelas passariam de {limite:.0f} MB (limite em dataset.limite_memoria_mb). "
                    "Aumente janela.passo, use features.face.modo: nenhuma/subconjunto, ou aumente o limite "
                    "se houver RAM sobrando."
                )
            Xs.append(jv.X.astype(np.float16))
            ys.append(np.full(len(jv.X), classes.index(l["classe"]), np.int32))
            vs.append(np.full(len(jv.X), l["video_id"], dtype=object))
            ss.append(np.full(len(jv.X), l["signer_id"], dtype=object))
            vars_.append(np.full(len(jv.X), var, dtype=object))
        if progresso:
            progresso(i, len(fontes), l["video_id"], len(jv.X))
    if not Xs:
        raise ErroDadosInsuficientes(
            "Nenhum video gerou janelas validas (pose ausente em todos?). "
            "Rode scripts/validar_dados.py para ver os motivos."
        )
    return ConjuntoJanelas(
        np.concatenate(Xs),
        np.concatenate(ys),
        np.concatenate(vs).astype(str),
        np.concatenate(ss).astype(str),
        np.concatenate(vars_).astype(str),
        classes,
        espec,
        infos,
    )


def pesos_classes(y: np.ndarray, n_classes: int) -> dict[int, float]:
    """Pesos inversamente proporcionais a frequencia (padrao 'balanced' do sklearn)."""
    cont = np.bincount(y, minlength=n_classes).astype(float)
    pesos = {}
    for c in range(n_classes):
        pesos[c] = float(len(y) / (n_classes * cont[c])) if cont[c] > 0 else 0.0
    return pesos


def gerador_lotes(conj, indices: np.ndarray, batch_size: int, embaralhar: bool, seed: int, transformar=None):
    """Fabrica de geradores; cada chamada (uma por epoca) embaralha com seed + epoca.

    conj: qualquer objeto com atributos X (N, L, D) e y (N,).

    transformar(x_lote_float32, rng) -> x_lote: gancho para data augmentation (so no treino).
    """
    indices = np.asarray(indices)
    estado = {"epoca": 0}

    def gerar():
        rng = np.random.default_rng(seed + estado["epoca"])
        estado["epoca"] += 1
        ordem = rng.permutation(indices) if embaralhar else indices
        for i in range(0, len(ordem), batch_size):
            idx = np.sort(ordem[i : i + batch_size]) if not embaralhar else ordem[i : i + batch_size]
            x = conj.X[idx].astype(np.float32)
            if transformar is not None:
                x = transformar(x, rng)
            yield x, conj.y[idx]

    return gerar


def criar_tf_dataset(conj, indices: np.ndarray, batch_size: int, embaralhar: bool, seed: int, transformar=None):
    import tensorflow as tf

    L, D = conj.X.shape[1], conj.X.shape[2]
    ds = tf.data.Dataset.from_generator(
        gerador_lotes(conj, indices, batch_size, embaralhar, seed, transformar),
        output_signature=(tf.TensorSpec((None, L, D), tf.float32), tf.TensorSpec((None,), tf.int32)),
    )
    # cardinalidade conhecida: o Keras sabe quantos lotes ha por epoca (sem avisos de "fim de dados")
    n_lotes = int(np.ceil(len(indices) / batch_size))
    ds = ds.apply(tf.data.experimental.assert_cardinality(n_lotes))
    return ds.prefetch(2)
