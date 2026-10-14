"""Previa da data augmentation: o original de um take e N variacoes, para olhar antes de treinar.

So visualizacao. Nada daqui entra no treino nem e salvo como dado: o treino continua sorteando as
variacoes do jeito de sempre (online, a cada epoca, ou offline com copias fixas; decisoes D8).

A janela mostrada e a de maior movimento das maos do take, normalizada pelo mesmo caminho do treino
(janelas_do_video). Na previa, cada familia escolhida e aplicada sempre (probabilidade 1), para cada
variacao mostrar o efeito; no treino, cada familia entra com a probabilidade do config.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..erros import ErroLibras
from ..normalizacao import I_FACE, I_MAO_D, I_MAO_E, I_POSE, desmontar, espec_de_config, janelas_do_video
from . import FAMILIAS_ONLINE, Aumentador

# BGR; a previa e uma imagem gerada (nao e interface Qt), como o esqueleto de landmarks.desenhar_esqueleto
FUNDO = (24, 19, 20)
COR_POSE, COR_MAO_E, COR_MAO_D, COR_FACE = (235, 235, 235), (246, 130, 59), (237, 58, 124), (150, 150, 150)
# pontos da pose ate a cintura: o enquadramento da coleta corta as pernas
POSE_VISIVEL = set(range(0, 25))


@dataclass
class Previa:
    video_id: str
    original: dict[str, np.ndarray]  # janela desmontada (ver normalizacao.desmontar)
    variacoes: list[dict[str, np.ndarray]]
    inicio: int  # primeiro frame da janela, na grade de fps_alvo
    familias: list[str]
    intensidade: str

    @property
    def n_frames(self) -> int:
        return len(self.original["presenca"])

    @property
    def caixa(self) -> tuple[float, float, float, float]:
        """Extensao (x0, y0, x1, y1) do original; a mesma para todas as variacoes, para compara-las."""
        return extensao(self.original)


def extensao(d: dict[str, np.ndarray]) -> tuple[float, float, float, float]:
    pts = []
    for nome, k in (("pose", I_POSE), ("mao_esq", I_MAO_E), ("mao_dir", I_MAO_D), ("face", I_FACE)):
        if nome in d:
            arr = d[nome][d["presenca"][:, k] > 0.5]
            if nome == "pose":
                arr = arr[:, sorted(POSE_VISIVEL)]
            if len(arr):
                pts.append(arr[..., :2].reshape(-1, 2))
    if not pts:
        return (-2.0, -2.0, 2.0, 2.0)
    todos = np.concatenate(pts)
    (x0, y0), (x1, y1) = todos.min(0), todos.max(0)
    return float(x0), float(y0), float(x1), float(y1)


def gerar_previa(
    cfg,
    linha: dict,
    n: int = 5,
    seed: int = 0,
    familias: list[str] | None = None,
    intensidade: str | None = None,
) -> Previa:
    from ..extracao import cache_valido, caminho_npz, carregar_npz, meta_esperada

    npz = caminho_npz(cfg, linha)
    if not cache_valido(npz, meta_esperada(cfg, linha)):
        raise ErroLibras(f"{linha['video_id']} nao tem landmarks validos. Rode Dataset > Extrair.")
    dados = carregar_npz(npz)
    espec = espec_de_config(cfg)
    jv = janelas_do_video(
        dados.partes,
        dados.tempos,
        dados.largura,
        dados.altura,
        espec,
        cfg,
        eh_nenhum_sinal=linha["classe"] == cfg.nome_nenhum_sinal,
    )
    if not len(jv.X):
        raise ErroLibras(f"{linha['video_id']} nao tem nenhuma janela valida (pose insuficiente).")
    k = int(np.argmax(jv.energias))
    familias = list(FAMILIAS_ONLINE) if familias is None else list(familias)
    aum = Aumentador(cfg, espec, familias=familias, intensidade=intensidade)
    aum.cfg_aug = {**aum.cfg_aug, **{f: {**aum.cfg_aug[f], "prob": 1.0} for f in familias}}
    rng = np.random.default_rng(seed)
    X = jv.X[k].astype(np.float32)
    variacoes = [desmontar(aum.janela(X, rng), espec) for _ in range(n)]
    return Previa(linha["video_id"], desmontar(X, espec), variacoes, int(jv.inicios[k]), familias, aum.intensidade)


def desenhar(
    d: dict[str, np.ndarray],
    i: int,
    largura: int = 320,
    altura: int = 300,
    titulo: str = "",
    caixa: tuple[float, float, float, float] | None = None,
) -> np.ndarray:
    """Quadro i de uma janela normalizada (origem no centro dos ombros, largura de ombro = 1).

    caixa: extensao a enquadrar (padrao: a da propria janela). Com folga de 30%, para as variacoes
    que giram ou esticam o corpo continuarem dentro do quadro."""
    import cv2

    from ..landmarks import _conexoes

    img = np.full((altura, largura, 3), FUNDO, np.uint8)
    x0, y0, x1, y1 = caixa or extensao(d)
    esc = min((largura - 24) / max(x1 - x0, 1e-3), (altura - 36) / max(y1 - y0, 1e-3)) / 1.3
    cx, cy = largura / 2 - esc * (x0 + x1) / 2, (altura + 12) / 2 - esc * (y0 + y1) / 2
    pres = d["presenca"][i]
    con = _conexoes()

    def pts(arr):
        return [(int(cx + x * esc), int(cy + y * esc)) for x, y in arr[:, :2]]

    if "face" in d and pres[I_FACE] > 0.5:
        for q in pts(d["face"][i]):
            cv2.circle(img, q, 1, COR_FACE, -1, cv2.LINE_AA)
    if "pose" in d and pres[I_POSE] > 0.5:
        p = pts(d["pose"][i])
        for a, b in con["pose"]:
            if a in POSE_VISIVEL and b in POSE_VISIVEL:
                cv2.line(img, p[a], p[b], COR_POSE, 2, cv2.LINE_AA)
    for nome, k, cor in (("mao_esq", I_MAO_E, COR_MAO_E), ("mao_dir", I_MAO_D, COR_MAO_D)):
        if nome in d and pres[k] > 0.5:
            p = pts(d[nome][i])
            for a, b in con["mao"]:
                cv2.line(img, p[a], p[b], cor, 2, cv2.LINE_AA)
    if titulo:
        cv2.putText(img, titulo, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COR_POSE, 1, cv2.LINE_AA)
    return img


def mosaico(previa: Previa, i: int, colunas: int = 3, largura: int = 320, altura: int = 300) -> np.ndarray:
    """Original e variacoes lado a lado, no quadro i (para exportar como figura)."""
    cx = previa.caixa
    paineis = [desenhar(previa.original, i, largura, altura, "original", cx)] + [
        desenhar(v, i, largura, altura, f"variacao {k + 1}", cx) for k, v in enumerate(previa.variacoes)
    ]
    while len(paineis) % colunas:
        paineis.append(np.full((altura, largura, 3), FUNDO, np.uint8))
    linhas = [np.hstack(paineis[r : r + colunas]) for r in range(0, len(paineis), colunas)]
    return np.vstack(linhas)
