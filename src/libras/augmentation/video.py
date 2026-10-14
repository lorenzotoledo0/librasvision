"""Familia D: augmentation em nivel de video (offline).

Degrada os frames dos videos brutos (iluminacao, desfoque, ruido, pequena rotacao e
zoom) e REEXTRAI os landmarks com o MediaPipe, simulando a variacao de iluminacao de
uma sala de aula (limitacao assumida no capitulo 3.6 do TCC I). Guarda apenas os
landmarks resultantes em data/landmarks_aug/<variante>/<classe>/<video_id>.npz; os
frames degradados nunca sao salvos.

As mesmas degradacoes servem ao teste de robustez da Fase 5 (aplicadas aos videos de
teste). No treino, as variantes de um video so entram quando o video original esta no
treino da particao (o dataset marca cada janela com a variante de origem).
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

# Parametros fixos e documentados de cada variante. Os aleatorios (angulo, zoom, ruido)
# sao sorteados uma vez por video com seed derivada do video_id: reprodutivel.
VARIANTES: dict[str, dict] = {
    "escuro": {"gama": 1.8, "brilho": -25, "contraste": 0.8},
    "claro": {"gama": 0.6, "brilho": 25, "contraste": 1.0},
    "desfoque": {"kernel": 7},
    "ruido_imagem": {"sigma": 12.0},
    "rotacao_zoom": {"graus_max": 8.0, "zoom": [0.9, 1.15]},
}


def seed_video(video_id: str, variante: str, seed_base: int) -> int:
    h = hashlib.sha256(f"{seed_base}:{variante}:{video_id}".encode()).hexdigest()
    return int(h[:8], 16)


def hash_variante(variante: str) -> str:
    return hashlib.sha256(json.dumps(VARIANTES[variante], sort_keys=True).encode()).hexdigest()[:12]


class Degradador:
    """Aplica uma variante a frames BGR. Parametros sorteados uma vez por video."""

    def __init__(self, variante: str, seed: int):
        if variante not in VARIANTES:
            from ..erros import ErroConfig

            raise ErroConfig(f"Variante de video desconhecida: {variante}. Opcoes: {sorted(VARIANTES)}")
        self.variante = variante
        self.p = dict(VARIANTES[variante])
        self.rng = np.random.default_rng(seed)
        if variante == "rotacao_zoom":
            self.angulo = float(self.rng.uniform(-self.p["graus_max"], self.p["graus_max"]))
            self.zoom = float(self.rng.uniform(*self.p["zoom"]))
        self._lut = None

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        import cv2

        v = self.variante
        if v in ("escuro", "claro"):
            if self._lut is None:
                x = np.arange(256) / 255.0
                y = (x ** self.p["gama"]) * 255.0 * self.p["contraste"] + self.p["brilho"]
                self._lut = np.clip(y, 0, 255).astype(np.uint8)
            return cv2.LUT(frame, self._lut)
        if v == "desfoque":
            k = int(self.p["kernel"]) | 1
            return cv2.GaussianBlur(frame, (k, k), 0)
        if v == "ruido_imagem":
            ruido = self.rng.normal(0, self.p["sigma"], frame.shape)
            return np.clip(frame.astype(np.float32) + ruido, 0, 255).astype(np.uint8)
        if v == "rotacao_zoom":
            h, w = frame.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2, h / 2), self.angulo, self.zoom)
            return cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        raise AssertionError(v)


def caminho_variante(cfg, linha: dict, variante: str) -> Path:
    return cfg.caminho("landmarks_aug") / variante / linha["classe"] / f"{linha['video_id']}.npz"


def meta_variante(cfg, linha: dict, variante: str) -> dict:
    from ..extracao import meta_esperada

    return dict(
        meta_esperada(cfg, linha),
        variante=variante,
        variante_hash=hash_variante(variante),
        seed=seed_video(linha["video_id"], variante, int(cfg["projeto"]["seed"])),
    )


def extrair_variante(cfg_mediapipe: dict, caminho_video: str, destino: str, meta: dict) -> dict:
    """Le o video, degrada cada frame em memoria, extrai landmarks e salva o npz."""
    import cv2

    from ..extracao import salvar_npz
    from ..landmarks import ExtratorHolistic, empilhar, ler_frames

    t0 = time.perf_counter()
    cap = cv2.VideoCapture(caminho_video)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cap.release()
    if fps <= 0:
        return {"ok": False, "motivo": "video sem FPS"}
    deg = Degradador(meta["variante"], meta["seed"])
    frames_lm, larg, alt = [], 0, 0
    with ExtratorHolistic(cfg_mediapipe, modo_estatico=False) as ext:
        for f in ler_frames(Path(caminho_video)):
            alt, larg = f.shape[:2]
            frames_lm.append(ext.processar(deg(f))[0])
    if not frames_lm:
        return {"ok": False, "motivo": "video sem frames"}
    seq = empilhar(frames_lm, larg, alt, fps)
    meta = dict(meta, n_frames=seq.n_frames, segundos_extracao=round(time.perf_counter() - t0, 2))
    salvar_npz(Path(destino), seq, np.arange(seq.n_frames) / fps, meta)
    return {
        "ok": True,
        "n_frames": seq.n_frames,
        "taxa_mao": seq.taxa_alguma_mao,
        "taxa_pose": seq.taxa("pose"),
        "segundos": meta["segundos_extracao"],
    }


def aumentar_lote(
    cfg,
    variantes: list[str],
    video_ids: list[str] | None = None,
    forcar: bool = False,
    processos: int = 1,
    progresso=None,
) -> dict:
    """Gera (ou reaproveita do cache) os landmarks das variantes para os videos elegiveis."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed

    from ..coleta.manifesto import Manifesto
    from ..erros import ErroDadosInsuficientes
    from ..extracao import _inicializar_processo, cache_valido
    from ..splits import videos_elegiveis

    for v in variantes:
        if v not in VARIANTES:
            from ..erros import ErroConfig

            raise ErroConfig(f"Variante desconhecida: {v}. Opcoes: {sorted(VARIANTES)}")
    linhas, _ = videos_elegiveis(cfg)
    if video_ids is not None:
        alvo_ids = set(video_ids)
        linhas = [l for l in linhas if l["video_id"] in alvo_ids]
    if not linhas:
        raise ErroDadosInsuficientes(
            "Nenhum video elegivel (status 'ok' com landmarks extraidos). Rode scripts/extrair.py primeiro."
        )
    man = Manifesto(cfg)
    tarefas, em_cache = [], 0
    for l in linhas:
        for v in variantes:
            dest, meta = caminho_variante(cfg, l, v), meta_variante(cfg, l, v)
            if not forcar and cache_valido(dest, meta):
                em_cache += 1
                continue
            tarefas.append((l["video_id"], v, str(man.caminho_absoluto(l)), str(dest), meta))
    res = {"processados": 0, "em_cache": em_cache, "falhas": {}}

    def reg(i, vid, v, r):
        if r.get("ok"):
            res["processados"] += 1
        else:
            res["falhas"][f"{vid}/{v}"] = r.get("motivo")
        if progresso:
            progresso(i, len(tarefas), vid, v, r)

    if processos <= 1:
        for i, (vid, v, src, dst, meta) in enumerate(tarefas, 1):
            try:
                r = extrair_variante(cfg["mediapipe"], src, dst, meta)
            except Exception as e:  # um video ruim nao derruba o lote
                r = {"ok": False, "motivo": f"{type(e).__name__}: {e}"}
            reg(i, vid, v, r)
    else:
        with ProcessPoolExecutor(
            processos, initializer=_inicializar_processo, mp_context=mp.get_context("spawn")
        ) as ex:
            fut = {
                ex.submit(extrair_variante, dict(cfg["mediapipe"]), s, d, m): (vid, v) for (vid, v, s, d, m) in tarefas
            }
            for i, f in enumerate(as_completed(fut), 1):
                vid, v = fut[f]
                try:
                    r = f.result()
                except Exception as e:
                    r = {"ok": False, "motivo": f"{type(e).__name__}: {e}"}
                reg(i, vid, v, r)
    return res
