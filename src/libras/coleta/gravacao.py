"""Salvamento de takes gravados pela webcam e registro no manifesto."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import Config
from ..erros import ErroLibras
from .manifesto import Manifesto
from .qualidade import InfoVideo
from .sinalizadores import CadastroSinalizadores


@dataclass
class MetaTake:
    classe: str
    signer_id: str
    iluminacao: str
    distancia: str
    fundo: str
    mao_dominante: str
    observacoes: str = ""


def validar_meta(cfg: Config, meta: MetaTake) -> None:
    op = cfg["coleta"]["opcoes"]
    if meta.classe not in cfg.classes:
        raise ErroLibras(f"Classe {meta.classe!r} nao esta no config. Opcoes: {cfg.classes}")
    for campo, chave in (
        ("iluminacao", "iluminacao"),
        ("distancia", "distancia"),
        ("fundo", "fundo"),
        ("mao_dominante", "mao_dominante"),
    ):
        v = getattr(meta, campo)
        if v not in op[chave]:
            raise ErroLibras(f"Valor invalido para {campo}: {v!r}. Opcoes: {op[chave]}")


def salvar_clipe(frames: list[np.ndarray], fps: float, destino: Path, codec: str = "mp4v") -> None:
    """Grava os frames EXATAMENTE como capturados (sem espelhar) e confere a leitura.

    Escreve primeiro em arquivo temporario; so renomeia para o destino se o arquivo
    abrir de volta com o mesmo numero de frames. Nunca sobrescreve um arquivo existente.
    """
    import cv2

    if not frames:
        raise ErroLibras("Clipe vazio: nada para salvar.")
    destino = Path(destino)
    if destino.exists():
        raise ErroLibras(f"Ja existe um arquivo em {destino}; nada foi sobrescrito.")
    destino.parent.mkdir(parents=True, exist_ok=True)
    alt, larg = frames[0].shape[:2]
    tmp = destino.with_name(destino.stem + ".parcial" + destino.suffix)
    w = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*codec), float(fps), (larg, alt))
    if not w.isOpened():
        raise ErroLibras(f"O OpenCV nao conseguiu criar o video com o codec {codec!r}.")
    try:
        for f in frames:
            if f.shape[:2] != (alt, larg):
                raise ErroLibras("A resolucao da camera mudou durante a gravacao; grave de novo.")
            w.write(f)
    finally:
        w.release()
    cap = cv2.VideoCapture(str(tmp))
    n = 0
    while cap.read()[0]:
        n += 1
    cap.release()
    if n != len(frames):
        tmp.unlink(missing_ok=True)
        raise ErroLibras(f"Video salvo com {n} frames em vez de {len(frames)}; gravacao descartada.")
    os.replace(tmp, destino)


def registrar_take(
    cfg: Config,
    manifesto: Manifesto,
    cadastro: CadastroSinalizadores,
    frames: list[np.ndarray],
    info: InfoVideo,
    meta: MetaTake,
    status: str = "ok",
) -> dict:
    """Salva o take no caminho padrao e cria o registro no manifesto (tudo ou nada). A trava do manifesto
    vale do numero do take ate o registro: duas gravacoes simultaneas nao pegam o mesmo numero."""
    with manifesto.trava():
        manifesto.recarregar()
        return _registrar_take(cfg, manifesto, cadastro, frames, info, meta, status)


def _registrar_take(
    cfg: Config,
    manifesto: Manifesto,
    cadastro: CadastroSinalizadores,
    frames: list[np.ndarray],
    info: InfoVideo,
    meta: MetaTake,
    status: str = "ok",
) -> dict:
    from ..utils import sha256_arquivo

    cadastro.exigir_tcle(meta.signer_id)
    validar_meta(cfg, meta)
    if info.erros:
        raise ErroLibras("Take com erro nao pode ser salvo: " + "; ".join(info.erros))
    take = manifesto.proximo_take(meta.signer_id, meta.classe)
    destino = manifesto.caminho_take(meta.classe, meta.signer_id, take)
    fps_arquivo = info.fps_medido if info.fps_medido > 0 else (info.fps_nominal or 30.0)
    salvar_clipe(frames, fps_arquivo, destino, cfg["coleta"]["gravacao"]["codec"])
    sha = sha256_arquivo(destino)
    ja = manifesto.hash_registrado(sha)
    if ja is not None:
        destino.unlink(missing_ok=True)
        raise ErroLibras(f"Este clipe e identico a {ja['video_id']}; nada foi salvo.")
    video_id = f"{meta.signer_id}_{meta.classe}_{take:03d}"
    npz = None
    if info.sequencia is not None:
        # landmarks dos frames brutos (antes da compressao do mp4); meta igual a da extracao (cache valido)
        from ..extracao import caminho_npz, meta_esperada, salvar_npz

        base = {"video_id": video_id, "classe": meta.classe, "sha256": sha}
        npz = caminho_npz(cfg, base)
        m = dict(meta_esperada(cfg, base), n_frames=int(info.sequencia.n_frames), origem_landmarks="gravacao")
        try:
            salvar_npz(npz, info.sequencia, info.tempos, m)
        except Exception:
            destino.unlink(missing_ok=True)
            raise
    registro = {
        "video_id": video_id,
        "sha256": sha,
        "sha256_origem": sha,
        "classe": meta.classe,
        "signer_id": meta.signer_id,
        "take": take,
        "origem": "gravado",
        "arquivo": manifesto.relativo(destino),
        "arquivo_original": "",
        "iluminacao": meta.iluminacao,
        "distancia": meta.distancia,
        "fundo": meta.fundo,
        "mao_dominante": meta.mao_dominante,
        "status": status,
        "observacoes": meta.observacoes,
        **info.campos_manifesto(),
    }
    if npz is not None:
        registro["landmarks_path"] = manifesto.relativo(npz)
        registro["sha256_npz"] = sha256_arquivo(npz)
    try:
        return manifesto.adicionar(registro)
    except Exception:
        destino.unlink(missing_ok=True)  # sem registro, sem arquivo: evita orfaos
        if npz is not None:
            npz.unlink(missing_ok=True)
        raise
