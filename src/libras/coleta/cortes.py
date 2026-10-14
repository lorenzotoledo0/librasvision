"""Corte de takes na revisao (item 5).

- cortar_take(video_id, ini, fim): mantem so os frames [ini, fim).
- dividir_take(video_id, ponto): frames [0, ponto) ficam no take original e [ponto, n) viram
  um take novo do mesmo sinalizador e classe.

Para cada trecho: novo .mp4 (mesmo FPS do original) e novo .npz com os landmarks do trecho e os
instantes reindexados a partir de 0. Os landmarks sao FATIADOS do .npz original (nao ha nova
passagem do MediaPipe), o que preserva o rastreamento continuo. O original (.mp4 e .npz) vai para
data/lixeira/cortes/<video_id>_<data>/; o manifesto registra os frames originais e o corte.

Minimo de 30 frames por trecho (uma janela). Remocao de trecho INTERNO nao existe de proposito:
juntar o antes e o depois criaria um salto que o modelo leria como movimento rapido.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

import numpy as np

from ..config import Config
from ..erros import ErroLibras
from ..utils import carimbo, sha256_arquivo
from .manifesto import Manifesto

log = logging.getLogger("libras.cortes")
MIN_FRAMES = 30


def _carregar_landmarks(cfg: Config, man: Manifesto, linha: dict):
    """Le o .npz do take; se ausente ou desatualizado, extrai do proprio video antes."""
    from ..extracao import cache_valido, caminho_npz, carregar_npz, extrair_um, meta_esperada

    npz = caminho_npz(cfg, linha)
    if not cache_valido(npz, meta_esperada(cfg, linha)):
        r = extrair_um(cfg["mediapipe"], str(man.caminho_absoluto(linha)), str(npz), meta_esperada(cfg, linha))
        if not r.get("ok"):
            raise ErroLibras(f"Nao foi possivel extrair os landmarks de {linha['video_id']}: {r.get('motivo')}")
    return carregar_npz(npz)


def _escrever_trecho_video(origem: Path, destino: Path, ini: int, fim: int, fps: float) -> int:
    import cv2

    cap = cv2.VideoCapture(str(origem))
    w = None
    n_escritos = 0
    try:
        i = 0
        while True:
            ok, f = cap.read()
            if not ok or i >= fim:
                break
            if i >= ini:
                if w is None:
                    alt, larg = f.shape[:2]
                    w = cv2.VideoWriter(str(destino), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (larg, alt))
                w.write(f)
                n_escritos += 1
            i += 1
    finally:
        cap.release()
        if w is not None:
            w.release()
    return n_escritos


def _fatiar(dados, ini: int, fim: int):
    from ..landmarks import SequenciaLandmarks

    p = dados.partes
    seq = SequenciaLandmarks(
        p.pose[ini:fim],
        p.mao_esq[ini:fim],
        p.mao_dir[ini:fim],
        p.face[ini:fim],
        p.presenca[ini:fim],
        dados.largura,
        dados.altura,
        dados.fps,
    )
    tempos = np.asarray(dados.tempos[ini:fim], dtype=np.float64)
    return seq, tempos - tempos[0]


def _contar_frames(caminho: Path) -> int:
    import cv2

    cap = cv2.VideoCapture(str(caminho))
    n = 0
    while cap.read()[0]:
        n += 1
    cap.release()
    return n


def _gerar_trechos(cfg: Config, video_id: str, trechos: list[tuple[int, int]]) -> list[dict]:
    with Manifesto(cfg).trava():  # numeros de take novos e troca do registro sem concorrencia
        return _gerar_trechos_travado(cfg, video_id, trechos)


def _gerar_trechos_travado(cfg: Config, video_id: str, trechos: list[tuple[int, int]]) -> list[dict]:
    """Nucleo comum: gera arquivos temporarios de cada trecho, move o original para a lixeira e
    atualiza o manifesto. O primeiro trecho fica com o video_id original; os demais viram takes novos."""
    from ..extracao import caminho_npz, meta_esperada, salvar_npz

    man = Manifesto(cfg)
    l = man.obter(video_id)
    if l is None:
        raise ErroLibras(f"Take {video_id} nao encontrado.")
    origem = man.caminho_absoluto(l)
    dados = _carregar_landmarks(cfg, man, l)
    n = dados.partes.n
    n_video = _contar_frames(origem)
    if n_video != n:
        raise ErroLibras(
            f"O video tem {n_video} frames e os landmarks {n}. Reextraia (scripts/extrair.py --forcar) antes de cortar."
        )
    for ini, fim in trechos:
        if not (0 <= ini < fim <= n):
            raise ErroLibras(f"Corte invalido: frames {ini}-{fim} (o take tem {n}).")
        if fim - ini < MIN_FRAMES:
            raise ErroLibras(
                f"Cada trecho precisa de pelo menos {MIN_FRAMES} frames (1 janela); "
                f"o trecho {ini}-{fim} teria {fim - ini}."
            )
    fps = float(l.get("fps_medido") or 0) or float(dados.fps) or 30.0
    original_n = int(l.get("frames_originais") or n)
    planos = []
    take_livre = man.proximo_take(l["signer_id"], l["classe"])
    for k, (ini, fim) in enumerate(trechos):
        if k == 0:
            vid, take, destino = video_id, int(l["take"]), origem
        else:
            take = take_livre
            take_livre += 1
            vid = f"{l['signer_id']}_{l['classe']}_{take:03d}"
            destino = man.caminho_take(l["classe"], l["signer_id"], take)
        planos.append((vid, take, destino, ini, fim))

    temporarios: list[Path] = []
    try:
        gerados = []
        for vid, take, destino, ini, fim in planos:
            tmp_v = destino.with_name(destino.stem + ".corte.parcial" + destino.suffix)
            escritos = _escrever_trecho_video(origem, tmp_v, ini, fim, fps)
            temporarios.append(tmp_v)
            if escritos != fim - ini or _contar_frames(tmp_v) != fim - ini:
                raise ErroLibras("O video cortado nao ficou com o numero esperado de frames; nada foi alterado.")
            seq, tempos = _fatiar(dados, ini, fim)
            gerados.append((vid, take, destino, ini, fim, tmp_v, seq, tempos))
        # original -> lixeira/cortes
        lix = cfg.caminho("lixeira") / "cortes" / f"{video_id}_{carimbo()}"
        lix.mkdir(parents=True, exist_ok=False)
        npz_orig = caminho_npz(cfg, l)
        shutil.move(str(origem), str(lix / origem.name))
        if npz_orig.exists():
            shutil.move(str(npz_orig), str(lix / npz_orig.name))
        resultado = []
        for vid, take, destino, ini, fim, tmp_v, seq, tempos in gerados:
            os.replace(tmp_v, destino)
            sha = sha256_arquivo(destino)
            base = {"video_id": vid, "classe": l["classe"], "sha256": sha}
            npz = caminho_npz(cfg, base)
            salvar_npz(
                npz,
                seq,
                tempos,
                dict(
                    meta_esperada(cfg, base),
                    n_frames=fim - ini,
                    origem_landmarks="corte",
                    corte_de=video_id,
                    corte_frames=[ini, fim],
                ),
            )
            campos = {
                "sha256": sha,
                "arquivo": man.relativo(destino),
                "n_frames": fim - ini,
                "duracao_s": f"{(fim - ini) / fps:.3f}",
                "landmarks_path": man.relativo(npz),
                "sha256_npz": sha256_arquivo(npz),
                "frames_originais": original_n,
                "taxa_mao": f"{float((seq.presenca[:, 1] | seq.presenca[:, 2]).mean()):.3f}",
                "taxa_pose": f"{float(seq.presenca[:, 0].mean()):.3f}",
                "corte": f"frames {ini}-{fim} de {n} ({video_id})",
            }
            if vid == video_id:
                man.atualizar(vid, **campos)
            else:
                novo = {
                    k: l[k]
                    for k in (
                        "sha256_origem",
                        "classe",
                        "signer_id",
                        "origem",
                        "arquivo_original",
                        "iluminacao",
                        "distancia",
                        "fundo",
                        "mao_dominante",
                        "largura",
                        "altura",
                        "fps_nominal",
                        "fps_medido",
                        "brilho_medio",
                        "flags_qualidade",
                        "status",
                    )
                }
                novo.update(
                    campos,
                    video_id=vid,
                    take=take,
                    observacoes=(l.get("observacoes", "") + " | " if l.get("observacoes") else "")
                    + f"dividido de {video_id}",
                )
                man.adicionar(novo)
            resultado.append(man.obter(vid))
        log.info("Corte aplicado em %s: %s; original em %s", video_id, [(g[3], g[4]) for g in gerados], lix)
        return resultado
    finally:
        for t in temporarios:
            t.unlink(missing_ok=True)


def cortar_take(cfg: Config, video_id: str, ini: int, fim: int) -> dict:
    return _gerar_trechos(cfg, video_id, [(int(ini), int(fim))])[0]


def dividir_take(cfg: Config, video_id: str, ponto: int) -> tuple[dict, dict]:
    man = Manifesto(cfg)
    l = man.obter(video_id)
    if l is None:
        raise ErroLibras(f"Take {video_id} nao encontrado.")
    n = int(l.get("n_frames") or 0) or _contar_frames(man.caminho_absoluto(l))
    a, b = _gerar_trechos(cfg, video_id, [(0, int(ponto)), (int(ponto), n)])
    return a, b


def aplicar_edicao(cfg: Config, video_id: str, inicio: int, fim: int, divisao: int | None = None) -> list[dict]:
    """Aplica o resultado do editor de corte (item 9) de uma vez: trecho mantido [inicio, fim) e,
    se houver, divisao em dois takes. Sem alteracao, nada e gravado."""
    man = Manifesto(cfg)
    l = man.obter(video_id)
    if l is None:
        raise ErroLibras(f"Take {video_id} nao encontrado.")
    n = int(l.get("n_frames") or 0) or _contar_frames(man.caminho_absoluto(l))
    inicio, fim = int(inicio), int(fim)
    if divisao is None:
        if (inicio, fim) == (0, n):
            return [l]
        return _gerar_trechos(cfg, video_id, [(inicio, fim)])
    divisao = int(divisao)
    return _gerar_trechos(cfg, video_id, [(inicio, divisao), (divisao, fim)])
