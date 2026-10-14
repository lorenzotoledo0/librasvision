"""Extracao de landmarks em lote, incremental, com cache por video.

Cada video do manifesto gera data/landmarks/<classe>/<video_id>.npz com as
coordenadas cruas, a presenca por parte, os instantes de cada frame e os metadados
que definem o cache (hash do video, versao do MediaPipe, parametros do Holistic).

Se qualquer um desses metadados mudar, o npz e considerado desatualizado e o video
e reprocessado. Mudar o vetor de features ou a normalizacao NAO exige reextrair,
porque o npz guarda os dados crus.
"""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path

import numpy as np

from .config import Config
from .erros import ErroDadosInsuficientes
from .normalizacao import Partes

VERSAO_EXTRACAO = 1  # incremente se o formato do npz ou a forma de extrair mudar


def versao_mediapipe() -> str:
    try:
        return metadata.version("mediapipe")
    except metadata.PackageNotFoundError:
        return "nao instalado"


def caminho_npz(cfg: Config, linha: dict) -> Path:
    return cfg.caminho("landmarks") / linha["classe"] / f"{linha['video_id']}.npz"


def meta_esperada(cfg: Config, linha: dict) -> dict:
    return {
        "versao_extracao": VERSAO_EXTRACAO,
        "sha256": linha["sha256"],
        "mediapipe": versao_mediapipe(),
        "mediapipe_cfg": cfg.hash_secao("mediapipe"),
    }


@dataclass
class DadosVideo:
    partes: Partes
    tempos: np.ndarray
    largura: int
    altura: int
    fps: float
    meta: dict

    def taxa(self, k: int) -> float:
        return float(self.partes.presenca[:, k].mean()) if self.partes.n else 0.0

    @property
    def taxa_alguma_mao(self) -> float:
        p = self.partes.presenca
        return float((p[:, 1] | p[:, 2]).mean()) if self.partes.n else 0.0


def salvar_npz(destino: Path, seq, tempos: np.ndarray, meta: dict) -> None:
    """Escrita atomica (arquivo temporario + substituicao)."""
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    tmp = destino.with_name(destino.stem + ".parcial.npz")
    np.savez_compressed(
        tmp,
        pose=seq.pose.astype(np.float32),
        mao_esq=seq.mao_esq.astype(np.float32),
        mao_dir=seq.mao_dir.astype(np.float32),
        face=seq.face.astype(np.float32),
        presenca=seq.presenca.astype(bool),
        tempos=np.asarray(tempos, np.float64),
        largura=np.int32(seq.largura),
        altura=np.int32(seq.altura),
        fps=np.float64(seq.fps),
        meta=np.array(json.dumps(meta, sort_keys=True)),
    )
    os.replace(tmp, destino)


def carregar_npz(caminho: Path) -> DadosVideo:
    with np.load(caminho, allow_pickle=False) as z:
        partes = Partes(z["pose"], z["mao_esq"], z["mao_dir"], z["face"], z["presenca"])
        return DadosVideo(
            partes, z["tempos"], int(z["largura"]), int(z["altura"]), float(z["fps"]), json.loads(str(z["meta"]))
        )


def ler_meta(caminho: Path) -> dict | None:
    try:
        with np.load(caminho, allow_pickle=False) as z:
            return json.loads(str(z["meta"]))
    except Exception:  # arquivo corrompido ou de formato antigo: trata como ausente
        return None


def cache_valido(caminho: Path, esperado: dict) -> bool:
    if not Path(caminho).exists():
        return False
    meta = ler_meta(caminho)
    return meta is not None and all(meta.get(k) == v for k, v in esperado.items())


# ----------------------------------------------------------------------
# Trabalho de um video (funcao de nivel de modulo: precisa ser "picklable" no Windows)
# ----------------------------------------------------------------------
def _inicializar_processo() -> None:
    # O MediaPipe 0.10.21 importa o TensorFlow sem precisar; bloquear economiza ~500 MB por processo.
    if "tensorflow" not in sys.modules:
        sys.modules["tensorflow"] = None


def extrair_um(cfg_mediapipe: dict, caminho_video: str, destino: str, meta: dict) -> dict:
    from .landmarks import extrair_video

    t0 = time.perf_counter()
    seq = extrair_video(Path(caminho_video), cfg_mediapipe)
    if seq.n_frames == 0:
        return {"ok": False, "motivo": "video sem frames legiveis"}
    if seq.fps <= 0:
        return {"ok": False, "motivo": "video sem FPS no conteiner"}
    tempos = np.arange(seq.n_frames, dtype=np.float64) / seq.fps
    meta = dict(
        meta, n_frames=seq.n_frames, segundos_extracao=round(time.perf_counter() - t0, 2), origem_landmarks="arquivo"
    )
    salvar_npz(Path(destino), seq, tempos, meta)
    return {
        "ok": True,
        "n_frames": seq.n_frames,
        "taxa_mao": seq.taxa_alguma_mao,
        "taxa_pose": seq.taxa("pose"),
        "segundos": meta["segundos_extracao"],
    }


# ----------------------------------------------------------------------
# Lote
# ----------------------------------------------------------------------
@dataclass
class ResumoExtracao:
    processados: list[str] = field(default_factory=list)
    em_cache: list[str] = field(default_factory=list)
    falhas: dict[str, str] = field(default_factory=dict)
    segundos: float = 0.0


def extrair_lote(
    cfg: Config, status: list[str] | None = None, forcar: bool = False, processos: int | None = None, progresso=None
) -> ResumoExtracao:
    from .coleta.manifesto import Manifesto

    status = list(status or cfg["extracao"]["status_incluidos"])
    processos = int(processos or cfg["extracao"].get("processos", 1))
    man = Manifesto(cfg)
    alvo = [l for l in man.linhas if l["status"] in status and l["classe"] in cfg.classes]
    if not alvo:
        raise ErroDadosInsuficientes(
            f"Nenhum video com status {status} no manifesto ({len(man)} registros no total). "
            "Grave ou importe videos primeiro (LibrasVisionLab.bat, aba Gravar, ou scripts/importar.py), e revise os pendentes."
        )
    resumo = ResumoExtracao()
    t0 = time.perf_counter()
    tarefas = []
    for l in alvo:
        destino = caminho_npz(cfg, l)
        esperado = meta_esperada(cfg, l)
        if not forcar and cache_valido(destino, esperado):
            resumo.em_cache.append(l["video_id"])
            continue
        video = man.caminho_absoluto(l)
        if not video.exists():
            resumo.falhas[l["video_id"]] = f"arquivo de video nao encontrado: {l['arquivo']}"
            continue
        tarefas.append((l, str(video), str(destino), esperado))

    total = len(tarefas)

    def registrar(i, l, r):
        if r.get("ok"):
            resumo.processados.append(l["video_id"])
        else:
            resumo.falhas[l["video_id"]] = r.get("motivo", "falha")
        if progresso:
            progresso(i, total, l["video_id"], r)

    if processos <= 1 or total <= 1:
        for i, (l, v, d, m) in enumerate(tarefas, 1):
            try:
                r = extrair_um(cfg["mediapipe"], v, d, m)
            except Exception as e:  # um video ruim nao derruba o lote
                r = {"ok": False, "motivo": f"{type(e).__name__}: {e}"}
            registrar(i, l, r)
    else:
        import multiprocessing as mp

        # "spawn" e o unico modo do Windows; usado em todo sistema para o comportamento ser o mesmo
        with ProcessPoolExecutor(
            max_workers=processos, initializer=_inicializar_processo, mp_context=mp.get_context("spawn")
        ) as ex:
            futuros = {ex.submit(extrair_um, dict(cfg["mediapipe"]), v, d, m): l for (l, v, d, m) in tarefas}
            for i, fut in enumerate(as_completed(futuros), 1):
                l = futuros[fut]
                try:
                    r = fut.result()
                except Exception as e:
                    r = {"ok": False, "motivo": f"{type(e).__name__}: {e}"}
                registrar(i, l, r)

    # registra o caminho dos landmarks no manifesto (releitura evita sobrescrever edicoes da interface)
    if resumo.processados or resumo.em_cache:
        man.recarregar()
        feitos = set(resumo.processados) | set(resumo.em_cache)
        mudou = False
        for l in man._linhas:
            if l["video_id"] in feitos:
                from .utils import sha256_arquivo

                npz = caminho_npz(cfg, l)
                rel = man.relativo(npz)
                sha_npz = sha256_arquivo(npz)
                if l.get("landmarks_path") != rel or l.get("sha256_npz") != sha_npz:
                    l["landmarks_path"], l["sha256_npz"] = rel, sha_npz
                    mudou = True
        if mudou:
            man.salvar()
    resumo.segundos = time.perf_counter() - t0
    return resumo
