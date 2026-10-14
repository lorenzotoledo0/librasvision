"""Inspecao de videos e clipes: metadados tecnicos e indicadores de qualidade.

Usado tanto pela gravacao (clipe em memoria) quanto pela importacao (arquivo).
Separa ERROS (o video nao serve: rejeitar) de AVISOS (serve, mas merece revisao).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..landmarks import ExtratorHolistic


@dataclass
class InfoVideo:
    largura: int = 0
    altura: int = 0
    fps_nominal: float = 0.0
    fps_medido: float = 0.0
    n_frames: int = 0
    duracao_s: float = 0.0
    brilho_medio: float = 0.0
    taxa_mao: float = 0.0
    taxa_pose: float = 0.0
    erros: list[str] = field(default_factory=list)
    avisos: list[str] = field(default_factory=list)
    # landmarks de TODOS os frames do take (gravacao) e instantes relativos ao 1o frame (s);
    # nao vao para o manifesto: sao salvos no .npz (decisoes D12)
    sequencia: object = field(default=None, repr=False)
    tempos: object = field(default=None, repr=False)

    @property
    def flags(self) -> str:
        return ";".join(self.avisos)

    def campos_manifesto(self) -> dict:
        return {
            "largura": self.largura,
            "altura": self.altura,
            "fps_nominal": f"{self.fps_nominal:.2f}",
            "fps_medido": f"{self.fps_medido:.2f}",
            "n_frames": self.n_frames,
            "duracao_s": f"{self.duracao_s:.3f}",
            "brilho_medio": f"{self.brilho_medio:.1f}",
            "taxa_mao": f"{self.taxa_mao:.3f}",
            "taxa_pose": f"{self.taxa_pose:.3f}",
            "flags_qualidade": self.flags,
        }


def brilho(frame_bgr: np.ndarray) -> float:
    """Luminancia media (0-255), aproximacao Rec.601."""
    f = frame_bgr.astype(np.float32)
    return float((0.114 * f[..., 0] + 0.587 * f[..., 1] + 0.299 * f[..., 2]).mean())


def fps_por_timestamps(ts: list[float]) -> float:
    """FPS efetivo a partir dos instantes de chegada dos frames (segundos)."""
    if len(ts) < 2:
        return 0.0
    dur = ts[-1] - ts[0]
    return (len(ts) - 1) / dur if dur > 0 else 0.0


def indices_amostra(n: int, k: int) -> list[int]:
    if n <= 0:
        return []
    k = min(k, n)
    return sorted({int(round(x)) for x in np.linspace(0, n - 1, k)})


def avaliar_clipe(
    frames: list[np.ndarray],
    timestamps: list[float],
    fps_nominal: float,
    cfg_coleta: dict,
    extrator: ExtratorHolistic | None,
    progresso=None,
) -> InfoVideo:
    """Qualidade de um clipe recem-gravado. TODOS os frames passam pelo MediaPipe; os landmarks
    ficam em info.sequencia (salvos no .npz quando o take e aceito).

    progresso(i, n): chamado a cada frame processado (barra de progresso da interface).
    """
    q = cfg_coleta["qualidade"]
    info = InfoVideo()
    if not frames:
        info.erros.append("nenhum frame capturado")
        return info
    info.altura, info.largura = frames[0].shape[:2]
    info.n_frames = len(frames)
    info.fps_nominal = float(fps_nominal or 0.0)
    info.fps_medido = fps_por_timestamps(timestamps)
    info.duracao_s = (info.n_frames / info.fps_medido) if info.fps_medido > 0 else 0.0
    amostra = indices_amostra(len(frames), 15)
    info.brilho_medio = float(np.mean([brilho(frames[i]) for i in amostra]))
    if extrator is not None:
        from ..landmarks import empilhar

        lms = []
        for i, f in enumerate(frames):
            lms.append(extrator.processar(f)[0])
            if progresso:
                progresso(i + 1, len(frames))
        seq = empilhar(lms, info.largura, info.altura, info.fps_medido)
        ts = np.asarray(timestamps, dtype=np.float64)
        info.sequencia = seq
        info.tempos = (ts - ts[0]) if len(ts) == len(frames) else np.arange(len(frames)) / max(info.fps_medido, 1.0)
        info.taxa_mao = seq.taxa_alguma_mao
        info.taxa_pose = seq.taxa("pose")
    _avisos_comuns(info, q, analisou_landmarks=extrator is not None)
    return info


def _avisos_comuns(info: InfoVideo, q: dict, analisou_landmarks: bool) -> None:
    if info.fps_medido and info.fps_medido < q["fps_minimo"]:
        info.avisos.append(f"fps_baixo({info.fps_medido:.1f})")
    if info.brilho_medio < q["brilho_minimo"]:
        info.avisos.append(f"escuro({info.brilho_medio:.0f})")
    elif info.brilho_medio > q["brilho_maximo"]:
        info.avisos.append(f"claro_demais({info.brilho_medio:.0f})")
    if analisou_landmarks:
        if info.taxa_mao < q["taxa_minima_mao"]:
            info.avisos.append(f"mao_pouco_detectada({info.taxa_mao:.0%})")
        if info.taxa_pose < q["taxa_minima_pose"]:
            info.avisos.append(f"corpo_pouco_detectado({info.taxa_pose:.0%})")


def sondar_arquivo(caminho: Path, cfg, extrator_estatico: ExtratorHolistic | None) -> InfoVideo:
    """Valida um arquivo de video para importacao. Nao altera o arquivo."""
    import cv2

    imp = cfg["importacao"]
    q = cfg["coleta"]["qualidade"]
    info = InfoVideo()
    cap = cv2.VideoCapture(str(caminho))
    if not cap.isOpened():
        info.erros.append("nao foi possivel abrir o video (formato/codec nao suportado ou arquivo corrompido)")
        return info
    info.fps_nominal = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    limite = int(imp["duracao_max_s"] * max(info.fps_nominal, 30) * 1.5) + 10
    frames_amostra: dict[int, np.ndarray] = {}
    brilhos = []
    n = 0
    # Contagem por decodificacao: CAP_PROP_FRAME_COUNT e pouco confiavel em varios codecs.
    total_estimado = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    alvo = set(indices_amostra(total_estimado, imp["amostras_validacao"])) if total_estimado > 0 else set()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if n == 0:
            info.altura, info.largura = frame.shape[:2]
        if n in alvo or (not alvo and n % 10 == 0):
            frames_amostra[n] = frame
        if n % 5 == 0:
            brilhos.append(brilho(frame))
        n += 1
        if n > limite:
            break
    cap.release()
    info.n_frames = n
    info.fps_medido = info.fps_nominal  # para arquivo, o FPS efetivo e o declarado no conteiner
    if n == 0:
        info.erros.append("o video nao tem frames legiveis")
        return info
    if info.fps_nominal <= 0:
        info.erros.append("o video nao informa FPS; nao e possivel calcular duracao")
        return info
    info.duracao_s = n / info.fps_nominal
    info.brilho_medio = float(np.mean(brilhos)) if brilhos else 0.0

    if n > limite:
        info.erros.append(f"video longo demais (> {imp['duracao_max_s']} s); grave um sinal por arquivo")
    elif info.duracao_s < imp["duracao_min_s"]:
        info.erros.append(f"video curto demais ({info.duracao_s:.2f} s < {imp['duracao_min_s']} s)")
    elif info.duracao_s > imp["duracao_max_s"]:
        info.erros.append(f"video longo demais ({info.duracao_s:.1f} s > {imp['duracao_max_s']} s)")
    else:
        lo, hi = imp["duracao_ideal_s"]
        if not (lo <= info.duracao_s <= hi):
            info.avisos.append(f"duracao_fora_do_ideal({info.duracao_s:.1f}s)")
    if info.fps_nominal < imp["fps_min_aceito"]:
        info.erros.append(f"FPS baixo demais ({info.fps_nominal:.1f} < {imp['fps_min_aceito']})")
    if info.altura < imp["altura_min"]:
        info.avisos.append(f"resolucao_baixa({info.largura}x{info.altura})")

    if extrator_estatico is not None and frames_amostra:
        amostras = [frames_amostra[k] for k in sorted(frames_amostra)][: imp["amostras_validacao"]]
        res = [extrator_estatico.processar(f)[0] for f in amostras]
        info.taxa_pose = float(np.mean([r.presenca[0] for r in res]))
        info.taxa_mao = float(np.mean([r.tem_mao for r in res]))
        if info.taxa_pose < imp["taxa_minima_pessoa"]:
            info.erros.append(
                f"pessoa nao detectada na maioria das amostras ({info.taxa_pose:.0%}); "
                "confira enquadramento e iluminacao"
            )
    if not info.erros:
        _avisos_comuns(info, q, analisou_landmarks=extrator_estatico is not None)
    return info
