"""Inferencia em tempo real: buffer deslizante, MESMA normalizacao do treino, modelo, decisao.

Fluxo por frame (identico ao do treino, ver normalizacao.py):
    frame BGR (nao espelhado) -> MediaPipe Holistic -> buffer com instantes reais
    -> reamostrar para fps_alvo -> ultimos N frames -> preparar_janela -> LSTM
    -> media movel das probabilidades -> limiar / nenhum_sinal -> texto

Os parametros de MediaPipe, janela, normalizacao e vetor de features vem dos METADADOS
do modelo (metadados.json), nao do config atual: o tempo real usa exatamente o que o
modelo viu no treino. Sem modelo, nada e predito.
"""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .decisao import PARAMETROS_PADRAO as _PAD
from .erros import ErroDadosInsuficientes, ErroLibras
from .landmarks import LandmarksFrame
from .normalizacao import EspecFeatures, Partes, preparar_janela, reamostrar


# ======================================================================
# Modelo
# ======================================================================
@dataclass
class ModeloCarregado:
    pasta: Path
    meta: dict
    espec: EspecFeatures
    classes: list[str]
    keras: object = None

    @property
    def comprimento(self) -> int:
        return int(self.meta["janela"]["comprimento"])

    @property
    def fps_alvo(self) -> float:
        return float(self.meta["janela"]["fps_alvo"])


def localizar_modelo(cfg) -> Path:
    """Pasta indicada em tempo_real.modelo ou o modelo final mais recente em runs/."""
    alvo = (cfg.get("tempo_real") or {}).get("modelo")
    if alvo:
        p = Path(alvo)
        p = p if p.is_absolute() else cfg.raiz / p
        if not (p / "metadados.json").exists():
            raise ErroDadosInsuficientes(f"Nao ha modelo em {p} (faltam modelo.keras/metadados.json).")
        return p
    runs = cfg.caminho("runs")
    candidatos = sorted(runs.glob("*/metadados.json")) if runs.exists() else []
    candidatos = [c.parent for c in candidatos if (c.parent / "modelo.keras").exists()]
    if not candidatos:
        raise ErroDadosInsuficientes(
            "Nenhum modelo treinado encontrado. Depois de coletar e extrair os dados, rode: "
            "scripts/treinar.py --nome final --final"
        )
    return candidatos[-1]  # nomes comecam com data_hora: o ultimo e o mais recente


def carregar_modelo(pasta: Path, carregar_keras: bool = True) -> ModeloCarregado:
    pasta = Path(pasta)
    meta = json.loads((pasta / "metadados.json").read_text(encoding="utf-8"))
    espec = EspecFeatures.de_dict(meta["features"])
    if espec.dim != meta["features"]["dim"]:
        raise ErroLibras("metadados.json inconsistente: dimensao do vetor de features nao confere.")
    m = ModeloCarregado(pasta, meta, espec, list(meta["classes"]))
    if carregar_keras:
        import tensorflow as tf

        m.keras = tf.keras.models.load_model(pasta / "modelo.keras", compile=False)
        esperado = (None, m.comprimento, espec.dim)
        if tuple(m.keras.input_shape) != esperado:
            raise ErroLibras(f"O modelo espera entrada {m.keras.input_shape}, metadados indicam {esperado}.")
        m.keras.predict_on_batch(np.zeros((1, m.comprimento, espec.dim), np.float32))  # aquecimento
    return m


# ======================================================================
# Reconhecedor (independente de camera e de interface)
# ======================================================================
@dataclass
class Predicao:
    t: float  # instante do frame mais recente usado
    classe: str | None  # classe aceita apos limiar/suavizacao (None = nada a mostrar)
    confianca: float
    probs: np.ndarray | None  # probabilidades suavizadas
    motivo: str = ""  # "", "sem_maos", "sem_pose", "baixa_confianca", "nenhum_sinal", "aquecendo"
    ms_preparo: float = 0.0
    ms_modelo: float = 0.0


@dataclass
class Reconhecedor:
    modelo: ModeloCarregado
    cfg_tr: dict = field(default_factory=dict)
    preditor: object = None  # funcao (1, L, D) -> (1, C); padrao: modelo.keras.predict_on_batch

    def __post_init__(self):
        L = self.modelo.comprimento
        fps = self.modelo.fps_alvo
        # guarda ~1,5x a duracao da janela (com folga para FPS irregular)
        self.duracao_buffer = 1.5 * L / fps
        self._frames: deque = deque()
        self._probs: deque = deque(maxlen=max(1, int(self.cfg_tr.get("suavizacao", _PAD["suavizacao"]))))
        self._contador = 0
        self.largura = self.altura = 0
        if self.preditor is None:
            if self.modelo.keras is None:
                raise ErroDadosInsuficientes("Modelo nao carregado: nenhuma predicao e feita sem modelo treinado.")
            self.preditor = self.modelo.keras.predict_on_batch

    def limpar(self) -> None:
        self._frames.clear()
        self._probs.clear()

    def ajustar(self, limiar: float | None = None, suavizacao: int | None = None) -> None:
        """Ajuste durante o teste ao vivo (item 10). Nao altera o config nem o modelo."""
        self.cfg_tr = dict(self.cfg_tr)
        if limiar is not None:
            self.cfg_tr["limiar_confianca"] = float(limiar)
        if suavizacao is not None and int(suavizacao) != self._probs.maxlen:
            self.cfg_tr["suavizacao"] = int(suavizacao)
            self._probs = deque(list(self._probs)[-int(suavizacao) :], maxlen=max(1, int(suavizacao)))

    def adicionar(self, lm: LandmarksFrame, t: float, largura: int, altura: int) -> Predicao | None:
        """Acrescenta um frame processado. Retorna uma Predicao quando o modelo roda neste frame."""
        self.largura, self.altura = largura, altura
        self._frames.append((t, lm))
        while self._frames and t - self._frames[0][0] > self.duracao_buffer:
            self._frames.popleft()
        self._contador += 1
        if self._contador % max(1, int(self.cfg_tr.get("predizer_a_cada", 1))):
            return None
        return self.predizer()

    def _janela_crua(self) -> Partes | None:
        L, fps = self.modelo.comprimento, self.modelo.fps_alvo
        if len(self._frames) < 2:
            return None
        tempos = np.array([t for t, _ in self._frames])
        if tempos[-1] - tempos[0] < (L - 1) / fps - 1e-6:
            return None  # ainda nao ha 1 janela de historia
        lms = [f for _, f in self._frames]
        p = Partes(
            np.stack([f.pose for f in lms]),
            np.stack([f.mao_esq for f in lms]),
            np.stack([f.mao_dir for f in lms]),
            np.stack([f.face for f in lms]),
            np.stack([f.presenca for f in lms]),
        )
        r = reamostrar(p, tempos, fps)
        return r.fatia(r.n - L, r.n)

    def predizer(self) -> Predicao:
        t_ult = self._frames[-1][0] if self._frames else 0.0
        t0 = time.perf_counter()
        crua = self._janela_crua()
        if crua is None:
            return Predicao(t_ult, None, 0.0, None, "aquecendo")
        tr = self.cfg_tr
        frac_mao = float((crua.presenca[:, 1] | crua.presenca[:, 2]).mean())
        if frac_mao < float(tr.get("min_fracao_mao", _PAD["min_fracao_mao"])):
            self._probs.clear()
            return Predicao(t_ult, None, 0.0, None, "sem_maos", (time.perf_counter() - t0) * 1000)
        res = preparar_janela(crua, self.largura, self.altura, self.modelo.espec, self.modelo.meta["normalizacao"])
        if not res.valida:
            return Predicao(t_ult, None, 0.0, None, "sem_pose", (time.perf_counter() - t0) * 1000)
        ms_prep = (time.perf_counter() - t0) * 1000
        t1 = time.perf_counter()
        p = np.asarray(self.preditor(res.X[None].astype(np.float32)))[0]
        ms_mod = (time.perf_counter() - t1) * 1000
        self._probs.append(p)
        ps = np.mean(self._probs, axis=0)
        k = int(np.argmax(ps))
        conf = float(ps[k])
        classe = self.modelo.classes[k]
        if classe == self.modelo.meta.get("nome_nenhum_sinal"):
            return Predicao(t_ult, None, conf, ps, "nenhum_sinal", ms_prep, ms_mod)
        if conf < float(tr.get("limiar_confianca", _PAD["limiar_confianca"])):
            return Predicao(t_ult, None, conf, ps, "baixa_confianca", ms_prep, ms_mod)
        return Predicao(t_ult, classe, conf, ps, "", ms_prep, ms_mod)


class Transcritor:
    """Transforma a sequencia de predicoes em texto: aceita um sinal apos N predicoes iguais
    seguidas e nao repete o mesmo sinal ate que haja uma pausa (nenhum sinal)."""

    def __init__(self, frames_estaveis: int = _PAD["frames_estaveis"], max_frases: int = _PAD["max_frases"]):
        self.n = max(1, frames_estaveis)
        self.max = max_frases
        self.texto: list[str] = []
        self._cand, self._cont = None, 0
        self._ultimo_aceito: str | None = None

    def atualizar(self, pred: Predicao) -> str | None:
        """Retorna a classe recem-aceita (ou None)."""
        c = pred.classe
        if c is None:
            self._cand, self._cont = None, 0
            if pred.motivo in ("sem_maos", "nenhum_sinal"):
                self._ultimo_aceito = None  # pausa: o mesmo sinal pode ser repetido depois
            return None
        if c == self._cand:
            self._cont += 1
        else:
            self._cand, self._cont = c, 1
        if self._cont >= self.n and c != self._ultimo_aceito:
            self._ultimo_aceito = c
            self.texto.append(c)
            self.texto = self.texto[-self.max :]
            return c
        return None

    def limpar(self) -> None:
        self.texto.clear()
        self._cand, self._cont, self._ultimo_aceito = None, 0, None

    def estado(self) -> tuple[str | None, int, int]:
        """(candidato atual, predicoes iguais seguidas, necessarias para confirmar)."""
        return self._cand, self._cont, self.n


# ======================================================================
# Medicao de latencia
# ======================================================================
class Cronometro:
    """Acumula tempos por etapa (ms) e gera resumo com mediana e p95."""

    ETAPAS = ("captura_ate_inicio", "mediapipe", "preparo", "modelo", "ponta_a_ponta")

    def __init__(self, janela_recente_s: float = 10.0):
        self.dados: dict[str, list[float]] = {e: [] for e in self.ETAPAS}
        self.frames = 0
        self.inicio = time.perf_counter()
        self.memoria_mb: list[float] = []
        self.janela_recente_s = float(janela_recente_s)
        self._recentes: dict[str, deque] = {}

    def registrar(self, **ms) -> None:
        agora = time.perf_counter()
        for k, v in ms.items():
            if v is not None:
                self.dados.setdefault(k, []).append(float(v))
                d = self._recentes.setdefault(k, deque())
                d.append((agora, float(v)))
                while d and agora - d[0][0] > self.janela_recente_s:
                    d.popleft()

    def resumo_recente(self) -> dict:
        """Mediana e p95 dos ultimos `janela_recente_s` segundos, por etapa (barra tecnica do app real, 13.8)."""
        agora = time.perf_counter()
        out = {}
        for k, d in self._recentes.items():
            v = [x for t, x in d if agora - t <= self.janela_recente_s]
            if v:
                a = np.array(v)
                out[k] = {"mediana_ms": round(float(np.median(a)), 1), "p95_ms": round(float(np.percentile(a, 95)), 1)}
        return out

    def resumo(self) -> dict:
        dur = time.perf_counter() - self.inicio
        out = {
            "frames_processados": self.frames,
            "segundos": round(dur, 1),
            "fps_processado": round(self.frames / dur, 2) if dur > 0 else 0.0,
            "memoria_mb_max": round(max(self.memoria_mb), 1) if self.memoria_mb else None,
        }
        for k, v in self.dados.items():
            if v:
                a = np.array(v)
                out[k] = {
                    "n": len(a),
                    "mediana_ms": round(float(np.median(a)), 2),
                    "p95_ms": round(float(np.percentile(a, 95)), 2),
                    "media_ms": round(float(a.mean()), 2),
                }
        return out


def salvar_medicao(cfg, cron: Cronometro, modelo: ModeloCarregado, origem: str, extra: dict | None = None) -> Path:
    from .utils import carimbo, escrever_texto_atomico, hardware, versoes

    dados = {
        "origem": origem,
        "modelo": str(modelo.pasta),
        "hardware": hardware(),
        "versoes": versoes(),
        "mediapipe": modelo.meta["mediapipe"],
        "tempo_real": cfg.get("tempo_real"),
        **cron.resumo(),
        **(extra or {}),
    }
    p = cfg.caminho("runs") / "tempo_real" / f"medicao_{carimbo()}.json"
    escrever_texto_atomico(p, json.dumps(dados, indent=2, ensure_ascii=False, default=float))
    return p


# ======================================================================
# Execucao sobre um arquivo de video (medicao reprodutivel, sem interface)
# ======================================================================
def processar_video(cfg, modelo: ModeloCarregado, caminho: Path, progresso=None) -> tuple[list[dict], Cronometro]:
    """Roda o pipeline de tempo real sobre um video gravado, usando os instantes do video.

    Mede o custo de cada etapa nesta maquina (o video e lido do disco, entao 'captura'
    nao e medida aqui). Retorna a linha do tempo das predicoes.
    """
    import cv2

    from .landmarks import ExtratorHolistic, ler_frames
    from .utils import memoria_processo_mb

    cap = cv2.VideoCapture(str(caminho))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    cap.release()
    if fps <= 0:
        raise ErroLibras(f"O video {caminho} nao informa FPS.")
    rec = Reconhecedor(modelo, cfg.get("tempo_real") or {})
    trans = Transcritor(int((cfg.get("tempo_real") or {}).get("frames_estaveis", _PAD["frames_estaveis"])))
    cron = Cronometro()
    linhas = []
    with ExtratorHolistic(modelo.meta["mediapipe"], modo_estatico=False) as ext:
        for i, frame in enumerate(ler_frames(caminho)):
            t_video = i / fps
            t0 = time.perf_counter()
            lm, _ = ext.processar(frame)
            ms_mp = (time.perf_counter() - t0) * 1000
            alt, larg = frame.shape[:2]
            pred = rec.adicionar(lm, t_video, larg, alt)
            cron.frames += 1
            total = (time.perf_counter() - t0) * 1000
            cron.registrar(mediapipe=ms_mp)
            if pred is not None:
                aceito = trans.atualizar(pred)
                cron.registrar(
                    preparo=pred.ms_preparo,
                    modelo=pred.ms_modelo if pred.probs is not None else None,
                    ponta_a_ponta=total,
                )
                linhas.append(
                    {
                        "frame": i,
                        "t_video_s": round(t_video, 3),
                        "classe": pred.classe or "",
                        "confianca": round(pred.confianca, 4),
                        "motivo": pred.motivo,
                        "aceito": aceito or "",
                        "ms_total": round(total, 2),
                    }
                )
            if i % 30 == 0:
                m = memoria_processo_mb()
                if m:
                    cron.memoria_mb.append(m)
            if progresso:
                progresso(i)
    return linhas, cron


# ======================================================================
# Processador continuo (camera -> MediaPipe -> reconhecedor), sem interface
# ======================================================================
class Processador:
    """Le o ultimo frame da captura, roda MediaPipe + reconhecedor e entrega o resultado a
    funcao `publicar(frame, resultado_mediapipe, LandmarksFrame, pred, t_chegada, ms_mediapipe)`.
    Frames que chegam enquanto o anterior e processado sao descartados (sempre o mais recente)."""

    def __init__(self, captura, modelo: ModeloCarregado, cfg_tr: dict, cron: Cronometro, publicar, extrator=None):
        from .landmarks import ExtratorHolistic

        self.captura = captura
        self.ext = extrator or ExtratorHolistic(modelo.meta["mediapipe"], modo_estatico=False)
        self.rec = Reconhecedor(modelo, cfg_tr)
        self.cron = cron
        self.publicar = publicar
        self._parar = False
        self.erro: str | None = None
        self.n_atual = 0  # contador da captura no frame em processamento (frames capturados desde que a camera ligou)

    @property
    def descartados(self) -> int:
        """Frames que chegaram enquanto o anterior era processado (comportamento normal de D15)."""
        return max(0, self.n_atual - self.cron.frames)

    def executar(self) -> None:
        ultimo_n = -1
        try:
            while not self._parar:
                frame, n, t_chegada = self.captura.ultimo_com_tempo()
                if self.captura.erro:
                    self.erro = self.captura.erro
                    break
                if frame is None or n == ultimo_n:
                    time.sleep(0.002)
                    continue
                ultimo_n = n
                self.n_atual = n
                t0 = time.perf_counter()
                lm, res = self.ext.processar(frame)
                t1 = time.perf_counter()
                alt, larg = frame.shape[:2]
                pred = self.rec.adicionar(lm, t_chegada, larg, alt)
                self.cron.frames += 1
                self.cron.registrar(captura_ate_inicio=(t0 - t_chegada) * 1000, mediapipe=(t1 - t0) * 1000)
                if pred is not None:
                    self.cron.registrar(
                        preparo=pred.ms_preparo, modelo=pred.ms_modelo if pred.probs is not None else None
                    )
                self.publicar(frame, res, lm, pred, t_chegada, (t1 - t0) * 1000)
        except Exception as e:  # noqa: BLE001 - repassado a quem chamou
            self.erro = f"{type(e).__name__}: {e}"
            import logging

            logging.getLogger("libras.tempo_real").exception("Falha no processamento")
        finally:
            self.ext.fechar()

    def parar(self) -> None:
        self._parar = True
