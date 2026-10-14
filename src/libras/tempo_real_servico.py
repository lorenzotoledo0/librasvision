"""Servico de reconhecimento em tempo real, executado em PROCESSO SEPARADO pela interface.

Motivo: o TensorFlow ocupa cerca de 1 GB; isolado em outro processo, ele so e carregado quando a
aba Tempo real e aberta e e liberado ao sair dela. A aba de coleta continua leve.

Protocolo (stdout binario, um quadro por mensagem):
    b"LVQ1" + uint32(len(json)) + json(utf-8) + uint32(len(jpeg)) + jpeg
O json tem tipo "quadro" (predicao, confianca, top-3, texto, tempos) ou "erro"/"info".
Comandos pela entrada padrao, um por linha: "landmarks 0|1", "limpar", "salvar", "sair" e
"parametros limiar=0.6 suavizacao=5 confirmacoes=3" (ajuste so durante o teste ao vivo: nao altera o
config nem o modelo). Cada quadro leva as probabilidades de TODAS as classes e a etapa da decisao.
Nada e predito sem modelo treinado: nesse caso o servico envia "erro" e termina.

Campos opcionais do quadro (item 13; quem nao os conhece os ignora, o protocolo continua o mesmo):
    "quadros": capturados, processados e descartados desde que a camera ligou (capturados =
               processados + descartados; descartar o frame que chega durante o processamento e normal)
    "tempos_recentes": mediana e p95 por etapa nos ultimos 10 s
    "enquadramento": pessoa, ombros, lados com mao cortada, brilho, caixa da cabeca e dos ombros
                     (libras.enquadramento; coordenadas da imagem NAO espelhada)

Modo app real (--pacote): o modelo vem de um pacote publicado, conferido pelo hash antes de carregar,
e os parametros de decisao vem do cartao do pacote. Com --sem-disco nada e gravado em disco (nem log
em arquivo, nem medicao): a interface do app real promete isso ao usuario (13.9-B).
"""

from __future__ import annotations

import json
import logging
import struct
import sys
import threading
import time

import numpy as np

MAGICO = b"LVQ1"


def codificar(meta: dict, jpeg: bytes = b"") -> bytes:
    j = json.dumps(meta, ensure_ascii=False, default=float).encode("utf-8")
    return MAGICO + struct.pack("<I", len(j)) + j + struct.pack("<I", len(jpeg)) + jpeg


class Decodificador:
    """Acumula bytes e devolve as mensagens completas (meta, jpeg)."""

    def __init__(self):
        self.buf = bytearray()

    def alimentar(self, dados: bytes) -> list[tuple[dict, bytes]]:
        self.buf += dados
        out = []
        while True:
            i = self.buf.find(MAGICO)
            if i < 0:
                self.buf = self.buf[-3:]
                return out
            if i > 0:
                del self.buf[:i]
            if len(self.buf) < 8:
                return out
            nj = struct.unpack("<I", self.buf[4:8])[0]
            if len(self.buf) < 8 + nj + 4:
                return out
            nimg = struct.unpack("<I", self.buf[8 + nj : 12 + nj])[0]
            fim = 12 + nj + nimg
            if len(self.buf) < fim:
                return out
            meta = json.loads(bytes(self.buf[8 : 8 + nj]).decode("utf-8"))
            img = bytes(self.buf[12 + nj : fim])
            del self.buf[:fim]
            out.append((meta, img))


def executar_servico(
    cfg,
    saida,
    entrada=None,
    abrir_camera=None,
    max_fps_envio: float = 20.0,
    duracao_max_s: float | None = None,
    gravar_em_disco: bool = True,
    extrator=None,
) -> int:
    """Laco principal. `saida`: stream binario; `entrada`: stream de texto com comandos."""
    import cv2

    from .camera import Captura
    from .camera import abrir_camera_ou_video as abrir_padrao
    from .config_real import PADRAO as PADRAO_REAL
    from .decisao import PARAMETROS_PADRAO as PAD
    from .decisao import etapa_decisao
    from .enquadramento import MedidorEnquadramento
    from .erros import ErroLibras
    from .inferencia import Cronometro, Processador, Transcritor, carregar_modelo, localizar_modelo, salvar_medicao
    from .utils import memoria_processo_mb

    log = logging.getLogger("libras.tempo_real")
    trava = threading.Lock()

    def enviar(meta, jpeg=b""):
        with trava:
            saida.write(codificar(meta, jpeg))
            saida.flush()

    try:
        modelo = carregar_modelo(localizar_modelo(cfg))
    except ErroLibras as e:
        enviar({"tipo": "erro", "mensagem": str(e), "sem_modelo": True})
        return 2
    m = modelo.meta
    cfg_tr0 = dict(cfg.get("tempo_real") or {})
    enviar(
        {
            "tipo": "info",
            "modelo": modelo.pasta.name,
            "pasta": str(modelo.pasta),
            "classes": modelo.classes,
            "nome_nenhum_sinal": m.get("nome_nenhum_sinal"),
            "videos": m.get("videos"),
            "sinalizadores": len(m.get("sinalizadores", [])),
            "lista_sinalizadores": m.get("sinalizadores", []),
            "ensaio": bool(m.get("ensaio")),
            "parametros": {
                "limiar": float(cfg_tr0.get("limiar_confianca", PAD["limiar_confianca"])),
                "suavizacao": int(cfg_tr0.get("suavizacao", PAD["suavizacao"])),
                "confirmacoes": int(cfg_tr0.get("frames_estaveis", PAD["frames_estaveis"])),
            },
        }
    )
    try:
        cap, info = (abrir_camera or abrir_padrao)(cfg["coleta"]["camera"])
    except ErroLibras as e:
        enviar({"tipo": "erro", "mensagem": str(e)})
        return 3
    captura = Captura(cap, info)  # a leitura comeca depois do MediaPipe pronto (contadores de 13.8)
    cfg_tr = dict(cfg.get("tempo_real") or {})
    janela_lat = float((cfg.get("informacoes_tecnicas") or PADRAO_REAL["informacoes_tecnicas"])["janela_latencia_s"])
    cron = Cronometro(janela_lat)
    medidor = MedidorEnquadramento(cfg.get("enquadramento") or PADRAO_REAL["enquadramento"], m, cfg_tr)
    enviar(
        {
            "tipo": "info",
            "camera": {"indice": info.indice, "backend": info.backend, "largura": info.largura, "altura": info.altura},
        }
    )
    trans = Transcritor(
        int(cfg_tr.get("frames_estaveis", PAD["frames_estaveis"])), int(cfg_tr.get("max_frases", PAD["max_frases"]))
    )
    estado = {"landmarks": False, "ultimo_envio": 0.0, "sair": False, "ultimo_pred": None}
    from .landmarks import desenhar_esqueleto

    def publicar(frame, res, lm, pred, t_chegada, ms_mp):
        enq = medidor.medir(frame, lm, t_chegada)
        aceito = None
        if pred is not None:
            estado["ultimo_pred"] = pred
            aceito = trans.atualizar(pred)
        agora = time.perf_counter()
        if agora - estado["ultimo_envio"] < 1.0 / max_fps_envio and aceito is None:
            return
        estado["ultimo_envio"] = agora
        img = desenhar_esqueleto(frame, lm.pose, lm.mao_esq, lm.mao_dir, lm.presenca) if estado["landmarks"] else frame
        img = cv2.flip(img, 1)  # espelhado SO para exibicao
        ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        p = estado["ultimo_pred"]
        cand, cont, nec = trans.estado()
        meta = {
            "tipo": "quadro",
            "classe": p.classe if p else None,
            "confianca": p.confianca if p else 0.0,
            "motivo": p.motivo if p else "aquecendo",
            "aceito": aceito,
            "texto": list(trans.texto),
            "etapa": etapa_decisao(p.motivo if p else "aquecendo", cand, cont, nec, aceito),
            "probs": None if p is None or p.probs is None else [float(x) for x in p.probs],
        }
        if p is not None and p.probs is not None:
            ordem = np.argsort(p.probs)[::-1][:3]
            meta["top"] = [[modelo.classes[i], float(p.probs[i])] for i in ordem]
        cron.registrar(ponta_a_ponta=(time.perf_counter() - t_chegada) * 1000)
        if cron.frames % 30 == 0:
            mm = memoria_processo_mb()
            if mm:
                cron.memoria_mb.append(mm)
        r = cron.resumo()
        meta["tempos"] = {
            "fps_camera": round(captura.fps_atual, 1),
            "fps_processado": r["fps_processado"],
            **{k: r[k]["mediana_ms"] for k in ("mediapipe", "modelo", "ponta_a_ponta") if k in r},
            "memoria_mb": cron.memoria_mb[-1] if cron.memoria_mb else None,
        }
        meta["tempos_recentes"] = cron.resumo_recente()
        meta["quadros"] = {"capturados": proc.n_atual, "processados": cron.frames, "descartados": proc.descartados}
        meta["enquadramento"] = enq
        enviar(meta, jpg.tobytes() if ok else b"")

    proc = Processador(captura, modelo, cfg_tr, cron, publicar, extrator)  # extrator: so nos testes (SINTETICO)
    captura.iniciar()

    def ler_comandos():
        for linha in entrada:
            cmd = linha.strip().split()
            if not cmd:
                continue
            if cmd[0] == "landmarks":
                estado["landmarks"] = len(cmd) > 1 and cmd[1] == "1"
            elif cmd[0] == "limpar":
                trans.limpar()
            elif cmd[0] == "salvar":
                if not gravar_em_disco:
                    continue  # app real: nada e gravado em disco
                p = salvar_medicao(cfg, cron, modelo, "camera", {"camera": {"backend": info.backend}})
                enviar({"tipo": "info", "medicao_salva": str(p)})
            elif cmd[0] == "parametros":
                kv = dict(x.split("=", 1) for x in cmd[1:] if "=" in x)
                try:
                    proc.rec.ajustar(
                        limiar=float(kv["limiar"]) if "limiar" in kv else None,
                        suavizacao=int(kv["suavizacao"]) if "suavizacao" in kv else None,
                    )
                    if "confirmacoes" in kv:
                        trans.n = max(1, int(kv["confirmacoes"]))
                except ValueError:
                    log.warning("Parametros invalidos: %s", linha.strip())
            elif cmd[0] == "sair":
                break
        estado["sair"] = True
        proc.parar()

    if entrada is not None:
        threading.Thread(target=ler_comandos, daemon=True, name="comandos").start()
    if duracao_max_s:
        threading.Timer(duracao_max_s, proc.parar).start()
    try:
        proc.executar()
    finally:
        captura.parar()
        if cron.frames > 30 and gravar_em_disco:
            try:
                salvar_medicao(cfg, cron, modelo, "camera", {"camera": {"backend": info.backend}})
            except Exception:  # noqa: BLE001
                log.exception("Falha ao salvar medicao")
    if proc.erro:
        enviar({"tipo": "erro", "mensagem": proc.erro})
        return 4
    return 0


def main() -> int:
    import argparse
    import os

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--modelo", default=None)
    ap.add_argument("--video", default=None, help="usa um arquivo de video no lugar da webcam (demonstracao/teste)")
    ap.add_argument("--camera", type=int, default=None, help="indice da camera (padrao: o do config)")
    ap.add_argument("--pacote", default=None, help="app real: pasta de um pacote publicado (publicados/<id>)")
    ap.add_argument("--config-real", default=None, help="app real: configs/app_real.yaml")
    ap.add_argument("--sem-disco", action="store_true", help="nao grava nada em disco (log so na saida de erro)")
    a = ap.parse_args()
    if a.sem_disco:
        logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    if a.pacote:
        from . import config_real
        from .pacote import ErroPacote, verificar_pacote

        try:
            cartao = verificar_pacote(a.pacote)  # confere o hash ANTES de carregar o modelo
        except ErroPacote as e:
            sys.stdout.buffer.write(
                codificar({"tipo": "erro", "mensagem": str(e), "sem_modelo": True, "pacote_invalido": True})
            )
            sys.stdout.buffer.flush()
            return 2
        cfg = config_real.config_servico(config_real.carregar(a.config_real), a.pacote, cartao, a.camera)
    else:
        from .config import carregar_config

        cfg = carregar_config(a.config)
        if a.modelo:
            cfg.setdefault("tempo_real", {})["modelo"] = a.modelo
        if a.camera is not None:
            cfg["coleta"]["camera"]["indice"] = a.camera
    if not a.sem_disco:
        from .registro import configurar_log

        configurar_log(cfg.caminho("logs"), "tempo_real")
    abrir = None
    if a.video:
        from .camera import abrir_arquivo_como_camera

        abrir = lambda _cfg_camera: abrir_arquivo_como_camera(a.video)  # noqa: E731
    return executar_servico(cfg, sys.stdout.buffer, sys.stdin, abrir_camera=abrir, gravar_em_disco=not a.sem_disco)


if __name__ == "__main__":
    sys.exit(main())
