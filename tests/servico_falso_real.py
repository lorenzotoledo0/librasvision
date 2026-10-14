"""SINTETICO: servico falso de tempo real para os testes de interface do app real.

Nao usa camera, MediaPipe nem modelo. Envia o mesmo protocolo do servico real (D15) com quadros de
imagem cinza gerada por codigo e, se pedido, palavras "confirmadas" e medidas de enquadramento
FICTICIAS (variaveis de ambiente). Serve so para testar a interface; nao e predicao de modelo.
    LIBRAS_FALSO_PALAVRAS   "biblioteca,prova"  (uma por segundo, depois de 1 s)
    LIBRAS_FALSO_ENQ        JSON do campo "enquadramento"
    LIBRAS_FALSO_ERRO       mensagem de erro de camera (envia "erro" e termina)
    LIBRAS_TESTE_COMANDOS   arquivo onde os comandos recebidos sao gravados
"""

import json
import os
import sys
import threading
import time

sys.path.insert(0, __file__.rsplit("tests", 1)[0] + "src")
import cv2  # noqa: E402
import numpy as np  # noqa: E402

from libras.tempo_real_servico import codificar  # noqa: E402

args = sys.argv[1:]
pacote = args[args.index("--pacote") + 1] if "--pacote" in args else None
arq = os.environ.get("LIBRAS_TESTE_COMANDOS")
sair = threading.Event()
out = sys.stdout.buffer


def comandos():
    for linha in sys.stdin:
        if arq:
            with open(arq, "a", encoding="utf-8") as f:
                f.write(linha)
        if linha.strip() == "sair":
            break
    sair.set()


threading.Thread(target=comandos, daemon=True).start()
cartao = json.loads(open(os.path.join(pacote, "cartao.json"), encoding="utf-8").read()) if pacote else {}
out.write(
    codificar(
        {
            "tipo": "info",
            "modelo": cartao.get("id", "x"),
            "classes": cartao.get("classes", []),
            "nome_nenhum_sinal": cartao.get("nome_nenhum_sinal"),
            "ensaio": False,
            "parametros": {"limiar": 0.6, "suavizacao": 5, "confirmacoes": 3},
        }
    )
)
erro = os.environ.get("LIBRAS_FALSO_ERRO")
if erro:
    out.write(codificar({"tipo": "erro", "mensagem": erro}))
    out.flush()
    sys.exit(3)
out.write(codificar({"tipo": "info", "camera": {"indice": 0, "backend": "FALSO", "largura": 640, "altura": 480}}))
out.flush()
img = np.full((480, 640, 3), 128, np.uint8)
img[:, :320] = 200
ok, jpg = cv2.imencode(".jpg", img)
palavras = [p for p in os.environ.get("LIBRAS_FALSO_PALAVRAS", "").split(",") if p]
enq = json.loads(os.environ.get("LIBRAS_FALSO_ENQ", "null") or "null")
t0 = time.time()
n = 0
while not sair.is_set():
    n += 1
    aceito = None
    k = int(time.time() - t0) - 1
    if palavras and 0 <= k < len(palavras) and not getattr(sys.modules[__name__], f"_dado_{k}", False):
        setattr(sys.modules[__name__], f"_dado_{k}", True)
        aceito = palavras[k]
    out.write(
        codificar(
            {
                "tipo": "quadro",
                "classe": aceito,
                "confianca": 0.9 if aceito else 0.0,
                "motivo": "" if aceito else "baixa_confianca",
                "aceito": aceito,
                "texto": [],
                "etapa": {"etapa": "confirmado" if aceito else "baixa_confianca", "texto": ""},
                "probs": None,
                "tempos": {"fps_camera": 30.0, "fps_processado": 15.0, "memoria_mb": 900.0},
                "tempos_recentes": {"ponta_a_ponta": {"mediana_ms": 80.0, "p95_ms": 120.0}},
                "quadros": {"capturados": 2 * n, "processados": n, "descartados": n},
                "enquadramento": enq,
            },
            jpg.tobytes(),
        )
    )
    out.flush()
    time.sleep(0.05)
