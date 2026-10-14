"""SINTETICO: servico falso de tempo real para o teste de interface do modo "Testar acertos".
Nao usa camera, MediaPipe nem modelo: envia um modelo ficticio (ENSAIO) e quadros com
probabilidades fixas. Os comandos recebidos sao gravados no arquivo LIBRAS_TESTE_COMANDOS."""

import os
import sys
import threading
import time

sys.path.insert(0, __file__.rsplit("tests", 1)[0] + "src")
from libras.tempo_real_servico import codificar  # noqa: E402

arq = os.environ.get("LIBRAS_TESTE_COMANDOS")
sair = threading.Event()


def comandos():
    for linha in sys.stdin:
        if arq:
            with open(arq, "a", encoding="utf-8") as f:
                f.write(linha)
        if linha.strip() == "sair":
            break
    sair.set()


threading.Thread(target=comandos, daemon=True).start()
out = sys.stdout.buffer
out.write(
    codificar(
        {
            "tipo": "info",
            "modelo": "modelo_ficticio",
            "pasta": "x",
            "classes": ["a", "b", "nenhum_sinal"],
            "nome_nenhum_sinal": "nenhum_sinal",
            "videos": 6,
            "sinalizadores": 1,
            "lista_sinalizadores": ["S01"],
            "ensaio": True,
            "parametros": {"limiar": 0.6, "suavizacao": 5, "confirmacoes": 3},
        }
    )
)
out.flush()
while not sair.is_set():
    out.write(
        codificar(
            {
                "tipo": "quadro",
                "classe": "a",
                "confianca": 0.7,
                "motivo": "",
                "aceito": None,
                "texto": [],
                "probs": [0.7, 0.2, 0.1],
                "etapa": {"etapa": "aguardando", "texto": "Aguardando predições iguais (1 de 3): a"},
                "tempos": {"fps_camera": 30.0, "fps_processado": 15.0},
            }
        )
    )
    out.flush()
    time.sleep(0.05)
