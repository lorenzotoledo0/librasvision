"""SINTETICO: servico falso de tempo real para testes de interface. Nao usa camera nem modelo.
Envia um quadro de informacao e, ao receber "sair", demora 2 s para terminar (simula o
encerramento lento da camera e do TensorFlow, que travava a troca de aba antes do item 7)."""

import sys
import time

sys.path.insert(0, __file__.rsplit("tests", 1)[0] + "src")
from libras.tempo_real_servico import codificar  # noqa: E402

sys.stdout.buffer.write(codificar({"tipo": "info", "modelo": "falso", "classes": ["a", "b"]}))
sys.stdout.buffer.flush()
for linha in sys.stdin:
    if linha.strip() == "sair":
        break
time.sleep(2.0)
