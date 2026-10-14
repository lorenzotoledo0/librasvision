"""Roda o servico de tempo real REAL (MediaPipe e modelo de verdade) com a rede bloqueada, para o teste
do selo de privacidade (13.9-B). Qualquer tentativa de conexao e registrada em LIBRAS_TESTE_REDE e
falha. Ao sair, grava em LIBRAS_TESTE_MODULOS a lista de modulos importados (teste de arquitetura)."""

import json
import os
import runpy
import socket
import sys

sys.path.insert(0, __file__.rsplit("tests", 1)[0] + "src")
registro = os.environ.get("LIBRAS_TESTE_REDE")


def _bloqueado(*a, **k):
    if registro:
        with open(registro, "a", encoding="utf-8") as f:
            f.write(f"tentativa de rede: {a!r}\n")
    raise OSError("rede bloqueada no teste de privacidade")


_init_original = socket.socket.__init__


SONDA_IPV6 = "_has_ipv6"  # urllib3 (importado pelo TensorFlow) cria um socket local so para saber se ha IPv6


def _criar(self, *a, **k):
    """Criar socket falha sempre. A unica criacao aceita no teste e a sonda de IPv6 do urllib3 ao importar
    o TensorFlow (nao conecta em nada); ela e registrada em LIBRAS_TESTE_SONDAS e tambem falha."""
    import traceback

    pilha = "".join(traceback.format_stack())
    sondas = os.environ.get("LIBRAS_TESTE_SONDAS")
    if SONDA_IPV6 in pilha and "urllib3" in pilha:
        if sondas:
            with open(sondas, "a", encoding="utf-8") as f:
                f.write("sonda de IPv6 do urllib3 (bloqueada)\n")
        raise OSError("socket bloqueado no teste de privacidade")
    _bloqueado("criar socket", *a)


socket.socket.__init__ = _criar
socket.socketpair = _bloqueado
socket.socket.connect = _bloqueado
socket.socket.connect_ex = _bloqueado
socket.socket.sendto = _bloqueado
socket.create_connection = _bloqueado
socket.getaddrinfo = _bloqueado

try:
    sys.argv = ["libras.tempo_real_servico"] + sys.argv[1:]
    runpy.run_module("libras.tempo_real_servico", run_name="__main__")
except SystemExit:
    pass
finally:
    mods = os.environ.get("LIBRAS_TESTE_MODULOS")
    if mods:
        with open(mods, "w", encoding="utf-8") as f:
            json.dump(sorted(sys.modules), f)
