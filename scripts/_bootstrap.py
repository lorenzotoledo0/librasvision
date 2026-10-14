"""Permite rodar os scripts sem instalar o pacote: coloca src/ no sys.path
e transforma ErroLibras em mensagem limpa (sem traceback) e codigo de saida 2."""

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ / "src") not in sys.path:
    sys.path.insert(0, str(RAIZ / "src"))

import os  # noqa: E402

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # menos ruido do TensorFlow
os.environ.setdefault("GLOG_minloglevel", "2")  # menos ruido do MediaPipe


def sem_tensorflow():
    """Impede que o MediaPipe 0.10.21 carregue o TensorFlow sem necessidade.

    O MediaPipe importa o TensorFlow so para um utilitario opcional de documentacao
    (tasks/python/core/optional_dependencies.py), que ja tem alternativa caso ele falte.
    Bloquear economiza cerca de 500 MB de RAM e ~4 s na abertura das ferramentas de
    coleta, que nao usam o TensorFlow. Nao usar em scripts de treino/inferencia.
    """
    if "tensorflow" not in sys.modules:
        sys.modules["tensorflow"] = None


def rodar(funcao):
    """Executa funcao() tratando erros esperados com mensagem amigavel."""
    from libras.erros import ErroLibras

    try:
        codigo = funcao()
    except ErroLibras as e:
        sys.stderr.write(f"\n[ERRO] {e}\n")
        sys.exit(2)
    except KeyboardInterrupt:
        sys.stderr.write("\nInterrompido.\n")
        sys.exit(130)
    sys.exit(codigo or 0)
