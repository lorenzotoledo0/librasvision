"""Log em arquivo e tratamento global de excecoes.

- runs/logs/app.log: log com rotacao (1 MB x 5 arquivos), com data, nivel, thread e modulo;
- runs/logs/falhas_nativas.log: faulthandler (registra quedas dentro de bibliotecas em C,
  como MediaPipe e OpenCV, que nao geram excecao Python);
- sys.excepthook e threading.excepthook: toda excecao nao tratada vai para o log e, se
  houver interface, para um dialogo (via funcao registrada com definir_notificador).
"""

from __future__ import annotations

import faulthandler
import logging
import sys
import threading
from logging.handlers import RotatingFileHandler
from pathlib import Path

_estado = {"configurado": False, "arquivo": None, "notificar": None, "falhas": None}


def configurar_log(pasta_logs: Path, nome: str = "app", nivel: int = logging.INFO) -> Path:
    """Idempotente. Retorna o caminho do arquivo de log."""
    if _estado["configurado"]:
        return _estado["arquivo"]
    pasta_logs = Path(pasta_logs)
    pasta_logs.mkdir(parents=True, exist_ok=True)
    arq = pasta_logs / f"{nome}.log"
    h = RotatingFileHandler(arq, maxBytes=1_000_000, backupCount=5, encoding="utf-8")
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s [%(threadName)s] %(name)s: %(message)s"))
    raiz = logging.getLogger()
    raiz.setLevel(nivel)
    raiz.addHandler(h)
    erro = logging.StreamHandler(sys.stderr)
    erro.setLevel(logging.WARNING)
    erro.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    raiz.addHandler(erro)
    try:
        _estado["falhas"] = open(pasta_logs / "falhas_nativas.log", "a", encoding="utf-8")  # noqa: SIM115
        faulthandler.enable(file=_estado["falhas"], all_threads=True)
    except Exception:  # pragma: no cover - ambiente sem suporte
        pass
    sys.excepthook = _excecao_global
    threading.excepthook = lambda a: _excecao_global(a.exc_type, a.exc_value, a.exc_traceback, a.thread)
    _estado.update(configurado=True, arquivo=arq)
    from . import __version__
    from .utils import versoes

    logging.getLogger("libras").info("Inicio (%s) versao %s | %s", nome, __version__, versoes())
    return arq


def arquivo_log() -> Path | None:
    return _estado["arquivo"]


def definir_notificador(funcao) -> None:
    """funcao(titulo, mensagem) mostra o erro ao usuario (a interface registra a sua)."""
    _estado["notificar"] = funcao


def _excecao_global(tipo, valor, tb, thread=None) -> None:
    if issubclass(tipo, KeyboardInterrupt):
        sys.__excepthook__(tipo, valor, tb)
        return
    onde = f" na thread {thread.name}" if thread is not None else ""
    logging.getLogger("libras").error("Excecao nao tratada%s", onde, exc_info=(tipo, valor, tb))
    notificar(tipo, valor)


def notificar(tipo, valor) -> None:
    from .erros import ErroLibras

    f = _estado["notificar"]
    if f is None:
        return
    if issubclass(tipo, ErroLibras):
        msg = str(valor)
    else:
        msg = (
            f"Ocorreu um erro inesperado ({tipo.__name__}: {valor}).\n\nO programa continua aberto. "
            f"Detalhes completos foram gravados em:\n{_estado['arquivo']}\n\n"
            "Envie esse arquivo para a equipe de desenvolvimento."
        )
    try:
        f("Erro", msg)
    except Exception:  # o proprio dialogo falhou: ja esta no log
        pass
