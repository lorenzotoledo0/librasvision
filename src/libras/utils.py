"""Utilitarios compartilhados: seeds, versoes, hash, escrita atomica, datas."""

from __future__ import annotations

import csv
import hashlib
import os
import platform
import random
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np


def fixar_seeds(seed: int) -> None:
    """Fixa seeds de random, numpy e (se ja importado ou disponivel) TensorFlow."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if "tensorflow" in sys.modules:
        import tensorflow as tf

        tf.random.set_seed(seed)


def versoes() -> dict:
    """Versoes relevantes para reprodutibilidade (sem importar pacotes pesados a toa)."""
    out = {
        "python": platform.python_version(),
        "sistema": f"{platform.system()} {platform.release()}",
        "processador": platform.processor() or platform.machine(),
        "numpy": np.__version__,
    }
    for mod in ("cv2", "mediapipe", "tensorflow", "keras", "sklearn"):
        try:
            m = __import__(mod)
            out[mod] = getattr(m, "__version__", "?")
        except Exception:  # pacote ausente ou falha de import nao deve derrubar o registro
            out[mod] = "nao instalado"
    return out


def sha256_arquivo(caminho: Path, bloco: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        for parte in iter(lambda: f.read(bloco), b""):
            h.update(parte)
    return h.hexdigest()


def agora_iso() -> str:
    return datetime.now().replace(microsecond=0).isoformat()


def carimbo() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def escrever_csv_atomico(caminho: Path, colunas: list[str], linhas: Iterable[dict]) -> None:
    """Escreve CSV em arquivo temporario e substitui o original de uma vez.

    Evita manifesto corrompido se o programa fechar no meio da escrita.
    No Windows, falha com mensagem clara se o arquivo estiver aberto no Excel.
    """
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=caminho.name + ".", suffix=".tmp", dir=caminho.parent)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=colunas, extrasaction="raise")
            w.writeheader()
            for linha in linhas:
                w.writerow({c: linha.get(c, "") for c in colunas})
        try:
            os.replace(tmp, caminho)
        except PermissionError as e:
            from .erros import ErroManifesto

            raise ErroManifesto(
                f"Nao foi possivel salvar {caminho.name}: o arquivo esta aberto em outro programa "
                "(provavelmente o Excel). Feche-o e tente de novo."
            ) from e
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def escrever_texto_atomico(caminho: Path, texto: str) -> None:
    caminho = Path(caminho)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=caminho.name + ".", suffix=".tmp", dir=caminho.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(texto)
        os.replace(tmp, caminho)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def ler_csv(caminho: Path) -> list[dict]:
    caminho = Path(caminho)
    if not caminho.exists():
        return []
    with open(caminho, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def eh_windows() -> bool:
    return sys.platform.startswith("win")


def hardware() -> dict:
    """Descricao do computador (para registrar onde tempos e latencias foram medidos)."""
    info = {
        "sistema": f"{platform.system()} {platform.release()}",
        "maquina": platform.machine(),
        "processador": platform.processor() or "desconhecido",
        "nucleos_logicos": os.cpu_count(),
    }
    ram = None
    try:
        if sys.platform.startswith("win"):
            import ctypes

            class _Mem(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            m = _Mem()
            m.dwLength = ctypes.sizeof(_Mem)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            ram = m.ullTotalPhys
            nome = os.environ.get("PROCESSOR_IDENTIFIER")
            if nome:
                info["processador"] = nome
        elif hasattr(os, "sysconf"):
            ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
            try:
                with open("/proc/cpuinfo", encoding="utf-8") as f:
                    for linha in f:
                        if linha.startswith("model name"):
                            info["processador"] = linha.split(":", 1)[1].strip()
                            break
            except OSError:
                pass
    except Exception:  # informacao auxiliar: nunca derruba a execucao
        pass
    info["ram_gb"] = round(ram / 2**30, 1) if ram else None
    return info


def memoria_processo_mb() -> float | None:
    """Memoria residente (RSS) deste processo, em MB. None se nao for possivel medir."""
    try:
        if sys.platform.startswith("win"):
            import ctypes
            from ctypes import wintypes

            class _PMC(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            c = _PMC()
            c.cb = ctypes.sizeof(_PMC)
            h = ctypes.windll.kernel32.GetCurrentProcess()
            ctypes.windll.psapi.GetProcessMemoryInfo(h, ctypes.byref(c), c.cb)
            return c.WorkingSetSize / 2**20
        with open("/proc/self/status", encoding="utf-8") as f:
            for linha in f:
                if linha.startswith("VmRSS:"):
                    return int(linha.split()[1]) / 1024
    except Exception:  # medicao auxiliar
        return None
    return None
