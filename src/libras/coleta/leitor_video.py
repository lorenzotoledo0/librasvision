"""Leitura de quadros sob demanda para o editor de corte (item 9), sem interface grafica.

O video nao e carregado inteiro na memoria: os quadros sao decodificados quando pedidos, com um
cache pequeno (LRU) e leitura sequencial quando o pedido e o proximo quadro (reproducao). Saltos
usam a busca do OpenCV (no backend FFmpeg ela decodifica a partir do quadro-chave anterior, entao
o quadro devolvido e o pedido). As miniaturas usam uma segunda leitura, sequencial, e ficam em um
cache em memoria por video (nada e gravado em disco: miniaturas tem o rosto das pessoas).
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np

_CACHE_MINIATURAS: "OrderedDict[tuple, list]" = OrderedDict()
_TRAVA_MINIATURAS = threading.Lock()
MAX_VIDEOS_EM_CACHE = 12


class LeitorVideo:
    def __init__(self, caminho: str | Path, tamanho_cache: int = 48):
        import cv2

        self.caminho = str(caminho)
        self._cap = cv2.VideoCapture(self.caminho)
        if not self._cap.isOpened():
            raise OSError(f"Nao foi possivel abrir o video {caminho}")
        self.n = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 0) or 30.0
        self.largura = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.altura = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        self._proximo = 0  # indice do quadro que um read() devolveria
        self._cache: "OrderedDict[int, np.ndarray]" = OrderedDict()
        self._max = int(tamanho_cache)
        self._trava = threading.Lock()
        self.buscas = 0  # quantas vezes foi preciso buscar (para testes e diagnostico)

    def quadro(self, i: int) -> np.ndarray | None:
        import cv2

        i = int(i)
        if i < 0 or (self.n and i >= self.n):
            return None
        with self._trava:
            f = self._cache.get(i)
            if f is not None:
                self._cache.move_to_end(i)
                return f
            if i != self._proximo:
                if 0 < i - self._proximo <= 8:  # pouco a frente: avancar e mais barato que buscar
                    while self._proximo < i and self._cap.grab():
                        self._proximo += 1
                else:
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                    self.buscas += 1
                    self._proximo = i
            ok, f = self._cap.read()
            if not ok:
                return None
            self._proximo = i + 1
            self._cache[i] = f
            if len(self._cache) > self._max:
                self._cache.popitem(last=False)
            return f

    def fechar(self) -> None:
        with self._trava:
            self._cap.release()
            self._cache.clear()


def indices_miniaturas(n: int, quantidade: int) -> list[int]:
    if n <= 0 or quantidade <= 0:
        return []
    quantidade = min(n, quantidade)
    return sorted({int(round(x)) for x in np.linspace(0, n - 1, quantidade)})


def miniaturas(
    caminho: str | Path,
    quantidade: int,
    altura: int = 54,
    chave: tuple | None = None,
    cancelado=lambda: False,
    progresso=lambda i, n: None,
) -> list[tuple[int, np.ndarray]]:
    """(indice, imagem BGR reduzida) para `quantidade` quadros espalhados pelo video. Leitura
    sequencial (grab sem decodificar a imagem dos quadros pulados). Resultado em cache em memoria."""
    import cv2

    if chave is None:
        st = Path(caminho).stat()
        chave = (str(caminho), st.st_mtime_ns, st.st_size, quantidade, altura)  # arquivo trocado = outra chave
    with _TRAVA_MINIATURAS:
        if chave in _CACHE_MINIATURAS:
            _CACHE_MINIATURAS.move_to_end(chave)
            return _CACHE_MINIATURAS[chave]
    cap = cv2.VideoCapture(str(caminho))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    alvos = indices_miniaturas(n, quantidade)
    out: list[tuple[int, np.ndarray]] = []
    try:
        i = 0
        for k, alvo in enumerate(alvos):
            while i < alvo:
                if not cap.grab():
                    break
                i += 1
            if cancelado():
                return out
            ok, f = cap.read()
            if not ok:
                break
            i += 1
            h, w = f.shape[:2]
            out.append((alvo, cv2.resize(f, (max(1, int(w * altura / h)), altura), interpolation=cv2.INTER_AREA)))
            progresso(k + 1, len(alvos))
    finally:
        cap.release()
    with _TRAVA_MINIATURAS:
        _CACHE_MINIATURAS[chave] = out
        while len(_CACHE_MINIATURAS) > MAX_VIDEOS_EM_CACHE:
            _CACHE_MINIATURAS.popitem(last=False)
    return out


def limpar_cache_miniaturas() -> None:
    with _TRAVA_MINIATURAS:
        _CACHE_MINIATURAS.clear()
