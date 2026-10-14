"""Estado do editor de corte (item 9), sem interface grafica.

O editor so faz uma coisa: escolher o trecho mantido (inicio e fim) e, opcionalmente, um ponto de
divisao em dois takes. Nada e gravado em disco ate salvar; salvar chama coleta.cortes.aplicar_edicao,
que mantem as regras de D9 (original para a lixeira, .mp4 e .npz novos com instantes reindexados).

Convencao: quadros indexados a partir de 0; o trecho mantido e [inicio, fim) (fim exclusivo), e a
divisao, se houver, separa [inicio, divisao) e [divisao, fim). Todo trecho tem pelo menos
MIN_FRAMES quadros; as funcoes de ajuste prendem o valor no limite (a alca "nao passa").
Remocao de trecho interno nao existe (criaria um salto que o modelo leria como movimento rapido).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cortes import MIN_FRAMES


@dataclass(frozen=True)
class EstadoCorte:
    inicio: int
    fim: int
    divisao: int | None = None


def formatar_tempo(segundos: float) -> str:
    """mm:ss.cc"""
    segundos = max(0.0, float(segundos))
    m = int(segundos // 60)
    s = segundos - 60 * m
    return f"{m:02d}:{s:05.2f}"


class SessaoCorte:
    def __init__(self, n_frames: int, fps: float, tempos: np.ndarray | None = None, minimo: int = MIN_FRAMES):
        if n_frames < minimo:
            raise ValueError(f"O take tem {n_frames} quadros; o minimo para editar e {minimo}.")
        self.n = int(n_frames)
        self.fps = float(fps) if fps and fps > 0 else 30.0
        self.tempos = None if tempos is None or len(tempos) != n_frames else np.asarray(tempos, np.float64)
        self.minimo = int(minimo)
        self.original = EstadoCorte(0, self.n, None)
        self.estado = self.original
        self._desfazer: list[EstadoCorte] = []
        self._refazer: list[EstadoCorte] = []
        self._marca: EstadoCorte | None = None

    # ------------------------------------------------------------------ consultas
    @property
    def inicio(self) -> int:
        return self.estado.inicio

    @property
    def fim(self) -> int:
        return self.estado.fim

    @property
    def divisao(self) -> int | None:
        return self.estado.divisao

    @property
    def alterado(self) -> bool:
        return self.estado != self.original

    @property
    def pode_desfazer(self) -> bool:
        return bool(self._desfazer)

    @property
    def pode_refazer(self) -> bool:
        return bool(self._refazer)

    def trechos(self) -> list[tuple[int, int]]:
        e = self.estado
        if e.divisao is None:
            return [(e.inicio, e.fim)]
        return [(e.inicio, e.divisao), (e.divisao, e.fim)]

    def tempo(self, quadro: int) -> float:
        """Instante do quadro em segundos (instantes reais do .npz quando existem)."""
        q = int(np.clip(quadro, 0, self.n - 1))
        if self.tempos is not None:
            return float(self.tempos[q] - self.tempos[0])
        return q / self.fps

    def duracao(self) -> float:
        return self.tempo(self.n - 1) + 1.0 / self.fps

    def quadros_por_segundo(self) -> int:
        return max(1, int(round(self.fps)))

    def limites_inicio(self) -> tuple[int, int]:
        lim = (self.divisao if self.divisao is not None else self.fim) - self.minimo
        return 0, max(0, lim)

    def limites_fim(self) -> tuple[int, int]:
        base = self.divisao if self.divisao is not None else self.inicio
        return min(self.n, base + self.minimo), self.n

    def limites_divisao(self) -> tuple[int, int]:
        return self.inicio + self.minimo, self.fim - self.minimo

    def pode_dividir_em(self, quadro: int) -> bool:
        a, b = self.limites_divisao()
        return a <= int(quadro) <= b

    def descricao(self) -> str:
        partes = []
        for k, (a, b) in enumerate(self.trechos(), 1):
            rot = f"Take {k}: " if len(self.trechos()) > 1 else "Trecho mantido: "
            partes.append(
                f"{rot}quadros {a + 1} a {b} ({b - a} quadros, {formatar_tempo(self.tempo(a))} a "
                f"{formatar_tempo(self.tempo(b - 1) + 1 / self.fps)})"
            )
        return "; ".join(partes) + f". Mínimo de {self.minimo} quadros por trecho."

    # ------------------------------------------------------------------ alteracoes
    def _mudar(self, novo: EstadoCorte, registrar: bool) -> bool:
        if novo == self.estado:
            return False
        if registrar:
            self._desfazer.append(self.estado)
            self._refazer.clear()
        self.estado = novo
        return True

    def iniciar_arrasto(self) -> None:
        """Um arrasto inteiro vira UM passo de desfazer."""
        self._marca = self.estado

    def concluir_arrasto(self) -> None:
        if self._marca is not None and self._marca != self.estado:
            self._desfazer.append(self._marca)
            self._refazer.clear()
        self._marca = None

    def definir_inicio(self, quadro: int, registrar: bool = True) -> int:
        a, b = self.limites_inicio()
        q = int(np.clip(int(round(quadro)), a, b))
        self._mudar(EstadoCorte(q, self.fim, self.divisao), registrar and self._marca is None)
        return q

    def definir_fim(self, quadro_exclusivo: int, registrar: bool = True) -> int:
        a, b = self.limites_fim()
        q = int(np.clip(int(round(quadro_exclusivo)), a, b))
        self._mudar(EstadoCorte(self.inicio, q, self.divisao), registrar and self._marca is None)
        return q

    def definir_inicio_no_quadro(self, quadro: int) -> int:
        """'Definir inicio aqui': o quadro do playhead passa a ser o primeiro mantido."""
        return self.definir_inicio(quadro)

    def definir_fim_no_quadro(self, quadro: int) -> int:
        """'Definir fim aqui': o quadro do playhead passa a ser o ultimo mantido."""
        return self.definir_fim(int(quadro) + 1) - 1

    def dividir(self, quadro: int) -> bool:
        """Divide no quadro do playhead: ele passa a ser o primeiro quadro do segundo take."""
        if not self.pode_dividir_em(quadro):
            return False
        return self._mudar(EstadoCorte(self.inicio, self.fim, int(quadro)), True)

    def remover_divisao(self) -> bool:
        return self._mudar(EstadoCorte(self.inicio, self.fim, None), True)

    def desfazer(self) -> bool:
        if not self._desfazer:
            return False
        self._refazer.append(self.estado)
        self.estado = self._desfazer.pop()
        return True

    def refazer(self) -> bool:
        if not self._refazer:
            return False
        self._desfazer.append(self.estado)
        self.estado = self._refazer.pop()
        return True

    def cancelar(self) -> None:
        """Volta ao take sem edicao (nada foi gravado)."""
        self._mudar(self.original, True)
