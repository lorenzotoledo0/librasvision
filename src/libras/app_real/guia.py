"""Guia de enquadramento (13.9-A): avisos com histerese, um por vez, na ordem de prioridade.

So INFORMA: nunca bloqueia o reconhecimento nem muda o que o modelo recebe (D15). As medidas vem do
servico de tempo real (libras.enquadramento); aqui so se decide qual aviso mostrar e quando.
Logica pura (relogio passado por quem chama), testada sem interface.
"""

from __future__ import annotations

ORDEM = ("sem_pessoa", "ombros", "mao_cortada", "pouca_luz")
TEXTOS = {
    "sem_pessoa": "Ninguém na imagem: fique de frente para a câmera",
    "ombros": "Mostre os ombros: afaste-se um pouco da câmera",
    "mao_cortada": "Uma das mãos sai da imagem: aproxime-se do centro",
    "pouca_luz": "Pouca luz: acenda uma luz à sua frente",
}
ITENS_GUIA = ("Ombros visíveis", "Mãos visíveis ao sinalizar", "Luz suficiente")


def condicoes(enq: dict) -> dict[str, bool]:
    """Condicoes de aviso do quadro. Mao NAO detectada nao gera aviso (repouso fora da imagem e normal)."""
    pessoa = bool(enq.get("pessoa"))
    return {
        "sem_pessoa": not pessoa,
        "ombros": pessoa and enq.get("ombros") != "ok",
        "mao_cortada": bool(enq.get("mao_cortada")),
        "pouca_luz": bool(enq.get("pouca_luz")),
    }


class Histerese:
    """A condicao precisa durar `s` segundos para ligar e ficar `s` segundos ausente para desligar."""

    def __init__(self, s: float):
        self.s = float(s)
        self.ativo = False
        self._desde: float | None = None

    def atualizar(self, cond: bool, t: float) -> bool:
        if cond != self.ativo:
            if self._desde is None:
                self._desde = t
            if t - self._desde >= self.s:
                self.ativo, self._desde = cond, None
        else:
            self._desde = None
        return self.ativo


class GuiaAvisos:
    def __init__(self, histerese_s: float = 1.0):
        self.h = {c: Histerese(histerese_s) for c in ORDEM}
        self.habilitado = True
        self.lados: list[str] = []

    def atualizar(self, enq: dict | None, t: float) -> str | None:
        """Chave do aviso a mostrar agora (ou None). Sem medidas (camera parada) nada aparece."""
        if not enq:
            for h in self.h.values():
                h.atualizar(False, t)
            return None
        cond = condicoes(enq)
        if enq.get("mao_cortada"):
            self.lados = list(enq["mao_cortada"])
        ativos = [c for c in ORDEM if self.h[c].atualizar(cond[c], t)]
        if not self.habilitado or not ativos:
            return None
        return ativos[0]

    def texto(self, chave: str | None) -> str:
        return TEXTOS.get(chave or "", "")


def itens_primeira_abertura(enq: dict | None) -> list[tuple[str, bool]]:
    """Lista ao vivo do guia de posicionamento (13.9-A): (texto, ok)."""
    enq = enq or {}
    return [
        (ITENS_GUIA[0], enq.get("ombros") == "ok"),
        (ITENS_GUIA[1], bool(enq.get("maos_suficientes"))),
        (ITENS_GUIA[2], bool(enq) and not enq.get("pouca_luz")),
    ]
