"""Legenda sobre o video: componente UNICO da bancada (aba Tempo real) e do app real (13.4, 13.5, 13.9).

- RollUp: logica pura (sem Qt) da legenda ao vivo no estilo do YouTube: janela de 2 ou 3 linhas; as
  palavras confirmadas entram na linha de baixo; quando ela enche, tudo sobe e a mais antiga sai; sem
  atividade por T segundos o texto esmaece. Tambem a pausa (13.9-D) e o destaque da palavra recem-
  confirmada (13.9-C), com no maximo 3 mudancas por segundo (WCAG 2.3.1).
- EstiloLegenda: tamanho, fonte, cores, contorno, largura, linhas, posicao (das preferencias do app real
  e das predefinicoes dos tokens).
- CamadaLegenda: junta RollUp, estilo e a linha discreta de "em analise" (D15) e desenha num QPainter.
- TelaLegendada: TelaVideo da bancada com a mesma camada.

A legenda e desenhada pela interface, nunca no JPEG do servico, e nunca e espelhada (a previa e
espelhada so na tela, D3 e D15). Nada aqui muda a decisao: so mostra o que o Transcritor confirmou.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from . import tema
from .componentes import TelaVideo

TOK = tema.TOKENS["legenda"]
POSICOES = ("embaixo", "pessoa", "topo")
NOMES_POSICAO = {
    "embaixo": "Embaixo, como legenda comum",
    "pessoa": "Em cima da pessoa",
    "topo": "Faixa fixa no alto da imagem",
}
CONTORNOS = {"nenhum": "Nenhum", "contorno": "Contorno", "sombra": "Sombra"}


# ======================================================================
# Texto do quadro
# ======================================================================
def parcial_do_quadro(meta: dict) -> str:
    """Linha discreta (D15): so aparece quando ajuda quem sinaliza (sem maos, aquecendo, em analise)."""
    etapa = meta.get("etapa") or {}
    e = etapa.get("etapa") or meta.get("motivo") or ""
    if e == "aquecendo":
        return "Preparando..."
    if e in ("sem_maos", "sem_pose"):
        return "Mostre as mãos para a câmera"
    if e == "aguardando":
        return f"Em análise: {etapa.get('texto', '').split(': ')[-1]}"
    return ""


def legenda_do_quadro(meta: dict) -> tuple[str, str]:
    """(texto confirmado, linha parcial). Mantida para a bancada e para os testes antigos."""
    return " ".join(meta.get("texto") or []), parcial_do_quadro(meta)


def cauda_que_cabe(palavras: list[str], cabe) -> str:
    """Ultimas palavras que cabem, com reticencias (usado so no texto completo da aba Decisao)."""
    if not palavras or cabe(" ".join(palavras)):
        return " ".join(palavras)
    for k in range(1, len(palavras)):
        s = "... " + " ".join(palavras[k:])
        if cabe(s):
            return s
    return palavras[-1]


# ======================================================================
# Roll-up (logica pura, relogio injetado)
# ======================================================================
@dataclass
class EstadoLegenda:
    linhas: list[list[tuple[str, bool]]]  # (palavra, destacada)
    saindo: list[tuple[str, bool]] | None  # linha que esta saindo pelo alto durante a rolagem
    progresso_rolagem: float  # 0 = acabou de rolar (a linha nova esta embaixo), 1 = parado
    opacidade: float
    destaque: str | None  # "pulso" | "reduzido" | None
    pausado: bool
    novas: int


class RollUp:
    def __init__(
        self,
        linhas: int = 2,
        sumir_s: float = 4.0,
        reduzir_movimento: bool = False,
        destacar: bool = True,
        relogio=time.monotonic,
    ):
        self.n = max(1, int(linhas))
        self.sumir_s = float(sumir_s)
        self.reduzir = bool(reduzir_movimento)
        self.destacar = bool(destacar)
        self.relogio = relogio
        self.dur_rolagem = tema.TOKENS["movimento"]["duracao_ms"] / 1000
        self.dur_esmaecer = TOK["esmaecer_ms"] / 1000
        self.cabe = lambda _texto: True  # definido pela camada com as medidas da fonte
        self._serial = 0
        self.limpar()

    # ------------------------------------------------------------------ configuracao
    def configurar(
        self,
        linhas: int | None = None,
        sumir_s: float | None = None,
        reduzir_movimento: bool | None = None,
        destacar: bool | None = None,
    ) -> None:
        if linhas is not None and int(linhas) != self.n:
            self.n = max(1, int(linhas))
            self.linhas = self.linhas[-self.n :]
        if sumir_s is not None:
            self.sumir_s = float(sumir_s)
        if reduzir_movimento is not None:
            self.reduzir = bool(reduzir_movimento)
        if destacar is not None:
            self.destacar = bool(destacar)

    def limpar(self) -> None:
        """Limpa a tela, o historico e o buffer da pausa, e retoma (13.9-D)."""
        self.linhas: list[list[tuple[int, str]]] = []
        self.ultimas: list[list[tuple[int, str]]] = []
        self.historico: list[str] = []
        self.buffer: list[str] = []
        self.pausado = False
        self.t_ultima: float | None = None
        self.t_rolagem: float | None = None
        self.saindo: list[tuple[int, str]] | None = None
        self.destaque_serial: int | None = None
        self.t_destaque: float | None = None
        self.mudancas: list[float] = []  # instantes em que o destaque liga ou desliga (WCAG 2.3.1)

    # ------------------------------------------------------------------ entrada
    def adicionar(self, palavra: str, t: float | None = None) -> None:
        t = self.relogio() if t is None else t
        self.historico.append(palavra)
        if self.pausado:
            self.buffer.append(palavra)
            return
        self._inserir(palavra, t, animar=True, destacar=True)

    def _inserir(self, palavra: str, t: float, animar: bool, destacar: bool) -> None:
        self.atualizar(t)
        self._serial += 1
        item = (self._serial, palavra)
        if not self.linhas:
            self.linhas = [[item]]
        else:
            tentativa = " ".join(w for _, w in self.linhas[-1] + [item])
            if self.cabe(tentativa):
                self.linhas[-1].append(item)
            else:
                self.linhas.append([item])
                if len(self.linhas) > self.n:
                    self.saindo = self.linhas.pop(0)
                    self.t_rolagem = t if (animar and not self.reduzir) else None
        self.ultimas = []
        self.t_ultima = t
        if destacar and self.destacar and self._pode_destacar(t):
            self.destaque_serial, self.t_destaque = self._serial, t
            self.mudancas += [t, t + self._dur_destaque()]

    def _dur_destaque(self) -> float:
        return (TOK["destaque_reduzido_ms"] if self.reduzir else TOK["destaque_ms"]) / 1000

    def _pode_destacar(self, t: float) -> bool:
        """Um pulso = 2 mudancas (liga e desliga). So comeca se nenhuma janela de 1 s passar de 3 mudancas."""
        if self.t_destaque is not None and t - self.t_destaque < TOK["intervalo_minimo_destaques_ms"] / 1000:
            return False
        fim = t + self._dur_destaque()
        todas = sorted(self.mudancas + [t, fim])
        return all(sum(1 for m in todas if a <= m < a + 1.0) <= 3 for a in todas)

    # ------------------------------------------------------------------ pausa
    def pausar(self) -> None:
        if not self.pausado:
            self.pausado = True
            if not self.linhas and self.ultimas:  # ja tinha esmaecido: volta o ultimo texto
                self.linhas, self.ultimas = self.ultimas, []
            self.saindo, self.t_rolagem = None, None

    def retomar(self, t: float | None = None) -> None:
        t = self.relogio() if t is None else t
        if not self.pausado:
            return
        self.pausado = False
        self.t_ultima = t  # o texto congelado continua na tela; as novas entram depois dele
        novas, self.buffer = self.buffer, []
        muitas = len(novas) > 1  # mais de uma palavra entra de uma vez, sem animacao nem destaque
        for p in novas:
            self._inserir(p, t, animar=not muitas, destacar=not muitas)
        self.t_ultima = t

    def alternar_pausa(self, t: float | None = None) -> bool:
        if self.pausado:
            self.retomar(t)
        else:
            self.pausar()
        return self.pausado

    # ------------------------------------------------------------------ tempo
    def atualizar(self, t: float | None = None) -> None:
        """Transicoes no tempo: fim da rolagem, fim do destaque, texto que esmaeceu (vira 'ultimas')."""
        t = self.relogio() if t is None else t
        if self.t_rolagem is not None and t - self.t_rolagem >= self.dur_rolagem:
            self.t_rolagem, self.saindo = None, None
        elif self.t_rolagem is None:
            self.saindo = None
        self.mudancas = [m for m in self.mudancas if m > t - 2.0]
        if self.pausado or self.t_ultima is None or not self.linhas:
            return
        fim = self.t_ultima + self.sumir_s + (0 if self.reduzir else self.dur_esmaecer)
        if t >= fim:
            self.ultimas, self.linhas = self.linhas, []
            self.saindo = self.t_rolagem = None

    def precisa_animar(self, t: float | None = None) -> bool:
        """True enquanto ha rolagem, destaque ou esmaecimento em andamento (a tela redesenha mais vezes)."""
        t = self.relogio() if t is None else t
        if self.pausado:
            return False
        if self.t_rolagem is not None or (self.t_destaque is not None and t - self.t_destaque < self._dur_destaque()):
            return True
        return bool(self.linhas and self.t_ultima is not None and t >= self.t_ultima + self.sumir_s)

    def estado(self, t: float | None = None) -> EstadoLegenda:
        t = self.relogio() if t is None else t
        opac = 1.0
        if not self.pausado and self.t_ultima is not None and self.linhas:
            passou = t - self.t_ultima - self.sumir_s
            if passou >= 0:
                opac = 0.0 if self.reduzir else max(0.0, 1.0 - passou / self.dur_esmaecer)
        dest = None
        if (
            not self.pausado
            and self.destaque_serial is not None
            and self.t_destaque is not None
            and t - self.t_destaque < self._dur_destaque()
        ):
            dest = "reduzido" if self.reduzir else "pulso"

        def conv(linha):
            return [(w, dest is not None and s == self.destaque_serial) for s, w in linha]

        prog = 1.0
        if self.t_rolagem is not None and not self.pausado:
            prog = min(1.0, max(0.0, (t - self.t_rolagem) / self.dur_rolagem))
        return EstadoLegenda(
            [conv(l) for l in self.linhas],
            conv(self.saindo) if self.saindo and prog < 1 else None,
            prog,
            opac,
            dest,
            self.pausado,
            len(self.buffer),
        )

    # ------------------------------------------------------------------ texto
    def texto_visivel(self) -> str:
        """O que esta na tela agora (durante a pausa, exatamente o texto congelado)."""
        linhas = self.linhas or (self.ultimas if self.pausado else [])
        return "\n".join(" ".join(w for _, w in l) for l in linhas)

    def texto_completo(self) -> str:
        return " ".join(self.historico)

    def refluir(self) -> None:
        """Quebra de novo as palavras da tela (mudou a largura, a fonte ou o numero de linhas)."""
        for nome in ("linhas", "ultimas"):
            itens = [it for l in getattr(self, nome) for it in l]
            novas: list[list[tuple[int, str]]] = []
            for it in itens:
                if novas and self.cabe(" ".join(w for _, w in novas[-1] + [it])):
                    novas[-1].append(it)
                else:
                    novas.append([it])
            setattr(self, nome, novas[-self.n :])
        self.saindo, self.t_rolagem = None, None


# ======================================================================
# Estilo
# ======================================================================
PREFERENCIAS_PADRAO = {
    "predefinicao": "padrao",
    "tamanho": TOK["tamanho_padrao"],
    "fonte": "padrao",
    "cor_texto": None,
    "cor_fundo": None,
    "opacidade": None,  # so na predefinicao "personalizada"
    "contorno": "nenhum",
    "largura_pct": TOK["largura_pct"]["padrao"],
    "linhas": 2,
    "sumir_s": 4.0,
    "posicao": "embaixo",
    "destacar": True,
}


@dataclass
class EstiloLegenda:
    tamanho_px: int
    familias: list[str]
    texto: str
    fundo: str
    opacidade: float
    parcial: str
    destaque: str | None
    contorno: str = "nenhum"
    largura_frac: float = 0.8
    linhas: int = 2
    sumir_s: float = 4.0
    posicao: str = "embaixo"
    destacar: bool = True
    avisos: list[str] = field(default_factory=list)

    def contraste(self) -> float:
        return tema.contraste_pior_caso(self.texto, self.fundo, self.opacidade)


def estilo_de(prefs: dict | None, escala: float = 1.0) -> EstiloLegenda:
    """Estilo a partir das preferencias (dict da secao 'legenda'); valores ausentes vem do padrao."""
    p = {**PREFERENCIAS_PADRAO, **(prefs or {})}
    tamanhos = TOK["tamanhos_px"]
    tam = int(round(tamanhos[min(len(tamanhos) - 1, max(0, int(p["tamanho"])))] * escala))
    familias = TOK["fontes"].get(p["fonte"], TOK["fontes"]["padrao"])["familias"]
    pred = TOK["predefinicoes"].get(p["predefinicao"])
    if pred:
        texto, fundo, opac, parcial, dest = (
            pred["texto"],
            pred["fundo"],
            pred["opacidade"],
            pred["parcial"],
            pred["destaque"],
        )
    else:  # personalizada: as cores do usuario vem do arquivo de preferencias
        base = TOK["predefinicoes"]["padrao"]
        texto = p.get("cor_texto") or base["texto"]
        fundo = p.get("cor_fundo") or base["fundo"]
        opac = base["opacidade"] if p.get("opacidade") is None else float(p["opacidade"])
        parcial = texto
        sel = tema.COR["selecao"]
        dest = sel if tema.contraste(texto, sel) >= 4.5 else None  # sem fundo de destaque: so contorno e peso
    est = EstiloLegenda(
        tam,
        list(familias),
        texto,
        fundo,
        float(opac),
        parcial,
        dest,
        p["contorno"],
        min(1.0, max(0.2, float(p["largura_pct"]) / 100)),
        3 if int(p["linhas"]) >= 3 else 2,
        float(p["sumir_s"]),
        p["posicao"] if p["posicao"] in POSICOES else "embaixo",
        bool(p["destacar"]),
    )
    if est.contraste() < 4.5:
        est.avisos.append(
            f"Contraste {est.contraste():.1f}:1, abaixo de 4,5:1 sobre imagens claras ou escuras: a "
            "legenda pode ficar difícil de ler."
        )
    return est


# ======================================================================
# Camada (medir e desenhar com Qt)
# ======================================================================
class CamadaLegenda:
    """RollUp + estilo + linha parcial, medidos e desenhados num QPainter."""

    def __init__(self, prefs: dict | None = None, relogio=time.monotonic):
        self.relogio = relogio
        self.estilo = estilo_de(prefs)
        self.rollup = RollUp(
            self.estilo.linhas, self.estilo.sumir_s, tema.ESTADO["reduzir_movimento"], self.estilo.destacar, relogio
        )
        self.parcial = ""
        self.exemplo: list[str] | None = None  # legenda de exemplo da previa de Configuracoes
        self._largura = -1.0
        self._fonte_chave = None
        self.ultimo_retangulo = None  # onde foi desenhada (testes e gerenciador de posicoes)

    # ------------------------------------------------------------------ dados
    def aplicar_preferencias(self, prefs: dict | None) -> None:
        self.estilo = estilo_de(prefs)
        self.rollup.configurar(
            self.estilo.linhas, self.estilo.sumir_s, tema.ESTADO["reduzir_movimento"], self.estilo.destacar
        )
        self._largura = -1.0  # forca refluir

    def receber_quadro(self, meta: dict, rotulo=lambda c: c, t: float | None = None) -> None:
        if meta.get("aceito"):
            self.rollup.adicionar(rotulo(meta["aceito"]), t)
        self.parcial = parcial_do_quadro(meta)

    def parcial_efetivo(self) -> str:
        """Linha discreta acima do texto: na pausa vira a etiqueta "Legenda pausada" com o contador (13.9-D)."""
        if self.exemplo is not None:
            return ""
        if self.rollup.pausado:
            n = len(self.rollup.buffer)
            return "Legenda pausada" + (f" · {n} {'nova' if n == 1 else 'novas'}" if n else "")
        return self.parcial

    # ------------------------------------------------------------------ fontes
    def fontes(self):
        from PySide6.QtGui import QFont

        f = QFont()
        f.setFamilies(self.estilo.familias)
        f.setPixelSize(self.estilo.tamanho_px)
        f.setWeight(QFont.DemiBold)
        fd = QFont(f)
        fd.setWeight(QFont.Black)
        fp = QFont(f)
        fp.setWeight(QFont.Normal)
        fp.setPixelSize(max(tema.px("corpo"), int(self.estilo.tamanho_px * 0.55)))
        return f, fd, fp

    def _preparar(self, largura_max: float) -> None:
        from PySide6.QtGui import QFontMetricsF

        chave = (self.estilo.tamanho_px, tuple(self.estilo.familias), self.estilo.linhas)
        if abs(largura_max - self._largura) < 0.5 and chave == self._fonte_chave:
            return
        f, _, _ = self.fontes()
        fm = QFontMetricsF(f)
        util = max(10.0, largura_max - 2 * self.pad_x())
        self.rollup.cabe = lambda s: fm.horizontalAdvance(s) <= util
        self._largura, self._fonte_chave = largura_max, chave
        self.rollup.refluir()

    def pad_x(self) -> float:
        return max(tema.ESPACO[3], self.estilo.tamanho_px * 0.45)

    def pad_y(self) -> float:
        return max(tema.ESPACO[2], self.estilo.tamanho_px * 0.25)

    # ------------------------------------------------------------------ medir
    def linhas_para_desenhar(self, t: float | None = None) -> tuple[EstadoLegenda, list[list[tuple[str, bool]]]]:
        if self.exemplo is not None:  # previa de Configuracoes: quebra com a mesma regra da legenda real
            linhas: list[list[str]] = []
            for w in " ".join(self.exemplo).split():
                if linhas and self.rollup.cabe(" ".join(linhas[-1] + [w])):
                    linhas[-1].append(w)
                else:
                    linhas.append([w])
            linhas_ = [[(w, False) for w in l] for l in linhas[-self.estilo.linhas :]]
            return EstadoLegenda(linhas_, None, 1.0, 1.0, None, False, 0), linhas_
        self.rollup.atualizar(t)
        est = self.rollup.estado(t)
        return est, est.linhas

    def medir(self, largura_max: float, t: float | None = None) -> tuple[float, float] | None:
        """(largura, altura) da caixa, ou None se nao ha nada a mostrar."""
        from PySide6.QtGui import QFontMetricsF

        self._preparar(largura_max)
        est, linhas = self.linhas_para_desenhar(t)
        parcial = self.parcial_efetivo()
        if not linhas and not parcial:
            return None
        f, fd, fp = self.fontes()
        fm, fmp = QFontMetricsF(fd), QFontMetricsF(fp)
        larg = max([fm.horizontalAdvance(" ".join(w for w, _ in l)) for l in linhas] + [0.0])
        if parcial:
            larg = max(larg, fmp.horizontalAdvance(parcial) + (fmp.height() if self.rollup.pausado else 0))
        larg = min(largura_max, larg + 2 * self.pad_x())
        alt = 2 * self.pad_y() + len(linhas) * fm.lineSpacing() + (fmp.lineSpacing() + tema.ESPACO[1] if parcial else 0)
        return larg, alt

    # ------------------------------------------------------------------ desenhar
    def desenhar(self, p, caixa, t: float | None = None) -> None:
        """Desenha na caixa (QRectF) calculada pelo gerenciador de posicoes."""
        from PySide6.QtCore import QPointF, QRectF, Qt
        from PySide6.QtGui import QColor, QFontMetricsF, QPainterPath, QPen

        est, linhas = self.linhas_para_desenhar(t)
        parcial = self.parcial_efetivo()
        self.ultimo_retangulo = caixa
        if not linhas and not parcial:
            return
        e = self.estilo
        f, fd, fp = self.fontes()
        fm, fmp = QFontMetricsF(f), QFontMetricsF(fp)
        p.save()
        p.setOpacity(est.opacidade if linhas else 1.0)
        fundo = QColor(e.fundo)
        fundo.setAlphaF(e.opacidade)
        p.setPen(Qt.NoPen)
        p.setBrush(fundo)
        raio = tema.RAIO["m"]
        p.drawRoundedRect(caixa, raio, raio)
        y = caixa.top() + self.pad_y()
        if parcial:
            p.setFont(fp)
            p.setPen(QColor(e.parcial))
            r = QRectF(caixa.left(), y, caixa.width(), fmp.lineSpacing())
            if self.rollup.pausado and self.exemplo is None:  # icone de pausa (duas barras) + texto
                larg_txt = fmp.horizontalAdvance(parcial)
                ic = fmp.height() * 0.7
                x0 = caixa.center().x() - (larg_txt + ic * 1.3) / 2
                p.setPen(Qt.NoPen)
                p.setBrush(QColor(e.parcial))
                yb = y + (fmp.lineSpacing() - ic) / 2
                p.drawRect(QRectF(x0, yb, ic * 0.32, ic))
                p.drawRect(QRectF(x0 + ic * 0.6, yb, ic * 0.32, ic))
                p.setPen(QColor(e.parcial))
                r = QRectF(x0 + ic * 1.3, y, larg_txt + 2, fmp.lineSpacing())
                p.drawText(r, Qt.AlignLeft | Qt.AlignVCenter, parcial)
            else:
                p.drawText(r, Qt.AlignHCenter | Qt.AlignVCenter, parcial)
            y += fmp.lineSpacing() + tema.ESPACO[1]
        lh = fm.lineSpacing()
        area = QRectF(caixa.left(), y, caixa.width(), len(linhas) * lh)
        p.setClipRect(area)
        filas = ([est.saindo] if est.saindo else []) + linhas
        y0 = y - est.progresso_rolagem * lh if est.saindo else y
        for i, linha in enumerate(filas):
            self._desenhar_linha(
                p, linha, caixa, y0 + i * lh, lh, f, fd, fm, est.destaque, QPainterPath, QPen, QPointF, QRectF
            )
        p.restore()

    def _desenhar_linha(self, p, linha, caixa, y, lh, f, fd, fm, destaque, QPainterPath, QPen, QPointF, QRectF):
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QColor, QFontMetricsF

        e = self.estilo
        fmd = QFontMetricsF(fd)
        larguras = [(fmd if d else fm).horizontalAdvance(w) for w, d in linha]
        esp = fm.horizontalAdvance(" ")
        total = sum(larguras) + esp * max(0, len(linha) - 1)
        x = caixa.center().x() - total / 2
        base = y + (lh - fm.height()) / 2 + fm.ascent()
        for (w, d), lw in zip(linha, larguras):
            fonte = fd if d else f
            if d and destaque == "pulso":
                r = QRectF(x - 4, y + 2, lw + 8, lh - 4)
                if e.destaque:
                    p.setPen(Qt.NoPen)
                    p.setBrush(QColor(e.destaque))
                    p.drawRoundedRect(r, tema.RAIO["p"], tema.RAIO["p"])
                p.setPen(QPen(QColor(e.texto), 2))
                p.setBrush(Qt.NoBrush)
                p.drawRoundedRect(r, tema.RAIO["p"], tema.RAIO["p"])
            if d and destaque == "reduzido":
                fonte = type(fd)(fd)
                fonte.setUnderline(True)
            p.setFont(fonte)
            ponto = QPointF(x, base)
            if e.contorno in ("contorno", "sombra"):
                cam = QPainterPath()
                cam.addText(ponto + (QPointF(2, 2) if e.contorno == "sombra" else QPointF(0, 0)), fonte, w)
                cor = QColor(e.fundo)
                if e.contorno == "contorno":
                    p.setPen(QPen(cor, max(2.0, e.tamanho_px / 12), Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
                    p.setBrush(Qt.NoBrush)
                else:
                    p.setPen(Qt.NoPen)
                    p.setBrush(cor)
                p.drawPath(cam)
            p.setPen(QColor(e.texto))
            p.drawText(ponto, w)
            x += lw + esp


# ======================================================================
# Tela da bancada
# ======================================================================
class TelaLegendada(TelaVideo):
    """TelaVideo com a CamadaLegenda por cima (aba Tempo real da bancada). Por padrao a legenda fica
    embaixo; em "Pre-visualizar como no app real" (13.5) usa as preferencias e o gerenciador de
    posicoes do app real, com a caixa da cabeca enviada pelo servico."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.camada = CamadaLegenda()
        self.cabeca_norm = None  # caixa da cabeca (normalizada, imagem nao espelhada) para a posicao "pessoa"
        from .sobreposicoes import SuavizadorCabeca

        self.suavizador = SuavizadorCabeca()

    # compatibilidade com a bancada antiga ---------------------------------
    @property
    def legenda_visivel(self) -> str:
        return self.camada.rollup.texto_visivel()

    @property
    def parcial_legenda(self) -> str:
        return self.camada.parcial

    @property
    def texto_legenda(self) -> str:
        return self.camada.rollup.texto_completo()

    def receber_quadro(self, meta: dict, rotulo=lambda c: c) -> None:
        self.camada.receber_quadro(meta, rotulo)
        enq = meta.get("enquadramento") or {}
        self.cabeca_norm = self.suavizador.atualizar(enq.get("cabeca"), self.camada.relogio())
        self.update()

    def limpar(self) -> None:
        self.camada.rollup.limpar()
        self.camada.parcial = ""
        self.update()

    def area_imagem(self):
        from PySide6.QtCore import QRectF

        pm = self.pixmap()
        if pm is None or pm.isNull():
            return None
        cr = self.contentsRect()
        w, h = pm.width() / pm.devicePixelRatio(), pm.height() / pm.devicePixelRatio()
        return QRectF(cr.x() + (cr.width() - w) / 2, cr.y() + (cr.height() - h) / 2, w, h)

    def paintEvent(self, ev) -> None:  # noqa: N802
        from PySide6.QtCore import QRectF
        from PySide6.QtGui import QPainter

        from .sobreposicoes import Ret, cabeca_na_tela, posicionar

        super().paintEvent(ev)
        img = self.area_imagem()
        if img is None:
            return
        r_img = Ret(img.x(), img.y(), img.width(), img.height())
        e = self.camada.estilo
        tam = self.camada.medir(min(img.width() - 2 * tema.ESPACO[2], img.width() * e.largura_frac))
        if tam is None:
            self.camada.ultimo_retangulo = None
            return
        pos = posicionar(
            r_img,
            r_img,
            legenda=tam,
            posicao=e.posicao,
            cabeca=cabeca_na_tela(self.cabeca_norm, r_img),
            botao_config=False,
        )
        L = pos["legenda"]
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.TextAntialiasing)
        self.camada.desenhar(p, QRectF(L.x, L.y, L.w, L.h))
        p.end()
        if self.camada.rollup.precisa_animar():
            from PySide6.QtCore import QTimer

            QTimer.singleShot(16, self.update)
