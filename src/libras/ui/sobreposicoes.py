"""Gerenciador UNICO de posicoes das sobreposicoes sobre a camera (13.4, 13.8, 13.9).

Botao de Configuracoes, barra de informacoes tecnicas, legenda, etiqueta de enquadramento, barras de
mao cortada e barra minima (com o selo) sao posicionados aqui, numa ordem de prioridade, de modo que
nenhum retangulo cruze outro. Logica pura (sem Qt): recebe e devolve retangulos em pixels.

Prioridade: barra minima e botao de Configuracoes (fixos) > barra tecnica (reserva o topo) > legenda
(nunca se move por causa da etiqueta) > etiqueta de enquadramento (primeiro espaco livre perto do topo).
"""

from __future__ import annotations

from dataclasses import dataclass

MARGEM = 8
ALVO = 44  # alvo de toque (tokens: alvo_toque_px)
ESPESSURA_BORDA = 6


@dataclass(frozen=True)
class Ret:
    x: float
    y: float
    w: float
    h: float

    @property
    def direita(self) -> float:
        return self.x + self.w

    @property
    def base(self) -> float:
        return self.y + self.h

    @property
    def centro_x(self) -> float:
        return self.x + self.w / 2

    def cruza(self, o: "Ret | None") -> bool:
        """Sobreposicao com area (encostar na borda nao conta)."""
        if o is None:
            return False
        return self.x < o.direita and o.x < self.direita and self.y < o.base and o.y < self.base

    def dentro(self, o: "Ret") -> bool:
        return (
            self.x >= o.x - 0.5
            and self.y >= o.y - 0.5
            and self.direita <= o.direita + 0.5
            and self.base <= o.base + 0.5
        )


def cabeca_na_tela(cabeca_norm, imagem: Ret, espelhada: bool = True) -> Ret | None:
    """Caixa da cabeca (normalizada pela imagem ORIGINAL) em pixels da previa, que e espelhada."""
    if not cabeca_norm:
        return None
    x0, y0, x1, y1 = cabeca_norm
    if espelhada:
        x0, x1 = 1 - x1, 1 - x0
    return Ret(imagem.x + x0 * imagem.w, imagem.y + y0 * imagem.h, (x1 - x0) * imagem.w, (y1 - y0) * imagem.h)


def lado_na_tela(lado: str, espelhada: bool = True) -> str:
    """Lado da imagem original -> lado na previa espelhada (esquerda vira direita)."""
    if espelhada and lado in ("esquerda", "direita"):
        return "direita" if lado == "esquerda" else "esquerda"
    return lado


class SuavizadorCabeca:
    """Suavizacao exponencial da caixa da cabeca (a legenda nao treme). Sem pessoa por mais de
    `perda_s` segundos devolve None (a legenda volta a faixa fixa no alto)."""

    def __init__(self, alfa: float = 0.25, perda_s: float = 1.0):
        self.alfa, self.perda_s = alfa, perda_s
        self.atual: list[float] | None = None
        self.t_visto: float | None = None

    def atualizar(self, caixa, t: float):
        if caixa:
            self.t_visto = t
            self.atual = (
                list(caixa) if self.atual is None else [a + self.alfa * (c - a) for a, c in zip(self.atual, caixa)]
            )
        elif self.t_visto is None or t - self.t_visto > self.perda_s:
            self.atual = None
        return self.atual


def posicionar(
    janela: Ret,
    imagem: Ret,
    *,
    legenda: tuple[float, float] | None = None,
    posicao: str = "embaixo",
    cabeca: Ret | None = None,
    botao_config: bool = True,
    barra_tecnica: float = 0.0,
    etiqueta: tuple[float, float] | None = None,
    barra_minima: float = 0.0,
    bordas: list[str] | tuple = (),
) -> dict:
    """Retangulos de cada sobreposicao (ausente = None) e a posicao efetiva da legenda."""
    m = MARGEM
    out: dict = {
        "barra_minima": None,
        "botao_config": None,
        "barra_tecnica": None,
        "legenda": None,
        "etiqueta": None,
        "bordas": {},
        "posicao_efetiva": None,
    }
    if barra_minima > 0:
        out["barra_minima"] = Ret(janela.x, janela.base - barra_minima, janela.w, barra_minima)
    topo = imagem.y + m
    lim_inf = imagem.base - m
    if out["barra_minima"] is not None:
        lim_inf = min(lim_inf, out["barra_minima"].y - m)
    btn = None
    if botao_config:
        # canto superior direito da JANELA (nao da imagem): em tela larga a imagem 4:3 fica centralizada
        # e o botao no canto dela parecia solto no meio da tela
        btn = Ret(janela.direita - m - ALVO, janela.y + m, ALVO, ALVO)
        out["botao_config"] = btn
    linha_topo = btn.base if btn else topo - m  # a legenda no alto fica abaixo da fila do botao
    if barra_tecnica > 0:
        fim = min(btn.x, imagem.direita) - m if btn else imagem.direita - m
        out["barra_tecnica"] = Ret(imagem.x + m, imagem.y + m, max(0.0, fim - imagem.x - m), barra_tecnica)
        linha_topo = max(linha_topo, out["barra_tecnica"].base)
    topo_livre = linha_topo + m

    # ---------------------------------------------------------------- legenda
    if legenda:
        w, h = min(legenda[0], imagem.w - 2 * m), legenda[1]
        cx = imagem.x + imagem.w / 2

        def ret_em(cx_, y_):
            x_ = min(max(cx_ - w / 2, imagem.x + m), imagem.direita - m - w)
            return Ret(x_, y_, w, h)

        efetiva = posicao
        if posicao == "pessoa":
            if cabeca is None:
                efetiva = "topo"  # pessoa nao detectada (o chamador ja aplicou o 1 s)
            else:
                y = cabeca.y - m - h
                efetiva = "pessoa" if y >= topo_livre else "embaixo"  # sem espaco acima: nunca desce sobre o rosto
        if efetiva == "pessoa":
            L = ret_em(cabeca.centro_x, cabeca.y - m - h)
        elif efetiva == "topo":
            L = ret_em(cx, topo_livre)
        else:
            L = ret_em(cx, lim_inf - h)
        if L.base > lim_inf:  # imagem baixa: sobe ate caber, sem passar do topo livre
            L = Ret(L.x, max(topo_livre, lim_inf - h), L.w, L.h)
        out["legenda"], out["posicao_efetiva"] = L, efetiva

    ocupados = [r for r in (out["barra_minima"], btn, out["barra_tecnica"], out["legenda"]) if r is not None]

    # ---------------------------------------------------------------- etiqueta de enquadramento
    if etiqueta:
        ew, eh = etiqueta
        dir_max = (btn.x - m) if btn else imagem.direita - m
        ew = min(ew, max(0.0, dir_max - imagem.x - m))
        topo_etq = (out["barra_tecnica"].base + m) if out["barra_tecnica"] else imagem.y + m
        candidatos = [Ret(imagem.x + m, topo_etq, ew, eh)]
        if out["legenda"] is not None:
            candidatos.append(Ret(imagem.x + m, out["legenda"].base + m, ew, eh))
            candidatos.append(Ret(imagem.x + m, out["legenda"].y - m - eh, ew, eh))
        candidatos.append(Ret(imagem.x + m, topo_livre, min(etiqueta[0], imagem.w - 2 * m), eh))
        # faixa preta acima da imagem (janela estreita, imagem centralizada): fora do video, mas visivel
        candidatos.append(Ret(imagem.x + m, imagem.y - m - eh, min(etiqueta[0], imagem.w - 2 * m), eh))
        area = Ret(janela.x, janela.y, janela.w, lim_inf + m - janela.y)
        for c in candidatos:
            if c.w > 0 and c.dentro(area) and not any(c.cruza(o) for o in ocupados):
                out["etiqueta"] = c
                ocupados.append(c)
                break

    # ---------------------------------------------------------------- barras de mao cortada (na borda da imagem)
    e = ESPESSURA_BORDA
    for lado in bordas:
        if lado == "esquerda":
            r = Ret(imagem.x, topo_livre, e, max(0.0, lim_inf - topo_livre))
        elif lado == "direita":
            r = Ret(imagem.direita - e, topo_livre, e, max(0.0, lim_inf - topo_livre))
        elif lado == "topo":
            r = Ret(imagem.x + m, imagem.y, max(0.0, imagem.w - 2 * m), e)
            if btn is not None and r.cruza(btn):  # a barra termina antes do botao, que fica no canto da janela
                r = Ret(r.x, r.y, max(0.0, btn.x - m - r.x), e)
        else:
            r = Ret(imagem.x + m, lim_inf + m - e, max(0.0, imagem.w - 2 * m), e)
        out["bordas"][lado] = r
    return out
