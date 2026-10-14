"""Informacoes tecnicas do app real (13.8): textos da barra no topo e o relatorio para a equipe.

A interface so LE e FORMATA o que o servico mediu (D15); nenhuma medida e recalculada aqui, exceto as
da propria interface (FPS de exibicao e memoria deste processo). O relatorio nao tem imagens, IDs de
sinalizadores nem caminhos (D11) e nao e gravado em disco.
"""

from __future__ import annotations

from ..decisao import estado_resumido
from .preferencias import GRUPOS_TECNICOS

DIVISOR = "  |  "
EXPLICACAO_PULADOS = "quadros pulados para manter o tempo real"


def _num(v, casas: int = 0) -> str:
    if v is None:
        return "-"
    return f"{v:.{casas}f}".replace(".", ",")


def _mem(mb) -> str:
    if mb is None:
        return "-"
    return f"{_num(mb / 1024, 1)} GB" if mb >= 1024 else f"{_num(mb)} MB"


def grupos(meta: dict, extra: dict, explicar: bool = True) -> dict[str, list[str]]:
    """Segmentos de cada grupo. `meta`: ultimo quadro do servico; `extra`: medidas da interface e do
    contexto (fps_exibicao, memoria_interface_mb, modelo, camera)."""
    tp = meta.get("tempos") or {}
    rec = meta.get("tempos_recentes") or {}
    q = meta.get("quadros") or {}
    cap, proc, desc = q.get("capturados"), q.get("processados"), q.get("descartados")
    pct = (100 * desc / cap) if cap else None

    def lat(chave):
        r = rec.get(chave) or {}
        return r.get("mediana_ms"), r.get("p95_ms")

    pp_med, pp_p95 = lat("ponta_a_ponta")
    mp_med, _ = lat("mediapipe")
    mod_med, _ = lat("modelo")
    cam = extra.get("camera") or {}
    pulados = f"Pulados {_num(pct)}%" + (f" ({EXPLICACAO_PULADOS})" if explicar else "")
    return {
        "taxa": [
            f"FPS cam {_num(tp.get('fps_camera'))}",
            f"Proc {_num(tp.get('fps_processado'))}",
            f"Exib {_num(extra.get('fps_exibicao'))}",
        ],
        "quadros": [
            f"Capturados {cap if cap is not None else '-'}",
            f"Processados {proc if proc is not None else '-'}",
            f"Descartados {desc if desc is not None else '-'}",
            pulados,
        ],
        "tempo": [
            f"Lat {_num(pp_med)} ms (p95 {_num(pp_p95)})",
            f"MediaPipe {_num(mp_med)} ms",
            f"Modelo {_num(mod_med)} ms",
        ],
        "recursos": [
            f"RAM interface {_mem(extra.get('memoria_interface_mb'))}",
            f"RAM reconhecimento {_mem(tp.get('memoria_mb'))}",
        ],
        "sistema": [
            f"Modelo {extra.get('modelo') or '-'}",
            f"Câmera {cam.get('indice', '-')} {cam.get('largura', '-')}x{cam.get('altura', '-')}",
            f"Estado {estado_resumido((meta.get('etapa') or {}).get('etapa') or meta.get('motivo'))}",
        ],
    }


def texto_barra(meta: dict, extra: dict, ligados: dict, faixa: str, cabe, max_linhas: int = 1) -> list[str]:
    """Linhas da barra: grupos ligados na ordem Taxa, Quadros, Tempo, Recursos, Sistema. O que nao cabe
    (cabe(texto) -> bool) e omitido por prioridade: sai primeiro o ultimo grupo da ordem."""
    escolhidos = [c for c in GRUPOS_TECNICOS if ligados.get(c)]
    if faixa == "expandida":  # com a explicacao dos pulados, se nenhum grupo precisar sair por isso
        g = grupos(meta, extra, True)
        linhas = _montar(escolhidos, g, cabe, max_linhas)
        if linhas is not None:
            return linhas
    g = grupos(meta, extra, False)
    while escolhidos:
        linhas = _montar(escolhidos, g, cabe, max_linhas)
        if linhas is not None:
            return linhas
        escolhidos = escolhidos[:-1]  # omite o grupo de menor prioridade
    return []


def _montar(escolhidos: list[str], g: dict, cabe, max_linhas: int) -> list[str] | None:
    """Junta os grupos em ate `max_linhas` linhas; None se nao couberem."""
    if not escolhidos:
        return None
    linhas: list[str] = []
    atual = ""
    for c in escolhidos:
        seg = "  ".join(g[c])
        tent = seg if not atual else atual + DIVISOR + seg
        if cabe(tent):
            atual = tent
        else:
            if atual:
                linhas.append(atual)
            atual = seg
    if atual:
        linhas.append(atual)
    if len(linhas) <= max_linhas and all(cabe(l) for l in linhas):
        return linhas
    return None


def relatorio(meta: dict, extra: dict, cartao: dict | None, versao_app: str, hardware: dict) -> str:
    """Texto para a area de transferencia (a equipe pede quando um usuario relata problema)."""
    g = grupos(meta, extra, True)
    linhas = [
        f"LibrasVision {versao_app} - relatório técnico",
        f"Modelo: {(cartao or {}).get('id', '-')} (hash {((cartao or {}).get('hash') or {}).get('modelo.keras', '-')[:16]})",
        f"Computador: {hardware.get('sistema', '-')}, {hardware.get('processador', '-')}, "
        f"{hardware.get('nucleos_logicos', '-')} núcleos, {hardware.get('ram_gb', '-')} GB de RAM",
    ]
    nomes = {"taxa": "Taxa", "quadros": "Quadros", "tempo": "Tempo", "recursos": "Recursos", "sistema": "Sistema"}
    for c in GRUPOS_TECNICOS:
        linhas.append(f"{nomes[c]}: {'; '.join(g[c])}")
    linhas.append(
        "Quadros pulados são normais: o quadro que chega enquanto o anterior é processado é descartado "
        "para manter o tempo real."
    )
    return "\n".join(linhas)
