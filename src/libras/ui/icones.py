"""Icones vetoriais (SVG desenhados para este projeto, traco simples). Sem emojis."""

from __future__ import annotations

_SVG = {
    "inicio": '<path d="M3 11 L12 4 L21 11"/><path d="M5 10 V20 H19 V10"/><path d="M10 20 V14 H14 V20"/>',
    "sinalizadores": '<circle cx="9" cy="8" r="3.2"/><path d="M3 20 C3 15 15 15 15 20"/>'
    '<circle cx="17" cy="9" r="2.4"/><path d="M15.5 14.5 C19 14 21.5 16 21.5 19"/>',
    "vocabulario": '<path d="M4 5 H10 C11.5 5 12 6 12 7 V20 C12 19 11.5 18 10 18 H4 Z"/>'
    '<path d="M20 5 H14 C12.5 5 12 6 12 7 V20 C12 19 12.5 18 14 18 H20 Z"/>',
    "gravar": '<rect x="3" y="6" width="13" height="12" rx="2"/><path d="M16 10 L21 7 V17 L16 14"/>',
    "revisao": '<rect x="4" y="4" width="16" height="16" rx="3"/><path d="M8 12 L11 15 L16 9"/>',
    "dataset": '<ellipse cx="12" cy="6" rx="7" ry="2.5"/><path d="M5 6 V18 C5 19.4 8 20.5 12 20.5 C16 20.5 19 19.4 '
    '19 18 V6"/><path d="M5 12 C5 13.4 8 14.5 12 14.5 C16 14.5 19 13.4 19 12"/>',
    "treino": '<path d="M3 17 L8 11 L12 14 L17 7 L21 10"/><circle cx="8" cy="11" r="1.2"/>'
    '<circle cx="17" cy="7" r="1.2"/>',
    "resultados": '<path d="M5 20 V11"/><path d="M10 20 V5"/><path d="M15 20 V13"/><path d="M20 20 V8"/>'
    '<path d="M3 20 H21"/>',
    "tempo_real": '<circle cx="12" cy="12" r="3"/><path d="M6.5 6.5 C3.5 9.5 3.5 14.5 6.5 17.5"/>'
    '<path d="M17.5 6.5 C20.5 9.5 20.5 14.5 17.5 17.5"/>',
    "configuracoes": '<path d="M4 7 H14"/><path d="M18 7 H20"/><circle cx="16" cy="7" r="2"/>'
    '<path d="M4 17 H8"/><path d="M12 17 H20"/><circle cx="10" cy="17" r="2"/>',
    "alerta": '<path d="M12 4 L21 19 H3 Z"/><path d="M12 10 V14"/><path d="M12 16.5 V17"/>',
    "ok": '<circle cx="12" cy="12" r="8.5"/><path d="M8 12 L11 15 L16 9"/>',
    "pasta": '<path d="M3 7 C3 6 4 5 5 5 H9 L11 7 H19 C20 7 21 8 21 9 V17 C21 18 20 19 19 19 H5 C4 19 3 18 3 17 Z"/>',
    "play": '<path d="M8 5 L19 12 L8 19 Z"/>',
    "menu": '<path d="M4 7 H20"/><path d="M4 12 H20"/><path d="M4 17 H20"/>',
    "atualizar": '<path d="M19 12 A7 7 0 1 1 16.5 6.6"/><path d="M17 3 V7 H13"/>',
    "inicio_video": '<path d="M6 5 V19"/><path d="M18 5 L9 12 L18 19 Z"/>',
    "fim_video": '<path d="M18 5 V19"/><path d="M6 5 L15 12 L6 19 Z"/>',
    "quadro_ant": '<path d="M15 6 L9 12 L15 18"/>',
    "quadro_prox": '<path d="M9 6 L15 12 L9 18"/>',
    "desfazer": '<path d="M9 7 L4 12 L9 17"/><path d="M4 12 H14 C17.5 12 20 14.5 20 18"/>',
    "refazer": '<path d="M15 7 L20 12 L15 17"/><path d="M20 12 H10 C6.5 12 4 14.5 4 18"/>',
    "tesoura": '<circle cx="6" cy="7" r="2.5"/><circle cx="6" cy="17" r="2.5"/><path d="M8 8.5 L20 17"/>'
    '<path d="M8 15.5 L20 7"/>',
    "camera": '<rect x="3" y="7" width="18" height="13" rx="2"/><circle cx="12" cy="13.5" r="3.5"/>'
    '<path d="M8 7 L9.5 4.5 H14.5 L16 7"/>',
    "pausa": '<path d="M8 5 V19"/><path d="M16 5 V19"/>',
    # app real (13.3, 13.8, 13.9)
    "engrenagem": '<path d="M18.05 10.65 L20.53 10.89 L20.53 13.11 L18.05 13.35 L17.24 15.32 L18.82 17.24 L17.24 18.82 L15.32 17.24 L13.35 18.05 L13.11 20.53 L10.89 20.53 L10.65 18.05 L8.68 17.24 L6.76 18.82 L5.18 17.24 L6.76 15.32 L5.95 13.35 L3.47 13.11 L3.47 10.89 L5.95 10.65 L6.76 8.68 L5.18 6.76 L6.76 5.18 L8.68 6.76 L10.65 5.95 L10.89 3.47 L13.11 3.47 L13.35 5.95 L15.32 6.76 L17.24 5.18 L18.82 6.76 L17.24 8.68 Z"/><circle cx="12" cy="12" r="2.8"/>',
    "cadeado": '<rect x="5" y="11" width="14" height="10" rx="2"/><path d="M8 11 V8 C8 5.5 10 4 12 4 C14 4 16 5.5 16 8 V11"/>'
    '<path d="M12 15 V17"/>',
    "tela_cheia": '<path d="M4 9 V4 H9"/><path d="M15 4 H20 V9"/><path d="M20 15 V20 H15"/><path d="M9 20 H4 V15"/>',
    "sair_tela_cheia": '<path d="M9 4 V9 H4"/><path d="M20 9 H15 V4"/><path d="M15 20 V15 H20"/><path d="M4 15 H9 V20"/>',
    "lista": '<path d="M9 6 H20"/><path d="M9 12 H20"/><path d="M9 18 H20"/><circle cx="5" cy="6" r="1"/>'
    '<circle cx="5" cy="12" r="1"/><circle cx="5" cy="18" r="1"/>',
    "mao": '<path d="M8 13 V6.5 C8 5.5 9.5 5.5 9.5 6.5 V12"/><path d="M9.5 11 V5 C9.5 4 11 4 11 5 V11"/>'
    '<path d="M11 11 V5.5 C11 4.5 12.5 4.5 12.5 5.5 V11.5"/><path d="M12.5 11.5 V7 C12.5 6 14 6 14 7 V14 '
    'C14 18 12 20 9.5 20 C7 20 5.5 18.5 4.5 16 L3.5 13.5 C3 12.5 4.5 11.8 5.2 12.8 L8 16"/>',
    "parar": '<rect x="6" y="6" width="12" height="12" rx="1.5"/>',
    "copiar": '<rect x="8" y="8" width="12" height="12" rx="2"/><path d="M16 8 V6 C16 5 15 4 14 4 H6 C5 4 4 5 4 6 '
    'V14 C4 15 5 16 6 16 H8"/>',
    "importar": '<path d="M12 4 V15"/><path d="M7 10 L12 15 L17 10"/><path d="M4 18 V20 H20 V18"/>',
    "fechar": '<path d="M6 6 L18 18"/><path d="M18 6 L6 18"/>',
}


def svg(nome: str, cor: str | None = None, tamanho: int = 24) -> bytes:
    if cor is None:
        from .tema import CORES

        cor = CORES["texto2"]
    corpo = _SVG[nome]
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{tamanho}" height="{tamanho}" viewBox="0 0 24 24" '
        f'fill="none" stroke="{cor}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">'
        f"{corpo}</svg>"
    ).encode()


def icone(nome: str, cor: str | None = None, cor_ativo: str | None = None):
    from .tema import CORES

    cor = cor or CORES["texto2"]
    cor_ativo = cor_ativo or CORES["sobre_acento"]
    from PySide6.QtCore import QByteArray, QSize, Qt
    from PySide6.QtGui import QIcon, QPainter, QPixmap
    from PySide6.QtSvg import QSvgRenderer

    ic = QIcon()
    for c, modo, estado in (
        (cor, QIcon.Normal, QIcon.Off),
        (cor_ativo, QIcon.Normal, QIcon.On),
        (cor_ativo, QIcon.Active, QIcon.On),
    ):
        pm = QPixmap(QSize(48, 48))
        pm.fill(Qt.transparent)
        p = QPainter(pm)
        QSvgRenderer(QByteArray(svg(nome, c, 48))).render(p)
        p.end()
        ic.addPixmap(pm, modo, estado)
    return ic
