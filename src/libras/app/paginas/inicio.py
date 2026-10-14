"""Inicio: resumo do projeto e alertas de integridade."""

from __future__ import annotations

from collections import Counter

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QGridLayout, QHBoxLayout, QVBoxLayout, QWidget

from ..base import EstadoVazio, Pagina, botao, cartao_numero, etiqueta, iniciar_tarefa, linha, painel, rotulo


def coletar_resumo(cfg) -> dict:
    """Calcula o resumo (fora da thread da interface: confere hashes de todos os .npz)."""
    from ...coleta.manifesto import Manifesto
    from ...coleta.sinalizadores import CadastroSinalizadores
    from ...inferencia import localizar_modelo
    from ...splits import PROTOCOLOS, caminho_split
    from ...vocabulario import Vocabulario

    man, cad, voc = Manifesto(cfg), CadastroSinalizadores(cfg), Vocabulario(cfg)
    st = Counter(l["status"] for l in man.linhas)
    integ = man.verificar_integridade(cad.ids(), cad.ids_com_tcle())
    alertas = [("aviso", a) for a in man.avisos + cad.avisos]
    if not integ.ok:
        for linha_ in integ.texto().splitlines()[1:]:
            if linha_.startswith("- "):
                alertas.append(("erro", linha_[2:]))
    sem_lm = sum(1 for l in man.linhas if l["status"] == "ok" and not l.get("landmarks_path"))
    if sem_lm:
        alertas.append(("aviso", f"{sem_lm} video(s) 'ok' sem landmarks: use Dataset > Extrair landmarks."))
    if st.get("pendente"):
        alertas.append(("aviso", f"{st['pendente']} take(s) pendente(s) de revisao (aba Revisao)."))
    sem_tcle = [s.signer_id for s in cad.listar() if not s.tcle_confirmado]
    if sem_tcle:
        alertas.append(("aviso", f"Sinalizadores sem TCLE confirmado: {', '.join(sem_tcle)}."))
    splits = [p for p in PROTOCOLOS if caminho_split(cfg, p).exists()]
    try:
        modelo = localizar_modelo(cfg).name
    except Exception:  # noqa: BLE001
        modelo = None
    from ...publicacao import publicado_atual
    from ...selecao import minimos, sinalizadores_por_classe

    publicado = publicado_atual(cfg)

    sig = sinalizadores_por_classe(cfg)
    r = {
        "sinalizadores": len(cad.ids()),
        "com_tcle": len(cad.ids_com_tcle()),
        "status": dict(st),
        "classes": len(voc.ativas()),
        "arquivadas": len(voc.todas()) - len(voc.ativas()),
        "alertas": alertas,
        "splits": splits,
        "modelo": modelo,
        "integro": integ.ok,
        "classes_com_videos": sorted(sig),
        "minimos": minimos(cfg),
        "abaixo_minimo": sorted(c for c, s in sig.items() if len(s) < minimos(cfg)["sinalizadores_por_classe"]),
        "nenhum_sinal": cfg.nome_nenhum_sinal,
        "publicado": publicado,
    }
    r["proximo"] = proximo_passo(r)
    return r


def proximo_passo(r: dict) -> tuple[str, str, str, str]:
    """(titulo, explicacao, texto do botao, aba) do proximo passo, a partir do resumo."""
    st = r["status"]
    if not r["com_tcle"]:
        return (
            "Cadastre o primeiro sinalizador",
            "Antes de gravar, cada pessoa assina o TCLE e recebe um código (S01, S02...).",
            "Ir para Sinalizadores",
            "sinalizadores",
        )
    if not st.get("ok") and not st.get("pendente"):
        return (
            "Grave os primeiros vídeos",
            "Escolha a pessoa e a palavra na aba Gravar. Cada take tem cerca de 3 s.",
            "Ir para Gravar",
            "gravar",
        )
    if st.get("pendente"):
        return (
            f"Revise {st['pendente']} take(s) pendente(s)",
            "Só vídeos aprovados entram no treino.",
            "Ir para Revisão",
            "revisao",
        )
    com = r["classes_com_videos"]
    if len(com) < r["minimos"]["classes"]:
        falta = r["nenhum_sinal"] if r["nenhum_sinal"] and r["nenhum_sinal"] not in com else "outra palavra"
        return (
            "Ainda não dá para treinar",
            f"É preciso ter vídeos de pelo menos {r['minimos']['classes']} classes. "
            f"Hoje: {', '.join(com) or 'nenhuma'}. Grave '{falta}' para poder treinar.",
            "Ir para Gravar",
            "gravar",
        )
    if r["abaixo_minimo"]:
        return (
            "Grave com mais pessoas",
            "Estas palavras têm vídeos de menos de "
            f"{r['minimos']['sinalizadores_por_classe']} pessoas: {', '.join(r['abaixo_minimo'])}. O teste "
            "precisa de alguém que não esteve no treino.",
            "Ir para Gravar",
            "gravar",
        )
    if not r["modelo"]:
        return (
            "Treine o modelo",
            "Rode a avaliação por partições (números do TCC) e o modelo final para o tempo real.",
            "Ir para Treino",
            "treino",
        )
    return (
        "Teste ao vivo",
        "O modelo final está pronto. Teste com a câmera na aba Tempo real.",
        "Ir para Tempo real",
        "tempo_real",
    )


class GradeCartoes(QWidget):
    """Cartoes de resumo que se reorganizam: 4 colunas em janela larga, 2 na media, 1 na estreita."""

    LARGURA_MIN = 190

    def __init__(self):
        super().__init__()
        self.grade = QGridLayout(self)
        self.grade.setContentsMargins(0, 0, 0, 0)
        self.grade.setSpacing(12)
        self.cartoes: list[QWidget] = []
        self.colunas = 0

    def adicionar(self, w: QWidget) -> None:
        self.cartoes.append(w)
        self.colunas = 0
        self._arrumar(4)

    def _arrumar(self, n: int) -> None:
        if n == self.colunas:
            return
        self.colunas = n
        for w in self.cartoes:
            self.grade.removeWidget(w)
        for i, w in enumerate(self.cartoes):
            self.grade.addWidget(w, i // n, i % n)
        for c in range(4):
            self.grade.setColumnStretch(c, 1 if c < n else 0)

    def resizeEvent(self, ev) -> None:  # noqa: N802
        larg = ev.size().width()
        self._arrumar(4 if larg >= 4 * self.LARGURA_MIN else 2 if larg >= 2 * self.LARGURA_MIN else 1)
        super().resizeEvent(ev)

    def minimumSizeHint(self):  # noqa: N802
        from PySide6.QtCore import QSize

        return QSize(self.LARGURA_MIN, super().minimumSizeHint().height())


def _alerta(tipo: str, texto: str) -> QWidget:
    """Alerta com etiqueta (Erro, Atenção, ok) e texto: o tipo nunca e indicado so pela cor."""
    w = QWidget()
    h = QHBoxLayout(w)
    h.setContentsMargins(0, 0, 0, 0)
    h.setSpacing(10)
    h.addWidget(
        etiqueta({"erro": "Erro", "ok": "ok"}.get(tipo, "Atenção"), {"erro": "erro", "ok": "ok"}.get(tipo, "pendente")),
        0,
        Qt.AlignTop,
    )
    h.addWidget(rotulo(texto, quebra=True), 1)
    return w


class PaginaClasse(Pagina):
    titulo = "Início"
    subtitulo = "Visão geral do dataset, alertas de integridade e próximos passos."

    def __init__(self, app):
        super().__init__(app)
        self.proximo = EstadoVazio("Verificando...", "", "Abrir", self._ir_proximo, icone_nome="ok")
        self.proximo.setObjectName("cartao")
        self._aba_proxima = "inicio"
        self.raiz.addWidget(self.proximo)
        self.grade = GradeCartoes()
        self.c_sig = cartao_numero("Sinalizadores")
        self.c_vid = cartao_numero("Vídeos 'ok'")
        self.c_cls = cartao_numero("Classes ativas")
        self.c_mod = cartao_numero("Modelo para tempo real")
        for c in (self.c_sig, self.c_vid, self.c_cls, self.c_mod):
            c[0].setObjectName("cartao")
            self.grade.adicionar(c[0])
        self.raiz.addWidget(self.grade)
        # modelo publicado para o app real (13.5)
        pp, lpp = painel()
        lpp.addWidget(rotulo("Modelo publicado no app LibrasVision", "titulo_painel", quebra=True))
        self.lb_publicado = rotulo("Verificando...", quebra=True)
        lpp.addWidget(self.lb_publicado)
        lpp.addLayout(linha(botao("Publicar na aba Resultados", acao=lambda: self.app.ir_para("resultados"))))
        self.raiz.addWidget(pp)
        p, lay = painel()
        lay.addWidget(rotulo("Alertas de integridade", "titulo_painel"))
        self.lista = QVBoxLayout()
        self.lista.setSpacing(8)
        lay.addLayout(self.lista)
        self.raiz.addWidget(p)
        p2, lay2 = painel()
        lay2.addWidget(rotulo("Fluxo de trabalho", "titulo_painel"))
        lay2.addWidget(
            rotulo(
                "1. Sinalizadores: cadastre cada participante com o TCLE assinado.\n"
                "2. Vocabulário: confira as palavras e as metas de vídeos.\n"
                "3. Gravar: grave os takes (vídeo e landmarks são salvos juntos).\n"
                "4. Revisão: aprove, corte ou divida os takes pendentes.\n"
                "5. Dataset: extraia o que faltar e valide.\n"
                "6. Treino: escolha as classes e treine; acompanhe as curvas ao vivo.\n"
                "7. Resultados e Tempo real.",
                "secundario",
                quebra=True,
            )
        )
        self.raiz.addWidget(p2)
        self.raiz.addStretch(1)

    def _ir_proximo(self) -> None:
        self.app.ir_para(self._aba_proxima)

    def ao_entrar(self) -> None:
        self._limpar()
        self.lista.addWidget(rotulo("Verificando...", "secundario"))
        iniciar_tarefa(
            self, lambda prog, canc: coletar_resumo(self.cfg), self._mostrar, descricao="Conferindo o dataset..."
        )

    def atualizar_dados(self, tipos: set) -> None:
        self.ao_entrar()

    def _limpar(self) -> None:
        while self.lista.count():
            w = self.lista.takeAt(0).widget()
            if w:
                w.deleteLater()

    def _mostrar(self, r: dict) -> None:
        self._limpar()
        st = r["status"]
        self.c_sig[1].setText(str(r["sinalizadores"]))
        self.c_sig[2].setText(f"{r['com_tcle']} com TCLE confirmado")
        self.c_vid[1].setText(str(st.get("ok", 0)))
        self.c_vid[2].setText(f"{st.get('pendente', 0)} pendentes, {st.get('rejeitado', 0)} rejeitados")
        self.c_cls[1].setText(str(r["classes"]))
        self.c_cls[2].setText(f"{r['arquivadas']} arquivada(s)" if r["arquivadas"] else "")
        pub = r.get("publicado")
        if pub:
            c = pub["cartao"]
            from ... import pacote as pk

            sinais = ", ".join(pk.rotulo(c, x) for x in c["classes"] if x != c.get("nome_nenhum_sinal"))
            self.lb_publicado.setText(
                f"{c['id']} (de {c.get('modelo_origem', '-')}), publicado em "
                f"{c.get('publicado_em', '-')}. Sinais: {sinais}."
            )
        else:
            self.lb_publicado.setText(
                "Nenhum modelo publicado. O app LibrasVision só reconhece com um pacote publicado: "
                "selecione um modelo final na aba Resultados e use Publicar modelo."
            )
        self.c_mod[1].setText("Sim" if r["modelo"] else "Não")
        self.c_mod[2].setText(r["modelo"] or "Treine o modelo final na aba Treino.")
        tit, expl, bt, aba = r["proximo"]
        self.proximo.definir(f"Próximo passo: {tit}", expl)
        self.proximo.botao.setText(bt)
        self._aba_proxima = aba
        if not r["alertas"]:
            self.lista.addWidget(_alerta("ok", "Nenhum problema encontrado."))
        for tipo, txt in r["alertas"][:40]:
            self.lista.addWidget(_alerta(tipo, txt))
