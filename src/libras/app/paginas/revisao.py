"""Revisao: lista de takes (aprovar, rejeitar, excluir) e o editor de corte (item 9).

O editor so corta: trecho mantido (inicio e fim) e, se quiser, divisao em dois takes. Nada e
gravado ate "Salvar corte". Ao salvar: novo .mp4 e novo .npz com instantes reindexados a partir
de 0, original em data/lixeira/cortes/, manifesto registrando o corte (regras de D9, em
coleta.cortes). Nao existe remocao de trecho interno. Minimo de 30 quadros por trecho.
"""

from __future__ import annotations

from PySide6.QtWidgets import QComboBox

from ...coleta.cortes import aplicar_edicao
from ...coleta.manifesto import Manifesto
from ...ui import tema
from ...ui.dinamica import Colunas
from ..base import Pagina, Tabela, botao, info, iniciar_tarefa, linha, painel, perguntar, rotulo
from ..editor_corte import EditorCorte


class PaginaClasse(Pagina):
    titulo = "Revisão"
    subtitulo = (
        "Assista cada take com o esqueleto e aprove, rejeite ou exclua. Para cortar, arraste as alças da "
        "linha do tempo ou use I e O; S divide em dois takes. Só vídeos 'ok' entram no treino."
    )

    def __init__(self, app):
        super().__init__(app)
        self.video_id: str | None = None
        h = Colunas(1000)
        p, lay = painel()
        self.f_status = QComboBox()
        self.f_status.addItems(["pendente", "todos", "ok", "rejeitado"])
        self.f_cls, self.f_sig = QComboBox(), QComboBox()
        for cb in (self.f_status, self.f_cls, self.f_sig):
            cb.currentIndexChanged.connect(self._carregar_lista)
        lay.addLayout(linha(rotulo("Status"), self.f_status))
        lay.addLayout(linha(rotulo("Palavra"), self.f_cls, rotulo("Sinalizador"), self.f_sig))
        self.tabela = Tabela(["video_id", "Status", "Quadros", "Duração", "Mão", "Avisos", "Corte"])
        self.tabela.itemSelectionChanged.connect(self._selecionar)
        lay.addWidget(self.tabela, 1)
        lay.addLayout(
            linha(
                botao("Aprovar (ok)", primario=True, acao=lambda: self._status("ok")),
                botao("Rejeitar", acao=lambda: self._status("rejeitado")),
                botao("Excluir (lixeira)", perigo=True, acao=self._excluir),
            )
        )
        self.detalhes = rotulo("", "secundario", quebra=True)
        lay.addWidget(self.detalhes)
        h.addWidget(p, 2)
        pe, le = painel()
        self.editor = EditorCorte()
        self.editor.salvar_pedido.connect(self._salvar_corte)
        le.addWidget(self.editor, 1)
        h.addWidget(pe, 3)
        self.raiz.addWidget(h, 1)

    # ------------------------------------------------------------------
    def ao_entrar(self) -> None:
        self._atualizar_filtros()
        self._carregar_lista()

    def ao_sair(self) -> None:
        self.editor.pausar()

    def encerrar(self) -> None:
        self.editor.encerrado = True
        self.editor.liberar()

    def atualizar_dados(self, tipos: set) -> None:
        if tipos & {"manifesto", "manual"}:
            self._atualizar_filtros()
            self._carregar_lista()

    def _atualizar_filtros(self) -> None:
        man = Manifesto(self.cfg)
        for cb, chave in ((self.f_cls, "classe"), (self.f_sig, "signer_id")):
            atual = cb.currentText()
            cb.blockSignals(True)
            cb.clear()
            cb.addItems(["todos"] + sorted({l[chave] for l in man.linhas}))
            cb.setCurrentText(atual if atual else "todos")
            cb.blockSignals(False)

    def _carregar_lista(self) -> None:
        st, cl, sg = self.f_status.currentText(), self.f_cls.currentText(), self.f_sig.currentText()
        linhas, cores = [], []
        for l in reversed(Manifesto(self.cfg).linhas):
            if (
                (st != "todos" and l["status"] != st)
                or (cl not in ("todos", "") and l["classe"] != cl)
                or (sg not in ("todos", "") and l["signer_id"] != sg)
            ):
                continue
            linhas.append(
                [
                    l["video_id"],
                    l["status"],
                    l["n_frames"],
                    l["duracao_s"],
                    l["taxa_mao"],
                    l["flags_qualidade"],
                    l.get("corte", ""),
                ]
            )
            cores.append({"pendente": tema.CORES["aviso"], "rejeitado": tema.CORES["erro"]}.get(l["status"]))
        self.tabela.preencher(linhas, cores)

    def _selecionar(self) -> None:
        vid = self.tabela.valor_selecionado(0)
        if not vid or vid == self.video_id:
            return
        if self.editor.tem_alteracoes and not perguntar(
            self, f"Há um corte não salvo em {self.video_id}. Descartar e abrir {vid}?"
        ):
            self.tabela.blockSignals(True)
            self.tabela.selecionar_valor(self.video_id)
            self.tabela.blockSignals(False)
            return
        self.carregar(vid)

    def carregar(self, vid: str) -> None:
        from ...extracao import cache_valido, caminho_npz, carregar_npz, meta_esperada

        man = Manifesto(self.cfg)
        l = man.obter(vid)
        if l is None:
            self.editor.limpar()
            self.video_id = None
            return
        self.video_id = vid
        npz = caminho_npz(self.cfg, l)
        dados = carregar_npz(npz) if cache_valido(npz, meta_esperada(self.cfg, l)) else None
        fps = float(l.get("fps_medido") or 0) or 30.0
        try:
            self.editor.carregar(vid, str(man.caminho_absoluto(l)), dados, fps)
        except (OSError, ValueError) as e:
            self.editor.limpar()
            self.detalhes.setText(f"{vid}: não foi possível abrir ({e}).")
            return
        self.detalhes.setText(
            f"{vid} | {l['classe']} | {l['signer_id']} | origem {l['origem']} | FPS {l['fps_medido']} | "
            f"mão {l['taxa_mao']} | corpo {l['taxa_pose']} | iluminação {l['iluminacao'] or '-'} | "
            f"distância {l['distancia'] or '-'}"
            + ("" if dados is not None else " | landmarks indisponíveis (rode Dataset > Extrair)")
            + (f"\nCorte: {l['corte']} (quadros originais: {l['frames_originais']})" if l.get("corte") else "")
        )

    # ------------------------------------------------------------------
    def _status(self, st: str) -> None:
        if self.video_id:
            Manifesto(self.cfg).definir_status(self.video_id, st, observacao=f"revisado: {st}")
            self._carregar_lista()

    def _excluir(self) -> None:
        if self.video_id and perguntar(self, f"Mover {self.video_id} para data/lixeira e tirar do manifesto?"):
            self.editor.liberar()  # fecha o arquivo antes de mover
            Manifesto(self.cfg).excluir(self.video_id)
            self.video_id = None
            self.editor.limpar()
            self._carregar_lista()

    def _salvar_corte(self, inicio: int, fim: int, divisao) -> None:
        vid, cfg = self.video_id, self.cfg
        if not vid:
            return
        partes = f"manter os quadros {inicio + 1} a {fim}" + (
            f" e dividir no quadro {divisao + 1}" if divisao is not None else ""
        )
        if not perguntar(self, f"Salvar o corte de {vid}: {partes}? O original vai para data/lixeira/cortes/."):
            return
        self.editor.liberar()  # o arquivo original precisa estar fechado para ir para a lixeira

        def fim_(r):
            ids = ", ".join(f"{x['video_id']} ({x['n_frames']} quadros)" for x in r)
            info(self, f"Corte salvo: {ids}.")
            self.video_id = None
            self._carregar_lista()
            self.tabela.selecionar_valor(vid)
            self.carregar(vid)

        def falhou(e):
            from ..base import erro

            erro(self, e)
            self.video_id = None
            self.carregar(vid)

        iniciar_tarefa(
            self,
            lambda p, c: aplicar_edicao(cfg, vid, inicio, fim, divisao),
            fim_,
            falhou,
            descricao="Salvando o corte...",
        )
