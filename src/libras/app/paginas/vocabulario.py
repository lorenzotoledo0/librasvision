"""Vocabulario: palavras (classes), metas, arquivamento e remocao."""

from __future__ import annotations

from collections import defaultdict

from PySide6.QtWidgets import QComboBox, QLineEdit, QSpinBox

from ...coleta.gestao import remover_classe
from ...coleta.manifesto import Manifesto
from ...coleta.sinalizadores import CadastroSinalizadores
from ...ui import tema
from ...ui.dinamica import Colunas
from ...vocabulario import Vocabulario
from ..base import DialogoDigitar, Pagina, Tabela, botao, erro, formulario, info, linha, painel, rotulo


class PaginaClasse(Pagina):
    titulo = "Vocabulário"
    subtitulo = (
        "Palavras que o sistema aprende. O nome técnico (slug) é gerado sem acento e não muda depois; "
        "o rótulo pode ser editado. Classes arquivadas saem da gravação e do treino, mas os vídeos ficam."
    )

    def __init__(self, app):
        super().__init__(app)
        h = Colunas(850)
        h.setSpacing(14)
        p, lay = painel()
        self.tabela = Tabela(
            [
                "Slug",
                "Rótulo",
                "Tipo",
                "Meta por sinalizador",
                "Situação",
                "Vídeos 'ok'",
                "Sinalizadores",
                "Exceções de meta",
            ]
        )
        self.tabela.itemSelectionChanged.connect(self._ao_selecionar)
        lay.addWidget(self.tabela)
        lay.addLayout(
            linha(botao("Arquivar / reativar", acao=self.arquivar), botao("Remover", perigo=True, acao=self.remover))
        )
        h.addWidget(p, 3)
        f, fl = painel()
        fl.addWidget(rotulo("Nova palavra", "titulo_painel"))
        form = formulario()
        self.novo = QLineEdit()
        self.novo.setPlaceholderText("ex.: sala de aula")
        self.meta_nova = QSpinBox()
        self.meta_nova.setRange(1, 500)
        self.meta_nova.setValue(10)
        self.tipo = QComboBox()
        self.tipo.addItems(["sinal", "nenhum_sinal"])
        form.addRow("Palavra", self.novo)
        form.addRow("Meta por sinalizador", self.meta_nova)
        form.addRow("Tipo", self.tipo)
        fl.addLayout(form)
        fl.addWidget(botao("Adicionar", primario=True, acao=self.adicionar))
        fl.addSpacing(10)
        fl.addWidget(rotulo("Editar a selecionada", "titulo_painel"))
        f2 = formulario()
        self.rotulo_ed = QLineEdit()
        self.meta_ed = QSpinBox()
        self.meta_ed.setRange(1, 500)
        f2.addRow("Rótulo", self.rotulo_ed)
        f2.addRow("Meta", self.meta_ed)
        fl.addLayout(f2)
        fl.addWidget(botao("Salvar alterações", acao=self.editar))
        fl.addSpacing(10)
        fl.addWidget(rotulo("Meta específica de um sinalizador", "titulo_painel"))
        f3 = formulario()
        self.sig = QComboBox()
        self.meta_sig = QSpinBox()
        self.meta_sig.setRange(1, 500)
        f3.addRow("Sinalizador", self.sig)
        f3.addRow("Meta", self.meta_sig)
        fl.addLayout(f3)
        fl.addLayout(
            linha(
                botao("Aplicar", acao=self.meta_sinalizador),
                botao("Usar a meta da classe", acao=lambda: self.meta_sinalizador(remover=True)),
            )
        )
        fl.addStretch(1)
        h.addWidget(f, 2)
        self.raiz.addWidget(h, 1)

    def ao_entrar(self) -> None:
        self._preencher()
        self.meta_nova.setValue(int(self.cfg["coleta"]["metas"].get("takes_por_classe_por_sinalizador", 10)))
        self.sig.clear()
        self.sig.addItems(CadastroSinalizadores(self.cfg).ids())

    def atualizar_dados(self, tipos: set) -> None:
        if tipos & {"vocabulario", "manifesto", "manual"}:
            self._preencher()
        if tipos & {"sinalizadores", "manual"}:
            atual = self.sig.currentText()
            self.sig.clear()
            self.sig.addItems(CadastroSinalizadores(self.cfg).ids())
            self.sig.setCurrentText(atual)

    def _preencher(self) -> None:
        v = Vocabulario(self.cfg)
        n, sig = defaultdict(int), defaultdict(set)
        for l in Manifesto(self.cfg).linhas:
            if l["status"] == "ok":
                n[l["classe"]] += 1
                sig[l["classe"]].add(l["signer_id"])
        linhas, cores = [], []
        for c in v.todas():
            exc = ", ".join(f"{k}={m}" for k, m in c.metas_sinalizador.items())
            linhas.append(
                [
                    c.slug,
                    c.rotulo,
                    c.tipo,
                    c.meta_por_sinalizador,
                    "ativa" if c.ativa else "arquivada",
                    n[c.slug],
                    len(sig[c.slug]),
                    exc,
                ]
            )
            cores.append(None if c.ativa else tema.CORES["texto3"])
        self.tabela.preencher(linhas, cores)

    def _sel(self) -> str | None:
        s = self.tabela.valor_selecionado(0)
        if not s:
            info(self, "Selecione uma palavra na tabela.")
        return s

    def _ao_selecionar(self) -> None:
        s = self.tabela.valor_selecionado(0)
        if s:
            c = Vocabulario(self.cfg).obter(s)
            self.rotulo_ed.setText(c.rotulo)
            self.meta_ed.setValue(c.meta_por_sinalizador)

    def adicionar(self) -> None:
        try:
            c = Vocabulario(self.cfg).adicionar(self.novo.text(), self.tipo.currentText(), self.meta_nova.value())
        except Exception as e:  # noqa: BLE001
            erro(self, e)
            return
        self.novo.clear()
        self.ao_entrar()
        info(self, f"Palavra criada: {c.rotulo} (slug {c.slug}).")

    def editar(self) -> None:
        s = self._sel()
        if s:
            try:
                Vocabulario(self.cfg).editar(s, self.rotulo_ed.text(), self.meta_ed.value())
            except Exception as e:  # noqa: BLE001
                erro(self, e)
            self.ao_entrar()

    def meta_sinalizador(self, remover: bool = False) -> None:
        s = self._sel()
        if s and self.sig.currentText():
            try:
                Vocabulario(self.cfg).definir_meta_sinalizador(
                    s, self.sig.currentText(), None if remover else self.meta_sig.value()
                )
            except Exception as e:  # noqa: BLE001
                erro(self, e)
            self.ao_entrar()

    def arquivar(self) -> None:
        s = self._sel()
        if s:
            v = Vocabulario(self.cfg)
            v.arquivar(s, ativa=not v.obter(s).ativa)
            self.ao_entrar()

    def remover(self) -> None:
        s = self._sel()
        if not s:
            return
        d = DialogoDigitar(
            self,
            "Remover palavra",
            f"Se {s} tiver vídeos, eles e seus landmarks vão para a lixeira e saem do manifesto; "
            "os splits serão invalidados. Se estiver vazia, só sai do vocabulário.",
            s,
        )
        if d.exec():
            try:
                r = remover_classe(self.cfg, s, d.texto())
                info(
                    self,
                    f"Palavra removida. Vídeos movidos: {r.videos}."
                    + (f"\nLixeira: {r.pasta_lixeira}" if r.pasta_lixeira else ""),
                )
            except Exception as e:  # noqa: BLE001
                erro(self, e)
            self.ao_entrar()
