"""Sinalizadores: cadastro anonimo, TCLE, remocao e exclusao definitiva."""

from __future__ import annotations

from PySide6.QtCore import QDate
from PySide6.QtWidgets import QCheckBox, QComboBox, QDateEdit, QLineEdit

from ...coleta.gestao import PALAVRA_EXCLUSAO, excluir_definitivamente, remover_sinalizador
from ...coleta.manifesto import Manifesto
from ...coleta.sinalizadores import CadastroSinalizadores
from ...ui import tema
from ...ui.dinamica import Colunas
from ..base import DialogoDigitar, Pagina, Tabela, botao, erro, formulario, info, linha, painel, rotulo


class PaginaClasse(Pagina):
    titulo = "Sinalizadores"
    subtitulo = (
        "Cada pessoa recebe um ID anônimo (S01, S02...). Não escreva nomes: a ligação entre ID e pessoa "
        "fica só no TCLE assinado. Sem TCLE confirmado, nada é gravado nem importado."
    )

    def __init__(self, app):
        super().__init__(app)
        h = Colunas(850)
        h.setSpacing(14)
        p, lay = painel()
        self.tabela = Tabela(["ID", "TCLE", "Data do TCLE", "Mão dominante", "Vídeos", "Cadastro", "Observação"])
        lay.addWidget(self.tabela)
        lay.addLayout(
            linha(
                botao("Confirmar TCLE", acao=self.confirmar),
                botao("Revogar TCLE", acao=self.revogar),
                botao("Remover (lixeira)", perigo=True, acao=self.remover),
                botao("Excluir definitivamente", perigo=True, acao=self.excluir),
            )
        )
        h.addWidget(p, 3)
        f, fl = painel()
        fl.addWidget(rotulo("Novo sinalizador", "titulo_painel"))
        form = formulario()
        self.mao = QComboBox()
        self.tcle = QCheckBox("TCLE assinado")
        self.data = QDateEdit(QDate.currentDate())
        self.data.setCalendarPopup(True)
        self.data.setDisplayFormat("yyyy-MM-dd")
        self.obs = QLineEdit()
        self.obs.setPlaceholderText("sem nomes")
        form.addRow("Mão dominante", self.mao)
        form.addRow("", self.tcle)
        form.addRow("Data do TCLE", self.data)
        form.addRow("Observação", self.obs)
        fl.addLayout(form)
        fl.addWidget(botao("Cadastrar", primario=True, acao=self.cadastrar))
        fl.addWidget(
            rotulo(
                "A remoção move vídeos e landmarks para data/lixeira/<ID>_<data>/. A exclusão "
                "definitiva (retirada de consentimento) apaga o que estiver na lixeira em nome "
                "do sinalizador. As duas invalidam os splits.",
                "secundario",
                quebra=True,
            )
        )
        fl.addStretch(1)
        h.addWidget(f, 2)
        self.raiz.addWidget(h, 1)

    def ao_entrar(self) -> None:
        self.mao.clear()
        self.mao.addItems(self.cfg["coleta"]["opcoes"]["mao_dominante"])
        self._preencher()

    def atualizar_dados(self, tipos: set) -> None:
        if tipos & {"sinalizadores", "manifesto", "manual"}:
            self._preencher()

    def _preencher(self) -> None:
        cad = CadastroSinalizadores(self.cfg)
        n = {}
        for l in Manifesto(self.cfg).linhas:
            n[l["signer_id"]] = n.get(l["signer_id"], 0) + 1
        linhas, cores = [], []
        for s in cad.listar():
            linhas.append(
                [
                    s.signer_id,
                    "sim" if s.tcle_confirmado else "NÃO",
                    s.data_tcle,
                    s.mao_dominante,
                    n.get(s.signer_id, 0),
                    s.data_cadastro,
                    s.observacoes,
                ]
            )
            cores.append(None if s.tcle_confirmado else tema.CORES["aviso"])
        self.tabela.preencher(linhas, cores)

    def _sel(self) -> str | None:
        s = self.tabela.valor_selecionado(0)
        if not s:
            info(self, "Selecione um sinalizador na tabela.")
        return s

    def cadastrar(self) -> None:
        try:
            s = CadastroSinalizadores(self.cfg).cadastrar(
                self.mao.currentText(),
                self.tcle.isChecked(),
                self.data.date().toString("yyyy-MM-dd") if self.tcle.isChecked() else "",
                self.obs.text(),
            )
        except Exception as e:  # noqa: BLE001
            erro(self, e)
            return
        self.obs.clear()
        self.ao_entrar()
        info(self, f"Sinalizador cadastrado como {s.signer_id}. Anote este ID no TCLE físico dessa pessoa.")

    def confirmar(self) -> None:
        s = self._sel()
        if s:
            try:
                CadastroSinalizadores(self.cfg).confirmar_tcle(s, self.data.date().toString("yyyy-MM-dd"))
            except Exception as e:  # noqa: BLE001
                erro(self, e)
            self.ao_entrar()

    def revogar(self) -> None:
        s = self._sel()
        if s:
            CadastroSinalizadores(self.cfg).revogar_tcle(s)
            self.ao_entrar()

    def remover(self) -> None:
        s = self._sel()
        if not s:
            return
        d = DialogoDigitar(
            self,
            "Remover sinalizador",
            f"Todos os vídeos e landmarks de {s} vão para a lixeira, e {s} sai do manifesto e do "
            "cadastro. Os splits serão invalidados.",
            s,
        )
        if d.exec():
            try:
                r = remover_sinalizador(self.cfg, s, d.texto())
                info(
                    self,
                    f"{r.videos} vídeo(s) e {r.landmarks} arquivo(s) de landmarks movidos para "
                    f"{r.pasta_lixeira}.\n\n" + "\n".join(r.avisos),
                )
            except Exception as e:  # noqa: BLE001
                erro(self, e)
            self.ao_entrar()

    def excluir(self) -> None:
        sid = self.tabela.valor_selecionado(0)
        if sid is None:
            from ..base import pedir_texto

            sid = pedir_texto(self, "Excluir definitivamente", "ID do sinalizador já removido (ex.: S03):")
            if not sid:
                return
        d1 = DialogoDigitar(
            self,
            "Excluir definitivamente (1/2)",
            f"Isto APAGA DE VEZ tudo o que estiver na lixeira em nome de {sid}. Use para retirada "
            "de consentimento. Não há como desfazer.",
            sid,
        )
        if not d1.exec():
            return
        d2 = DialogoDigitar(self, "Excluir definitivamente (2/2)", "Última confirmação.", PALAVRA_EXCLUSAO)
        if not d2.exec():
            return
        try:
            n = excluir_definitivamente(self.cfg, sid, d1.texto(), d2.texto())
            info(self, f"{n} arquivo(s) apagado(s) definitivamente.")
        except Exception as e:  # noqa: BLE001
            erro(self, e)
        self.ao_entrar()
