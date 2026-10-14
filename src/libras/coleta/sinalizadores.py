"""Cadastro anonimo de sinalizadores com registro obrigatorio de TCLE.

Nenhum nome real e armazenado aqui. A correspondencia entre ID (S01, S02...) e a
pessoa fica apenas no TCLE fisico/assinado, guardado pela equipe fora do repositorio.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path

from ..config import Config
from ..erros import ErroConsentimento, ErroLibras
from ..tabelas import guardar_backup, ler_tabela
from ..utils import agora_iso, carimbo, escrever_csv_atomico

COLUNAS = ["signer_id", "tcle_confirmado", "data_tcle", "mao_dominante", "data_cadastro", "observacoes"]
_RX_ID = re.compile(r"^S\d{2,3}$")


@dataclass
class Sinalizador:
    signer_id: str
    tcle_confirmado: bool
    data_tcle: str
    mao_dominante: str
    data_cadastro: str
    observacoes: str = ""

    def como_linha(self) -> dict:
        d = asdict(self)
        d["tcle_confirmado"] = "sim" if self.tcle_confirmado else "nao"
        return d


def id_valido(signer_id: str) -> bool:
    return bool(_RX_ID.match(signer_id or ""))


class CadastroSinalizadores:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.arquivo: Path = cfg.caminho("sinalizadores")
        self._itens: dict[str, Sinalizador] = {}
        self.avisos: list[str] = []
        self.recarregar()

    # ------------------------------------------------------------------
    def recarregar(self) -> None:
        import logging

        self._itens = {}
        leitura = ler_tabela(
            self.arquivo,
            COLUNAS,
            ["signer_id"],
            "Cadastro de sinalizadores",
            validar_linha=lambda d: None if id_valido(d["signer_id"].strip()) else f"ID invalido {d['signer_id']!r}",
        )
        self.avisos = list(leitura.avisos)
        if leitura.precisa_backup:
            b = guardar_backup(self.arquivo, leitura, self.cfg.caminho("backup"), carimbo())
            if b:
                self.avisos.append(f"Copia do arquivo original guardada em {b}.")
        for a in self.avisos:
            logging.getLogger("libras.sinalizadores").warning(a)
        for linha in leitura.linhas:
            linha = {k: (v or "").strip() for k, v in linha.items()}
            s = Sinalizador(
                signer_id=linha["signer_id"],
                tcle_confirmado=linha.get("tcle_confirmado", "").strip().lower() == "sim",
                data_tcle=linha.get("data_tcle", ""),
                mao_dominante=linha.get("mao_dominante", ""),
                data_cadastro=linha.get("data_cadastro", ""),
                observacoes=linha.get("observacoes", ""),
            )
            self._itens[s.signer_id] = s

    def _salvar(self) -> None:
        escrever_csv_atomico(self.arquivo, COLUNAS, (s.como_linha() for s in self.listar()))

    # ------------------------------------------------------------------
    def listar(self) -> list[Sinalizador]:
        return [self._itens[k] for k in sorted(self._itens)]

    def obter(self, signer_id: str) -> Sinalizador | None:
        return self._itens.get(signer_id)

    def ids(self) -> list[str]:
        return sorted(self._itens)

    def ids_com_tcle(self) -> list[str]:
        return [s.signer_id for s in self.listar() if s.tcle_confirmado]

    def proximo_id(self) -> str:
        nums = [int(i[1:]) for i in self._itens if id_valido(i)]
        n = (max(nums) + 1) if nums else 1
        return f"S{n:02d}" if n < 100 else f"S{n:03d}"

    def cadastrar(
        self, mao_dominante: str, tcle_confirmado: bool, data_tcle: str = "", observacoes: str = ""
    ) -> Sinalizador:
        opcoes = self.cfg["coleta"]["opcoes"]["mao_dominante"]
        if mao_dominante not in opcoes:
            raise ErroLibras(f"Mao dominante deve ser uma de {opcoes}.")
        if tcle_confirmado and not data_tcle:
            raise ErroLibras("Informe a data de assinatura do TCLE.")
        s = Sinalizador(
            signer_id=self.proximo_id(),
            tcle_confirmado=bool(tcle_confirmado),
            data_tcle=data_tcle if tcle_confirmado else "",
            mao_dominante=mao_dominante,
            data_cadastro=agora_iso(),
            observacoes=observacoes.strip(),
        )
        self._itens[s.signer_id] = s
        self._salvar()
        return s

    def confirmar_tcle(self, signer_id: str, data_tcle: str) -> Sinalizador:
        s = self._exigir_existente(signer_id)
        if not data_tcle:
            raise ErroLibras("Informe a data de assinatura do TCLE.")
        s.tcle_confirmado = True
        s.data_tcle = data_tcle
        self._salvar()
        return s

    def revogar_tcle(self, signer_id: str) -> Sinalizador:
        """Registra retirada de consentimento. Os videos devem ser removidos (ver docs)."""
        s = self._exigir_existente(signer_id)
        s.tcle_confirmado = False
        self._salvar()
        return s

    def remover(self, signer_id: str) -> None:
        """Tira o sinalizador do cadastro (use coleta.gestao.remover_sinalizador, que cuida dos dados)."""
        self._exigir_existente(signer_id)
        del self._itens[signer_id]
        self._salvar()

    def exigir_tcle(self, signer_id: str) -> Sinalizador:
        """Bloqueia qualquer gravacao/importacao sem TCLE confirmado."""
        s = self._exigir_existente(signer_id)
        if not s.tcle_confirmado:
            raise ErroConsentimento(
                f"O sinalizador {signer_id} nao tem TCLE confirmado. Nada pode ser gravado ou "
                "importado para ele ate que o termo esteja assinado e registrado."
            )
        return s

    def _exigir_existente(self, signer_id: str) -> Sinalizador:
        s = self._itens.get(signer_id)
        if s is None:
            raise ErroLibras(
                f"Sinalizador {signer_id!r} nao esta cadastrado. Cadastre-o primeiro "
                "(aba Sinalizadores ou scripts/sinalizadores.py)."
            )
        return s
