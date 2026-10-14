"""Modelos do app real (13.2): pacote ativo, pacote novo, importar e voltar ao anterior.

Os pacotes ficam na pasta de modelos (padrao `publicados/` ao lado do app; configuravel). O hash e
conferido antes de qualquer uso (libras.pacote). Nunca ha predicao simulada: sem pacote valido, a tela
explica e oferece importar.
"""

from __future__ import annotations

from pathlib import Path

from .. import config_real
from .. import pacote as pk


class ControleModelos:
    def __init__(self, cfg_real, prefs):
        self.cfg = cfg_real
        self.prefs = prefs

    @property
    def base(self) -> Path:
        p = self.prefs["pasta_modelos"]
        return Path(p) if p else config_real.pasta_modelos(self.cfg)

    def listar(self) -> list[dict]:
        return pk.listar_pacotes(self.base)

    def validos(self) -> list[dict]:
        return [it for it in self.listar() if it["erro"] is None]

    def ativo(self) -> dict | None:
        """Pacote ativo (conferido). Se o escolhido esta na pasta mas invalido (alterado, incompleto),
        devolve None: a tela mostra o problema e o usuario escolhe o que fazer (nunca troca calado).
        Se o escolhido sumiu da pasta (ou nenhum foi escolhido), usa o mais recente valido."""
        ident = self.prefs["modelo"]["ativo"]
        todos = self.listar()
        for it in todos:
            if it["id"] == ident:
                return it if it["erro"] is None else None
        vals = [it for it in todos if it["erro"] is None]
        if vals:
            self.prefs["modelo"]["ativo"] = vals[0]["id"]
            self.prefs.salvar()
            return vals[0]
        return None

    def problema_do_ativo(self) -> str | None:
        """Mensagem clara se o pacote escolhido existe mas esta invalido (alterado, incompleto)."""
        ident = self.prefs["modelo"]["ativo"]
        for it in self.listar():
            if it["id"] == ident and it["erro"]:
                return it["erro"]
        return None

    def mais_recente_valido(self) -> dict | None:
        vals = self.validos()
        return vals[0] if vals else None

    def novo_disponivel(self) -> dict | None:
        """Pacote valido publicado depois do ativo, ainda nao oferecido e recusado."""
        atual = self.ativo()
        if atual is None:
            return None
        data_atual = (atual["cartao"] or {}).get("publicado_em") or ""
        for it in self.validos():
            if it["id"] == atual["id"] or it["id"] in self.prefs["modelo"]["recusados"]:
                continue
            if ((it["cartao"] or {}).get("publicado_em") or "") > data_atual:
                return it
        return None

    def trocar(self, ident: str) -> None:
        m = self.prefs["modelo"]
        if m["ativo"] != ident:
            m["anterior"], m["ativo"] = m["ativo"], ident
            self.prefs.salvar()

    def recusar(self, ident: str) -> None:
        if ident not in self.prefs["modelo"]["recusados"]:
            self.prefs["modelo"]["recusados"].append(ident)
            self.prefs.salvar()

    def pode_voltar(self) -> bool:
        ant = self.prefs["modelo"]["anterior"]
        return bool(ant) and any(it["id"] == ant for it in self.validos())

    def voltar_anterior(self) -> str | None:
        if not self.pode_voltar():
            return None
        ant = self.prefs["modelo"]["anterior"]
        self.trocar(ant)
        return ant

    def importar(self, arquivo_zip: Path) -> dict:
        """Confere e importa um .zip; o pacote importado passa a ser o ativo (o anterior fica guardado)."""
        pasta = pk.importar_zip(arquivo_zip, self.base)
        cartao = pk.verificar_pacote(pasta)
        self.trocar(cartao["id"])
        return {"pasta": pasta, "id": cartao["id"], "cartao": cartao, "erro": None}
