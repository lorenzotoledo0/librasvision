"""Teste ao vivo do modelo com a camera (item 10), sem interface grafica.

- listar_modelos: modelos finais disponiveis em runs/ (modelo.keras + metadados.json).
- etapa_decisao: em que etapa da decisao a predicao parou (para o usuario entender por que nada
  aparece): sem mao, confianca abaixo do limiar, aguardando N predicoes iguais, sinal confirmado.
- SessaoTeste: modo "Testar acertos". A pessoa escolhe o sinal esperado, faz o sinal e registra a
  tentativa; compara com a classe de maior probabilidade media (media movel das ultimas predicoes,
  sem o limiar de exibicao). Contagens por classe, acuracia e matriz de confusao da sessao.

Isto e um teste QUALITATIVO: nao substitui a avaliacao por particoes (D13) e nao vira numero do
TCC. Todas as saidas levam esse aviso. Nenhum codigo de augmentation roda aqui (D8), e a
normalizacao, as janelas e a decisao sao as mesmas do tempo real (inferencia.py).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from .decisao import TEXTOS_ETAPA, etapa_decisao  # noqa: F401 - reexportados para a bancada

AVISO_TESTE = "Teste qualitativo ao vivo; não substitui a avaliação por partições (D13) e não pode virar número do TCC."
SEM_PREDICAO = "(sem predição)"


# ----------------------------------------------------------------------
# Modelos disponiveis
# ----------------------------------------------------------------------
@dataclass
class InfoModelo:
    pasta: Path
    nome: str
    data: str
    classes: list[str]
    protocolo: str
    ensaio: bool
    sinalizadores: list[str]
    videos: int | None

    def rotulo(self) -> str:
        return f"{self.data} | {len(self.classes)} classes: {', '.join(self.classes)}" + (
            " | ENSAIO" if self.ensaio else ""
        )


def _data_da_pasta(nome: str) -> str:
    m = re.match(r"^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})", nome)
    if not m:
        return "-"
    a, mes, d, h, mi, _s = m.groups()
    return f"{d}/{mes}/{a} {h}:{mi}"


def listar_modelos(cfg) -> list[InfoModelo]:
    """Modelos finais em runs/ (mais recente primeiro)."""
    runs = cfg.caminho("runs")
    out = []
    if not runs.exists():
        return out
    for meta_arq in runs.glob("*/metadados.json"):
        pasta = meta_arq.parent
        if not (pasta / "modelo.keras").exists():
            continue
        try:
            meta = json.loads(meta_arq.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        val = meta.get("validacao_por")
        out.append(
            InfoModelo(
                pasta,
                pasta.name,
                meta.get("criado_em") and _formatar_iso(meta["criado_em"]) or _data_da_pasta(pasta.name),
                list(meta.get("classes", [])),
                meta.get("protocolo") or ("final, validação por " + (val or "-")),
                bool(meta.get("ensaio")),
                list(meta.get("sinalizadores", [])),
                meta.get("videos"),
            )
        )
    out.sort(key=lambda m: m.nome, reverse=True)
    return out


def _formatar_iso(s: str) -> str:
    try:
        return datetime.fromisoformat(s).strftime("%d/%m/%Y %H:%M")
    except ValueError:
        return s


def aviso_sinalizador_visto(meta: dict, signer_id: str | None) -> str | None:
    """Se quem esta testando teve videos no treino do modelo, o resultado tende a ser otimista (D6)."""
    if signer_id and signer_id in (meta.get("sinalizadores") or []):
        return (
            f"{signer_id} tem vídeos no treino deste modelo: o resultado tende a ser otimista, porque o modelo "
            "já viu esta pessoa (D6). Para uma impressão mais honesta, teste com alguém que não está no treino."
        )
    return None


# ----------------------------------------------------------------------
# Etapa da decisao
# ----------------------------------------------------------------------
# TEXTOS_ETAPA e etapa_decisao ficam em libras.decisao (compartilhado com o app real)


def classe_prevista(probs, motivo: str, classes: list[str], nome_nenhum_sinal: str | None) -> str:
    """Classe de maior probabilidade media (ignora o limiar de exibicao). Sem maos visiveis conta como
    nenhum_sinal quando o modelo tem essa classe; sem janela valida, nao ha predicao."""
    if probs is not None and len(probs) == len(classes):
        return classes[int(np.argmax(probs))]
    if motivo == "sem_maos" and nome_nenhum_sinal in classes:
        return nome_nenhum_sinal
    return SEM_PREDICAO


# ----------------------------------------------------------------------
# Sessao "Testar acertos"
# ----------------------------------------------------------------------
@dataclass
class SessaoTeste:
    classes: list[str]
    nome_nenhum_sinal: str | None = None
    modelo: str = ""
    ensaio: bool = False
    tentativas: list[dict] = field(default_factory=list)

    @property
    def colunas(self) -> list[str]:
        """Classes previstas possiveis: as do modelo e, se ocorreu, 'sem predicao'."""
        extra = [SEM_PREDICAO] if any(t["previsto"] == SEM_PREDICAO for t in self.tentativas) else []
        return list(self.classes) + extra

    def registrar(self, esperado: str, probs, motivo: str, confianca: float = 0.0, t: float | None = None) -> dict:
        if esperado not in self.classes:
            raise ValueError(f"A classe {esperado} nao pertence ao modelo.")
        previsto = classe_prevista(probs, motivo, self.classes, self.nome_nenhum_sinal)
        tent = {
            "n": len(self.tentativas) + 1,
            "esperado": esperado,
            "previsto": previsto,
            "acerto": previsto == esperado,
            "confianca": round(float(confianca), 4),
            "motivo": motivo,
            "probs": None if probs is None else [round(float(x), 4) for x in probs],
            "instante": datetime.now().isoformat(timespec="seconds") if t is None else t,
        }
        self.tentativas.append(tent)
        return tent

    def zerar(self) -> None:
        self.tentativas.clear()

    def por_classe(self) -> dict[str, dict]:
        out = {c: {"acertos": 0, "erros": 0} for c in self.classes}
        for t in self.tentativas:
            out[t["esperado"]]["acertos" if t["acerto"] else "erros"] += 1
        return out

    def acuracia(self) -> float | None:
        if not self.tentativas:
            return None
        return sum(t["acerto"] for t in self.tentativas) / len(self.tentativas)

    def matriz(self) -> np.ndarray:
        """Linhas: esperado (classes do modelo). Colunas: previsto (self.colunas)."""
        cols = self.colunas
        m = np.zeros((len(self.classes), len(cols)), dtype=int)
        for t in self.tentativas:
            m[self.classes.index(t["esperado"]), cols.index(t["previsto"])] += 1
        return m

    def como_dict(self, extra: dict | None = None) -> dict:
        from .utils import hardware, versoes

        d = {
            "aviso": AVISO_TESTE,
            "ensaio": self.ensaio,
            "modelo": self.modelo,
            "classes": self.classes,
            "colunas_matriz": self.colunas,
            "matriz_esperado_x_previsto": self.matriz().tolist(),
            "por_classe": self.por_classe(),
            "acuracia_sessao": self.acuracia(),
            "tentativas": self.tentativas,
            "hardware": hardware(),
            "versoes": versoes(),
            "salvo_em": datetime.now().isoformat(timespec="seconds"),
        }
        if self.ensaio:
            d["etiqueta"] = "ENSAIO: modelo treinado abaixo dos mínimos; não usar no TCC."
        d.update(extra or {})
        return d

    def salvar(self, cfg, extra: dict | None = None) -> Path:
        from .utils import carimbo, escrever_texto_atomico

        p = cfg.caminho("runs") / "tempo_real" / f"teste_{carimbo()}.json"
        escrever_texto_atomico(p, json.dumps(self.como_dict(extra), indent=2, ensure_ascii=False, default=float))
        return p
