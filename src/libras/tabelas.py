"""Leitura tolerante das tabelas CSV do projeto (manifesto e cadastro de sinalizadores).

Problema real que motivou este modulo: abrir e salvar o manifest.csv no Excel em
portugues regrava o arquivo em cp1252 e com ponto e virgula. A leitura estrita em
UTF-8 quebrava o programa na abertura seguinte. Aqui:

- aceita UTF-8 (com ou sem BOM) e cp1252; aceita virgula, ponto e virgula ou tab;
- linhas em branco sao ignoradas em silencio;
- linhas com numero errado de campos ou sem os campos obrigatorios sao ignoradas
  COM AVISO indicando o numero da linha, e preservadas em um arquivo de backup;
- colunas novas que o arquivo ainda nao tem (versao anterior) sao preenchidas vazias;
- se faltar uma coluna obrigatoria, o erro explica como restaurar.

Nada e regravado na leitura. Na proxima escrita, o arquivo volta ao formato padrao
(UTF-8, virgula); antes disso, uma copia do original fica em data/backup/.
"""

from __future__ import annotations

import csv
import io
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from .erros import ErroManifesto


@dataclass
class LeituraTabela:
    linhas: list[dict] = field(default_factory=list)
    avisos: list[str] = field(default_factory=list)
    ignoradas: list[tuple[int, str, str]] = field(default_factory=list)  # (linha, motivo, conteudo)
    formato_alterado: bool = False
    codificacao: str = "utf-8"
    delimitador: str = ","
    colunas_adicionadas: list[str] = field(default_factory=list)

    @property
    def precisa_backup(self) -> bool:
        return self.formato_alterado or bool(self.ignoradas)


def _decodificar(dados: bytes) -> tuple[str, str]:
    for cod in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return dados.decode(cod), ("utf-8" if cod == "utf-8-sig" else cod)
        except UnicodeDecodeError:
            continue
    raise AssertionError("latin-1 decodifica qualquer byte")


def _delimitador(cabecalho: str) -> str:
    contagem = {d: cabecalho.count(d) for d in (",", ";", "\t")}
    return max(contagem, key=contagem.get) if max(contagem.values()) > 0 else ","


def ler_tabela(
    caminho: Path, colunas: list[str], obrigatorias: list[str], nome: str, validar_linha=None
) -> LeituraTabela:
    """Le o CSV de forma tolerante. validar_linha(dict) -> motivo (str) ou None."""
    r = LeituraTabela()
    caminho = Path(caminho)
    if not caminho.exists():
        return r
    dados = caminho.read_bytes()
    if not dados.strip():
        return r
    texto, r.codificacao = _decodificar(dados)
    primeira = texto.splitlines()[0] if texto.splitlines() else ""
    r.delimitador = _delimitador(primeira)
    r.formato_alterado = r.codificacao != "utf-8" or r.delimitador != "," or dados.startswith(b"\xef\xbb\xbf")
    leitor = csv.reader(io.StringIO(texto, newline=""), delimiter=r.delimitador)
    try:
        cab = next(leitor)
    except StopIteration:
        return r
    cab = [c.strip().lstrip("﻿") for c in cab]
    faltam_obrig = [c for c in obrigatorias if c not in cab]
    if faltam_obrig:
        raise ErroManifesto(
            f"O arquivo {caminho.name} nao tem as colunas obrigatorias {faltam_obrig}. Ele pode ter sido "
            f"alterado por outro programa. Restaure a copia mais recente de data/backup/ (se houver) ou "
            f"envie o arquivo para analise. Colunas encontradas: {cab[:8]}"
        )
    r.colunas_adicionadas = [c for c in colunas if c not in cab]
    linhas_brutas = texto.splitlines()
    for campos in leitor:
        n = leitor.line_num
        if not any(c.strip() for c in campos):
            continue
        bruto = linhas_brutas[n - 1] if 0 < n <= len(linhas_brutas) else ";".join(campos)
        if len(campos) != len(cab):
            r.ignoradas.append((n, f"{len(campos)} campos, esperado {len(cab)}", bruto))
            continue
        d = {c: "" for c in colunas}
        d.update({k: (v if v is not None else "") for k, v in zip(cab, campos) if k})
        vazios = [c for c in obrigatorias if not str(d.get(c, "")).strip()]
        if vazios:
            r.ignoradas.append((n, f"campo(s) obrigatorio(s) vazio(s): {', '.join(vazios)}", bruto))
            continue
        motivo = validar_linha(d) if validar_linha else None
        if motivo:
            r.ignoradas.append((n, motivo, bruto))
            continue
        r.linhas.append(d)
    if r.formato_alterado:
        r.avisos.append(
            f"{nome}: o arquivo foi salvo por outro programa (provavelmente o Excel) com codificacao "
            f"{r.codificacao} e separador '{r.delimitador}'. Foi lido mesmo assim e sera regravado no formato "
            "padrao na proxima alteracao. Para so consultar, abra uma copia."
        )
    for n, motivo, _ in r.ignoradas:
        r.avisos.append(f"{nome}: linha {n} ignorada ({motivo}).")
    return r


def guardar_backup(caminho: Path, leitura: LeituraTabela, pasta_backup: Path, carimbo: str) -> Path | None:
    """Copia o arquivo original e as linhas ignoradas para data/backup/ (antes de qualquer regravacao)."""
    if not leitura.precisa_backup or not Path(caminho).exists():
        return None
    pasta_backup.mkdir(parents=True, exist_ok=True)
    destino = pasta_backup / f"{Path(caminho).stem}_{carimbo}{Path(caminho).suffix}"
    if not destino.exists():
        shutil.copy2(caminho, destino)
    if leitura.ignoradas:
        with open(destino.with_name(destino.stem + "_linhas_ignoradas.txt"), "w", encoding="utf-8") as f:
            for n, motivo, bruto in leitura.ignoradas:
                f.write(f"linha {n} ({motivo}): {bruto}\n")
    return destino
