"""Manifesto do dataset (data/manifest.csv): fonte unica de verdade sobre os videos.

Por que CSV e nao SQLite: o volume e pequeno (centenas a poucos milhares de linhas),
a equipe consegue abrir e conferir no Excel, e diffs sao legiveis. A integridade e
garantida por escrita atomica e pela verificacao em verificar_integridade().
"""

from __future__ import annotations

import contextlib
import os
import shutil
import threading
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from ..config import Config
from ..erros import ErroManifesto
from ..tabelas import guardar_backup, ler_tabela
from ..utils import agora_iso, carimbo, escrever_csv_atomico

COLUNAS = [
    "video_id",
    "sha256",
    "sha256_origem",
    "classe",
    "signer_id",
    "take",
    "origem",
    "arquivo",
    "arquivo_original",
    "iluminacao",
    "distancia",
    "fundo",
    "mao_dominante",
    "largura",
    "altura",
    "fps_nominal",
    "fps_medido",
    "n_frames",
    "duracao_s",
    "taxa_mao",
    "taxa_pose",
    "brilho_medio",
    "data_registro",
    "flags_qualidade",
    "status",
    "landmarks_path",
    "observacoes",
    "sha256_npz",
    "frames_originais",
    "corte",
]
OBRIGATORIAS = ["video_id", "classe", "signer_id", "arquivo", "status"]
STATUS_VALIDOS = ("ok", "pendente", "rejeitado")

# Trava de escrita: toda alteracao e "reler, mudar, gravar". Sem trava, duas escritas ao mesmo tempo
# (thread da interface e tarefa em segundo plano, ou a interface e um script) fazem a segunda apagar
# a linha que a primeira acabou de gravar. RLock para as threads deste processo; arquivo .lock
# (criacao exclusiva) para outros processos, so na camada mais externa.
_TRAVA_THREADS = threading.RLock()
_PROFUNDIDADE = threading.local()
TRAVA_ESPERA_S = 15.0
TRAVA_ABANDONADA_S = 120.0  # .lock mais velho que isso e de um processo que morreu


@dataclass
class RelatorioIntegridade:
    linhas_sem_arquivo: list[str] = field(default_factory=list)
    arquivos_sem_linha: list[str] = field(default_factory=list)
    sinalizadores_desconhecidos: list[str] = field(default_factory=list)
    sinalizadores_sem_tcle: list[str] = field(default_factory=list)
    classes_desconhecidas: list[str] = field(default_factory=list)
    ids_duplicados: list[str] = field(default_factory=list)
    hashes_duplicados: list[str] = field(default_factory=list)
    status_invalidos: list[str] = field(default_factory=list)
    landmarks_orfaos: list[str] = field(default_factory=list)
    landmarks_ausentes: list[str] = field(default_factory=list)
    landmarks_alterados: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(getattr(self, f) for f in self.__dataclass_fields__)

    def texto(self) -> str:
        if self.ok:
            return "Manifesto integro: nenhum problema encontrado."
        partes = ["Problemas encontrados no manifesto:"]
        rotulos = {
            "linhas_sem_arquivo": "Registros cujo arquivo de video nao existe",
            "arquivos_sem_linha": "Videos em data/raw_videos sem registro no manifesto",
            "sinalizadores_desconhecidos": "Registros com sinalizador nao cadastrado",
            "sinalizadores_sem_tcle": "Registros de sinalizador sem TCLE confirmado",
            "classes_desconhecidas": "Registros com classe que nao existe no vocabulario",
            "ids_duplicados": "video_id repetido",
            "hashes_duplicados": "Mesmo conteudo de video registrado mais de uma vez",
            "status_invalidos": "Status diferente de ok/pendente/rejeitado",
            "landmarks_orfaos": "Arquivos de landmarks (.npz) de videos que nao estao no manifesto",
            "landmarks_ausentes": "Registros que apontam para um .npz que nao existe (rode scripts/extrair.py)",
            "landmarks_alterados": "Arquivos .npz diferentes do registrado no manifesto (hash nao confere)",
        }
        for f, rot in rotulos.items():
            itens = getattr(self, f)
            if itens:
                partes.append(f"- {rot} ({len(itens)}):")
                partes += [f"    {i}" for i in itens[:50]]
                if len(itens) > 50:
                    partes.append(f"    ... e mais {len(itens) - 50}")
        return "\n".join(partes)


class Manifesto:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.arquivo: Path = cfg.caminho("manifesto")
        self.raiz = cfg.raiz
        self._linhas: list[dict] = []
        self.avisos: list[str] = []
        self.backup: Path | None = None
        self.recarregar()

    # ------------------------------------------------------------------
    def recarregar(self) -> None:
        """Le o manifesto de forma tolerante (ver libras.tabelas). Problemas viram self.avisos."""
        import logging

        leitura = ler_tabela(self.arquivo, COLUNAS, OBRIGATORIAS, "Manifesto")
        self.avisos = list(leitura.avisos)
        if leitura.precisa_backup:
            self.backup = guardar_backup(self.arquivo, leitura, self.cfg.caminho("backup"), carimbo())
            if self.backup:
                self.avisos.append(f"Copia do arquivo original guardada em {self.backup}.")
        for a in self.avisos:
            logging.getLogger("libras.manifesto").warning(a)
        self._linhas = leitura.linhas

    def salvar(self) -> None:
        escrever_csv_atomico(self.arquivo, COLUNAS, self._linhas)

    @contextlib.contextmanager
    def trava(self):
        """Exclusividade para "reler, alterar e salvar". Reentrante na mesma thread."""
        with _TRAVA_THREADS:
            prof = getattr(_PROFUNDIDADE, "n", 0)
            _PROFUNDIDADE.n = prof + 1
            try:
                if prof:
                    yield
                    return
                lock = self.arquivo.with_name(self.arquivo.name + ".lock")
                self._adquirir_arquivo(lock)
                try:
                    yield
                finally:
                    lock.unlink(missing_ok=True)
            finally:
                _PROFUNDIDADE.n = prof

    @staticmethod
    def _adquirir_arquivo(lock: Path) -> None:
        lock.parent.mkdir(parents=True, exist_ok=True)
        limite = time.monotonic() + TRAVA_ESPERA_S
        while True:
            try:
                os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
                return
            except FileExistsError:
                try:
                    if time.time() - lock.stat().st_mtime > TRAVA_ABANDONADA_S:
                        lock.unlink(missing_ok=True)
                        continue
                except FileNotFoundError:
                    continue
                if time.monotonic() > limite:
                    raise ErroManifesto(
                        f"O manifesto esta em uso por outro processo ({lock.name}). Tente de novo; se nada "
                        "estiver rodando, apague esse arquivo."
                    ) from None
                time.sleep(0.05)

    # ------------------------------------------------------------------
    @property
    def linhas(self) -> list[dict]:
        return list(self._linhas)

    def __len__(self) -> int:
        return len(self._linhas)

    def obter(self, video_id: str) -> dict | None:
        for l in self._linhas:
            if l["video_id"] == video_id:
                return l
        return None

    def caminho_absoluto(self, linha: dict) -> Path:
        return self.raiz / Path(linha["arquivo"])

    def relativo(self, caminho: Path) -> str:
        """Caminho relativo a raiz com barras '/', valido em Windows e Linux."""
        return Path(caminho).resolve().relative_to(self.raiz.resolve()).as_posix()

    def hash_registrado(self, sha: str) -> dict | None:
        for l in self._linhas:
            if sha and (l["sha256"] == sha or l.get("sha256_origem") == sha):
                return l
        return None

    def proximo_take(self, signer_id: str, classe: str) -> int:
        """Proximo numero de take livre: considera manifesto, arquivos no disco e lixeira."""
        usados = {
            int(l["take"])
            for l in self._linhas
            if l["signer_id"] == signer_id and l["classe"] == classe and l["take"].isdigit()
        }
        pasta = self.cfg.caminho("videos") / classe
        if pasta.exists():
            for p in pasta.glob(f"{signer_id}_*"):
                sufixo = p.stem.split("_")[-1]
                if sufixo.isdigit():
                    usados.add(int(sufixo))
        lixeira = self.cfg.caminho("lixeira") / classe
        if lixeira.exists():
            for p in lixeira.glob(f"{signer_id}_*"):
                partes = p.stem.split("_")
                if len(partes) >= 2 and partes[1].isdigit():
                    usados.add(int(partes[1]))
        return (max(usados) + 1) if usados else 1

    def caminho_take(self, classe: str, signer_id: str, take: int) -> Path:
        ext = self.cfg["coleta"]["gravacao"]["extensao"]
        return self.cfg.caminho("videos") / classe / f"{signer_id}_{take:03d}{ext}"

    # ------------------------------------------------------------------
    def adicionar(self, registro: dict) -> dict:
        with self.trava():
            return self._adicionar(registro)

    def _adicionar(self, registro: dict) -> dict:
        self.recarregar()  # outra ferramenta pode ter alterado o arquivo (ex.: extracao)
        faltando = [
            c
            for c in ("video_id", "sha256", "classe", "signer_id", "take", "arquivo", "status")
            if not str(registro.get(c, "")).strip()
        ]
        if faltando:
            raise ErroManifesto(f"Registro incompleto, faltam: {faltando}")
        if registro["status"] not in STATUS_VALIDOS:
            raise ErroManifesto(f"Status invalido: {registro['status']}")
        if self.obter(registro["video_id"]):
            raise ErroManifesto(f"video_id ja existe no manifesto: {registro['video_id']}")
        if not (self.raiz / registro["arquivo"]).exists():
            raise ErroManifesto(f"O arquivo {registro['arquivo']} nao existe; registro nao criado.")
        linha = {c: "" for c in COLUNAS}
        linha.update({k: ("" if v is None else str(v)) for k, v in registro.items() if k in COLUNAS})
        linha.setdefault("data_registro", agora_iso())
        if not linha["data_registro"]:
            linha["data_registro"] = agora_iso()
        self._linhas.append(linha)
        self.salvar()
        return linha

    def atualizar(self, video_id: str, **campos) -> dict:
        with self.trava():
            return self._atualizar(video_id, **campos)

    def _atualizar(self, video_id: str, **campos) -> dict:
        """Altera campos de um registro (so colunas conhecidas)."""
        desconhecidos = [k for k in campos if k not in COLUNAS]
        if desconhecidos:
            raise ErroManifesto(f"Colunas desconhecidas: {desconhecidos}")
        self.recarregar()
        l = self.obter(video_id)
        if l is None:
            raise ErroManifesto(f"video_id nao encontrado: {video_id}")
        l.update({k: ("" if v is None else str(v)) for k, v in campos.items()})
        self.salvar()
        return l

    def definir_status(self, video_id: str, status: str, observacao: str | None = None) -> dict:
        with self.trava():
            return self._definir_status(video_id, status, observacao)

    def _definir_status(self, video_id: str, status: str, observacao: str | None = None) -> dict:
        if status not in STATUS_VALIDOS:
            raise ErroManifesto(f"Status invalido: {status}")
        self.recarregar()
        l = self.obter(video_id)
        if l is None:
            raise ErroManifesto(f"video_id nao encontrado: {video_id}")
        l["status"] = status
        if observacao:
            l["observacoes"] = (l["observacoes"] + " | " if l["observacoes"] else "") + observacao
        self.salvar()
        return l

    def definir_status_lote(self, video_ids: list[str], status: str, observacao: str | None = None) -> int:
        """Mesmo status para varios takes numa unica leitura e escrita (revisao em lote)."""
        if status not in STATUS_VALIDOS:
            raise ErroManifesto(f"Status invalido: {status}")
        with self.trava():
            self.recarregar()
            alvo = set(video_ids)
            n = 0
            for l in self._linhas:
                if l["video_id"] in alvo:
                    l["status"] = status
                    if observacao:
                        l["observacoes"] = (l["observacoes"] + " | " if l["observacoes"] else "") + observacao
                    n += 1
            if n:
                self.salvar()
            return n

    def excluir(self, video_id: str) -> Path:
        with self.trava():
            return self._excluir(video_id)

    def _excluir(self, video_id: str) -> Path:
        """Remove o registro e move o video para data/lixeira (nunca apaga direto)."""
        self.recarregar()
        l = self.obter(video_id)
        if l is None:
            raise ErroManifesto(f"video_id nao encontrado: {video_id}")
        origem = self.caminho_absoluto(l)
        destino_dir = self.cfg.caminho("lixeira") / l["classe"]
        destino_dir.mkdir(parents=True, exist_ok=True)
        destino = destino_dir / f"{origem.stem}_{carimbo()}{origem.suffix}"
        if origem.exists():
            shutil.move(str(origem), str(destino))
        from ..extracao import caminho_npz

        caminho_npz(self.cfg, l).unlink(missing_ok=True)  # landmarks derivados do video excluido
        self._apagar_variantes(l)
        self._linhas = [x for x in self._linhas if x["video_id"] != video_id]
        try:
            self.salvar()
        except Exception:
            if destino.exists():  # desfaz para nao deixar arquivo sem registro
                shutil.move(str(destino), str(origem))
            self._linhas.append(l)
            raise
        return destino

    def _apagar_variantes(self, l: dict) -> None:
        base = self.cfg.caminho("landmarks_aug")
        if base.exists():
            for p in base.glob(f"*/{l['classe']}/{l['video_id']}.npz"):
                p.unlink(missing_ok=True)

    # ------------------------------------------------------------------
    def contagens(self, status: tuple[str, ...] = ("ok",)) -> dict[tuple[str, str], int]:
        """Numero de takes por (classe, signer_id) com os status pedidos."""
        c: dict[tuple[str, str], int] = defaultdict(int)
        for l in self._linhas:
            if l["status"] in status:
                c[(l["classe"], l["signer_id"])] += 1
        return dict(c)

    def verificar_integridade(
        self, ids_sinalizadores: list[str] | None = None, ids_com_tcle: list[str] | None = None
    ) -> RelatorioIntegridade:
        r = RelatorioIntegridade()
        classes = set(self.cfg.classes_todas)
        registrados = set()
        for l in self._linhas:
            vid = l["video_id"]
            p = self.caminho_absoluto(l)
            registrados.add(p.resolve())
            if not p.exists():
                r.linhas_sem_arquivo.append(f"{vid} -> {l['arquivo']}")
            if l.get("landmarks_path"):
                npz = self.raiz / l["landmarks_path"]
                if not npz.exists():
                    r.landmarks_ausentes.append(f"{vid} -> {l['landmarks_path']}")
                elif l.get("sha256_npz"):
                    from ..utils import sha256_arquivo

                    if sha256_arquivo(npz) != l["sha256_npz"]:
                        r.landmarks_alterados.append(f"{vid} -> {l['landmarks_path']}")
            if l["classe"] not in classes:
                r.classes_desconhecidas.append(f"{vid} ({l['classe']})")
            if l["status"] not in STATUS_VALIDOS:
                r.status_invalidos.append(f"{vid} ({l['status']})")
            if ids_sinalizadores is not None and l["signer_id"] not in ids_sinalizadores:
                r.sinalizadores_desconhecidos.append(f"{vid} ({l['signer_id']})")
            elif ids_com_tcle is not None and l["signer_id"] not in ids_com_tcle:
                r.sinalizadores_sem_tcle.append(f"{vid} ({l['signer_id']})")
        for vid, n in Counter(l["video_id"] for l in self._linhas).items():
            if n > 1:
                r.ids_duplicados.append(vid)
        for sha, n in Counter(l["sha256"] for l in self._linhas).items():
            if n > 1:
                r.hashes_duplicados.append(sha)
        pasta = self.cfg.caminho("videos")
        if pasta.exists():
            exts = set(self.cfg["importacao"]["extensoes"]) | {self.cfg["coleta"]["gravacao"]["extensao"]}
            for p in sorted(pasta.rglob("*")):
                if p.is_file() and p.suffix.lower() in exts and p.resolve() not in registrados:
                    r.arquivos_sem_linha.append(self.relativo(p))
        pasta_lm = self.cfg.caminho("landmarks")
        if pasta_lm.exists():
            ids = {(l["classe"], l["video_id"]) for l in self._linhas}
            for p in sorted(pasta_lm.rglob("*.npz")):
                if (p.parent.name, p.stem) not in ids:
                    r.landmarks_orfaos.append(self.relativo(p))
        return r
