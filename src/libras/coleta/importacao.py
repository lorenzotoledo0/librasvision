"""Importacao de videos ja gravados para o dataset padronizado.

Garantias:
- o arquivo original do usuario nunca e alterado nem apagado (sempre copiamos);
- importacao idempotente: o hash SHA-256 do conteudo original impede duplicatas;
- nada e importado para sinalizador sem TCLE confirmado;
- cada execucao gera um relatorio CSV (importado / ignorado / rejeitado, com motivo).
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from ..config import Config
from ..erros import ErroLibras
from ..utils import carimbo, escrever_csv_atomico, sha256_arquivo
from .manifesto import Manifesto
from .qualidade import InfoVideo, sondar_arquivo
from .sinalizadores import CadastroSinalizadores, id_valido


@dataclass
class MetaPadrao:
    """Metadados aplicados a todos os arquivos (podem ser sobrescritos por pasta/nome)."""

    classe: str | None = None
    signer_id: str | None = None
    iluminacao: str = ""
    distancia: str = ""
    fundo: str = ""
    mao_dominante: str | None = None  # None = usa a do cadastro do sinalizador
    observacoes: str = ""


@dataclass
class ItemImportacao:
    origem: Path
    classe: str | None = None
    signer_id: str | None = None
    take_original: str = ""
    sha256_origem: str = ""
    situacao: str = "pendente"  # importado | ignorado | rejeitado
    motivo: str = ""
    video_id: str = ""
    destino: str = ""
    info: InfoVideo | None = None
    avisos: list[str] = field(default_factory=list)


# ----------------------------------------------------------------------
# Descoberta de arquivos e resolucao de metadados
# ----------------------------------------------------------------------
def listar_videos(entradas: list[Path], extensoes: list[str]) -> list[Path]:
    exts = {e.lower() for e in extensoes}
    out: list[Path] = []
    for e in entradas:
        e = Path(e)
        if e.is_dir():
            out += sorted(p for p in e.rglob("*") if p.is_file() and p.suffix.lower() in exts)
        elif e.is_file() and e.suffix.lower() in exts:
            out.append(e)
        elif e.is_file():
            pass  # extensao nao suportada: aparece no relatorio via resolver
    # remove repetidos preservando ordem
    vistos, unicos = set(), []
    for p in out:
        k = p.resolve()
        if k not in vistos:
            vistos.add(k)
            unicos.append(p)
    return unicos


def parse_mapa(pares: list[str] | None) -> dict[str, str]:
    """'usuario1=S03' -> {'usuario1': 'S03'}"""
    mapa = {}
    for par in pares or []:
        if "=" not in par:
            raise ErroLibras(f"Mapeamento invalido {par!r}; use rotulo=SXX, ex.: usuario1=S03")
        k, v = (x.strip() for x in par.split("=", 1))
        if not id_valido(v):
            raise ErroLibras(f"{v!r} nao e um ID de sinalizador valido (formato S01, S02...).")
        mapa[k] = v
    return mapa


def resolver_metadados(
    item: ItemImportacao,
    cfg: Config,
    padrao: MetaPadrao,
    usar_pasta_classe: bool,
    padrao_nome: str | None,
    mapa: dict[str, str],
) -> None:
    """Preenche classe e sinalizador do item. Em caso de falha, marca como rejeitado com motivo."""
    classes = cfg.classes
    grupos: dict[str, str] = {}
    if padrao_nome:
        rx = cfg["importacao"]["padroes_nome"].get(padrao_nome, padrao_nome)
        m = re.match(rx, item.origem.stem)
        if not m:
            item.situacao, item.motivo = "rejeitado", f"nome do arquivo nao segue o padrao {padrao_nome!r}"
            return
        grupos = {k: v for k, v in m.groupdict().items() if v is not None}

    # classe: grupo do nome > pasta > padrao do formulario
    classe = grupos.get("classe")
    if classe is None and usar_pasta_classe:
        pasta = item.origem.parent.name
        if pasta in classes:
            classe = pasta
    if classe is None:
        classe = padrao.classe
    if classe is None:
        item.situacao, item.motivo = "rejeitado", "classe nao informada (nem pela pasta nem pelo formulario)"
        return
    if classe not in classes:
        item.situacao, item.motivo = "rejeitado", f"classe {classe!r} nao existe no config"
        return

    # sinalizador: grupo do nome (ID ou rotulo mapeado) > padrao do formulario
    rotulo = grupos.get("signer")
    if rotulo is not None:
        if id_valido(rotulo):
            signer = rotulo
        elif rotulo in mapa:
            signer = mapa[rotulo]
        else:
            item.situacao = "rejeitado"
            item.motivo = f"rotulo de sinalizador {rotulo!r} sem mapeamento; use --mapa {rotulo}=SXX"
            return
    else:
        signer = padrao.signer_id
    if not signer:
        item.situacao, item.motivo = "rejeitado", "sinalizador nao informado"
        return
    item.classe, item.signer_id = classe, signer
    item.take_original = grupos.get("take", "")


# ----------------------------------------------------------------------
# Padronizacao opcional de formato
# ----------------------------------------------------------------------
def ffmpeg_disponivel() -> str | None:
    return shutil.which("ffmpeg")


def padronizar_video(origem: Path, destino: Path, cfg_pad: dict) -> str:
    """Re-codifica para o formato do projeto sem distorcer a imagem (escala + bordas).

    Escala uniforme e bordas nao alteram a geometria relativa do corpo; a normalizacao
    da Fase 3 (centro e largura dos ombros, em pixels) remove esse efeito.
    Retorna o metodo usado ('ffmpeg' ou 'opencv').
    """
    larg, alt, fps = int(cfg_pad["largura"]), int(cfg_pad["altura"]), float(cfg_pad["fps"])
    ff = ffmpeg_disponivel()
    destino.parent.mkdir(parents=True, exist_ok=True)
    if ff:
        vf = f"scale={larg}:{alt}:force_original_aspect_ratio=decrease,pad={larg}:{alt}:(ow-iw)/2:(oh-ih)/2,fps={fps}"
        cmd = [
            ff,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(origem),
            "-vf",
            vf,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-an",
            str(destino),
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode == 0 and destino.exists():
            return "ffmpeg"
        destino.unlink(missing_ok=True)
    # Fallback OpenCV: redimensiona com bordas, mas MANTEM o FPS original (OpenCV nao reamostra).
    import cv2
    import numpy as np

    cap = cv2.VideoCapture(str(origem))
    fps_orig = float(cap.get(cv2.CAP_PROP_FPS) or fps)
    w = cv2.VideoWriter(str(destino), cv2.VideoWriter_fourcc(*"mp4v"), fps_orig, (larg, alt))
    try:
        while True:
            ok, f = cap.read()
            if not ok:
                break
            h0, w0 = f.shape[:2]
            s = min(larg / w0, alt / h0)
            nw, nh = int(round(w0 * s)), int(round(h0 * s))
            r = cv2.resize(f, (nw, nh), interpolation=cv2.INTER_AREA)
            tela = np.zeros((alt, larg, 3), np.uint8)
            y0, x0 = (alt - nh) // 2, (larg - nw) // 2
            tela[y0 : y0 + nh, x0 : x0 + nw] = r
            w.write(tela)
    finally:
        cap.release()
        w.release()
    return "opencv"


# ----------------------------------------------------------------------
# Execucao
# ----------------------------------------------------------------------
COLUNAS_RELATORIO = [
    "situacao",
    "arquivo_origem",
    "motivo",
    "avisos",
    "classe",
    "signer_id",
    "video_id",
    "destino",
    "sha256_origem",
]


def importar(
    cfg: Config,
    entradas: list[Path],
    padrao: MetaPadrao,
    *,
    usar_pasta_classe: bool = True,
    padrao_nome: str | None = None,
    mapa: dict[str, str] | None = None,
    padronizar: bool | None = None,
    validar_pessoa: bool = True,
    simular: bool = False,
    progresso=None,
    extrair_landmarks: bool = True,
) -> tuple[list[ItemImportacao], Path | None]:
    """Importa videos. Retorna (itens, caminho_do_relatorio).

    simular=True so valida e mostra o que aconteceria, sem copiar nem registrar.
    progresso: funcao opcional (i, n, item) chamada a cada arquivo (para a interface).
    """
    manifesto = Manifesto(cfg)
    cadastro = CadastroSinalizadores(cfg)
    mapa = mapa or {}
    if padronizar is None:
        padronizar = bool(cfg["importacao"]["padronizar"]["ativo"])
    op = cfg["coleta"]["opcoes"]
    for campo in ("iluminacao", "distancia", "fundo"):
        v = getattr(padrao, campo)
        if v and v not in op[campo]:
            raise ErroLibras(f"Valor invalido para {campo}: {v!r}. Opcoes: {op[campo]}")

    arquivos = listar_videos(entradas, cfg["importacao"]["extensoes"])
    if not arquivos:
        raise ErroLibras(
            "Nenhum video encontrado nas entradas informadas "
            f"(extensoes aceitas: {', '.join(cfg['importacao']['extensoes'])})."
        )

    extrator = None
    if validar_pessoa:
        from ..landmarks import ExtratorHolistic

        extrator = ExtratorHolistic(cfg["mediapipe"], modo_estatico=True)

    itens: list[ItemImportacao] = []
    hashes_lote: dict[str, Path] = {}
    try:
        for i, arq in enumerate(arquivos):
            item = ItemImportacao(origem=arq)
            itens.append(item)
            try:
                _importar_um(
                    cfg,
                    item,
                    manifesto,
                    cadastro,
                    padrao,
                    usar_pasta_classe,
                    padrao_nome,
                    mapa,
                    padronizar,
                    extrator,
                    simular,
                    hashes_lote,
                )
                if extrair_landmarks and item.situacao == "importado" and not simular:
                    _gerar_landmarks(cfg, manifesto, item)
            except ErroLibras as e:
                item.situacao, item.motivo = "rejeitado", str(e)
            except Exception as e:  # um arquivo problematico nao derruba o lote
                item.situacao, item.motivo = "rejeitado", f"erro inesperado: {type(e).__name__}: {e}"
            if progresso:
                progresso(i + 1, len(arquivos), item)
    finally:
        if extrator is not None:
            extrator.fechar()

    rel = None
    if not simular:
        rel = cfg.caminho("relatorios_importacao") / f"importacao_{carimbo()}.csv"
        escrever_csv_atomico(rel, COLUNAS_RELATORIO, (_linha_relatorio(it) for it in itens))
    return itens, rel


def _gerar_landmarks(cfg: Config, manifesto: Manifesto, item: ItemImportacao) -> None:
    """Extrai os landmarks do video recem-importado (mesmo .npz de scripts/extrair.py)."""
    from ..extracao import caminho_npz, extrair_um, meta_esperada
    from ..utils import sha256_arquivo

    l = manifesto.obter(item.video_id)
    npz = caminho_npz(cfg, l)
    try:
        r = extrair_um(cfg["mediapipe"], str(manifesto.caminho_absoluto(l)), str(npz), meta_esperada(cfg, l))
    except Exception as e:  # o video fica importado; os landmarks podem ser gerados depois
        r = {"ok": False, "motivo": f"{type(e).__name__}: {e}"}
    if r.get("ok"):
        manifesto.atualizar(item.video_id, landmarks_path=manifesto.relativo(npz), sha256_npz=sha256_arquivo(npz))
    else:
        item.avisos.append(f"landmarks_nao_gerados({r.get('motivo')}); rode scripts/extrair.py")


def _linha_relatorio(it: ItemImportacao) -> dict:
    return {
        "situacao": it.situacao,
        "arquivo_origem": str(it.origem),
        "motivo": it.motivo,
        "avisos": ";".join(it.avisos),
        "classe": it.classe or "",
        "signer_id": it.signer_id or "",
        "video_id": it.video_id,
        "destino": it.destino,
        "sha256_origem": it.sha256_origem,
    }


def _importar_um(
    cfg,
    item,
    manifesto,
    cadastro,
    padrao,
    usar_pasta_classe,
    padrao_nome,
    mapa,
    padronizar,
    extrator,
    simular,
    hashes_lote,
) -> None:
    resolver_metadados(item, cfg, padrao, usar_pasta_classe, padrao_nome, mapa)
    if item.situacao == "rejeitado":
        return
    sinalizador = cadastro.exigir_tcle(item.signer_id)  # bloqueia sem TCLE

    item.sha256_origem = sha256_arquivo(item.origem)
    ja = manifesto.hash_registrado(item.sha256_origem)
    if ja is not None:
        item.situacao, item.motivo = "ignorado", f"ja importado como {ja['video_id']}"
        item.video_id = ja["video_id"]
        return
    if item.sha256_origem in hashes_lote:
        item.situacao, item.motivo = "ignorado", f"duplicata de {hashes_lote[item.sha256_origem]} neste lote"
        return
    hashes_lote[item.sha256_origem] = item.origem

    info = sondar_arquivo(item.origem, cfg, extrator)
    item.info = info
    item.avisos = list(info.avisos)
    if info.erros:
        item.situacao, item.motivo = "rejeitado", "; ".join(info.erros)
        return
    if simular:
        item.situacao, item.motivo = "importado", "(simulacao: nada foi copiado)"
        return

    take = manifesto.proximo_take(item.signer_id, item.classe)
    destino_padrao = manifesto.caminho_take(item.classe, item.signer_id, take)
    metodo = "copia"
    if padronizar:
        destino = destino_padrao.with_suffix(".mp4")
        metodo = padronizar_video(item.origem, destino, cfg["importacao"]["padronizar"])
        info = sondar_arquivo(destino, cfg, None)  # metadados do arquivo final
        info.taxa_mao, info.taxa_pose = item.info.taxa_mao, item.info.taxa_pose
        info.avisos = item.avisos + ([] if metodo == "ffmpeg" else ["padronizado_sem_ffmpeg_fps_original"])
    else:
        destino = destino_padrao.with_suffix(item.origem.suffix.lower())
        destino.parent.mkdir(parents=True, exist_ok=True)
        if destino.exists():
            raise ErroLibras(f"Ja existe {destino}; nada foi sobrescrito.")
        shutil.copy2(item.origem, destino)

    sha_final = sha256_arquivo(destino)
    obs = [x for x in (padrao.observacoes, f"importado ({metodo})") if x]
    if item.take_original:
        obs.append(f"take original {item.take_original}")
    registro = {
        "video_id": f"{item.signer_id}_{item.classe}_{take:03d}",
        "sha256": sha_final,
        "sha256_origem": item.sha256_origem,
        "classe": item.classe,
        "signer_id": item.signer_id,
        "take": take,
        "origem": "importado",
        "arquivo": manifesto.relativo(destino),
        # so o nome do arquivo: caminhos completos podem conter nomes de pessoas
        "arquivo_original": item.origem.name,
        "iluminacao": padrao.iluminacao,
        "distancia": padrao.distancia,
        "fundo": padrao.fundo,
        "mao_dominante": padrao.mao_dominante or sinalizador.mao_dominante,
        # avisos de qualidade deixam o video pendente de revisao humana
        "status": "pendente" if info.avisos else "ok",
        "observacoes": " | ".join(obs),
        **info.campos_manifesto(),
    }
    try:
        manifesto.adicionar(registro)
    except Exception:
        destino.unlink(missing_ok=True)
        raise
    item.situacao, item.video_id, item.destino = "importado", registro["video_id"], registro["arquivo"]
    item.avisos = list(info.avisos)
