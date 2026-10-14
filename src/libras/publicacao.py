"""Publicar e exportar um modelo final como pacote para o app real (item 13.2, D19). So na BANCADA.

Le o modelo final em runs/, o vocabulario (rotulos), o config (parametros de decisao) e, se houver,
a avaliacao por particoes (D13) com as mesmas classes. Grava o pacote (libras.pacote) e registra a
publicacao em data/auditoria.log. Modelo ENSAIO e modelo que nao e final sao recusados.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

from . import pacote as pk
from .decisao import parametros
from .erros import ErroLibras
from .utils import agora_iso, escrever_texto_atomico, sha256_arquivo


def pasta_publicados(cfg) -> Path:
    return cfg.caminho("publicados")


def eh_modelo_final(pasta: Path) -> bool:
    return (Path(pasta) / "modelo.keras").is_file() and (Path(pasta) / "metadados.json").is_file()


def avaliacao_correspondente(cfg, classes: list[str]) -> Path | None:
    """Avaliacao por particoes mais recente, concluida, nao ENSAIO, do protocolo principal e com as
    mesmas classes do modelo. None se nao houver (o pacote sai sem metricas)."""
    runs = cfg.caminho("runs")
    if not runs.is_dir():
        return None
    for d in sorted((p for p in runs.iterdir() if p.is_dir()), reverse=True):
        arq = d / "resumo.json"
        if not arq.is_file() or (d / "plano.json").exists():
            continue
        try:
            r = json.loads(arq.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if r.get("ensaio") or r.get("secundario") or sorted(r.get("classes") or []) != sorted(classes):
            continue
        return d
    return None


def _metricas(pasta_aval: Path) -> dict:
    r = json.loads((pasta_aval / "resumo.json").read_text(encoding="utf-8"))
    met = {}
    for chave in ("acuracia_video", "f1_macro_video", "acuracia_janela", "f1_macro_janela"):
        if f"{chave}_media" in r:
            met[chave] = {"media": r[f"{chave}_media"], "dp": r.get(f"{chave}_dp")}
    return {
        "execucao": pasta_aval.name,
        "protocolo": r.get("protocolo"),
        "n_particoes": r.get("n_resultados"),
        "seeds": r.get("seeds"),
        "n_sinalizadores": r.get("n_sinalizadores"),
        "n_videos": r.get("n_videos"),
        "metricas": met,
        "ressalva": pk.RESSALVA_AVALIACAO,
    }


def _rotulos(cfg, classes: list[str]) -> dict:
    try:
        from .vocabulario import vocabulario_em_cache

        voc = vocabulario_em_cache(cfg)
        out = {}
        for c in classes:
            try:
                out[c] = voc.obter(c).rotulo
            except ErroLibras:
                out[c] = c
        return out
    except Exception:  # noqa: BLE001 - sem vocabulario: usa o nome tecnico
        return {c: c for c in classes}


def preparar(cfg, pasta_modelo: Path, usar_avaliacao: bool = True) -> dict:
    """Confere o modelo e devolve o que vai no pacote (para mostrar e pedir confirmacao), sem gravar."""
    pasta_modelo = Path(pasta_modelo)
    if not eh_modelo_final(pasta_modelo):
        raise pk.ErroPacote(
            "Só modelos finais (modelo.keras + metadados.json) podem ser publicados. Treine o modelo "
            "final na aba Treino."
        )
    meta = json.loads((pasta_modelo / "metadados.json").read_text(encoding="utf-8"))
    if meta.get("ensaio"):
        raise pk.ErroPacote(
            "Este modelo é ENSAIO (treinado abaixo dos mínimos) e não pode ser publicado para o app real."
        )
    classes = list(meta.get("classes") or [])
    if len(classes) < 2:
        raise pk.ErroPacote("O modelo não tem classes suficientes para ser publicado.")
    meta_limpo = {k: meta[k] for k in pk.CHAVES_METADADOS if k in meta}
    meta_limpo["n_sinalizadores"] = len(meta.get("sinalizadores") or [])
    meta_limpo["versoes"] = {k: v for k, v in (meta.get("versoes") or {}).items() if k in pk.CHAVES_VERSOES}
    aval = avaliacao_correspondente(cfg, classes) if usar_avaliacao else None
    cartao = {
        "formato": pk.FORMATO,
        "classes": classes,
        "nome_nenhum_sinal": meta.get("nome_nenhum_sinal"),
        "rotulos": _rotulos(cfg, classes),
        "modelo_origem": pasta_modelo.name,
        "criado_em": meta.get("criado_em"),
        "protocolo": meta.get("protocolo"),
        "ensaio": False,
        "versoes": meta_limpo["versoes"],
        "mediapipe": meta.get("mediapipe"),
        "decisao": parametros(cfg.get("tempo_real")),
        "avaliacao": _metricas(aval) if aval else None,
        "aviso": pk.AVISO_USO,
    }
    return {"meta": meta_limpo, "cartao": cartao, "pasta_modelo": pasta_modelo}


def gravar(prep: dict, base: Path) -> Path:
    """Grava o pacote em base/<id>/ e confere. `prep` vem de preparar()."""
    base = Path(base)
    base.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".publicando_", dir=base))
    try:
        shutil.copy2(prep["pasta_modelo"] / "modelo.keras", tmp / "modelo.keras")
        escrever_texto_atomico(tmp / "metadados.json", json.dumps(prep["meta"], indent=2, ensure_ascii=False))
        cartao = dict(prep["cartao"])
        cartao["hash"] = {"algoritmo": "sha256", **{n: sha256_arquivo(tmp / n) for n in pk.ARQUIVOS_COM_HASH}}
        cartao["publicado_em"] = agora_iso()
        cartao["id"] = f"{datetime.now():%Y%m%d_%H%M%S}_{cartao['hash']['modelo.keras'][:8]}"
        escrever_texto_atomico(tmp / "cartao.json", json.dumps(cartao, indent=2, ensure_ascii=False))
        pk.verificar_pacote(tmp)
        destino = base / cartao["id"]
        if destino.exists():
            raise pk.ErroPacote(f"Já existe um pacote {cartao['id']}. Tente de novo em um segundo.")
        tmp.replace(destino)
        return destino
    finally:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)


def publicar(cfg, pasta_modelo: Path, usar_avaliacao: bool = True) -> Path:
    """Publica em publicados/<id>/ e registra em data/auditoria.log."""
    from .coleta.gestao import auditar

    prep = preparar(cfg, pasta_modelo, usar_avaliacao)
    destino = gravar(prep, pasta_publicados(cfg))
    c = pk.ler_cartao(destino)
    auditar(
        cfg,
        "publicar_modelo",
        c["id"],
        modelo_origem=c["modelo_origem"],
        sha256_modelo=c["hash"]["modelo.keras"],
        classes=c["classes"],
        avaliacao=(c.get("avaliacao") or {}).get("execucao"),
    )
    return destino


def exportar_zip(cfg, pasta_modelo: Path, destino_zip: Path, usar_avaliacao: bool = True) -> Path:
    """Monta o pacote numa pasta temporaria e grava so o .zip (nao publica em publicados/)."""
    from .coleta.gestao import auditar

    prep = preparar(cfg, pasta_modelo, usar_avaliacao)
    with tempfile.TemporaryDirectory() as tmp:
        pasta = gravar(prep, Path(tmp))
        c = pk.ler_cartao(pasta)
        pk.compactar(pasta, destino_zip)
    auditar(
        cfg,
        "exportar_pacote",
        c["id"],
        modelo_origem=c["modelo_origem"],
        sha256_modelo=c["hash"]["modelo.keras"],
        arquivo=Path(destino_zip).name,
    )
    return Path(destino_zip)


def publicado_atual(cfg) -> dict | None:
    """Pacote integro mais recente em publicados/ (para a tela Inicio da bancada)."""
    for it in pk.listar_pacotes(pasta_publicados(cfg)):
        if it["erro"] is None:
            return it
    return None


def descrever(prep: dict) -> str:
    """Texto do dialogo de confirmacao: o que vai e o que nao vai no pacote."""
    c = prep["cartao"]
    linhas = [
        "Vai no pacote:",
        "  modelo.keras (o modelo final)",
        "  metadados.json (normalização, janela e MediaPipe usados no treino; sem IDs de sinalizadores)",
        "  cartao.json:",
    ]
    linhas += [
        f"    {r}: {v}" for r, v in pk.resumo({**c, "hash": {}, "publicado_em": "(agora)"}) if r != "Hash do modelo"
    ]
    d = c["decisao"]
    linhas.append(
        f"    Decisão: limiar {d['limiar_confianca']:.2f}, média de {d['suavizacao']} predições, "
        f"{d['frames_estaveis']} iguais para confirmar, mão em {100 * d['min_fracao_mao']:.0f}% da janela"
    )
    if c["avaliacao"]:
        linhas.append(f"    Métricas da avaliação {c['avaliacao']['execucao']}, com a ressalva do protocolo")
    else:
        linhas.append("    Sem métricas: não há avaliação por partições concluída com as mesmas classes")
    linhas += ["", "Não vai: vídeos, landmarks, manifesto, cadastro, IDs de sinalizadores nem caminhos de pastas."]
    return "\n".join(linhas)
