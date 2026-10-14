"""Pacote de modelo publicado (item 13.2, D19): o que o app real recebe da bancada.

Um pacote e uma pasta `publicados/<id>/` com exatamente tres arquivos:
    modelo.keras    o modelo final treinado (D13/D15)
    metadados.json  os metadados do modelo que a inferencia usa (D15), SEM dados de pessoas
    cartao.json     rotulos das classes, datas, protocolo, versoes, parametros de decisao, hashes e,
                    se houver, metricas resumidas da avaliacao por particoes (D13) com a ressalva

Nunca vao no pacote: videos, landmarks, manifesto, cadastro, IDs de sinalizadores ou caminhos (D11).

Este modulo e COMPARTILHADO (bancada e app real): so le, confere e importa pacotes. Montar e
publicar (o que exige o vocabulario, os runs e a auditoria da bancada) fica em libras.publicacao.
Nao importa TensorFlow, MediaPipe nem Qt.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import zipfile
from pathlib import Path

from .erros import ErroLibras
from .utils import sha256_arquivo

FORMATO = "librasvision-pacote/1"
ARQUIVOS = ("modelo.keras", "metadados.json", "cartao.json")
ARQUIVOS_COM_HASH = ("modelo.keras", "metadados.json")
TAMANHO_MAXIMO_ZIP_MB = 300  # um LSTM pequeno tem poucos MB; acima disso o arquivo nao e um pacote nosso

# Chaves de metadados.json que vao para o pacote (lista branca). Ficam de fora, entre outras:
# "sinalizadores" (IDs, D11), "hardware_treino" e "avisos_selecao" (podem citar IDs).
CHAVES_METADADOS = (
    "classes",
    "nome_nenhum_sinal",
    "features",
    "janela",
    "normalizacao",
    "mediapipe",
    "versoes",
    "seed",
    "familias_augmentation",
    "validacao_por",
    "videos",
    "n_sinalizadores",
    "epocas",
    "aviso",
    "ensaio",
    "criado_em",
    "protocolo",
)
CHAVES_VERSOES = ("python", "numpy", "cv2", "mediapipe", "tensorflow", "keras")

AVISO_USO = "Reconhece só os sinais do vocabulário deste modelo e não substitui um intérprete de Libras."
RESSALVA_AVALIACAO = (
    "Métricas da avaliação por partições (D13): o teste foi feito com sinalizadores que ficaram "
    "fora do treino de cada partição. O modelo publicado é o modelo final, treinado com todos os "
    "sinalizadores, e não tem avaliação independente própria; estes números são a melhor "
    "estimativa disponível, não uma garantia para uma pessoa nova."
)


class ErroPacote(ErroLibras):
    """Pacote ausente, incompleto, alterado ou de formato desconhecido (mensagem para o usuario)."""


# ----------------------------------------------------------------------
# Leitura e conferencia
# ----------------------------------------------------------------------
def ler_cartao(pasta: Path) -> dict:
    arq = Path(pasta) / "cartao.json"
    if not arq.exists():
        raise ErroPacote("Este pacote não tem o arquivo cartao.json. Peça à equipe um pacote exportado pela bancada.")
    try:
        cartao = json.loads(arq.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ErroPacote("O cartao.json do pacote está corrompido. Peça um pacote novo à equipe.") from e
    if not isinstance(cartao, dict) or cartao.get("formato") != FORMATO:
        raise ErroPacote(
            "Este pacote é de um formato que esta versão do LibrasVision não conhece. "
            "Atualize o aplicativo ou peça um pacote novo à equipe."
        )
    return cartao


def verificar_pacote(pasta: Path) -> dict:
    """Confere arquivos, formato e hashes ANTES de carregar o modelo. Retorna o cartao.

    O hash detecta arquivo corrompido ou alterado depois da publicacao; nao e assinatura digital
    (quem altera o modelo tambem pode alterar o cartao). Ver D19."""
    pasta = Path(pasta)
    if not pasta.is_dir():
        raise ErroPacote("A pasta do modelo não existe mais. Importe o pacote de novo.")
    faltam = [a for a in ARQUIVOS if not (pasta / a).is_file()]
    if faltam:
        raise ErroPacote(f"Pacote incompleto: falta {', '.join(faltam)}. Importe o pacote (.zip) de novo.")
    cartao = ler_cartao(pasta)
    hashes = cartao.get("hash") or {}
    for nome in ARQUIVOS_COM_HASH:
        esperado = hashes.get(nome)
        if not esperado or sha256_arquivo(pasta / nome) != esperado:
            raise ErroPacote(
                f"O arquivo {nome} não confere com o cartão do pacote (hash diferente): o pacote foi "
                "alterado ou corrompido depois de publicado. Nada foi carregado. Peça um pacote novo à "
                "equipe."
            )
    try:
        meta = json.loads((pasta / "metadados.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise ErroPacote("O metadados.json do pacote está corrompido.") from e
    if list(meta.get("classes") or []) != list(cartao.get("classes") or []):
        raise ErroPacote("As classes do cartão não conferem com as do modelo. Peça um pacote novo à equipe.")
    if cartao.get("ensaio") or meta.get("ensaio"):
        raise ErroPacote(
            "Este pacote é de um modelo ENSAIO (treinado abaixo dos mínimos) e não pode ser usado no aplicativo."
        )
    if "sinalizadores" in meta:
        raise ErroPacote("Este pacote contém dados que não deveriam sair da bancada. Ele não será usado.")
    return cartao


def listar_pacotes(base: Path) -> list[dict]:
    """Pacotes em `base`, do publicado mais recente para o mais antigo. Cada item: pasta, id,
    cartao (ou None) e erro (texto, ou None se o pacote esta integro)."""
    base = Path(base)
    out = []
    if not base.is_dir():
        return out
    for pasta in sorted(p for p in base.iterdir() if p.is_dir() and not p.name.startswith(".")):
        try:
            cartao, erro = verificar_pacote(pasta), None
        except ErroPacote as e:
            erro = str(e)
            try:
                cartao = ler_cartao(pasta)
            except ErroPacote:
                cartao = None
        out.append({"pasta": pasta, "id": (cartao or {}).get("id") or pasta.name, "cartao": cartao, "erro": erro})
    out.sort(key=lambda d: ((d["cartao"] or {}).get("publicado_em") or "", d["id"]), reverse=True)
    return out


def rotulo(cartao: dict, classe: str) -> str:
    """Como a classe aparece na legenda (rotulo do vocabulario gravado no cartao)."""
    return (cartao.get("rotulos") or {}).get(classe, classe)


def parametros_decisao(cartao: dict) -> dict:
    """Parametros do Transcritor/Reconhecedor gravados na publicacao (D15). O usuario nao os ajusta."""
    return dict(cartao.get("decisao") or {})


def resumo(cartao: dict) -> list[tuple[str, str]]:
    """Linhas (rotulo, valor) para mostrar o pacote ao usuario."""
    sinais = [rotulo(cartao, c) for c in cartao.get("classes", []) if c != cartao.get("nome_nenhum_sinal")]
    linhas = [
        ("Sinais reconhecidos", ", ".join(sinais)),
        ("Treinado em", cartao.get("criado_em") or "-"),
        ("Publicado em", cartao.get("publicado_em") or "-"),
        ("Protocolo", cartao.get("protocolo") or "-"),
        ("MediaPipe", str((cartao.get("versoes") or {}).get("mediapipe", "-"))),
        ("Hash do modelo", ((cartao.get("hash") or {}).get("modelo.keras") or "-")[:16]),
    ]
    av = cartao.get("avaliacao")
    if av:
        m = av.get("metricas") or {}
        f1 = m.get("f1_macro_video") or {}
        acc = m.get("acuracia_video") or {}
        if f1.get("media") is not None:
            linhas.append(
                (
                    "Avaliação por partições",
                    f"F1 macro por vídeo {f1['media']:.2f} (dp {f1.get('dp', 0):.2f}); "
                    f"acurácia por vídeo {acc.get('media', float('nan')):.2f}; "
                    f"{av.get('n_particoes', '-')} partições",
                )
            )
    return linhas


# ----------------------------------------------------------------------
# Zip: exportar e importar
# ----------------------------------------------------------------------
def compactar(pasta: Path, destino_zip: Path) -> Path:
    """Gera o .zip de um pacote ja conferido (so os tres arquivos, sem pastas nem caminhos)."""
    verificar_pacote(pasta)
    destino_zip = Path(destino_zip)
    destino_zip.parent.mkdir(parents=True, exist_ok=True)
    tmp = destino_zip.with_name(destino_zip.name + ".tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for nome in ARQUIVOS:
            z.write(Path(pasta) / nome, arcname=nome)
    tmp.replace(destino_zip)
    return destino_zip


def importar_zip(arquivo_zip: Path, base: Path) -> Path:
    """Extrai e confere um pacote .zip em `base/<id>/`. Nada fica em `base` se o pacote for invalido.
    Retorna a pasta do pacote (a existente, se o mesmo pacote ja foi importado)."""
    arquivo_zip, base = Path(arquivo_zip), Path(base)
    if not zipfile.is_zipfile(arquivo_zip):
        raise ErroPacote("O arquivo escolhido não é um pacote .zip válido.")
    with zipfile.ZipFile(arquivo_zip) as z:
        infos = [i for i in z.infolist() if not i.is_dir()]
        nomes = {Path(i.filename).name: i for i in infos}
        estranhos = [
            i.filename
            for i in infos
            if Path(i.filename).name not in ARQUIVOS or ".." in Path(i.filename).parts or Path(i.filename).is_absolute()
        ]
        if estranhos or set(nomes) != set(ARQUIVOS) or len(infos) != len(ARQUIVOS):
            raise ErroPacote(
                "O .zip não tem o formato de um pacote do LibrasVision (deve conter só modelo.keras, "
                "metadados.json e cartao.json)."
            )
        if sum(i.file_size for i in infos) > TAMANHO_MAXIMO_ZIP_MB * 1024 * 1024:
            raise ErroPacote("O pacote é grande demais para ser um modelo do LibrasVision.")
        base.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(prefix=".importando_", dir=base))
        try:
            for nome, info in nomes.items():
                with z.open(info) as src, open(tmp / nome, "wb") as dst:
                    shutil.copyfileobj(src, dst)
            cartao = verificar_pacote(tmp)
            ident = str(cartao.get("id") or "")
            if not ident or Path(ident).name != ident or ident.startswith("."):
                raise ErroPacote("O cartão do pacote tem um identificador inválido.")
            destino = base / ident
            if destino.exists():
                try:
                    if verificar_pacote(destino).get("hash") == cartao.get("hash"):
                        return destino  # o mesmo pacote ja estava importado
                except ErroPacote:
                    pass
                raise ErroPacote(f"Já existe um pacote diferente com o identificador {ident} na pasta de modelos.")
            tmp.replace(destino)
            return destino
        finally:
            if tmp.exists():
                shutil.rmtree(tmp, ignore_errors=True)
