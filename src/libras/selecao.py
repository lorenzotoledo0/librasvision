"""Selecao das classes que entram no treino (itens 2 e 6).

- classes_treino(cfg): fonte unica da lista de classes do treino, na ordem do vocabulario.
  * Classes pedidas explicitamente (linha de comando, botao Iniciar): usadas como estao; classe
    inexistente ou arquivada e erro.
  * Selecao salva (configs/local.yaml ou config.yaml): classes inexistentes, arquivadas ou sem
    videos sao descartadas em silencio (selecao obsoleta).
  * Sem selecao: as classes ativas que tem videos 'ok' com landmarks. Antes do item 6 o padrao era
    o vocabulario inteiro, o que cobrava classes sem nenhum video.
- tabela_classes(cfg): por classe, videos 'ok', videos com landmarks, sinalizadores e janelas.
- avaliar_selecao(...): considera SO as classes marcadas e aplica os minimos (treino.minimos).
  Classe abaixo do minimo fica bloqueada com explicacao; treino.ensaio = true permite treinar
  mesmo assim, e todas as saidas passam a ser marcadas como ENSAIO.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from .config import Config
from .erros import ErroConfig, ErroDadosInsuficientes

TEXTO_ENSAIO = (
    "ENSAIO: treino com classes abaixo dos minimos configurados (treino.minimos). "
    "Estes numeros nao devem ser usados no TCC."
)

POR_QUE_MINIMO_CLASSES = (
    "Por que pelo menos 2 classes: com uma classe só, o modelo não tem o que distinguir e "
    "a saída é sempre 100% para ela. nenhum_sinal conta como classe."
)

_CHAVE_EXPLICITA = "_classes_explicitas"


def fixar_classes(cfg: Config, classes: list[str]) -> list[str]:
    """Selecao pedida explicitamente (linha de comando ou botao Iniciar). Erro se nao existir."""
    todas = cfg.classes
    desconhecidas = [c for c in classes if c not in todas]
    if desconhecidas:
        raise ErroConfig(
            f"Classes pedidas para o treino que nao existem (ou estao arquivadas): "
            f"{', '.join(desconhecidas)}. Disponiveis: {', '.join(todas)}."
        )
    cfg["treino"]["classes"] = list(classes)
    cfg["treino"][_CHAVE_EXPLICITA] = True
    return [c for c in todas if c in classes]


def sinalizadores_por_classe(cfg: Config) -> dict[str, set]:
    """Classe ativa -> sinalizadores com video 'ok' cujo arquivo de landmarks existe (verificacao
    barata; a validade completa do cache e conferida na tabela e nos videos elegiveis)."""
    from .coleta.manifesto import Manifesto
    from .extracao import caminho_npz

    ativas = set(cfg.classes)
    sig: dict[str, set] = defaultdict(set)
    for l in Manifesto(cfg).linhas:
        if l["status"] == "ok" and l["classe"] in ativas and caminho_npz(cfg, l).exists():
            sig[l["classe"]].add(l["signer_id"])
    return dict(sig)


def classes_com_videos(cfg: Config) -> list[str]:
    sig = sinalizadores_por_classe(cfg)
    return [c for c in cfg.classes if c in sig]


def limpar_selecao(cfg: Config, sel: list[str] | None, com_videos: list[str] | None = None) -> list[str]:
    """Remove da selecao salva o que nao existe, esta arquivado ou nao tem videos."""
    com_videos = classes_com_videos(cfg) if com_videos is None else com_videos
    return [c for c in cfg.classes if c in (sel or []) and c in com_videos]


def classes_treino(cfg: Config) -> list[str]:
    todas = cfg.classes
    tr = cfg.get("treino") or {}
    sel = tr.get("classes")
    if sel and tr.get(_CHAVE_EXPLICITA):
        desconhecidas = [c for c in sel if c not in todas]
        if desconhecidas:
            raise ErroConfig(
                f"Classes selecionadas para o treino que nao existem (ou estao arquivadas): "
                f"{', '.join(desconhecidas)}. Disponiveis: {', '.join(todas)}."
            )
        return [c for c in todas if c in sel]  # ordem do vocabulario
    sig = sinalizadores_por_classe(cfg)
    com = [c for c in todas if c in sig]
    limpa = limpar_selecao(cfg, sel, com) if sel else []
    if limpa:
        return limpa
    # sem selecao: classes com videos que atingem o minimo de sinalizadores (todas com videos se ENSAIO)
    # Se nenhuma atinge o minimo, devolve todas com videos: a validacao entao explica o minimo, em vez de
    # uma lista vazia que esconderia o motivo.
    n = minimos(cfg)["sinalizadores_por_classe"]
    ok = [c for c in com if eh_ensaio(cfg) or len(sig[c]) >= n]
    return ok or com


def eh_ensaio(cfg: Config) -> bool:
    return bool((cfg.get("treino") or {}).get("ensaio", False))


def minimos(cfg: Config) -> dict:
    m = (cfg.get("treino") or {}).get("minimos") or {}
    return {"sinalizadores_por_classe": int(m.get("sinalizadores_por_classe", 2)), "classes": int(m.get("classes", 2))}


@dataclass
class LinhaClasse:
    classe: str
    videos_ok: int = 0
    videos_com_landmarks: int = 0
    sinalizadores: int = 0
    janelas: int | None = None
    selecionada: bool = False
    bloqueio: str = ""

    def como_dict(self) -> dict:
        return {
            "classe": self.classe,
            "videos_ok": self.videos_ok,
            "videos_com_landmarks": self.videos_com_landmarks,
            "sinalizadores": self.sinalizadores,
            "janelas": self.janelas,
            "selecionada": self.selecionada,
            "bloqueio": self.bloqueio,
        }


@dataclass
class AvaliacaoSelecao:
    classes: list[str]
    linhas: list[LinhaClasse]
    bloqueios: dict[str, str] = field(default_factory=dict)
    avisos: list[str] = field(default_factory=list)
    erros: list[str] = field(default_factory=list)  # impedem ate o ensaio
    ensaio: bool = False

    @property
    def pode_treinar(self) -> bool:
        return not self.erros and (not self.bloqueios or self.ensaio)

    def texto(self) -> str:
        cab = f"{'classe':16} {'videos ok':>9} {'c/ landm.':>9} {'sinaliz.':>8} {'janelas':>8}  situacao"
        out = [cab]
        for l in self.linhas:
            sit = ("bloqueada: " + l.bloqueio) if l.bloqueio else ("selecionada" if l.selecionada else "-")
            jan = "-" if l.janelas is None else str(l.janelas)
            out.append(
                f"{l.classe:16} {l.videos_ok:>9} {l.videos_com_landmarks:>9} {l.sinalizadores:>8} {jan:>8}  {sit}"
            )
        out += [f"ERRO: {e}" for e in self.erros] + [f"AVISO: {a}" for a in self.avisos]
        return "\n".join(out)


def tabela_classes(cfg: Config, contar_janelas: bool = True) -> list[LinhaClasse]:
    """Contagens por classe (todas as classes ativas, selecionadas ou nao)."""
    from .coleta.manifesto import Manifesto
    from .extracao import cache_valido, caminho_npz, carregar_npz, meta_esperada
    from .normalizacao import espec_de_config, janelas_do_video

    linhas = {c: LinhaClasse(c) for c in cfg.classes}
    sinal: dict[str, set] = defaultdict(set)
    espec = espec_de_config(cfg) if contar_janelas else None
    nada = cfg.nome_nenhum_sinal
    for l in Manifesto(cfg).linhas:
        c = l["classe"]
        if c not in linhas or l["status"] != "ok":
            continue
        linhas[c].videos_ok += 1
        if not cache_valido(caminho_npz(cfg, l), meta_esperada(cfg, l)):
            continue
        linhas[c].videos_com_landmarks += 1
        sinal[c].add(l["signer_id"])
        if contar_janelas:
            d = carregar_npz(caminho_npz(cfg, l))
            jv = janelas_do_video(d.partes, d.tempos, d.largura, d.altura, espec, cfg, eh_nenhum_sinal=(c == nada))
            linhas[c].janelas = (linhas[c].janelas or 0) + len(jv.X)
    for c, ls in linhas.items():
        ls.sinalizadores = len(sinal[c])
        if contar_janelas and ls.janelas is None:
            ls.janelas = 0
    return list(linhas.values())


def mensagem_minimo_classes(n_min: int, marcadas: list[str], com_videos: list[str]) -> str:
    """Diz o que falta e como resolver quando ha menos classes marcadas que o minimo."""

    def lista(cs):
        return ", ".join(f"'{c}'" for c in cs)

    if not com_videos:
        return (
            "Nenhuma classe tem vídeos 'ok' com landmarks ainda. Grave na aba Gravar e aprove na aba Revisão "
            f"(pelo menos {n_min} palavras, contando nenhum_sinal)."
        )
    if len(com_videos) < n_min:
        verbo = "tem" if len(com_videos) == 1 else "têm"
        return (
            f"Marque pelo menos {n_min} classes com vídeos. Hoje só {lista(com_videos)} {verbo} vídeos; "
            "grave 'nenhum_sinal' ou outra palavra."
        )
    faltam = [c for c in com_videos if c not in marcadas]
    return (
        f"Marque pelo menos {n_min} classes com vídeos (marcadas: {len(marcadas)}). Também têm vídeos: {lista(faltam)}."
    )


def avaliar_selecao(
    cfg: Config, classes: list[str] | None = None, tabela: list[LinhaClasse] | None = None, ensaio: bool | None = None
) -> AvaliacaoSelecao:
    """Avalia SO as classes marcadas. Classes nao marcadas nunca geram erro nem bloqueio."""
    classes = list(classes) if classes is not None else classes_treino(cfg)
    ensaio = eh_ensaio(cfg) if ensaio is None else ensaio
    tabela = tabela if tabela is not None else tabela_classes(cfg, contar_janelas=False)
    mins = minimos(cfg)
    av = AvaliacaoSelecao(classes, tabela, ensaio=ensaio)
    por_nome = {l.classe: l for l in tabela}
    for l in tabela:
        l.selecionada, l.bloqueio = False, ""
    for c in classes:
        l = por_nome.get(c)
        if l is None:
            av.erros.append(f"Classe {c} nao existe no vocabulario.")
            continue
        l.selecionada = True
        if l.videos_com_landmarks == 0:
            l.bloqueio = "sem videos 'ok' com landmarks"
            av.erros.append(f"A classe {c} nao tem nenhum video utilizavel: desmarque-a ou colete videos.")
        elif l.sinalizadores < mins["sinalizadores_por_classe"]:
            l.bloqueio = (
                f"{l.sinalizadores} sinalizador(es); minimo {mins['sinalizadores_por_classe']} "
                "(sem isso nao ha como testar a classe em uma pessoa que nao esteve no treino)"
            )
            av.bloqueios[c] = l.bloqueio
    marcadas_validas = [c for c in classes if c in por_nome and por_nome[c].videos_com_landmarks > 0]
    if len(marcadas_validas) < mins["classes"]:
        com_videos = [l.classe for l in tabela if l.videos_com_landmarks > 0]
        av.erros.append(mensagem_minimo_classes(mins["classes"], marcadas_validas, com_videos))
    nada = cfg.nome_nenhum_sinal
    if nada and nada not in classes:
        av.avisos.append(
            f"A classe {nada} ficou fora do treino: no tempo real o modelo sempre escolhera um "
            "dos sinais, mesmo com a pessoa parada (ver decisoes D7)."
        )
    if av.bloqueios and ensaio:
        av.avisos.append(TEXTO_ENSAIO)
    return av


def situacao(l: LinhaClasse, mins: dict) -> tuple[str, bool]:
    """Texto curto da coluna Situacao e se a classe pode ser marcada (sem ENSAIO)."""
    if l.videos_ok == 0:
        return "sem vídeos", False
    if l.videos_com_landmarks == 0:
        return "sem landmarks (Dataset > Extrair)", False
    if l.sinalizadores < mins["sinalizadores_por_classe"]:
        return f"abaixo do mínimo ({l.sinalizadores} de {mins['sinalizadores_por_classe']} sinalizadores)", True
    return "elegível", True


def pode_marcar(l: LinhaClasse, mins: dict, ensaio: bool) -> tuple[bool, str]:
    """O bloqueio vale para MARCAR (e treinar), nunca para desmarcar."""
    if l.videos_com_landmarks == 0:
        return False, f"A classe {l.classe} ainda não tem vídeos utilizáveis."
    if l.sinalizadores < mins["sinalizadores_por_classe"] and not ensaio:
        return False, (
            f"A classe {l.classe} tem {l.sinalizadores} sinalizador(es); o mínimo é "
            f"{mins['sinalizadores_por_classe']}. Grave com outra pessoa ou marque ENSAIO antes."
        )
    return True, ""


def elegiveis(tabela: list[LinhaClasse], mins: dict, ensaio: bool = False) -> list[str]:
    return [l.classe for l in tabela if pode_marcar(l, mins, ensaio)[0]]


def exigir_selecao_valida(cfg: Config, linhas_elegiveis: list[dict]) -> AvaliacaoSelecao:
    """Usado por gerar_splits e treinar_final: erro claro se a selecao nao puder ser treinada."""
    tabela = _tabela_de_linhas(cfg, linhas_elegiveis)
    av = avaliar_selecao(cfg, tabela=tabela)
    if av.erros:
        raise ErroDadosInsuficientes("Selecao de classes invalida:\n" + "\n".join(f"- {e}" for e in av.erros))
    if av.bloqueios and not av.ensaio:
        txt = "\n".join(f"- {c}: {m}" for c, m in av.bloqueios.items())
        raise ErroDadosInsuficientes(
            "Classes abaixo do minimo configurado (treino.minimos):\n"
            + txt
            + "\nDesmarque essas classes, colete mais dados, ou rode como ENSAIO (treino.ensaio: true / --ensaio). "
            "Resultados de ensaio ficam marcados e nao servem para o TCC."
        )
    return av


def _tabela_de_linhas(cfg: Config, linhas: list[dict]) -> list[LinhaClasse]:
    t = {c: LinhaClasse(c) for c in cfg.classes}
    sig: dict[str, set] = defaultdict(set)
    for l in linhas:
        if l["classe"] in t:
            t[l["classe"]].videos_ok += 1
            t[l["classe"]].videos_com_landmarks += 1
            sig[l["classe"]].add(l["signer_id"])
    for c in t:
        t[c].sinalizadores = len(sig[c])
    return list(t.values())
