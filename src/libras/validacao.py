"""Relatorio de validacao do dataset (scripts/validar_dados.py).

Responde, antes de treinar: ha dados suficientes? estao balanceados? as maos
aparecem? algum video nao gera janelas? os splits salvos estao atualizados e sem
vazamento?
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .config import Config
from .utils import carimbo, escrever_csv_atomico, escrever_texto_atomico


@dataclass
class Relatorio:
    linhas: list[str] = field(default_factory=list)
    problemas: list[str] = field(default_factory=list)  # bloqueiam o treino
    avisos: list[str] = field(default_factory=list)
    videos: list[dict] = field(default_factory=list)

    def sec(self, titulo: str) -> None:
        self.linhas += ["", f"## {titulo}", ""]

    def add(self, txt: str = "") -> None:
        self.linhas.append(txt)

    def texto(self) -> str:
        fim = ["", "## Conclusao", ""]
        if self.problemas:
            fim.append(f"{len(self.problemas)} problema(s) que impedem o treino:")
            fim += [f"- {p}" for p in self.problemas]
        else:
            fim.append("Nenhum problema bloqueante.")
        if self.avisos:
            fim += ["", f"{len(self.avisos)} aviso(s):"] + [f"- {a}" for a in self.avisos]
        return "\n".join(["# Relatorio de validacao do dataset"] + self.linhas + fim) + "\n"


def _tabela(cab: list[str], linhas: list[list]) -> list[str]:
    out = ["| " + " | ".join(cab) + " |", "|" + "---|" * len(cab)]
    out += ["| " + " | ".join(str(x) for x in l) + " |" for l in linhas]
    return out


def gerar_relatorio(cfg: Config, montar_janelas: bool = True) -> Relatorio:
    from .coleta.manifesto import Manifesto
    from .coleta.sinalizadores import CadastroSinalizadores
    from .splits import PROTOCOLOS, assinatura, caminho_split, verificar_vazamento, videos_elegiveis

    r = Relatorio()
    man, cad = Manifesto(cfg), CadastroSinalizadores(cfg)
    classes = cfg.classes
    from .vocabulario import vocabulario_em_cache

    voc = vocabulario_em_cache(cfg)

    # 1. manifesto ------------------------------------------------------
    r.sec("Manifesto")
    st = Counter(l["status"] for l in man.linhas)
    r.add(
        f"Registros: {len(man)} ({', '.join(f'{k}={v}' for k, v in sorted(st.items())) or 'nenhum'}). "
        f"Sinalizadores cadastrados: {len(cad.ids())} ({len(cad.ids_com_tcle())} com TCLE)."
    )
    integ = man.verificar_integridade(cad.ids(), cad.ids_com_tcle())
    if not integ.ok:
        r.problemas.append("Manifesto com problemas de integridade (detalhes abaixo).")
        r.add("")
        r.add("```")
        r.add(integ.texto())
        r.add("```")
    if len(man) == 0:
        r.avisos.append(
            "Ainda nao ha videos. Grave (LibrasVisionLab.bat, aba Gravar) ou importe (aba Gravar ou scripts/importar.py)."
        )
        return r
    if st.get("pendente"):
        r.avisos.append(
            f"{st['pendente']} video(s) pendentes de revisao nao entram no treino (aba Progresso e revisao)."
        )

    # 2. contagem ok por classe x sinalizador -----------------------------
    r.sec("Videos 'ok' por classe e sinalizador")
    ok = man.contagens(("ok",))
    sig = sorted({s for (_, s) in ok})
    tab = []
    for c in classes:
        tab.append([c] + [ok.get((c, s), 0) for s in sig] + [sum(ok.get((c, s), 0) for s in sig)])
    r.linhas += _tabela(["classe"] + sig + ["total"], tab)
    abaixo = [(c, s) for c in classes for s in sig if ok.get((c, s), 0) < voc.meta(c, s)]
    if abaixo:
        r.avisos.append(f"{len(abaixo)} combinacao(oes) classe x sinalizador abaixo da meta do vocabulario.")
    tot_classe = {c: sum(ok.get((c, s), 0) for s in sig) for c in classes}
    nz = [n for n in tot_classe.values() if n > 0]
    if nz and max(nz) / min(nz) > 1.5:
        r.avisos.append(
            f"Desbalanceamento de videos entre classes (maior/menor = {max(nz) / min(nz):.1f}). "
            "Considere ativar pesos de classe no treino ou equilibrar a coleta."
        )
    if len(sig) < 2:
        r.problemas.append(
            f"Ha {len(sig)} sinalizador(es) com videos 'ok'; a avaliacao independente de sinalizador exige ao menos 2."
        )
    elif len(sig) < 3:
        r.avisos.append("Com 2 sinalizadores a avaliacao independente e pouco informativa (ideal: 5 ou mais).")

    # 2b. selecao de classes para o treino ---------------------------------
    from .selecao import avaliar_selecao, tabela_classes

    r.sec("Classes para o treino (selecao e minimos)")
    av = avaliar_selecao(cfg, tabela=tabela_classes(cfg, contar_janelas=False))
    r.linhas += _tabela(
        ["classe", "videos ok", "com landmarks", "sinalizadores", "situacao"],
        [
            [
                l.classe,
                l.videos_ok,
                l.videos_com_landmarks,
                l.sinalizadores,
                ("bloqueada: " + l.bloqueio) if l.bloqueio else ("selecionada" if l.selecionada else "fora"),
            ]
            for l in av.linhas
        ],
    )
    r.problemas += av.erros
    if av.bloqueios and not av.ensaio:
        r.problemas.append(
            f"Classes selecionadas abaixo do minimo: {', '.join(av.bloqueios)} (desmarque, colete "
            "mais dados ou treine como ensaio)."
        )
    r.avisos += av.avisos

    # 3. landmarks --------------------------------------------------------
    r.sec("Landmarks")
    elegiveis, excluidos = videos_elegiveis(cfg)
    r.add(f"Videos 'ok' com landmarks atualizados: {len(elegiveis)}. Fora do dataset: {len(excluidos)}.")
    sem_lm = [v for v, m in excluidos.items() if "landmarks" in m]
    if sem_lm:
        r.problemas.append(
            f"{len(sem_lm)} video(s) 'ok' sem landmarks extraidos ou desatualizados. Rode: scripts/extrair.py"
        )
    if not elegiveis or not montar_janelas:
        return r

    # 4. janelas ----------------------------------------------------------
    from .dataset import montar_conjunto
    from .erros import ErroLibras

    r.sec("Janelas")
    try:
        conj = montar_conjunto(cfg, elegiveis)
    except ErroLibras as e:
        r.problemas.append(str(e))
        return r
    lim_mao = cfg["coleta"]["qualidade"]["taxa_minima_mao"]
    jan_classe = Counter(conj.y.tolist())
    r.add(f"Janelas validas: {conj.n} ({conj.espec.descricao()}; {conj.memoria_mb():.0f} MB em memoria).")
    r.add("")
    r.linhas += _tabela(
        ["classe", "videos", "janelas"],
        [
            [c, sum(i.classe == c for i in conj.infos.values()), jan_classe.get(k, 0)]
            for k, c in enumerate(conj.classes)
        ],
    )
    nzj = [v for v in jan_classe.values() if v > 0]
    if nzj and max(nzj) / min(nzj) > 2:
        r.avisos.append(f"Desbalanceamento de janelas entre classes (maior/menor = {max(nzj) / min(nzj):.1f}).")
    descartes: Counter = Counter()
    for info in conj.infos.values():
        descartes.update(info.descartadas)
        motivos = []
        if info.n_janelas == 0:
            motivos.append("nenhuma janela valida")
        if info.taxa_mao < lim_mao and info.classe != cfg.nome_nenhum_sinal:
            motivos.append(f"mao detectada em {info.taxa_mao:.0%} dos frames")
        motivos += info.avisos
        r.videos.append(
            {
                "video_id": info.video_id,
                "classe": info.classe,
                "signer_id": info.signer_id,
                "n_frames": info.n_frames,
                "n_janelas": info.n_janelas,
                "taxa_mao": f"{info.taxa_mao:.3f}",
                "taxa_pose": f"{info.taxa_pose:.3f}",
                "problemas": "; ".join(motivos),
            }
        )
    if descartes:
        r.add("")
        r.add("Janelas descartadas: " + ", ".join(f"{m}: {n}" for m, n in descartes.most_common()))
    prob_videos = [v for v in r.videos if v["problemas"]]
    r.sec("Videos que merecem revisao")
    if prob_videos:
        r.linhas += _tabela(
            ["video_id", "janelas", "mao", "problemas"],
            [[v["video_id"], v["n_janelas"], v["taxa_mao"], v["problemas"]] for v in prob_videos[:60]],
        )
        if len(prob_videos) > 60:
            r.add(f"... e mais {len(prob_videos) - 60} (lista completa no CSV).")
        sem_jan = [v for v in prob_videos if v["n_janelas"] == 0]
        if sem_jan:
            r.avisos.append(f"{len(sem_jan)} video(s) nao geram nenhuma janela e serao ignorados no treino.")
    else:
        r.add("Nenhum.")

    # 5. splits -------------------------------------------------------------
    import json

    r.sec("Splits salvos")
    por_id = {l["video_id"]: l for l in elegiveis}
    from .selecao import classes_treino

    ass = assinatura(elegiveis, classes_treino(cfg))
    algum = False
    for prot in PROTOCOLOS:
        p = caminho_split(cfg, prot)
        if not p.exists():
            continue
        algum = True
        sp = json.loads(p.read_text(encoding="utf-8"))
        if sp["assinatura_dados"] != ass:
            r.avisos.append(
                f"Splits '{prot}' desatualizados (o conjunto de videos mudou). "
                f"Rode scripts/gerar_splits.py --protocolo {prot}."
            )
            r.add(f"- {prot}: DESATUALIZADO")
            continue
        vaz = verificar_vazamento(sp, por_id)
        if vaz:
            r.problemas.append(f"Vazamento nos splits '{prot}': " + "; ".join(vaz[:5]))
            r.add(f"- {prot}: VAZAMENTO")
        else:
            r.add(f"- {prot}: ok, {len(sp['folds'])} particoes, sem vazamento de video ou de sinalizador")
    if not algum:
        r.add("Nenhum split gerado ainda (scripts/gerar_splits.py).")
    return r


def salvar_relatorio(cfg: Config, r: Relatorio) -> tuple[Path, Path | None]:
    base = cfg.caminho("relatorios")
    ts = carimbo()
    md = base / f"validacao_{ts}.md"
    escrever_texto_atomico(md, r.texto())
    csv_p = None
    if r.videos:
        csv_p = base / f"validacao_{ts}_videos.csv"
        escrever_csv_atomico(csv_p, list(r.videos[0].keys()), r.videos)
    return md, csv_p
