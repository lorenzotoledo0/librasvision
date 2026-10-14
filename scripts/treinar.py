"""Treina e avalia o modelo em todas as particoes do protocolo (e seeds do config).

Cada execucao vai para runs/<data>_<nome>/ com configuracao, versoes, hardware,
metricas por particao, curvas, matrizes de confusao (so teste), predicoes e modelos.

Exemplos:
  python scripts/treinar.py --nome baseline --aug nenhuma            # baseline SEM augmentation
  python scripts/treinar.py --nome aug_espacial --aug espacial
  python scripts/treinar.py --nome aug_completa --aug espacial temporal ruido --intensidade media
  python scripts/treinar.py --nome final --final                     # modelo para o tempo real

Antes: landmarks extraidos (gravacao e importacao ja geram; senao scripts/extrair.py). Splits: scripts/gerar_splits.py ou --gerar-splits. Sem dados, o script para com mensagem clara.
"""

import argparse

import _bootstrap


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument(
        "--classes", nargs="*", default=None, help="classes do treino (padrao: as classes ativas com videos)"
    )
    ap.add_argument(
        "--ensaio",
        action="store_true",
        help="permite classes abaixo dos minimos; saidas marcadas como ENSAIO (nao usar no TCC)",
    )
    ap.add_argument("--nome", default="treino")
    ap.add_argument("--protocolo", default=None)
    ap.add_argument(
        "--aug",
        nargs="*",
        default=None,
        help="familias: espacial temporal ruido video | 'nenhuma' | omitido = segue o config",
    )
    ap.add_argument("--intensidade", default=None, choices=["fraca", "media", "forte"])
    ap.add_argument("--modo-aug", default=None, choices=["online", "offline"])
    ap.add_argument(
        "--variantes-video",
        nargs="*",
        default=None,
        help="variantes da familia 'video' (padrao: augmentation.video.variantes do config)",
    )
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--folds", nargs="*", type=int, default=None, help="so estas particoes (para testes rapidos)")
    ap.add_argument("--epocas", type=int, default=None)
    ap.add_argument("--final", action="store_true", help="treina o modelo final com todos os sinalizadores")
    ap.add_argument("--sem-modelos", action="store_true", help="nao salva o modelo de cada particao")
    ap.add_argument("--silencioso", action="store_true")
    ap.add_argument("--progresso", default=None, help="arquivo .jsonl de eventos de progresso (usado pela interface)")
    ap.add_argument("--gerar-splits", action="store_true", help="regera o split do protocolo com a selecao atual")
    a = ap.parse_args()

    from libras.config import carregar_config
    from libras.treino import OpcoesTreino, executar, treinar_final

    cfg = carregar_config(a.config)
    if a.classes:
        from libras.selecao import fixar_classes

        fixar_classes(cfg, a.classes)
    if a.ensaio:
        cfg["treino"]["ensaio"] = True
    familias = None
    if a.aug is not None:
        familias = [] if (not a.aug or a.aug == ["nenhuma"]) else a.aug
    sob = {}
    if a.epocas:
        sob["treino"] = {"epocas": a.epocas}
    if a.modo_aug:
        sob.setdefault("augmentation", {})["modo"] = a.modo_aug
    if a.variantes_video:
        sob.setdefault("augmentation", {})["video"] = {"variantes": a.variantes_video}
    op = OpcoesTreino(
        nome=a.nome,
        protocolo=a.protocolo,
        familias=familias,
        intensidade=a.intensidade,
        seeds=a.seeds,
        folds=a.folds,
        verbose=0 if a.silencioso else 2,
        salvar_modelos=False if a.sem_modelos else None,
        sobrescrever=sob,
        arquivo_progresso=a.progresso,
    )
    if a.final:
        pasta = treinar_final(cfg, op)
        print(f"\nModelo final salvo em {pasta}")
        print(
            "Atencao: este modelo usa todos os sinalizadores no treino; os numeros do TCC vem das execucoes "
            "por particao, nao dele."
        )
        return 0
    if a.gerar_splits:
        from libras.splits import gerar_splits, salvar_split, videos_elegiveis

        linhas, _ = videos_elegiveis(cfg)
        salvar_split(cfg, gerar_splits(cfg, linhas, protocolo=a.protocolo))
    pasta, r = executar(cfg, op)
    print("\n" + (pasta / "LEIA-ME.txt").read_text(encoding="utf-8"))
    print(f"Resultados em {pasta}")
    return 0


if __name__ == "__main__":
    _bootstrap.rodar(main)
