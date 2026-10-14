"""Estudo de ablacao da data augmentation, com teste estatistico e tabelas/figuras para o TCC.

Etapas (mesmo split e mesmas seeds em tudo): baseline; cada familia isolada; combinacao;
intensidades fraca/media/forte da melhor condicao (escolhida pela VALIDACAO); baseline no
protocolo secundario. Teste t corrigido para validacao cruzada repetida + Holm.

Saidas em runs/<data>_ablacao/: RELATORIO.md, tabela_*.csv e .tex (abnTeX2), fig_*.png,
resultados.json e uma pasta por condicao.

Custo: (condicoes ~8) x (seeds) x (particoes) treinos. Num i5 pode levar horas. Se for
interrompida, retome com --continuar <pasta>: condicoes concluidas sao reaproveitadas.

Exemplos:
  python scripts/ablacao.py
  python scripts/ablacao.py --continuar runs/20261020_101500_ablacao
  python scripts/ablacao.py --seeds 42 --folds 0 1 --epocas 20     # ensaio rapido (NAO usar no TCC)
"""

import argparse
from pathlib import Path

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
    ap.add_argument("--continuar", type=Path, default=None, help="pasta de uma ablacao interrompida")
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--folds", nargs="*", type=int, default=None)
    ap.add_argument("--epocas", type=int, default=None)
    ap.add_argument("--detalhado", action="store_true", help="mostra o progresso de cada epoca")
    ap.add_argument("--gerar-splits", action="store_true", help="regera o split principal com a selecao atual")
    ap.add_argument("--progresso", default=None, help="arquivo .jsonl de eventos de progresso (usado pela interface)")
    a = ap.parse_args()
    from libras.ablacao import executar_ablacao
    from libras.config import carregar_config

    cfg = carregar_config(a.config)
    if a.classes:
        from libras.selecao import fixar_classes

        fixar_classes(cfg, a.classes)
    if a.ensaio:
        cfg["treino"]["ensaio"] = True
    sob = {"treino": {"epocas": a.epocas}} if a.epocas else {}
    if a.seeds or a.folds or a.epocas:
        print("ATENCAO: ensaio reduzido (seeds/particoes/epocas alterados). Nao use estes numeros no TCC.\n")
    if a.gerar_splits:
        from libras.splits import gerar_splits, salvar_split, videos_elegiveis

        linhas, _ = videos_elegiveis(cfg)
        salvar_split(cfg, gerar_splits(cfg, linhas))
    pasta = executar_ablacao(
        cfg,
        pasta=a.continuar,
        seeds=a.seeds,
        folds=a.folds,
        sobrescrever=sob,
        verbose=2 if a.detalhado else 0,
        arquivo_progresso=a.progresso,
    )
    print("\n" + (pasta / "RELATORIO.md").read_text(encoding="utf-8"))
    print(f"Saidas em {pasta}")
    return 0


if __name__ == "__main__":
    _bootstrap.rodar(main)
