"""Mostra, por classe, videos 'ok', videos com landmarks, sinalizadores e janelas elegiveis,
e se a selecao de treino atende aos minimos do config (treino.minimos).

Uso:
  python scripts/classes.py
  python scripts/classes.py --classes biblioteca prova secretaria
"""

import argparse

import _bootstrap


def main():
    _bootstrap.sem_tensorflow()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--classes", nargs="*", default=None)
    ap.add_argument("--ensaio", action="store_true")
    a = ap.parse_args()
    from libras.config import carregar_config
    from libras.selecao import avaliar_selecao, tabela_classes

    cfg = carregar_config(a.config)
    av = avaliar_selecao(cfg, classes=a.classes, tabela=tabela_classes(cfg), ensaio=a.ensaio or None)
    print(av.texto())
    print("\nPode treinar." if av.pode_treinar else "\nNao pode treinar com esta selecao.")
    return 0 if av.pode_treinar else 1


if __name__ == "__main__":
    _bootstrap.rodar(main)
