"""Abre o aplicativo LibrasVision na aba Gravar (atalho para a linha de comando).

Uso:  python scripts/coletar.py [--config configs/config.yaml]
A bancada abre com LibrasVisionLab.bat ou `python -m libras.app`.
"""

import argparse

import _bootstrap


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    from libras.app.janela import main as app_main

    return app_main((["--config", a.config] if a.config else []) + ["--aba", "gravar"])


if __name__ == "__main__":
    _bootstrap.rodar(main)
