"""Ponto de entrada do app real CONGELADO (PyInstaller), so para a avaliacao do item 13.6.

No executavel congelado nao existe "python -m": a interface chama o proprio executavel com
"-m libras.tempo_real_servico ..." (ui.componentes.Subprocesso). Este arquivo despacha esse caso para o
servico e qualquer outro para a interface. Fora do executavel congelado ele nao e usado.
"""

import multiprocessing
import sys

if __name__ == "__main__":
    multiprocessing.freeze_support()
    if len(sys.argv) >= 3 and sys.argv[1] == "-m" and sys.argv[2] == "libras.tempo_real_servico":
        sys.argv = [sys.argv[0]] + sys.argv[3:]
        from libras.tempo_real_servico import main as servico

        sys.exit(servico())
    from libras.app_real.janela import main

    sys.exit(main())
