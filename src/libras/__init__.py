"""LibrasVision: reconhecimento de sinais dinamicos de Libras com MediaPipe Holistic + LSTM."""

import sys

__version__ = "0.1.0"

PYTHON_SUPORTADO = ((3, 10), (3, 12))  # inclusivo

if not (PYTHON_SUPORTADO[0] <= sys.version_info[:2] <= PYTHON_SUPORTADO[1]):
    sys.stderr.write(
        "\n[LibrasVision] Python {}.{} detectado. Este projeto precisa de Python 3.11 "
        "(3.10 a 3.12 aceitos), porque TensorFlow e MediaPipe ainda nao publicam pacotes "
        "para versoes mais novas.\nNo Windows: instale o Python 3.11 e crie o ambiente com "
        "'py -3.11 -m venv .venv'. Veja o README.\n\n".format(*sys.version_info[:2])
    )
