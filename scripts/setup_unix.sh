#!/usr/bin/env bash
# Instala o LibrasVision no Linux/macOS: cria .venv com Python 3.11 e instala as dependencias.
# Uso (na raiz do projeto):  bash scripts/setup_unix.sh [--dev] [--sem-camera]
set -euo pipefail
cd "$(dirname "$0")/.."

DEV=0
CAMERA=""
for arg in "$@"; do
  case "$arg" in
    --dev) DEV=1 ;;
    --sem-camera) CAMERA="--sem-camera" ;;
    *) echo "Opcao desconhecida: $arg" >&2; exit 2 ;;
  esac
done

if ! command -v python3.11 >/dev/null 2>&1; then
  echo "python3.11 nao encontrado. Instale o Python 3.11 (por exemplo: sudo apt install python3.11 python3.11-venv)." >&2
  exit 1
fi

[ -x .venv/bin/python ] || python3.11 -m venv .venv
PY=.venv/bin/python
"$PY" -m pip install --upgrade pip
"$PY" -m pip install -r requirements.txt
if [ "$DEV" = 1 ]; then
  "$PY" -m pip install -e ".[dev]"
  .venv/bin/pre-commit install
fi
"$PY" -m pip check

echo
echo "Verificando o ambiente..."
"$PY" scripts/verificar_ambiente.py $CAMERA
echo
echo "Bancada da equipe: PYTHONPATH=src .venv/bin/python -m libras.app"
echo "App do usuario final: PYTHONPATH=src .venv/bin/python -m libras.app_real"
