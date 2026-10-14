@echo off
REM Instala o LibrasVision no Windows: cria .venv com Python 3.11 e instala as dependencias.
cd /d "%~dp0"
py -3.11 --version >nul 2>&1
if errorlevel 1 (
  echo.
  echo Python 3.11 nao encontrado.
  echo Baixe o instalador "Windows installer (64-bit)" do Python 3.11 em https://www.python.org/downloads/windows/
  echo Na instalacao, mantenha marcadas as opcoes "py launcher" e "tcl/tk and IDLE".
  echo O Python 3.14 que ja esta instalado pode continuar; os dois convivem.
  pause
  exit /b 1
)
if not exist .venv (
  echo Criando ambiente .venv com Python 3.11...
  py -3.11 -m venv .venv || (echo Falha ao criar o ambiente. & pause & exit /b 1)
)
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt || (echo Falha na instalacao dos pacotes. & pause & exit /b 1)
echo.
echo Verificando o ambiente...
.venv\Scripts\python scripts\verificar_ambiente.py
echo.
echo Bancada da equipe: LibrasVisionLab.bat. App para o usuario final: LibrasVision.bat
pause
