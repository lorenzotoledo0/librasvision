@echo off
REM Abre o LibrasVision (app para o usuario final): so camera e legenda.
REM Precisa de um pacote de modelo publicado pela bancada (pasta publicados\ ou Importar pacote .zip).
cd /d "%~dp0"
if not exist .venv\Scripts\pythonw.exe (
  echo Ambiente nao instalado. Rode instalar.bat primeiro.
  pause
  exit /b 1
)
set PYTHONPATH=%~dp0src
start "" .venv\Scripts\pythonw.exe -m libras.app_real %*
