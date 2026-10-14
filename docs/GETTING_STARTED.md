# Primeiros passos

Guia do zero até a primeira gravação e o primeiro treino. Para o roteiro completo de uma sessão de coleta, veja [protocolo_coleta.md](protocolo_coleta.md).

## 1. Requisitos

| Item | Requisito |
|---|---|
| Sistema | Windows 10/11 (principal) ou Linux. macOS não é testado. |
| Python | **3.11** (3.13 e 3.14 não funcionam: TensorFlow e MediaPipe não publicam pacotes para eles) |
| Disco | cerca de 1 GB para as dependências |
| Webcam | 720p recomendada |
| Hardware alvo | Intel i5, 4 GB de RAM (desempenho nesse hardware: a ser medido) |

## 2. Instalação

### Windows (PowerShell)

Instale o Python 3.11 ("Windows installer (64-bit)" em <https://www.python.org/downloads/windows/>), mantendo as opções **py launcher** e **tcl/tk and IDLE**. Ele convive com outras versões já instaladas.

```powershell
git clone https://github.com/lorenzotoledo0/librasvision.git
cd librasvision
.\scripts\setup_windows.ps1
```

Se a política de execução do PowerShell bloquear o script, rode-o só nesta sessão:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_windows.ps1
```

Alternativa sem PowerShell: dê dois cliques em `instalar.bat`.

### Linux

```bash
git clone https://github.com/lorenzotoledo0/librasvision.git
cd librasvision
bash scripts/setup_unix.sh
```

O script exige `python3.11` no PATH. No Ubuntu, a interface precisa de bibliotecas do sistema: `sudo apt install libegl1 libgl1 libxkbcommon0 libfontconfig1 libdbus-1-3`.

### Com uv (opcional)

```powershell
uv venv --python 3.11 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
```

## 3. Verificação

```powershell
.venv\Scripts\python scripts\verificar_ambiente.py
.venv\Scripts\python -m pytest -q
```

O `verificar_ambiente.py` confere a versão do Python, os pacotes, o MediaPipe Holistic e a webcam (use `--sem-camera` para pular a câmera).

## 4. Primeira sessão (bancada)

```powershell
.\LibrasVisionLab.bat
```

1. **Sinalizadores:** cadastre a pessoa com o TCLE assinado e anote o ID (S01...) no termo físico.
2. **Gravar:** ligue a câmera, escolha sinalizador e palavra. Espaço grava, Enter aceita, Esc descarta. Para várias repetições seguidas, use **Gravar rajada**.
3. **Revisão:** aprove ou rejeite os takes pendentes e corte o início e o fim se precisar.
4. **Dataset:** gere o relatório de validação e os splits.
5. **Treino:** escolha as classes (mínimo de 2 sinalizadores por classe) e treine. Depois, treine o **Modelo final**.
6. **Resultados:** **Publicar modelo** gera o pacote que o app LibrasVision usa.

## 5. App do usuário final

```powershell
.\LibrasVision.bat
```

Ele precisa de um pacote publicado (pasta `publicados/`) ou de um `.zip` importado em Configurações > Modelo. Sem pacote, a tela explica o que falta. O app nunca simula predição.

## 6. Linha de comando

Todos os scripts em `scripts/` têm `--help`. Os principais:

```powershell
.venv\Scripts\python scripts\sinalizadores.py cadastrar --mao direita --tcle 2026-10-10
.venv\Scripts\python scripts\importar.py D:\videos --padrao projeto --simular
.venv\Scripts\python scripts\extrair.py
.venv\Scripts\python scripts\validar_dados.py
.venv\Scripts\python scripts\gerar_splits.py
.venv\Scripts\python scripts\treinar.py --nome baseline --aug nenhuma
.venv\Scripts\python scripts\treinar.py --nome final --final
.venv\Scripts\python scripts\ablacao.py
.venv\Scripts\python scripts\tempo_real.py --video teste.mp4
```

## Problemas comuns

- **"Python 3.11 não encontrado":** instale o 3.11 com o py launcher e rode a instalação de novo.
- **Câmera não abre:** feche Teams, Zoom ou o navegador, que podem estar usando a câmera. A tela de erro indica a causa provável.
- **Windows bloqueia o `.bat`:** antes de extrair o `.zip`, clique com o botão direito > Propriedades > Desbloquear. Para uma pasta já extraída: `Get-ChildItem -Recurse | Unblock-File`.
- **Logs:** `runs\logs\app.log` e `runs\logs\falhas_nativas.log`.
