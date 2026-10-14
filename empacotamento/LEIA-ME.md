# Empacotamento do app real (avaliação, item 13.6)

Nada aqui foi decidido como forma de distribuição (D19). Comando usado na avaliação, com o PyInstaller
instalado só durante o teste (ferramenta de build, fora do `requirements.txt`):

    pyinstaller --noconfirm --name LibrasVision --windowed --paths src ^
        --collect-data mediapipe --collect-binaries mediapipe ^
        --hidden-import libras.tempo_real_servico --add-data configs/app_real.yaml:configs ^
        --add-data src/libras/ui/tokens.json:libras/ui --add-data src/libras/ui/recursos:libras/ui/recursos ^
        empacotamento/entrada_app_real.py

`entrada_app_real.py` existe porque, congelado, não há `python -m`: a interface chama o próprio executável
com `-m libras.tempo_real_servico` e este arquivo despacha para o serviço. Resultados em docs/decisoes.md (D19).
