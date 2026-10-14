# ADR 0005: Dois aplicativos sobre um núcleo comum

- Situação: aceita
- Data: 2026-10-10
- Detalhes: [decisoes.md, D19](../decisoes.md#d19-dois-aplicativos-bancada-librasvision-lab-e-app-real-librasvision)

## Contexto

A equipe precisa de uma ferramenta completa de coleta, treino e avaliação. A pessoa que usa o sistema no dia a dia precisa apenas da câmera e da legenda.

## Decisão

Dois aplicativos no mesmo pacote `libras`:

- **LibrasVision Lab** (`libras.app`): a bancada da equipe.
- **LibrasVision** (`libras.app_real`): câmera e legenda. Não importa coleta, treino nem augmentation, e um teste de arquitetura garante isso.

O app real só consome **pacotes de modelo publicados** pela bancada. Cada pacote traz o modelo, os metadados e uma verificação de integridade, sem IDs de sinalizadores.

## Consequências

- Normalização, janelas e decisão continuam com uma única implementação, usada pelos dois.
- O app real pode ser empacotado e distribuído sem as ferramentas da equipe (avaliação do PyInstaller em `empacotamento/`).
