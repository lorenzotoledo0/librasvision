# LibrasVision (TCC II) - contexto para o Claude Code

Este arquivo reúne o contexto do projeto. Detalhes e justificativas das escolhas técnicas estão em @docs/decisoes.md. Atualize aquele documento sempre que uma decisão mudar.

## O que é o projeto

- Trabalho de Conclusão de Curso em Ciência da Computação, Centro Universitário do Distrito Federal (UDF), Brasília-DF.
- Título: "LibrasVision: Desenvolvimento de um sistema de visão computacional para o reconhecimento de sinais dinâmicos em ambientes educacionais".
- Objetivo: reconhecer sinais dinâmicos da Libras e traduzi-los para texto em tempo real, reduzindo barreiras de comunicação de estudantes surdos no ambiente acadêmico (escassez de intérpretes TILSP).
- Abordagem: MediaPipe Holistic (landmarks de pose, mãos e face) + LSTM empilhada (Keras/TensorFlow) sobre janelas de 30 frames.
- Vocabulário mínimo: biblioteca, prova, secretaria, mais a classe nenhum_sinal. Pode crescer. O vocabulário é configurável (data/vocabulario.json).
- Metas do TCC: acurácia acima de 85%, baixa latência, rodar em hardware comum (alvo: Intel i5, 4 GB de RAM, webcam 720p, Windows 10/11).
- Equipe: Augusto Cezar Araujo Saboia, Lorenzo Toledo, Lucas Caitano Barreto, Elizeu Barbosa Souza.
- Orientação: TCC I com Prof.ª Dra. Flavia Maria; TCC II (julho a dezembro de 2026) com Prof.ª Kadidja Valéria Reginaldo de Oliveira.
- Toda a documentação acadêmica é em português do Brasil (normas ABNT, tabelas compatíveis com abnTeX2).

## Estado atual (referência: 2026-10-06)

- O sistema do TCC II foi recriado do zero. O protótipo do TCC I foi descartado.
- Fases 0 a 6 implementadas, mais as rodadas de revisão (treino com seleção de classes, vocabulário configurável, landmarks salvos na gravação com cortes, interface única, editor de corte, teste ao vivo, dois aplicativos).
- Nenhum vídeo real coletado ainda. Não existe nenhum número de desempenho do TCC II.
- O resultado do TCC I (97% de acurácia, 80 a 100 ms) veio de um protótipo descartado e de uma avaliação que provavelmente não separava sinalizadores entre treino e teste. É referência histórica, nunca baseline.
- Meta de coleta: mais de 10 sinalizadores. Pessoas surdas fluentes em Libras, com TCLE.

## Dois aplicativos

- Bancada, "LibrasVision Lab" (`LibrasVisionLab.bat` ou `python -m libras.app`): coleta, revisão, dataset, treino, avaliação, teste ao vivo. Uso da equipe.
- App real, "LibrasVision" (`LibrasVision.bat` ou `python -m libras.app_real`): só câmera e legenda para o usuário final, usa um pacote de modelo publicado pela bancada. Não importa coleta, treino nem augmentation.
- Interface escura e preparada para uma futura versão mobile (três faixas de largura, alvos de toque de 44 px).

## Ambiente (não alterar versões sem verificar compatibilidade)

- Python 3.11 (o 3.14 da máquina da equipe não tem TensorFlow nem MediaPipe; usar `py -3.11`).
- mediapipe 0.10.21 (última com a API legada `mp.solutions.holistic`), numpy 1.26.4, protobuf 4.25.9, tensorflow 2.19.1, opencv-contrib-python 4.11.0.86 (não instalar `opencv-python` junto), PySide6-Essentials 6.11.2.
- Instalação: `instalar.bat`. Conferir com `pip check`.
- Windows é a plataforma da equipe; Linux também funciona para desenvolvimento.

## Estrutura do repositório

- `src/libras/`: pacote principal (normalizacao, extracao, splits, augmentation, coleta, inferencia, decisao, app, app_real, ui, pacote).
- `scripts/`: linha de comando (extrair, treinar, ablacao, tempo_real, verificar_manifesto, sinalizadores, aumentar_videos).
- `configs/config.yaml` é a referência documentada; `configs/local.yaml` guarda ajustes locais (fora do Git).
- `data/` e `runs/` ficam fora do Git (privacidade). `publicados/` também.
- `tests/`: suíte de testes (confirmar o comando; provavelmente `pytest`).
- `docs/`: decisoes.md, analise_interface.md.
- `empacotamento/`: avaliação do PyInstaller (não decidido).

## Regras que não podem ser quebradas

1. Um único módulo de normalização para treino, avaliação e tempo real. Um teste garante que os caminhos dão a mesma entrada ao modelo. Não duplicar normalização, janelas ou decisão.
2. Sem vazamento de dados: todos os vídeos de um sinalizador ficam do mesmo lado de qualquer divisão. Avaliação principal é validação cruzada agrupada por sinalizador (StratifiedGroupKFold, 5 partições). Augmentation só no treino, nunca em validação, teste ou inferência.
3. Nenhum frame é espelhado antes do MediaPipe. O espelhamento existe só na tela.
4. Escolhas de modelo e de "melhor condição" são feitas pela validação, nunca pelo teste. O teste não aparece na curva por época.
5. Ensaios (classes abaixo do mínimo, seeds ou épocas reduzidas) ficam marcados como ENSAIO em todas as saídas e não viram número do TCC. Teste ao vivo é qualitativo e também não vira número do TCC.
6. Nunca simular predição. Sem pacote de modelo válido, a tela explica o que falta.
7. Privacidade: sinalizadores identificados só por ID (S01, S02...). Sem TCLE confirmado, a ferramenta não grava. Logs, manifesto e pacotes não levam nomes, caminhos pessoais nem IDs de sinalizadores (no pacote). Imagens e vídeos nunca saem do computador; o modelo final usa só landmarks.
8. Nenhuma tarefa longa na thread da interface. Treino, ablação e tempo real rodam em subprocesso. O processo da interface nunca importa TensorFlow.
9. Sem emojis na interface, no código e nos logs. O público final é surdo: nenhum retorno depende de som.
10. Cores e medidas visuais vêm de `src/libras/app/tokens.json`. Não escrever cor hexadecimal solta no código da interface. Contraste mínimo WCAG 2.1 AA.
11. Ao fechar vídeos, câmera e subprocessos, esperar o encerramento antes de reabrir (no Windows, arquivo ou câmera presa falha ao abrir).

## Como trabalhar neste repositório

- Fazer mudanças pequenas e verificáveis. Rodar os testes antes de concluir.
- Qualquer decisão técnica nova ou alterada: registrar em docs/decisoes.md com a justificativa, na seção da decisão correspondente (D1 a D19) ou em "Decisões tomadas sem consulta à equipe".
- Não afirmar resultados de desempenho que não foram medidos. Latência e memória no i5 com 4 GB ainda precisam ser medidas pela equipe.
- Escrever textos acadêmicos em português do Brasil, com referências a conferir antes de entrar no texto final.

## Pendências em aberto

- Coletar vídeos reais (meta: mais de 10 sinalizadores) e calibrar limiares iniciais (seleção de janelas por atividade, guia de enquadramento: borda 3%, brilho 60, visibilidade do ombro 0,5).
- Medir latência e memória no hardware alvo; avaliar TensorFlow Lite se a memória apertar.
- Decidir distribuição do app real (instalador PyInstaller de cerca de 2,4 GB, distribuição remota de pacotes, assinatura digital) e opções para mobile (nada decidido).
- Verificar com o primeiro vídeo real se o rótulo `left_hand` do Holistic corresponde à mão esquerda da pessoa.
