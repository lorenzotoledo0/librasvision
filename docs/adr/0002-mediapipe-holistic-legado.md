# ADR 0002: MediaPipe Holistic 0.10.21 (API legada)

- Situação: aceita
- Data: 2026-10-05
- Detalhes: [decisoes.md, D1 e D2](../decisoes.md#d1-ambiente-e-versões)

## Contexto

O reconhecimento usa landmarks de pose, mãos e face do mesmo quadro. A API `mp.solutions.holistic` entrega os três juntos, com os modelos embutidos no pacote (sem download durante a execução), e é a API citada no texto do TCC. A partir do MediaPipe 0.10.30/1.0, essa API foi removida.

## Decisão

Fixar `mediapipe==0.10.21`, a última versão com `mp.solutions.holistic`. Isso também fixa `numpy<2` (1.26.4) e `protobuf<5` (4.25.9).

## Consequências

- Os landmarks são extraídos por um único módulo (`libras/landmarks.py`) na coleta, no treino e no tempo real.
- O `protobuf<5` limita o TensorFlow a 2.19.x (ADR 0003).
- Migrar para a API de tarefas do MediaPipe exigiria reextrair todos os vídeos e revalidar o vetor de features.
