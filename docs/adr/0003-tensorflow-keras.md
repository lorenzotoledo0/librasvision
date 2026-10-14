# ADR 0003: TensorFlow/Keras 2.19 para a LSTM

- Situação: aceita
- Data: 2026-10-05
- Detalhes: [decisoes.md, D1, D5 e D13](../decisoes.md#d5-arquitetura)

## Contexto

O modelo é uma LSTM empilhada pequena, treinada em CPU. O alvo de execução é um computador comum (Intel i5, 4 GB de RAM, Windows). O MediaPipe 0.10.21 exige `protobuf<5`.

## Decisão

Usar TensorFlow 2.19.1 com Keras, a versão mais recente compatível com `protobuf<5`. No Windows nativo ele roda em CPU, o que basta para uma LSTM desse porte. O PyTorch foi descartado para não somar um segundo conjunto de dependências pesadas ao MediaPipe.

## Consequências

- O TensorFlow ocupa bastante memória. Por isso a interface nunca importa TensorFlow: treino, ablação e tempo real rodam em subprocesso.
- Se a memória apertar no hardware alvo, a alternativa a avaliar é o TensorFlow Lite (pendência registrada).
