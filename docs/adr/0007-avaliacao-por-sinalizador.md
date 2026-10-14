# ADR 0007: Avaliação agrupada por sinalizador

- Situação: aceita
- Data: 2026-10-05
- Detalhes: [decisoes.md, D6 e D14](../decisoes.md#d6-protocolo-de-avaliação)

## Contexto

O resultado do TCC I (97% de acurácia) veio de uma avaliação que provavelmente não separava sinalizadores entre treino e teste. Assim, o modelo é testado em pessoas que já viu, e o número sai otimista.

## Decisão

- Avaliação principal: validação cruzada estratificada por classe e agrupada por sinalizador (`StratifiedGroupKFold`, 5 partições). Nenhum sinalizador aparece no treino e no teste da mesma partição.
- A divisão aleatória por vídeo fica só como protocolo secundário, rotulado como tal.
- Escolhas de modelo e da "melhor condição" são feitas pela validação, nunca pelo teste.
- Comparações usam o teste t corrigido para validação cruzada repetida, com correção de Holm.

## Consequências

- O treino exige pelo menos 2 sinalizadores por classe. A meta é de mais de 10 sinalizadores.
- Os números do TCC II não podem ser comparados diretamente com os do TCC I, que ficam como referência histórica.
