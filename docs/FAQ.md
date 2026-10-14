# Perguntas frequentes

## O LibrasVision substitui um intérprete de Libras?

Não. Ele reconhece apenas os sinais isolados do vocabulário do modelo carregado (inicialmente "biblioteca", "prova" e "secretaria") e não interpreta frases, gramática ou contexto. O app mostra esse aviso de forma permanente.

## Qual é a acurácia?

Ainda não foi medida. Não existem vídeos reais coletados. Os 97% citados no TCC I vieram de um protótipo descartado, com uma avaliação que provavelmente não separava sinalizadores entre treino e teste. Esse número é referência histórica, não resultado deste sistema. Veja [EVALUATION.md](EVALUATION.md).

## Por que Python 3.11 se já tenho o 3.14?

TensorFlow e MediaPipe não publicam pacotes para 3.13 e 3.14. Os dois Pythons podem ficar instalados juntos. Veja o [ADR 0001](adr/0001-python-3-11.md).

## Por que não usar a versão mais nova do MediaPipe?

A API `mp.solutions.holistic`, que entrega pose, mãos e face juntas, foi removida a partir da 0.10.30. Veja o [ADR 0002](adr/0002-mediapipe-holistic-legado.md).

## Como adiciono uma palavra nova?

Pela aba **Vocabulário** da bancada, pelo campo "Nova palavra" da aba Gravar ou com `scripts/vocabulario.py adicionar "Sala de aula"`. Não é preciso mudar código. Depois, grave vídeos da palavra com pelo menos 2 sinalizadores e treine de novo.

## Por que a prévia aparece espelhada e o vídeo salvo não?

A prévia espelhada é mais confortável para quem sinaliza, como um espelho. O arquivo salvo não é espelhado de propósito: espelhar trocaria mão direita por esquerda nos dados, e sinais assimétricos podem mudar de significado.

## Os vídeos vão para a internet?

Não. Vídeos e imagens nunca saem do computador da coleta e nunca entram no Git. O modelo publicado contém só pesos e metadados, sem IDs, nomes ou imagens. Veja [DATA_PRIVACY.md](DATA_PRIVACY.md).

## Por que a rajada só extrai os landmarks no fim?

Para os takes saírem um atrás do outro, sem esperar a extração de cada um. Os frames ficam na memória até o fim, com um limite configurável (`coleta.gravacao.memoria_max_rajada_mb`).

## Funciona em celular?

Ainda não. A interface foi desenhada para facilitar uma futura versão mobile (faixas de largura, alvos de toque de 44 px, pacote de modelo autocontido), mas nada foi decidido sobre a plataforma.

## Onde estão os logs?

Em `runs\logs\app.log` e `runs\logs\falhas_nativas.log`. Eles usam só IDs anônimos.
