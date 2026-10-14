# Modelo

Situação: arquitetura e treino implementados e testados com dados sintéticos. **Nenhum modelo foi treinado com dados reais. Desempenho: a ser medido.**

Todos os valores abaixo são configuráveis em `configs/config.yaml`. As justificativas estão em [decisoes.md](decisoes.md) (D2 a D8).

## Entrada

Uma janela de **30 frames x 550 features** (`float32`).

### Vetor por frame (padrão, seção `features`)

| Parte | Pontos | Dimensões |
|---|---|---|
| Pose (x, y, z, visibilidade) | 33 | 132 |
| Mãos esquerda e direita (x, y, z) | 42 | 126 |
| Face: subconjunto de lábios, sobrancelhas e contorno (x, y, z) | 96 | 288 |
| Flags de presença (pose, mão esquerda, mão direita, face) | - | 4 |
| **Total** | **171** | **550** |

A face completa (468 pontos) ocuparia 84% do vetor e tende ao sobreajuste com um dataset pequeno. A face completa e a remoção da face ficam como variantes de ablação (`features.face.modo`).

### Normalização (módulo único: `libras/normalizacao.py`)

1. Coordenadas convertidas para pixels (x vezes largura, y vezes altura, z vezes largura), para que vídeos 16:9 e 4:3 tenham a mesma geometria.
2. Origem no ponto médio dos ombros e escala pela distância entre os ombros. A referência é a **mediana da janela**, não de cada frame, para preservar o movimento do tronco.
3. Lacunas internas de até 5 frames sem detecção são interpoladas. Lacunas maiores ficam zeradas, com a flag de presença em 0.
4. Janelas com pose em menos de 50% dos frames são descartadas.
5. Nenhum frame é espelhado antes do MediaPipe, em nenhuma etapa.

O mesmo código prepara a janela no treino, na avaliação e no tempo real. Um teste garante que as duas trajetórias geram entradas idênticas.

### Janela temporal

- Todo vídeo (e o buffer da webcam) é reamostrado para 30 FPS pelos instantes reais dos frames. Uma janela sempre equivale a 1 s.
- Passo de 5 frames no treino.
- Seleção por atividade (padrão): em vídeos de sinal, só entram janelas com movimento médio das mãos de pelo menos 50% do maior movimento do vídeo. Vídeos de `nenhum_sinal` usam todas as janelas.

## Arquitetura (seção `modelo`)

```text
Entrada (30, 550)
LSTM 64 unidades, retorna sequência, tanh
Dropout 0,3
LSTM 32 unidades, tanh
Dropout 0,3
Densa softmax (número de classes)
```

| Hiperparâmetro | Padrão |
|---|---|
| Otimizador | Adam, taxa 0,001 |
| Perda | entropia cruzada categórica |
| Épocas | até 150 |
| Early stopping | `val_loss`, paciência 20, restaura os melhores pesos |
| Pesos de classe | inversamente proporcionais à frequência |
| Seeds da ablação | 42, 7, 2024 |

## Data augmentation (seção `augmentation`)

Aplicada **somente às janelas de treino** de cada partição. Validação, teste e inferência nunca recebem augmentation.

| Família | Transformações |
|---|---|
| A. Espacial | rotação 2D (até ±15°), escala anisotrópica (±10%), escala de cada mão em torno do pulso (±10%), deslocamento das mãos em relação ao corpo, cisalhamento leve, escala de z. Espelhamento implementado, mas desligado por padrão. |
| B. Temporal | time warping (velocidade 0,8 a 1,2), deslocamento da janela em até 3 frames, remoção de 10% dos frames internos |
| C. Ruído e falhas | jitter gaussiano (σ = 0,01 largura de ombro), perda de uma mão em frames contíguos, perda da face em um trecho |
| D. Vídeo (offline) | variantes escuro, claro, desfoque, ruído de imagem e rotação/zoom nos frames, com reextração dos landmarks (`scripts/aumentar_videos.py`) |

Rotação, zoom e translação globais em imagem perdem o efeito depois da normalização pelos ombros. Por isso foram substituídos por equivalentes no espaço dos landmarks normalizados (detalhes em D8).

Modos: online (variação nova a cada época, com seed + época) ou offline (cópias fixas). Intensidades fraca, média e forte.

## Decisão em tempo real (`libras/decisao.py`)

1. Sem mão em pelo menos 30% dos frames da janela, o modelo nem roda.
2. Média móvel das últimas 5 predições.
3. Abaixo de 60% de confiança, nada é exibido.
4. Um sinal entra no texto depois de 3 predições iguais seguidas e só se repete depois de uma pausa.
5. A classe `nenhum_sinal` nunca vira texto.

## Saídas de um treino

- `runs/<data>_<nome>/`: configuração efetiva, versões, hardware, seeds, cópia do split, métricas por partição, curvas, matrizes de confusão, predições e um `LEIA-ME.txt`.
- Modelo final (`--final`): `modelo.keras` + `metadados.json` (classes, features, janela, normalização e parâmetros do MediaPipe), usados pelo tempo real.
- Pacote publicado (`publicados/<id>/`): `modelo.keras`, `metadados.json` filtrado e `cartao.json`, com hashes. É o que o app real consome.
