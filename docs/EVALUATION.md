# Avaliação

Situação: protocolo implementado e testado com dados sintéticos. **Resultados: a ser medido.** Ainda não existem vídeos reais.

## Protocolo

| Protocolo | Uso | Divisão |
|---|---|---|
| Principal | números do TCC | validação cruzada estratificada por classe e **agrupada por sinalizador** (`StratifiedGroupKFold`), 5 partições |
| Opcional | conferência | leave-one-signer-out (uma partição por pessoa) |
| Secundário | mostrar o efeito do vazamento | divisão aleatória por vídeo, sempre rotulada como secundária |

Regras:

- Nenhum sinalizador aparece no treino e no teste da mesma partição. Todas as janelas de um vídeo ficam do mesmo lado.
- Validação para early stopping: cerca de 20% dos sinalizadores do treino de cada partição. Com menos de 3 sinalizadores no treino, a validação passa a ser 15% dos vídeos, estratificada, e isso fica registrado.
- Escolhas de modelo e de "melhor condição" são feitas pela validação. O teste nunca aparece na curva por época.
- O vazamento é conferido ao gerar, ao carregar e no relatório de validação.
- Splits são salvos com uma assinatura do conjunto de vídeos. Se o conjunto mudar, o treino recusa o split desatualizado.

## Métricas

| Nível | Métricas |
|---|---|
| Por vídeo (principal) | F1 macro, acurácia, matriz de confusão (média das probabilidades das janelas do vídeo) |
| Por janela de 1 s | F1 macro, acurácia, matriz de confusão |
| Robustez | as mesmas métricas em cópias degradadas do teste: oclusão de mão, ruído, queda de FPS, ausência de face e vídeo escuro |
| Tempo real | latência de ponta a ponta (mediana e p95), FPS processado e memória, por etapa |

## Ablação e estatística

`scripts/ablacao.py` compara, com o mesmo split e as mesmas seeds: baseline, cada família de augmentation isolada, a combinação, as intensidades e o baseline no protocolo secundário.

- Teste t corrigido para validação cruzada repetida (Nadeau e Bengio, 2003, na forma de Bouckaert e Frank, 2004) com correção de Holm (1979). As referências devem ser conferidas antes do texto final.
- Só se fala em "ganho" quando o p ajustado é menor que 0,05 e a diferença é positiva. Caso contrário: "sem evidência de diferença".
- Saídas: `RELATORIO.md`, tabelas CSV e LaTeX (compatíveis com abnTeX2) e figuras PNG.

## Resultados

| Experimento | F1 macro por vídeo | Acurácia por vídeo | Latência p95 (i5, 4 GB) |
|---|---|---|---|
| Baseline (protocolo principal) | a ser medido | a ser medido | a ser medido |
| Melhor condição de augmentation | a ser medido | a ser medido | a ser medido |
| Baseline (protocolo secundário) | a ser medido | a ser medido | - |

O resultado do TCC I (97% de acurácia, 80 a 100 ms) veio de um protótipo descartado e de uma avaliação que provavelmente não separava sinalizadores. Ele é referência histórica, **não baseline**.

## O que não conta como resultado

- Execuções marcadas como **ENSAIO** (classes abaixo do mínimo, seeds ou épocas reduzidas).
- O modo "Testar acertos" do teste ao vivo: é qualitativo.
- O modelo final treinado com todos os sinalizadores: ele não tem avaliação independente.
- Medições de latência fora do hardware alvo.
