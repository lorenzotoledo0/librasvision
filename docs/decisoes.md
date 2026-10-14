# Decisões de projeto

Registro das escolhas técnicas do LibrasVision (TCC II), com justificativa. Tudo o que é parametrizável está em `configs/config.yaml`; este documento explica o porquê dos valores padrão. Atualize-o sempre que uma decisão mudar.

Situação em 2026-10-06 (Fases 0 a 6 implementadas, mais os itens 1 a 5 da revisão de 2026-10-06: correção da reabertura, treino com seleção de classes, vocabulário configurável e remoção de sinalizadores, landmarks salvos na gravação com cortes, interface única; e os itens 6 a 10 da segunda rodada: seleção de classes corrigida, troca de abas rápida, interface dinâmica, editor de corte e teste ao vivo do modelo): nenhum vídeo real coletado. Nenhum número de desempenho existe ainda. O resultado do TCC I (97% de acurácia, 80-100 ms) veio de um protótipo descartado e de uma avaliação que provavelmente não separava sinalizadores entre treino e teste; ele é referência histórica a ser reavaliada, nunca baseline.

## D1. Ambiente e versões

| Item | Escolha | Motivo |
|---|---|---|
| Sistema | Windows 10/11 (equipe); Linux também funciona | Informado pela equipe |
| Python | 3.11 | Python 3.14 (instalado na máquina da equipe) não tem TensorFlow nem MediaPipe. Em 2026-10-05 havia uma issue aberta no repositório do TensorFlow confirmando a incompatibilidade, e o MediaPipe declara suporte a 3.9-3.11 no PyPI. O 3.11 convive com o 3.14 via `py -3.11`. |
| MediaPipe | 0.10.21 | Última versão com a API legada `mp.solutions.holistic`, que traz os modelos embutidos (sem download) e é a citada no TCC. A partir de 0.10.30/1.0 essa API foi removida (verificado: o pacote 1.0.1 não tem `solutions`). |
| NumPy / protobuf | 1.26.4 / 4.25.9 | Exigências do MediaPipe 0.10.21 (numpy < 2, protobuf < 5). |
| TensorFlow | 2.19.1 | Versão mais recente compatível com protobuf < 5. O 2.21 exige protobuf novo e quebra o MediaPipe. No Windows nativo roda em CPU, o que basta para uma LSTM pequena. |
| OpenCV | opencv-contrib-python 4.11.0.86 | Dependência do MediaPipe; instalar `opencv-python` junto causa conflito. |
| Interface | PySide6-Essentials 6.11.2 + shiboken6 6.11.2 | Ver D16. Wheels `abi3` para win_amd64 e cp311; só os módulos essenciais do Qt (sem WebEngine, 3D, Multimedia). Download de cerca de 78 MB no Windows; instalado ocupa cerca de 230 MB (medido no Linux de desenvolvimento). |

Verificação feita: instalação conjunta sem conflitos (`pip check`) em Python 3.11, resolução de todas as dependências para `win_amd64`/cp311 (65 pacotes, todos com wheel), Holistic processando frames e TensorFlow montando uma LSTM no mesmo ambiente.

Verificação feita com o PySide6-Essentials adicionado (2026-10-06): `pip check` sem conflitos junto com opencv-contrib 4.11.0.86, mediapipe 0.10.21, numpy 1.26.4 e protobuf 4.25.9. O OpenCV do pip para Windows não traz Qt próprio, então não há conflito de plugins Qt nessa plataforma (no Linux de desenvolvimento o cv2 define `QT_QPA_PLATFORM_PLUGIN_PATH`, o que a interface contorna).

Observação de memória: o MediaPipe 0.10.21 importa o TensorFlow inteiro só para um utilitário opcional de documentação. Nos processos que não usam TensorFlow (interface, coleta, extração), esse import é bloqueado, o que reduziu o uso de RAM de cerca de 700 MB para cerca de 170 MB na medição feita. Isso importa para o alvo de 4 GB.

Impacto de memória do PySide6, medido no ambiente de desenvolvimento (Linux, não o i5 alvo): a janela Qt custa cerca de 40 MB a mais que a antiga em Tkinter. O processo da interface ocupa cerca de 100 MB ao abrir, cerca de 500 MB na aba Gravar com a prévia do MediaPipe ligada e cerca de 450 MB na aba Treino (o TensorFlow fica no subprocesso de treino, não na interface). O subprocesso de tempo real (TensorFlow + MediaPipe) ocupa cerca de 1,0 GB e só existe enquanto a aba Tempo real está aberta. Pior caso simultâneo previsto: interface + treino, ou interface + tempo real, cerca de 1,5 GB a 2 GB; a equipe deve confirmar no i5 com 4 GB.

## D2. Vetor de features por frame

543 é o número de pontos do Holistic (33 pose + 2 x 21 mãos + 468 face), não o número de features. Com (x, y, z) seriam 1629 dimensões; com visibility na pose, 1662.

Padrão adotado (configurável em `features`):

| Parte | Pontos | Dimensões |
|---|---|---|
| Pose com visibility | 33 | 132 |
| Mãos (esquerda e direita) | 42 | 126 |
| Face, subconjunto lábios + sobrancelhas + contorno | 96 | 288 |
| Flags de presença (pose, mão esq., mão dir., face) | - | 4 |
| **Total** | **171** | **550** |

Justificativa: a face completa (1404 dimensões) ocuparia 84% do vetor e tende a gerar sobreajuste com um dataset pequeno, enquanto a informação linguística não manual mais relevante (boca, sobrancelhas, movimento de cabeça) cabe no subconjunto. A face completa e a remoção da face ficam como variantes de ablação (`features.face.modo: completa | nenhuma`). Os índices dos subconjuntos vêm das conexões oficiais da malha facial do MediaPipe; estão copiados em `normalizacao.FACE_REGIOES` (para o treino não depender do MediaPipe) e um teste confere a cópia contra o MediaPipe instalado.

As coordenadas são extraídas e guardadas cruas, junto com largura e altura do vídeo. Selecionar e normalizar é etapa posterior, então mudar o vetor não exige reextrair os vídeos.

## D3. Normalização, presença e lateralidade (implementado em `src/libras/normalizacao.py`)

- Um único módulo faz a normalização para treino, avaliação e tempo real. Um teste garante que a última janela de um vídeo processado pelo caminho de treino é idêntica à produzida pelo caminho do tempo real.
- Conversão para pixels antes de qualquer cálculo geométrico: x vezes largura, y vezes altura, z vezes largura (o z do MediaPipe usa aproximadamente a escala de x). Sem isso, vídeos 16:9 e 4:3 teriam geometrias diferentes. Um teste confirma que a mesma pessoa deslocada, ampliada e gravada em 1280x720 gera as mesmas features que em 640x480.
- Origem: ponto médio dos ombros (pose 11 e 12). Escala: distância entre os ombros. Assim, distância da câmera, resolução e bordas pretas de padronização deixam de importar.
- Referência por janela, não por frame: centro e escala são a mediana ao longo dos 30 frames da janela. Por frame, apagaria o movimento do tronco (que pode fazer parte do sinal); por vídeo, não existiria no tempo real, que só conhece a janela atual. Janela com pose em menos de 50% dos frames é descartada (sem referência confiável).
- Presença explícita por parte (pose, mão esquerda, mão direita, face): 4 flags concatenadas ao vetor de cada frame.
- Lacunas internas sem detecção de até 5 frames (configurável) são interpoladas linearmente e marcadas como presentes. Lacunas maiores, e lacunas nas bordas da janela, ficam zeradas com flag 0, para o modelo distinguir "mão parada" de "mão não vista". A interpolação é feita dentro da janela, porque é o que o tempo real consegue fazer.
- Lateralidade: nenhum frame é espelhado antes do MediaPipe, em nenhuma etapa (coleta, treino, tempo real). O espelhamento existe só na tela. Assim, os rótulos mão esquerda/direita são produzidos da mesma forma em todo o sistema. A mão dominante fica registrada no manifesto e no cadastro do sinalizador. O espelhamento como augmentation fica desabilitado por padrão (ver D8).
- A verificar com o primeiro vídeo real: se o rótulo `left_hand` do Holistic corresponde à mão esquerda da pessoa. Não afeta a consistência, mas afeta a redação do texto.

## D4. Janela temporal

- Comprimento: 30 frames (cerca de 1 s a 30 FPS), passo 5 no treino. A referência do TCC I era uma janela de 30 frames.
- Todo vídeo, e no tempo real o buffer da webcam com os instantes reais de cada frame, é reamostrado para 30 FPS antes de formar janelas. Assim uma janela é sempre 1 s, mesmo que a webcam entregue 20 FPS. A grade é ancorada no último frame (no tempo real, o frame mais recente sempre entra). Coordenadas são interpoladas quando os dois frames vizinhos têm a parte detectada; senão vale o frame mais próximo.
- Vídeos com menos de 30 frames (após reamostrar) são completados repetindo o primeiro e o último frame (pessoa parada) e recebem aviso.
- Seleção de janelas (`janela.selecao: atividade`, padrão): num vídeo de sinal de 3 s, parte das janelas contém só repouso. Rotular essas janelas com o sinal ensinaria o modelo a chamar repouso de sinal, justamente o que a classe `nenhum_sinal` deve capturar. Por isso, em vídeos de sinal, só ficam janelas cujo movimento médio das mãos é ao menos 50% do maior movimento do vídeo. Vídeos de `nenhum_sinal` usam todas as janelas. O limiar é uma escolha inicial, a ser conferida com os primeiros vídeos reais (o relatório de validação mostra quantas janelas foram descartadas); `selecao: todas` desliga o filtro e pode entrar na ablação.
- Regra de vazamento: todas as janelas de um vídeo ficam do mesmo lado de qualquer divisão, e todos os vídeos de um sinalizador também (ver D6).

## D5. Arquitetura

LSTM empilhada: 64 unidades (retorna sequência), dropout 0,3, 32 unidades, dropout 0,3, densa softmax. Adam (taxa 0,001), entropia cruzada, até 150 épocas com early stopping em val_loss (paciência 20, restaurando os melhores pesos), pesos de classe inversamente proporcionais à frequência. Ativação padrão tanh (o protótipo do TCC I usava ReLU dentro da LSTM, o que desativa o kernel otimizado e é menos estável). Frames e partes ausentes chegam zerados e indicados pelas flags de presença (D3). Tudo em YAML. O tamanho é compatível com CPU de i5 e 4 GB.

## D6. Protocolo de avaliação

Com a meta de mais de 10 sinalizadores:

- Principal: validação cruzada agrupada por sinalizador e estratificada por classe (StratifiedGroupKFold), 5 partições. Nenhum sinalizador aparece em treino e teste da mesma partição. Com 10+ pessoas, cada partição testa em 2 ou mais pessoas nunca vistas.
- Opcional: leave-one-signer-out (uma partição por pessoa). Mais fiel, mas com 10+ pessoas multiplica o custo de treino (problema num i5, especialmente na ablação com várias seeds).
- Secundária, rotulada como tal: divisão aleatória por vídeo (mostra o quanto o resultado fica otimista quando o sinalizador já foi visto, o que ajuda a explicar a diferença em relação ao TCC I).
- Validação para early stopping: cerca de 20% dos sinalizadores do treino de cada partição (deixando ao menos 2 no treino). Se o treino tiver menos de 3 sinalizadores, a validação passa a ser 15% dos vídeos do treino, estratificada por classe, e isso fica registrado no split (`validacao_por: videos`).
- Os splits são salvos em `data/splits/<protocolo>.json` com uma assinatura do conjunto de vídeos. Se houver coleta, revisão ou exclusão depois, o treino recusa splits desatualizados, porque resultados com divisões diferentes não são comparáveis.
- Vazamento é conferido na geração, ao carregar e no relatório de validação: nenhum vídeo em dois lados, nenhum sinalizador em treino e teste (ou validação e teste) nos protocolos principais, e cada vídeo no teste de exatamente uma partição.
- Recusa explícita com menos de 2 sinalizadores; aviso com menos de 3.

## D7. Classe "nenhum_sinal"

Presente por padrão no vocabulário (D17), com `tipo: nenhum_sinal`. Sem ela, o tempo real sempre escolhe uma das classes de sinal, mesmo com a pessoa parada. Conteúdo: mãos em repouso, coçar o rosto, ajustar óculos, transições entre sinais, movimentos aleatórios. Deve ser gravada por todos os sinalizadores, como as demais.

Desde a seleção de classes do treino (D13), ela é opcional: pode ser desmarcada, mas o treino, a interface e o `RELATORIO`/`LEIA-ME` da execução avisam que, sem ela, o modelo em tempo real sempre aponta um dos sinais, mesmo com a pessoa parada. Para o TCC, a recomendação continua sendo treinar com ela.

## D8. Data augmentation (implementado em `src/libras/augmentation/`)

Adaptação para sequências de landmarks da ideia de Islam, Hossain et al. (2019), que aplicaram rotação, zoom e deslocamento a imagens para uma CNN (92,87% para 97,12% naquele trabalho). A contribuição metodológica está em mapear cada transformação de imagem para o espaço de landmarks normalizados pelo corpo:

| Em imagem (Islam et al.) | Em landmarks normalizados (este trabalho) |
|---|---|
| Rotação | Rotação 2D única na janela inteira, em torno do centro dos ombros (até ±15°). Mantida: a normalização não remove rotação. |
| Zoom (escala global) | **Sem efeito**: a normalização divide pela largura dos ombros e desfaz qualquer escala global. Um teste demonstra isso. Substituído por escala anisotrópica (proporções corporais, ±10%) e escala de cada mão em torno do pulso (tamanho da mão, ±10%). |
| Deslocamento (translação global) | **Sem efeito**: a normalização centraliza nos ombros. Substituído por deslocamento das mãos em relação ao corpo (±0,05 largura de ombro), que move junto os pontos da mão presentes na pose. |
| (sem equivalente) | Cisalhamento leve; escala de z (o z do MediaPipe é relativo e ruidoso, então só escala multiplicativa). |

Transformações que mudam a referência corporal (escala anisotrópica, cisalhamento, jitter) são seguidas de renormalização com a mesma regra do módulo de normalização (`referencia_ombros`). Assim, uma janela aumentada tem as mesmas propriedades de uma janela real: ombros centrados na origem e largura de ombro 1.

Famílias (cada uma com probabilidade e magnitude no YAML):
- **A. Espaciais:** as da tabela acima. Espelhamento com troca esquerda/direita implementado, mas desligado por padrão e bloqueado enquanto a face estiver no vetor (não há mapa de pares da malha facial). Só ligar com aprovação da equipe: sinais assimétricos podem mudar de significado.
- **B. Temporais:** time warping (velocidade 0,8 a 1,2, volta ao comprimento fixo por interpolação), deslocamento da janela em até 3 frames (repetindo as bordas) e remoção de 10% dos frames internos com reinterpolação (simula queda de FPS). O recorte em posições arbitrárias do vídeo é coberto pelas janelas com passo 5 (D4).
- **C. Ruído e falhas de captura:** jitter gaussiano (σ = 0,01 largura de ombro, cerca de 1,6 px com ombros de 160 px), perda de uma mão em frames contíguos (até 8) e perda da face em frames contíguos. As flags de presença são atualizadas e as coordenadas zeradas. Desvio em relação ao enunciado: em vez de apagar partes da face mantendo a flag em 1 (situação que o MediaPipe nunca produz), a face inteira some em um trecho, como acontece de fato com oclusão.
- **D. Nível de vídeo (offline):** variantes escuro, claro, desfoque, ruído de imagem e rotação/zoom aplicadas aos frames brutos, com reextração dos landmarks (`scripts/aumentar_videos.py`). Só os landmarks são guardados. Custa uma extração completa por variante. As mesmas variantes servem ao teste de robustez da Fase 5, aplicadas aos vídeos de teste.
- Intensidades fraca, média e forte multiplicam as magnitudes por 0,5, 1 e 1,5 (as probabilidades ficam iguais).

Regras garantidas pelo código e cobertas por testes:
- só as janelas de treino de cada partição são aumentadas; validação e teste nunca. As variantes de vídeo só entram pelos vídeos do treino;
- antes de treinar cada partição, o treino confere que nenhum vídeo e nenhum sinalizador do teste estão no treino ou na validação;
- online por padrão (variação nova a cada época, sorteada com seed + época); modo offline com cópias fixas e reprodutíveis (fator efetivo 1 + cópias);
- o efeito real fica registrado em cada execução: janelas vistas, fração alterada e fração por família; no modo offline, o fator de multiplicação;
- as transformações não recebem o rótulo; propriedades testadas: forma igual, sem NaN, magnitudes dentro da faixa, determinismo por seed, distâncias dentro de uma mão mudando só por um fator único, máscaras coerentes;
- nenhum código de augmentation roda na inferência.

Prévia de augmentation (pedido da equipe, 2026-10-14). Na aba Revisão, "Prévia de augmentation" mostra o take e 5 variações lado a lado, animadas, com as famílias e a intensidade escolhidas na tela (`augmentation/previa.py` e `app/previa_augmentation.py`). Serve para conferir se as variações continuam parecendo um sinal real antes de treinar e para gerar figuras do TCC ("Salvar figura" grava o mosaico de um quadro em `runs/augmentation_previa/`, só com esqueletos, sem imagem da pessoa). Regras:
- é só visualização: nada é salvo como dado e o config não muda; o treino continua sorteando as variações como antes;
- a janela mostrada é a de maior movimento das mãos, normalizada pelo mesmo caminho do treino (`janelas_do_video`);
- na prévia, cada família marcada é aplicada sempre (probabilidade 1), para cada variação mostrar o efeito; no treino, cada família entra com a probabilidade do config;
- todas as variações usam o enquadramento do original, para as diferenças de rotação, proporção e profundidade ficarem visíveis.

Por que não 5 cópias fixas por take como padrão do treino: o modo online já gera uma variação nova a cada época (dezenas de versões de cada janela ao longo de um treino), e cópias fixas multiplicam a memória das janelas (fator 6 com 5 cópias) num alvo de 4 GB. O modo offline com `copias_offline: 5` continua disponível e pode entrar na ablação como mais uma condição; a escolha entre os dois é feita pela validação (D14).

## D9. Ferramenta de coleta

- A coleta é a aba Gravar da interface única (D16). A lógica (manifesto, cadastro, vocabulário, gravação, importação, qualidade, cortes) é separada da interface e coberta por testes; a interface só chama funções.
- Câmera no Windows: tenta DirectShow, depois Media Foundation, depois automático, com MJPG forçado (muitas webcams só atingem 30 FPS em 640x480 com MJPG). Falha de abertura gera mensagem com as causas comuns (câmera em uso, permissão de privacidade do Windows, índice errado). A variável de ambiente `LIBRAS_CAMERA_VIDEO=arquivo.mp4` usa um vídeo no lugar da câmera (demonstração e testes).
- Prévia: o MediaPipe da prévia roda numa thread própria sobre o frame mais recente; a interface nunca espera por ele. Ao sair da aba, o MediaPipe da prévia e a câmera são encerrados em segundo plano (item 7: antes isso segurava a troca de aba por cerca de 3 s). A câmera também abre em segundo plano, com indicador de carregamento, e espera o encerramento anterior terminar antes de reabrir (no Windows, câmera ainda presa a outro objeto falha ao abrir). Ao fechar o aplicativo, o encerramento é esperado.
- Gravação: frames guardados em memória com o instante real de chegada. Ao parar, o MediaPipe processa **todos** os frames, com barra de progresso, e a análise de qualidade usa essa extração completa (a amostragem de 1 a cada 3 frames foi removida). Os arquivos só são escritos quando o take é aceito: o `.mp4` com o FPS medido (não o declarado), em arquivo temporário, conferência do número de frames e renomeação final, nunca sobrescrevendo; depois o `.npz` de landmarks (D12) a partir dos mesmos frames e instantes. Se a escrita do `.npz` falhar, o vídeo é desfeito, para não sobrar take sem landmarks. Take descartado não deixa arquivo.
- Codec `mp4v`: o único que o OpenCV do Windows grava de forma confiável sem dependências externas. A compressão com perdas é aceitável porque o que se usa são landmarks, e os landmarks da gravação vêm dos frames sem compressão (ver D12).
- Gravar além da meta é permitido; a meta só orienta (contagem n/meta por palavra para o sinalizador atual).
- Takes aceitos na revisão ficam `ok` (com os avisos registrados em `flags_qualidade`). Takes do modo rajada com aviso ficam `pendente` para revisão humana.
- Rajada com extração no fim (pedido da equipe, 2026-10-14): na rajada, os takes ficam em memória (frames brutos e instantes) e só passam pelo MediaPipe depois do último take ou de "Parar rajada"; antes, cada take era extraído entre um e outro, e a rajada esperava a extração. No fim, todos são extraídos em segundo plano (um grafo novo por take, como antes, para o rastreamento não vazar entre clipes) e salvos um por vez na thread da interface (`.mp4` e `.npz` juntos, mesmas regras de sempre), liberando a memória de cada take ao salvar. Os landmarks continuam vindo dos frames sem compressão. "Parar rajada" deixa terminar o take que está sendo gravado e processa o que já foi gravado. Ao sair da aba no meio da rajada, os takes completos são processados e o take incompleto é perdido; ao fechar o aplicativo, a confirmação avisa quantos takes ainda não foram salvos. Um take com erro não interrompe os outros: as falhas aparecem juntas no fim.
- Limite de memória da rajada: `coleta.gravacao.memoria_max_rajada_mb` (padrão 1500 MB, pensando no alvo de 4 GB de RAM). A estimativa por take é largura x altura x 3 bytes x FPS x duração (640x480 a 30 FPS por 3 s: cerca de 80 MB, ou seja, até 18 takes). Se o número de takes pedido não couber, a rajada não começa e a tela diz quantos cabem.
- Revisão rápida da rajada (pedido da equipe, 2026-10-14): antes, take de rajada sem aviso de qualidade ia direto para `ok`, sem ninguém assistir, e os avisos só medem FPS, brilho e detecção; não percebem sinal errado ou take começado tarde. Agora, depois da extração, cada take passa na tela em loop com o esqueleto: Enter mantém e vai ao próximo, Esc descarta; a lista mostra quadros, detecção de mão e avisos de cada um, com "Marcar todos" e "Desmarcar todos". Só os marcados são salvos (`ok`, ou `pendente` se tiver aviso); take com erro não pode ser marcado. Ao sair da aba durante a revisão, ela continua ao voltar; ao fechar o aplicativo, a confirmação avisa quantos takes não foram salvos.
- O máximo do campo "Takes na rajada" acompanha o que cabe em `memoria_max_rajada_mb` com a resolução real da câmera (calculado ao ligar a câmera); a mensagem de erro continua como segunda barreira.
- Durante a rajada, a legenda da câmera mostra o progresso do sinalizador na palavra ("S03/prova: 7 de 10"), contando os já salvos e os gravados nesta rajada.
- "Parar rajada" deixa o take em gravação terminar e o leva à revisão; "Parar e descartar o take atual" (Esc durante a gravação) joga fora o take em andamento na hora.
- Revisão em lote (2026-10-14): na aba Revisão, cada take tem uma caixa de marcação; "Marcar pendentes", "Marcar todos" (da lista filtrada) e "Desmarcar"; "Aprovar marcados" e "Rejeitar marcados" pedem confirmação e gravam o status de todos de uma vez, com a observação "revisado em lote".
- Revisão com editor de corte (item 9, substitui os dois controles deslizantes do item 5). No estilo de editores de vídeo, com uma única função: cortar. Prévia grande com esqueleto (liga e desliga); barra de controles (início, quadro anterior, reproduzir/pausar, próximo quadro, fim, tempo atual e duração em mm:ss.cc, número do quadro); linha do tempo com régua, faixa de miniaturas, cabeça de reprodução arrastável, trecho mantido em destaque com alças nas duas pontas (fora dele, escurecido) e uma curva fina de movimento das mãos calculada do `.npz` para achar onde o sinal começa e termina. Ações: cortar início e fim (alças ou "definir início/fim aqui"), dividir em dois takes no playhead, desfazer e refazer (um arrasto inteiro é um passo), salvar e cancelar. Atalhos: Espaço, setas (quadro a quadro), Shift+setas (1 s), Home/End, I e O, S, Ctrl+Z e Ctrl+Y.
- Regras mantidas: mínimo de 30 quadros por trecho (a alça para no limite e a tela diz por quê), original para `data/lixeira/cortes/<video_id>_<data>/`, `.mp4` e `.npz` novos com os instantes reindexados a partir de 0, reaproveitando os landmarks já extraídos, e manifesto registrando o corte. Nada é gravado até "Salvar corte"; corte e divisão podem ser salvos juntos (`coleta.cortes.aplicar_edicao`). Não existe remoção de trecho interno: juntar as duas pontas criaria um salto que o modelo leria como movimento rápido.
- Desempenho do editor: o vídeo não é carregado inteiro na memória. Os quadros são decodificados sob demanda numa thread (só o pedido mais recente; leitura sequencial na reprodução, busca nos saltos, cache de 48 quadros); a reprodução segue o FPS real do take pelo tempo decorrido; as miniaturas são geradas em segundo plano, em baixa resolução, e ficam em cache só na memória (não em disco, porque mostram o rosto das pessoas). O estado do corte (`coleta.edicao_corte.SessaoCorte`) não depende da interface e é testado nos limites. O arquivo de vídeo é fechado antes de salvar ou excluir (no Windows, arquivo aberto não vai para a lixeira).
- A curva de movimento usa a mesma medida de `energia_maos` (deslocamento médio dos pontos da mão, em larguras de ombro), por quadro (`normalizacao.movimento_maos_por_frame`). Serve só para exibição.
- Importação: copia o vídeo para `data/raw_videos/` e gera o `.npz` na hora (pode ser desligado com `--sem-landmarks`; nesse caso a extração do Dataset completa depois).
- Arquivos temporários órfãos (`*.parcial.*`, `*.tmp`) com mais de 10 minutos são movidos para `data/lixeira/orfaos/` ao abrir o aplicativo. O limite de 10 minutos evita mexer em arquivo que outro processo ainda esteja escrevendo.
- Augmentation de vídeo (D8, família D) e robustez com vídeo escuro (D14) não mudaram: continuam partindo do `.mp4` guardado.

## D10. Manifesto

CSV em `data/manifest.csv`, escrito de forma atômica. Motivos para não usar SQLite: volume pequeno, a equipe consegue abrir e conferir no Excel e o histórico fica legível.

- Colunas acrescentadas: `sha256_npz` (hash do arquivo de landmarks), `frames_originais` e `corte` (trecho mantido, quando o take foi cortado ou dividido). Manifestos antigos sem essas colunas continuam sendo lidos.
- Leitura tolerante (correção do item 1): aceita UTF-8 com ou sem BOM e cp1252, separador `,` ou `;` (o Excel em português salva assim), linhas em branco e colunas faltando. Linha que não dá para usar (sem `video_id`, classe, sinalizador, arquivo ou status) é ignorada, com aviso que aponta o número da linha e o motivo; o aplicativo abre normalmente e mostra o aviso na tela inicial. Antes de regravar uma tabela que foi lida com problema, é feita uma cópia em `data/backup/`. O cadastro de sinalizadores usa a mesma leitura.
- Toda alteração relê o arquivo antes de escrever, para não perder o que outro processo gravou.
- Trava de escrita (2026-10-14): reler, alterar e salvar acontece sob uma trava (`Manifesto.trava()`): `RLock` para as threads do mesmo processo e arquivo `manifest.csv.lock`, criado de forma exclusiva, para outros processos (scripts). A trava cobre também a escolha do número do take até o registro (gravação, importação e cortes), para duas escritas simultâneas não pegarem o mesmo número. Um `.lock` com mais de 120 s é tratado como abandonado; um recente faz a escrita esperar até 15 s e então falhar com mensagem clara. Mudança de status em lote (`definir_status_lote`) faz uma única leitura e escrita.
- Integridade (`scripts/verificar_manifesto.py` e a tela Início): registros sem arquivo, arquivos sem registro, sinalizadores sem TCLE, duplicatas por hash, classes fora do vocabulário, landmarks órfãos, landmarks ausentes e landmarks alterados (hash do `.npz` diferente do registrado).
- Exclusões: excluir um take move vídeo e landmarks para `data/lixeira/`. A exclusão definitiva só existe para sinalizadores, no fluxo de privacidade de D11; é a única situação em que a ferramenta apaga arquivos.

## D11. Privacidade

- Sinalizadores identificados apenas por ID (S01, S02...). A ligação ID-pessoa existe só no TCLE físico, guardado pela equipe.
- Sem TCLE confirmado, a ferramenta não grava nem importa para o sinalizador.
- Todo `data/` e `runs/` fica fora do controle de versão (`.gitignore`). O modelo final usa apenas landmarks.
- O manifesto guarda apenas o nome do arquivo importado (não o caminho completo, que pode conter nomes de pessoas). Recomenda-se renomear arquivos que contenham nomes antes de importar.
- Retirada de consentimento ou erro de cadastro: remoção de sinalizador em duas etapas (aba Sinalizadores ou `scripts/sinalizadores.py`).
  1. **Remover**: a pessoa confirma digitando o ID. Vídeos e landmarks do sinalizador vão para `data/lixeira/<ID>_<data>/`; as linhas saem do manifesto e do cadastro (e as metas individuais do vocabulário). Reversível à mão.
  2. **Excluir definitivamente**: só para quem já foi removido; exige digitar o ID e a palavra EXCLUIR. Apaga de fato o conteúdo da lixeira daquele sinalizador. Não é reversível.
- Ambas as etapas invalidam os splits existentes (movidos para `data/splits/invalidos/<data>/`, com aviso), porque o conjunto de vídeos mudou. Execuções antigas em `runs/` não são apagadas; se precisarem sair por causa do consentimento, a equipe as remove à mão.
- Cada remoção e exclusão é registrada em `data/auditoria.log` com ação, ID anônimo ou classe, quantidades e data, sem nomes de pessoas.
- Logs do aplicativo (`runs/logs/app.log`, rotação de 1 MB com 5 arquivos, e `runs/logs/falhas_nativas.log` para falhas fora do Python) também usam só IDs.

## D12. Extração de landmarks e dataset

- `scripts/extrair.py` processa só vídeos novos ou alterados. Cada vídeo gera `data/landmarks/<classe>/<video_id>.npz` com coordenadas cruas, presença, instantes dos frames, largura, altura e metadados. O cache é invalidado se mudar o hash do vídeo, a versão do MediaPipe, a seção `mediapipe` do config ou a versão do formato. Mudar features, normalização ou janelas não exige reextrair.
- Paralelismo opcional (`--processos 2`), com o modo "spawn" em todos os sistemas para se comportar igual ao Windows.
- Origem dos `.npz`: gravação (gerado na hora a partir dos frames em memória, com os instantes reais de chegada), importação (gerado na cópia) ou extração do Dataset (vídeos antigos, ou importados com `--sem-landmarks`). O campo `origem_landmarks` registra qual. Os da gravação vêm dos frames sem a compressão do `mp4v`, então uma reextração a partir do `.mp4` daria valores ligeiramente diferentes; por isso a extração não sobrescreve um `.npz` válido, e o hash do `.npz` fica no manifesto.
- Cortes fatiam o `.npz` existente e reindexam os instantes a partir de 0 (D9).
- Excluir um vídeo remove também os landmarks; a verificação de integridade aponta `.npz` órfãos, ausentes ou alterados.
- As janelas de todos os vídeos elegíveis são calculadas uma vez e guardadas em float16 (a normalização é por janela, então não depende da partição). Com o vetor padrão, cada janela ocupa cerca de 33 KB; há um limite configurável (`dataset.limite_memoria_mb`, 1200 MB) com erro claro se for ultrapassado. O tf.data lê por gerador, em lotes, sem copiar o conjunto para dentro do grafo, e embaralha de forma reprodutível (seed + época). O gancho `transformar` recebe a augmentation na Fase 4, aplicada só ao treino.

## D13. Treino, avaliação e registro das execuções

- Seleção de classes: o treino não exige mais todas as classes do vocabulário. A pessoa escolhe as classes (aba Treino ou `--classes biblioteca,prova`); uma tabela mostra, por classe, vídeos `ok`, sinalizadores e janelas elegíveis. Mínimos configuráveis em `treino.minimos` (padrão: 2 sinalizadores por classe e 2 classes). Classe abaixo do mínimo fica bloqueada com a explicação; com 1 sinalizador não há como testar a classe em pessoa nova.
- "Treinar mesmo assim" (`--ensaio`) libera classes abaixo do mínimo só como ensaio: como nos ensaios reduzidos de D14, todas as saídas (pasta, `ambiente.json`, `LEIA-ME`, `metadados.json`, relatórios) ficam marcadas como ENSAIO, que não pode virar número do TCC. O ensaio não é salvo como padrão; vale só para a execução pedida.
- O split usa só as classes selecionadas, e a assinatura do split inclui o conjunto de classes; treinar com outras classes exige gerar splits de novo (o treino recusa, com mensagem). O `metadados.json` do modelo final registra as classes usadas, e o tempo real usa essa lista.
- A seleção feita na interface fica em `configs/local.yaml` (D16), não no `config.yaml`, e é gravada a cada mudança, para valer também na aba Dataset e na linha de comando.
- Correção do item 6. Causa encontrada: sem seleção salva, o padrão de `classes_treino` era o vocabulário inteiro. Com só "prova" tendo vídeos, a tabela abria com as quatro classes marcadas e a validação cobrava "sem vídeo utilizável" das três outras; a seleção da aba Treino só era gravada ao iniciar um treino válido, então gerar split na aba Dataset (que não tem caixas de classe) também usava o vocabulário inteiro. A validação em si já considerava só as classes marcadas. Agora:
  - classe sem vídeo `ok` com landmarks aparece desabilitada e desmarcada ("sem vídeos" ou "sem landmarks") e nunca entra na validação;
  - toda classe pode ser desmarcada a qualquer momento; o mínimo de sinalizadores bloqueia marcar (sem ENSAIO) e treinar, nunca desmarcar;
  - botões "Marcar todas as elegíveis" e "Limpar seleção"; a mensagem diz o que falta e como resolver (ex.: "Hoje só 'prova' tem vídeos; grave 'nenhum_sinal' ou outra palavra") e a tela explica por que o mínimo é 2 classes (com uma, a saída é sempre 100%); `nenhum_sinal` conta como classe;
  - sem seleção salva, o padrão são as classes com vídeos que atingem o mínimo de sinalizadores (com ENSAIO, todas as que têm vídeos; se nenhuma atinge o mínimo, todas as que têm vídeos, para a validação explicar o motivo);
  - seleção salva que cite classe inexistente, arquivada ou sem vídeos é limpa em silêncio; classes pedidas explicitamente (`--classes`, botão Iniciar) continuam estritas: classe inexistente é erro.
- O modelo final pode ser treinado pela aba Treino (modo "Modelo final com todos os sinalizadores", subprocesso com progresso; conferido por teste de integração). O `metadados.json` passou a registrar `criado_em` e `protocolo`.
- `scripts/treinar.py` treina e avalia em todas as partições do protocolo, para cada seed do config. Cada execução vai para `runs/<data>_<nome>/` com a configuração efetiva, versões, hardware, seeds, cópia do split, métricas por partição, curvas de treino e validação, matrizes de confusão (só teste), predições por janela e por vídeo e o modelo de cada partição.
- Métricas em dois níveis: por janela de 1 s e por vídeo (média das probabilidades das janelas do vídeo). A matriz de confusão é conferida: a soma tem de bater com o número de amostras de teste.
- Reprodutibilidade: seed fixada em Python, NumPy e TensorFlow, com determinismo de operações habilitado.
- Latência registrada aqui é só a do modelo (uma janela, `predict_on_batch`), medida na máquina do treino. A latência de ponta a ponta (câmera, MediaPipe, modelo, tela) é medida na Fase 6. Observação de implementação: chamar o modelo diretamente em modo eager foi cerca de 30 vezes mais lento na CPU (cerca de 145 ms contra cerca de 4,5 ms no ambiente de desenvolvimento), por isso a inferência usa `predict_on_batch`.
- `--final` treina um modelo com todos os sinalizadores (validação por sinalizadores para early stopping) e salva `modelo.keras` com `metadados.json` (classes, vetor de features, janela, normalização e parâmetros do MediaPipe) para o tempo real. Esse modelo não tem avaliação independente; os números do TCC vêm das execuções por partição.
- Acompanhamento ao vivo: com `--progresso arquivo.jsonl`, o treino escreve um evento JSON por linha: início, início de partição, cada época (só perda e acurácia de treino e de validação), fim de partição (aqui, e só aqui, entram as métricas de teste e a matriz de confusão) e fim. O teste nunca aparece na curva por época: mostrar o teste durante o treino convida a escolher o modelo por ele. A interface roda o treino em subprocesso e lê esse arquivo.
- Teste de integração feito: com vídeos montados a partir de uma única foto (sem diferença real entre classes), o treino completo com todas as famílias de augmentation deu acurácia de 25% com 4 classes, o nível do acaso. É o esperado e indica que o encanamento não vaza rótulos. Não é um resultado do TCC.

## D14. Ablação, estatística e robustez

- `scripts/ablacao.py` roda, com o mesmo split e as mesmas seeds (padrão: 42, 7, 2024): baseline; cada família isolada (espacial, temporal, ruído e vídeo, esta só se as variantes estiverem extraídas para todos os vídeos); a combinação; as intensidades fraca e forte da melhor condição; e o baseline no protocolo secundário.
- A "melhor condição" para a etapa de intensidades é escolhida pela métrica de **validação**, nunca pelo teste. Escolher pelo teste e depois reportar o teste inflaria o resultado.
- Métrica principal padrão: F1 macro por vídeo (configurável). Acurácia e F1 por janela também são reportadas.
- Teste estatístico: as partições compartilham dados de treino, então os resultados por partição não são independentes e o teste t comum subestima a variância. Usamos o teste t corrigido para validação cruzada repetida (NADEAU; BENGIO, 2003), na forma de Bouckaert e Frank (2004), com pares (seed, partição), mais correção de Holm (1979) para as várias comparações com o baseline. O Wilcoxon pareado sobre as médias por partição entra só como conferência: com 5 partições o menor p possível é 0,0625, então ele nunca indica significância nesse desenho, e o relatório diz isso. Referências a conferir antes de entrar no texto final: Nadeau e Bengio (2003, Machine Learning, v. 52); Bouckaert e Frank (2004, PAKDD); Holm (1979, Scandinavian Journal of Statistics).
- Linguagem do relatório: só se fala em "ganho" quando o p ajustado por Holm é menor que alfa (0,05) e a diferença é positiva. Caso contrário: "sem evidência de diferença".
- Robustez: cada modelo de cada partição é avaliado também em cópias degradadas do teste: oclusão de uma mão por 10 frames, ruído (σ = 0,03), queda de FPS (metade dos frames), ausência de face e, se extraída, a variante de vídeo escuro. A degradação é determinística (mesma cópia degradada para todas as condições). O modelo nunca treina nessas cópias.
- Saídas prontas para o TCC: tabelas em CSV e em LaTeX compatível com abnTeX2 (vírgula decimal, `\fonte{Elaborado pelos autores (2026).}`, `\nota{}`), figuras PNG em 200 dpi e um RELATORIO.md. A ablação pode ser retomada se for interrompida, e recusa misturar execuções com planos diferentes na mesma pasta. Ensaios reduzidos (menos seeds, partições ou épocas) são marcados como tal em todas as saídas.
- Custo estimado: cerca de 8 condições × 3 seeds × 5 partições = 120 treinos. O tempo no i5 ainda não foi medido.

## D15. Inferência em tempo real

- O tempo real é a aba Tempo real da interface única (`scripts/tempo_real.py` abre o aplicativo nessa aba; `--video` continua rodando sem janela). O processamento roda num subprocesso (`libras.tempo_real_servico`), iniciado só quando a aba é aberta e encerrado ao sair dela, para o TensorFlow não ocupar memória no resto do tempo. O subprocesso envia à interface, pela saída padrão, mensagens binárias (cabeçalho `LVQ1`, JSON com resultado e medições, imagem JPEG do frame) e recebe comandos simples pela entrada padrão (mostrar landmarks, limpar texto, salvar medição, sair).
- Dentro do subprocesso: uma thread captura, outra roda MediaPipe e modelo sobre o frame mais recente (frames que chegam enquanto o anterior é processado são descartados). A prévia é espelhada só na tela.
- Mesmo caminho do treino: buffer com os instantes reais de chegada dos frames, reamostragem para 30 FPS ancorada no frame mais recente, últimos 30 frames, `preparar_janela`, modelo. Um teste confirma que a entrada do modelo no tempo real é idêntica à janela correspondente do treino, e outro que FPS irregular (cerca de 20) continua gerando janelas de 1 s.
- Os parâmetros do MediaPipe, da janela, da normalização e do vetor de features vêm do `metadados.json` do modelo, não do config atual, para que o tempo real use exatamente o que o modelo viu no treino.
- Decisão por janela: sem mão em pelo menos 30% dos frames, o modelo nem roda ("nenhuma mão visível"); média móvel das últimas 5 predições; abaixo de 60% de confiança, nada é exibido; a classe `nenhum_sinal` nunca vira texto. Um sinal entra no texto após 3 predições iguais seguidas e só se repete depois de uma pausa.
- Sem modelo treinado, a interface explica o que falta e oferece "Treinar o modelo final" (leva à aba Treino com esse modo escolhido). As classes vêm do `metadados.json` do modelo. Não há predição simulada em nenhum ponto.
- Teste ao vivo do modelo (item 10), no mesmo subprocesso e no mesmo pipeline (nada de normalização, janelas ou decisão duplicadas; nenhum código de augmentation):
  - lista dos modelos finais em `runs/` (data, classes, protocolo e etiqueta ENSAIO), escolha do modelo e do índice da câmera na aba, "Atualizar lista" e o modelo ativo sempre visível;
  - painel de diagnóstico: barras de probabilidade de todas as classes (inclusive `nenhum_sinal`, que continua sem virar texto), com a marca do limiar; a etapa em que a decisão parou ("nenhuma mão visível", "confiança abaixo do limiar", "aguardando N predições iguais" com o contador, "sinal confirmado"); limiar, média móvel e confirmações ajustáveis na tela, só para o processo em execução (comando `parametros`; não alteram o config nem o modelo); esqueleto, FPS e latência;
  - modo "Testar acertos": a pessoa escolhe o sinal, faz e registra a tentativa (botão ou tecla T). Vale a classe de maior probabilidade média nas últimas predições, sem o limiar de exibição; sem mãos visíveis conta como `nenhum_sinal` (a tentativa de `nenhum_sinal` vale quando a pessoa fica sem sinalizar). Acertos e erros por classe, acurácia e matriz de confusão da sessão (esperado x previsto), zerar e salvar em `runs/tempo_real/teste_<data>.json` com modelo, classes, tentativas, matriz e hardware. Todas as saídas levam o texto "Teste qualitativo ao vivo; não substitui a avaliação por partições (D13) e não pode virar número do TCC." e, se for o caso, a etiqueta ENSAIO;
  - se quem testa está no cadastro e tem vídeos no treino do modelo, a tela avisa que o resultado tende a ser otimista (D6);
  - "Testar com arquivo de vídeo" usa o mesmo painel sobre um vídeo, para repetir o teste de forma reprodutível.
- Organização da aba (pedido após o item 12): a vista principal é só a câmera, com a palavra reconhecida como legenda desenhada sobre a parte de baixo do vídeo (texto confirmado em branco e negrito sobre fundo escuro translúcido; acima, menor, "Mostre as mãos para a câmera" ou "Em análise: <classe>"); quando o texto não cabe em duas linhas, as palavras mais antigas saem pela esquerda com reticências. A legenda é desenhada pela interface, não no JPEG do serviço, para não sair espelhada nem mudar o protocolo. O botão "Diagnóstico e teste" (tecla D) abre uma coluna (embaixo do vídeo em janela estreita) com três abas: "Decisão" (etapas, probabilidades, texto completo, limiar, média móvel, confirmações, esqueleto, FPS e tempos), "Testar acertos" e "Modelo" (modelo, câmera e arquivo de vídeo). Fora desse modo, a tecla T não registra tentativa. O nome do modelo ativo, a etiqueta ENSAIO, "Limpar legenda" e o aviso de que não substitui intérprete continuam na vista principal.
- Item 13 (D19), sem mudar a decisão: as constantes da decisão (limiar, média móvel, confirmações, mínimo de mãos de 30% e de pose de 50%) ficam num só módulo, `libras/decisao.py`, junto com os textos das etapas; inferência, serviço, normalização e app real importam de lá (mesmos valores; um teste falha se o guia de enquadramento repetir os números). O serviço ganhou campos OPCIONAIS no quadro (quem não os conhece os ignora; o protocolo `LVQ1` é o mesmo): `quadros` (capturados, processados e descartados desde que a câmera ligou, com capturados = processados + descartados), `tempos_recentes` (mediana e p95 por etapa nos últimos 10 s) e `enquadramento` (pessoa, ombros, lados com mão cortada, brilho e caixas da cabeça e dos ombros, em coordenadas da imagem não espelhada; `libras/enquadramento.py`), e uma mensagem `info` com índice, resolução e backend da câmera. Modo do app real: `--pacote <pasta>` confere o hash do pacote antes de carregar o modelo e usa os parâmetros de decisão do cartão; `--sem-disco` não grava log em arquivo nem medição (o comando `salvar` é ignorado). Custo das medidas novas, medido no serviço real com MediaPipe sobre o mesmo vídeo: cerca de 0,15 ms por quadro; latência de ponta a ponta igual com e sem elas (mediana de 15 a 18 ms nas quatro rodadas alternadas, vídeo sintético sem pessoa). Comparação com a versão anterior ao item 13 (commit b6345ab), mesmo vídeo e mesmo modelo, duas rodadas alternadas de 20 s: ponta a ponta 15,3 e 15,5 ms antes, 14,3 e 15,0 ms depois; cerca de 29,5 quadros por segundo nas duas (não piorou). A câmera só começa a ser lida depois que o MediaPipe está pronto, para os quadros da carga inicial não contarem como pulados.
- Sair da aba pede ao subprocesso para terminar e não espera por ele (item 7); voltar à aba aguarda o processo anterior soltar a câmera antes de abrir outro.
- Medição: por frame, espera entre chegada e início do processamento, MediaPipe, preparo, modelo e ponta a ponta (chegada do frame da câmera até a exibição do resultado), mais memória do processo. Mediana e p95 são salvas com o hardware em `runs/tempo_real/medicao_<data>.json` ao fechar ou pelo botão "Salvar medição". `--video arquivo.mp4` roda o mesmo pipeline sobre um vídeo gravado, para medições reprodutíveis.
- Medição no ambiente de desenvolvimento (Xeon 2,8 GHz, 2 núcleos, sem GPU; **não é o hardware alvo**), com câmera simulada: MediaPipe cerca de 56 ms por frame (mediana), modelo cerca de 5 ms, cerca de 14 frames processados por segundo, cerca de 1,1 GB de memória (TensorFlow + MediaPipe no mesmo processo). Medição repetida com o subprocesso da interface única: cerca de 60 ms por frame no MediaPipe, cerca de 5 ms no modelo, 13 a 15 frames processados por segundo, cerca de 1,0 GB no subprocesso. A latência e a memória no i5 com 4 GB precisam ser medidas pela equipe; se a memória apertar, a alternativa a avaliar é converter o modelo para TensorFlow Lite.

## D16. Interface única

- Um único aplicativo desktop substitui os `.bat` de cada etapa: `LibrasVision.bat` (ou `python -m libras.app`). Abas: Início (resumo e alertas de integridade), Sinalizadores, Vocabulário, Gravar (com importação), Revisão, Dataset (extração, validação, splits), Treino, Resultados (execuções, ablação, robustez), Tempo real e Configurações. Os scripts de linha de comando continuam existindo e chamam o mesmo código; só `instalar.bat` e `LibrasVision.bat` ficaram como `.bat`.
- PySide6 em vez de CustomTkinter. Os dois são compatíveis com as versões de D1 (o CustomTkinter é Python puro sobre o Tkinter). Escolhemos o PySide6 porque: (1) o visual pedido (fundo #070F1C, painéis #0D1830/#111F3D, gradiente #3B82F6 para #7C3AED nos botões primários e no item ativo da barra lateral, cantos arredondados, ícones vetoriais) sai direto de uma folha de estilo QSS e de SVG, enquanto o CustomTkinter não tem gradiente nem SVG nativos; (2) `QProcess` e sinais entre threads resolvem subprocessos de treino e tempo real sem travar a tela; (3) o matplotlib tem backend Qt (`QtAgg`) para a curva de treino ao vivo; (4) tabelas grandes (manifesto, revisão) ficam leves com `QTableWidget`. Custos: cerca de 78 MB de download (mais espaço instalado) e cerca de 40 MB de RAM a mais que o Tkinter (D1), e licença LGPL (uso sem modificar a biblioteca, compatível com um TCC).
- Tema centralizado em `src/libras/app/tema.py` (cores, fontes, folha de estilo). Ícones em SVG próprios (`src/libras/app/icones.py`). Sem emojis na interface, no código e nos logs.
- Nenhuma tarefa longa roda na thread da interface: gravação/extração/cortes/importação rodam em threads com barra de progresso e cancelamento; treino, ablação e tempo real rodam em subprocessos (o processo da interface nunca importa o TensorFlow). O subprocesso usa o `python.exe` ao lado do `pythonw.exe`, sem abrir console.
- Aba Treino: escolha de classes (tabela de D13), protocolo, seeds, épocas e augmentation; iniciar e parar. Curva ao vivo em aba própria, com perda e acurácia de treino e validação por época, seletor de partição e seed e marca na melhor época (menor perda de validação). As métricas de teste e a matriz de confusão aparecem só ao fim de cada partição.
- Configurações: a aba grava só os ajustes alterados em `configs/local.yaml`, que é mesclado sobre o `config.yaml` ao carregar e fica fora do controle de versão. O `config.yaml` continua sendo a referência documentada.
- Troca de abas (item 7). Medição antes da correção, no ambiente de desenvolvimento: as abas em si trocavam em 30 a 50 ms; a lentidão vinha da aba anterior. Sair do Tempo real (que fica logo acima de Configurações na barra) esperava o subprocesso terminar, cerca de 1,5 s; sair do Gravar com a câmera ligada esperava o MediaPipe da prévia e a câmera fecharem, cerca de 3 s; e a primeira pintura do gráfico vazio do matplotlib custava cerca de 200 ms na aba Treino. Correção: esses encerramentos rodam em segundo plano (esperados só ao fechar o aplicativo), os gráficos só são criados quando há o que desenhar e as abas são criadas uma vez, no tempo ocioso logo após abrir. Depois: as trocas medidas ficaram entre 25 e 60 ms nesta máquina, inclusive nos dois casos lentos (cerca de 30 ms); só a primeira visita a uma aba antes de o pré-carregamento terminar (os primeiros segundos após abrir) pode passar de 100 ms. Cada troca é registrada em `runs/logs/app.log` (sair, criar, entrar e até pintar), com aviso acima de 100 ms, para conferir no i5. Um teste offscreen mede a troca para cada aba e falha acima de 200 ms.
- Interface dinâmica (item 8):
  - layout responsivo: colunas com divisor arrastável que passam a ficar uma embaixo da outra em largura estreita; conteúdo das abas em área com rolagem (nada é cortado); tabelas que esticam; janela mínima de 860x540, para caber em 1366x768 com escala de 150%; conferido por capturas com escala de 100%, 125% e 150%;
  - barra lateral que recolhe para só ícones abaixo de 1180 px de largura (ou por Ctrl+B / botão, e então a escolha fica gravada); tamanho e posição da janela lembrados em `configs/interface.ini` (fora do Git);
  - estados de hover, pressionado, foco e desabilitado em botões, campos, caixas de seleção, abas, controles deslizantes e tabelas (no tema);
  - indicador de atividade (barra fina com texto) para qualquer tarefa em segundo plano que passe de 200 ms;
  - avisos simples como notificações que somem sozinhas; diálogos só para confirmações destrutivas e erros;
  - dados ao vivo: o aplicativo confere a cada segundo se manifesto, cadastro, vocabulário, splits ou `runs/` mudaram (data e tamanho dos arquivos; inclusive mudanças feitas pela linha de comando ou pelo treino) e a aba aberta se atualiza sozinha, preservando a seleção nas tabelas (contagens n/meta do Gravar, Início, listas). Polling em vez de vigia de arquivos porque a escrita atômica troca o arquivo e o vigia perde o arquivo no Windows;
  - transição de 150 ms (esmaecimento) na troca de abas, incluída na medição do item 7;
  - atalhos: Ctrl+1 a Ctrl+0 (abas), Ctrl+Tab e Ctrl+Shift+Tab, Ctrl+B, F5 (atualizar), F1 (lista de atalhos), C no Gravar (câmera), atalhos do editor de corte (D9) D no Tempo real (abre e fecha "Diagnóstico e teste") e T no teste ao vivo (D15).
- Robustez da interface (item 1): log em arquivo com rotação, tratador global de exceções (Python e threads) que mostra uma janela com a mensagem e o caminho do log, `faulthandler` para falhas nativas, e encerramento ordenado ao fechar (prévia parada, câmera liberada, subprocessos encerrados).

- Dois aplicativos (item 13, D19): este aplicativo virou a BANCADA, "LibrasVision Lab" na interface, aberta por `LibrasVisionLab.bat` (o antigo `LibrasVision.bat` foi renomeado) ou `python -m libras.app`, sem perder nenhuma função. O nome `LibrasVision.bat` passou a abrir o app real (`python -m libras.app_real`). Componentes compartilhados foram MOVIDOS (não copiados) para `libras/ui/` (tema, tokens, ícones, componentes, legenda, gerenciador de sobreposições); `app/base.py` mantém a classe `Pagina` e reexporta os componentes, então as páginas da bancada não mudaram. Resultados ganhou "Publicar modelo" e "Exportar pacote (.zip)" (13.2); Início mostra o modelo publicado; Tempo real usa a mesma legenda do app real e tem "Pré-visualizar como no app real" (13.5).
- Redesenho visual (item 12, terceira rodada): a interface passou a seguir o sistema de design de D18 (tokens, componentes, três faixas de largura, acessibilidade). A lógica (normalização, janelas, splits, treino, tempo real) não mudou; a suíte de testes continua passando sem alteração de comportamento. Diálogos agora são próprios (sem `QMessageBox`, que toca o som do sistema no Windows). Memória da interface, medida no ambiente de desenvolvimento com todas as abas pré-carregadas: cerca de 153 MB ao abrir e 163 MB depois de visitar Treino, Resultados e Revisão, contra 146 MB e 158 MB antes do redesenho (cerca de +6 MB). Em relação ao valor de D1 (cerca de 100 MB ao abrir), a diferença vem quase toda do pré-carregamento das abas (item 7), não do redesenho.

## D17. Vocabulário configurável

- Fonte única das classes: `data/vocabulario.json`. Cada classe tem `slug` (identificador sem acento, em minúsculas, usado em pastas e no manifesto), `rótulo` (texto exibido), `tipo` (`sinal` ou `nenhum_sinal`), meta por sinalizador, metas individuais opcionais por sinalizador e situação (ativa ou arquivada).
- Na primeira execução o arquivo é criado com os presets do `config.yaml` (biblioteca, prova, secretaria, nenhum_sinal), que podem ser editados ou removidos. Depois disso, a seção `classes` do `config.yaml` só serve para criar o vocabulário.
- Na aba Gravar, a palavra é escolhida numa lista com a contagem n/meta do sinalizador atual, ou criada no campo "nova palavra" (o slug é normalizado sem acento; duplicatas são recusadas).
- Arquivar esconde a classe da gravação e do treino sem apagar dados. Remover uma classe só é direto se ela não tiver takes; com dados, os takes vão para `data/lixeira/classe_<slug>_<data>/` com confirmação digitada, registro em auditoria e invalidação dos splits, como na remoção de sinalizador (D11).

## Decisões tomadas pela implementação sem consulta à equipe (revisão de 2026-10-06)

Para a equipe conferir e, se discordar, pedir mudança:

1. PySide6-Essentials em vez de CustomTkinter (D16).
2. Na gravação, os arquivos são escritos quando o take é aceito, não quando a gravação para; a extração e a análise de qualidade rodam antes da revisão do take (D9).
3. Cortes reaproveitam os landmarks já extraídos em vez de rodar o MediaPipe de novo (D9, D12).
4. O `.npz` da gravação vem dos frames sem compressão e dos instantes reais; a extração do Dataset não o sobrescreve (D12).
5. A seleção de classes da interface fica em `configs/local.yaml`; o modo ensaio não é persistido (D13).
6. A aba Configurações grava em `configs/local.yaml`, não no `config.yaml` (D16).
7. A importação foi para uma janela dentro da aba Gravar, não para uma aba própria.
8. Tempo real em subprocesso com protocolo binário próprio pela saída padrão (D15).
9. Variável `LIBRAS_CAMERA_VIDEO` para usar um vídeo como câmera (D9).
10. Limpeza de temporários órfãos só para arquivos com mais de 10 minutos (D9).
11. Mínimos padrão: 2 sinalizadores por classe e 2 classes (D13).
12. Os aplicativos antigos em Tkinter e os `.bat` de cada etapa foram removidos (D16).
13. A exclusão definitiva exige remoção prévia e a palavra EXCLUIR além do ID (D11).
14. Classe removida com dados vai para `data/lixeira/classe_<slug>_<data>/` (D17).
15. Splits invalidados são movidos para `data/splits/invalidos/<data>/`, não apagados (D11).
16. Execuções antigas em `runs/` não são apagadas pela remoção de sinalizador (D11).

## Decisões tomadas sem consulta à equipe (segunda rodada, itens 6 a 10)

1. Sem seleção salva, o padrão do treino são as classes com vídeos que atingem o mínimo de sinalizadores; com ENSAIO, todas as que têm vídeos (D13).
2. A seleção de classes é gravada a cada mudança na aba Treino (não só ao iniciar), para a aba Dataset e a linha de comando usarem a mesma (D13).
3. Seleção salva obsoleta é limpa em silêncio, mas `--classes` com classe inexistente continua sendo erro (D13).
4. Marcar uma classe abaixo do mínimo exige marcar ENSAIO antes; desmarcar é sempre livre (D13).
5. A linha inteira da tabela de classes marca e desmarca (não só a caixinha), para facilitar o clique em escala alta.
6. Encerramentos de câmera, MediaPipe e subprocesso em segundo plano na troca de aba, esperados só ao fechar o aplicativo (D16).
7. Abas criadas no tempo ocioso logo após abrir o aplicativo (D16).
8. Gráficos do matplotlib criados só quando há dados (D16).
9. Dados ao vivo por consulta a cada 1 s, em vez de vigia de arquivos (D16).
10. Barra lateral recolhe abaixo de 1180 px; janela mínima de 860x540 (D16).
11. Geometria da janela em `configs/interface.ini`, não no registro do Windows (D16).
12. Os botões da barra lateral não recebem foco por Tab; a navegação por teclado entre abas é por Ctrl+1 a Ctrl+0 e Ctrl+Tab (D16).
13. Miniaturas do editor de corte só em memória (até 12 vídeos), nunca em disco, por mostrarem rostos (D9).
14. Curva de movimento com a mesma medida de `energia_maos`, por quadro, suavizada em 5 quadros; só exibição (D9).
15. Corte e divisão podem ser salvos juntos numa operação só (D9).
16. Na tentativa do teste ao vivo vale a média móvel das predições (a mesma das barras), sem o limiar; sem mãos visíveis conta como `nenhum_sinal` quando o modelo tem essa classe; sem janela válida, a tentativa é registrada como "(sem predição)" e entra na matriz como coluna própria (D15).
17. O teste ao vivo pergunta "quem está testando" (ID do cadastro ou "outra pessoa") para decidir o aviso de sinalizador visto; o ID vai para o JSON salvo, nunca nome (D15).
18. A lista do teste ao vivo mostra só modelos finais (`modelo.keras` + `metadados.json` na raiz da execução); modelos de partição não aparecem (D15).

## D18. Sistema de design e preparo para mobile

Base da decisão: `docs/analise_interface.md` (padrões de apps de Libras, legendas ao vivo, ferramentas de ML e editores de vídeo, com fonte de cada afirmação).

**Tokens.** Tudo o que é visual vem de `src/libras/app/tokens.json`, lido por `tema.py`: cores, tipografia (corpo mínimo de 14 px; só eixos de gráfico e régua do editor usam 12 px), espaçamento em múltiplos de 4/8, raios, elevação por luminosidade (sem sombra), movimento (150 ms) e alvo de toque (44 px). Um teste falha se houver cor hexadecimal fora de `tokens.json` no código da interface. O desenho do esqueleto no vídeo (`landmarks.desenhar_esqueleto`) é da camada de lógica e continua com as cores próprias.

**Cores.**
- Mantidos: fundo #070F1C, painéis #0D1830 e #111F3D e o gradiente dos botões primários (#3B82F6 para #7C3AED).
- Rampa de 4 pontos amostrada da imagem de referência da curva (`docs/referencia_curva.png`): ciano #55CECA, azul #3B82F6 (o da marca; o trecho azul da imagem é dessaturado, #7273B6), violeta #9057C6 e magenta #B341B6 (amostras #8F55C5 e #A61FA9 clareadas o mínimo para 3:1 sobre todas as superfícies). Usada só em dados (curvas, barras de probabilidade) e em elementos de assinatura (faixa do cabeçalho do Início e da barra lateral, indicador de atividade). Nunca indica estado.
- Semânticas separadas: sucesso #22C55E, alerta #F59E0B, erro #F87171 (texto) e #EF4444 (borda), informação #A5B4FC.
- Nada depende só de cor: etiquetas têm texto e borda (tracejada em "pendente"), alertas têm etiqueta escrita, treino e validação no gráfico se distinguem por traço e marcador, a classe de maior probabilidade é marcada por contorno e pela palavra "maior", e as etapas da decisão no Tempo real têm forma e texto.

**Contraste (testado a partir dos tokens).** Texto a 4,5:1 e componentes e gráficos a 3:1 sobre fundo e superfícies (WCAG 2.1 AA). Ajustes que isso exigiu: texto terciário #8290AE (o antigo #5F7096 fica só para controles desabilitados, isentos pela WCAG); borda de controles #627194; seleção em tabela #2563EB. O branco sobre o azul do gradiente primário dá 3,68:1, abaixo de 4,5:1: o gradiente foi mantido e o rótulo dos botões primários passou a texto grande em negrito (19 px), para o qual o mínimo é 3:1; o item ativo da navegação usa o mesmo gradiente um tom mais escuro (#2563EB para #6D28D9), com texto em tamanho normal.

**Componentes padronizados** (`base.py`, `dinamica.py`, `galeria.py`): botão primário, secundário, fantasma e destrutivo; campo; tabela; cartão; etiqueta (ok, pendente, ENSAIO, erro, informação, captando); aviso não bloqueante; diálogo; estado vazio com o próximo passo; barra de progresso; dica sempre visível. A galeria (Ctrl+Shift+G, ou Configurações > Galeria de componentes) mostra todos em normal, passar o mouse, pressionado, foco e desabilitado.

**Telas (12.4).**
- Tempo real: legenda sobre o vídeo (até 40 px, proporcional à altura da imagem), com contraste testado sobre o pior caso de imagem (fundo translúcido `legenda_fundo` composto sobre branco e sobre preto, 4,5:1 nos dois); o sinal "em análise" aparece numa linha menor e mais discreta e o texto confirmado em cor plena (padrão de legenda ao vivo); no modo "Diagnóstico e teste", barras de probabilidade com uma cor da rampa por classe (só para distinguir; `nenhum_sinal` em cinza neutro); indicador das etapas da decisão (D15) em sequência, com forma e texto; etiqueta de captação ("mãos detectadas" ou "sem mãos visíveis"); esqueleto ligável; aviso de que o sistema não substitui intérprete.
- Treino e Resultados: curva no estilo da referência. Validação com um ponto por época, cor percorrendo a rampa ao longo das épocas e haste fina até a linha de base; treino em linha tracejada com quadrados vazados em cor neutra; melhor época com anel e rótulo. Sem suavização por padrão (a linha é o dado real); a opção "Suavizar (média móvel de 5 épocas)" fica visível e rotulada e, quando ligada, desenha a curva cinza da referência e as hastes passam a ligar cada ponto real à curva. O teste continua fora da curva por época (D13). Estilo do matplotlib vindo dos tokens, fundo igual ao painel.
- Gravar, Revisão e editor de corte: mesmos componentes; os botões do editor têm ícone e rótulo, e há botões de -1 s e +1 s (equivalentes por toque de Shift+setas).
- Início: faixa de assinatura, cartão "Próximo passo" (ex.: "Grave 'nenhum_sinal' para poder treinar"), cartões de resumo que se reorganizam em 4, 2 ou 1 coluna e alertas com etiqueta escrita.

**Três faixas de largura (12.5).**
- Compacta (até 600 px): barra de navegação inferior com 5 destinos (Início, Gravar, Tempo real, Resultados, Mais); "Mais" lista as outras abas em botões grandes.
- Média (600 a 1180 px): trilho de ícones com rótulo embaixo (o nome do destino não fica só na dica).
- Expandida: barra lateral completa.
- Alvos de toque de 44 px nas faixas compacta e média (testado); formulários põem o rótulo acima do campo e filas de botões quebram linha quando não cabem; nenhuma tela tem rolagem horizontal em 390, 900 ou 1366 px (testado).
- Todo atalho de teclado tem equivalente por toque ou clique (lista em F1 e no botão "Atalhos").
- Telas que fariam sentido no celular: Tempo real e Gravar. Dataset e Treino, e as abas de Ablação e Robustez de Resultados, ficam só no desktop; na faixa compacta mostram um aviso claro em vez de layout quebrado.

**Acessibilidade e público (12.6).** Foco visível em todos os controles; "Reduzir movimento" (desliga a transição entre abas e as animações dos avisos) e "Tamanho do texto" de 100% a 150% em Configurações, aplicados na hora e gravados em `configs/local.yaml` (seção `interface`). O público final é surdo: o aplicativo não emite som e nenhum retorno depende de som.

**Fachada para um cliente móvel.** Funções da camada de lógica que a interface chama (um cliente móvel reaproveitaria estas, não a interface):
- Cadastro e consentimento: `coleta.sinalizadores.CadastroSinalizadores`; remoção: `coleta.gestao.remover_sinalizador`, `excluir_definitivamente`, `remover_classe`.
- Vocabulário: `vocabulario.Vocabulario`.
- Gravação: `camera.abrir_camera_ou_video`, `Captura` (movidos de `coleta.camera` no item 13); `coleta.previa.TrabalhadorPrevia`; `coleta.qualidade.avaliar_clipe`; `coleta.gravacao.registrar_take` (vídeo + landmarks); importação: `coleta.importacao.importar`.
- Manifesto e integridade: `coleta.manifesto.Manifesto` (`verificar_integridade`, `definir_status`, `excluir`).
- Revisão e corte: `coleta.edicao_corte.SessaoCorte`, `coleta.leitor_video.LeitorVideo` e `miniaturas`, `coleta.cortes.aplicar_edicao`, `normalizacao.movimento_maos_por_frame`.
- Dataset: `extracao` (`extrair_lote`, `carregar_npz`, `cache_valido`), `validacao.gerar_relatorio`, `splits.gerar_splits`, `selecao` (`tabela_classes`, `avaliar_selecao`, `pode_marcar`).
- Treino e ablação: `scripts/treinar.py` e `scripts/ablacao.py` em subprocesso, com eventos JSON por linha (`--progresso`, D13).
- Tempo real: `inferencia` (`Reconhecedor`, `Transcritor`, `localizar_modelo`, `carregar_modelo`) e o serviço `tempo_real_servico` (protocolo binário de D15); teste ao vivo: `teste_ao_vivo` (`listar_modelos`, `SessaoTeste`, `etapa_decisao`, `aviso_sinalizador_visto`).

**Opções para mobile (estado em 06/10/2026, pesquisado; NÃO decidido).** As páginas foram lidas por uma ferramenta que resume; trechos podem não estar literais.
- PySide6 no Android: suportado desde o Qt for Python 6.5 com `pyside6-android-deploy` (host Linux ou macOS; ponto de entrada `main.py`; usa python-for-android) ([blog Qt](https://www.qt.io/blog/taking-qt-for-python-to-android); [doc 6.8](https://doc.qt.io/qtforpython-6.8/deployment/deployment-pyside6-android-deploy.html)). Não verificado: se TensorFlow ou LiteRT empacotam por esse caminho.
- PySide6 no iOS: anunciado para o Qt 6.12, ainda sem pacotes de terceiros com extensões em C (NumPy, Pillow) ([blog Qt, 30/07/2026](https://www.qt.io/blog/python-mobile-app-development-bringing-pyside6-on-ios)); a versão mais recente no PyPI era a 6.11.2 ([PyPI](https://pypi.org/project/PySide6/#history)).
- MediaPipe: a API legada (`mp.solutions`, usada em D1) teve o suporte encerrado em 01/03/2023 e foi removida das versões novas do pacote ([guia](https://developers.google.com/edge/mediapipe/solutions/guide); [issue](https://github.com/google-ai-edge/mediapipe/issues/6261)). O Holistic Landmarker da API Tasks tem guias para Android, iOS e Python ([página](https://developers.google.com/edge/mediapipe/solutions/vision/holistic_landmarker)); devolve 478 pontos de rosto. Não verificado: se pontos e coordenadas batem com o `mp.solutions.holistic` 0.10.21; se não baterem, o vetor de 550 features muda e seria preciso reextrair e retreinar.
- TensorFlow Lite, agora LiteRT: roda em Android e iOS ([LiteRT](https://developers.google.com/edge/litert)); converte LSTM do Keras sem estado para a operação `UnidirectionalSequenceLSTM` ([RNN](https://developers.google.com/edge/litert/conversion/tensorflow/rnn)); há relatos de falha na conversão de LSTM com Keras 3 (TF 2.16+), contornada com `tf_keras` ([issue](https://github.com/tensorflow/tensorflow/issues/68797)); o fallback de operações do TensorFlow aumenta o pacote ([ops_select](https://developers.google.com/edge/litert/conversion/tensorflow/ops_select)). A conversão do modelo deste projeto (TF 2.19) não foi testada.
- Outra via, apenas registrada: cliente nativo (Kotlin/Swift) com MediaPipe Tasks e LiteRT, reescrevendo o pré-processamento das features fora do Python.

**Capturas e problemas de layout que restam (12.7).** As capturas de cada aba em 1366, 900 e 390 px, nas escalas 100%, 125% e 150%, ficam em `runs/capturas/` (fora do Git; 99 imagens, incluindo a galeria). Nenhuma aba tem rolagem horizontal da página em nenhuma combinação (conferido pelo script de captura e por teste). Restam:
1. Mudar de faixa ao redimensionar (passar por 600 ou 1180 px) ou mudar o tamanho do texto leva cerca de 0,8 s nesta máquina, porque a folha de estilo é reaplicada a todos os controles. A troca de abas não é afetada (25 a 60 ms).
2. Em 390 px, as tabelas largas (Sinalizadores, Vocabulário, Revisão, Resultados) rolam na horizontal dentro da própria tabela; o resto da página não.
3. A galeria de componentes não se adapta a 390 px (ferramenta de conferência, não tela de uso).
4. Na aba Treino, a coluna "Situação" da tabela de classes fica parcialmente cortada quando a coluna da esquerda é estreita; a explicação completa aparece no texto abaixo da tabela.
5. As capturas usam vídeo sintético (quadros coloridos), então o esqueleto e o texto reconhecido não aparecem nelas; a legenda delas mostra "Mostre as mãos para a câmera", que é o que o pipeline devolve para esse vídeo. A legenda com palavras só foi conferida no teste do componente, com texto sintético.
6. Em 390 px, o vídeo e as abas do modo "Diagnóstico e teste" não cabem juntos na tela: ao abrir o modo, a página rola até as abas e o vídeo fica acima.

**Item 13: legenda e sobreposições do app real.**
- Tokens novos em `tokens.json`: seção `legenda` (5 tamanhos de 20 a 52 px, 3 fontes do sistema, uma sem serifa de alta legibilidade, predefinições, largura de 40% a 100%, duração do destaque) e as cores `imagem_pior_caso_clara`/`imagem_pior_caso_escura` usadas no teste de contraste. Predefinições, com contraste medido a partir dos tokens sobre o pior caso de imagem (fundo translúcido composto sobre branco e sobre preto): branco sobre fundo escuro translúcido (9,3:1; parcial 6,6:1; texto sobre o destaque 5,2:1), amarelo sobre preto (16,1:1; destaque 7,9:1) e preto sobre branco (17,6:1; destaque 16,1:1). Combinação personalizada abaixo de 4,5:1 mostra aviso e não é bloqueada; as cores do usuário ficam só no arquivo de preferências.
- Um único gerenciador de posições (`ui/sobreposicoes.py`) para botão de Configurações, barra técnica, legenda, etiqueta de enquadramento, barras de mão cortada e barra mínima com o selo. Prioridade: barra mínima e botão (fixos), barra técnica (reserva o topo e termina antes do botão), legenda (nunca se move por causa da etiqueta), etiqueta (primeiro espaço livre perto do topo; na janela estreita pode ocupar a faixa preta acima da imagem). Testado nas três posições da legenda, nas três faixas e com a cabeça simulada no topo, na borda e perdida.
- Destaque da palavra confirmada: fundo de acento, contorno na cor do texto e peso mais forte por 300 ms; com "Reduzir movimento", sem animação: peso mais forte e sublinhado por 1,2 s. No máximo 3 mudanças por segundo (WCAG 2.3.1): um pulso só começa 750 ms depois do anterior e se nenhuma janela de 1 s passar de 3 mudanças.
- App real nas três faixas: compacta (até 600 px) com rótulos curtos embaixo dos ícones, selo só com o ícone (tocar abre o texto) e Configurações como folha que sobe de baixo; média e expandida com painel lateral. Alvos de toque de 44 px (testado). Ícones novos desenhados para o projeto: engrenagem, cadeado, tela cheia, lista, mão, parar, copiar, importar e fechar.

## Decisões tomadas sem consulta à equipe (terceira rodada, item 12)

1. A imagem de referência da curva (enviada depois do pedido) foi a fonte da rampa; violeta e magenta foram clareados o mínimo para 3:1, e o azul do meio foi ancorado no azul da marca.
2. Rótulo dos botões primários em texto grande e negrito (19 px), para manter o gradiente pedido com contraste válido (3:1 para texto grande).
3. Item ativo da navegação com o gradiente um tom mais escuro (#2563EB para #6D28D9), para texto normal a 4,5:1.
4. Texto terciário e borda de controles clareados (#8290AE e #627194); o cinza antigo ficou só para desabilitados.
5. Diálogos próprios no lugar do `QMessageBox` (o do Windows toca som).
6. Gráfico sem suavização por padrão; com "Suavizar", hastes ligam o ponto real à curva cinza (como na referência), e sem suavizar ligam à linha de base (como no pedido).
7. Trilho da faixa média com rótulo embaixo do ícone, para o nome não ficar só na dica.
8. "Mais" na faixa compacta como uma página com botões grandes, e não como menu suspenso.
9. Formulários com rótulo acima do campo e filas de botões que quebram linha quando não cabem.
10. Botões de -1 s e +1 s no editor, como equivalente por toque de Shift+setas.
11. Eixos de gráfico e régua do editor em 12 px (abaixo do corpo mínimo de 14 px), por serem rótulos densos de dados.
12. Tamanho do texto em quatro passos (100%, 110%, 125% e 150%).
13. Capturas da 12.7 com a largura lógica fixa (1366, 900 e 390 px) em cada escala, simulando a densidade de pixels; no Windows, 125% e 150% também reduzem a largura lógica disponível.
14. Rótulos curtos na barra inferior e no trilho ("Pessoas", "Palavras", "Ao vivo", "Ajustes"), para caberem em 390 px sem cortar; na barra lateral completa e nos títulos das abas os nomes continuam os mesmos.
15. A sobreposição de conteúdo em largura estreita (filas de botões e textos que quebram dentro das colunas) foi corrigida fazendo as colunas e as filas reservarem a altura necessária na largura real.

## Decisões tomadas sem consulta à equipe (pedido da legenda no Tempo real)

1. Legenda na parte de baixo da imagem, como legenda de TV, e não acima da cabeça: não depende de localizar o rosto e não cobre as mãos durante a maior parte dos sinais.
2. Duas linhas no máximo para o texto confirmado; as palavras antigas saem pela esquerda. O texto completo fica na aba "Decisão".
3. A linha discreta da legenda só aparece quando ajuda quem sinaliza (sem mãos na imagem, aquecendo ou sinal em análise); "confiança abaixo do limiar" e `nenhum_sinal` não aparecem na legenda.
4. Seleção de modelo e câmera foi para a aba "Modelo" do modo "Diagnóstico e teste"; por padrão continua o modelo final mais recente.
5. Atalho D para abrir e fechar o modo; T só registra tentativa com o modo aberto.
6. Três tokens novos (`legenda_fundo` #000000B8, `legenda_texto` #FFFFFF, `legenda_parcial` #D3DAE6).
7. Uma mensagem de erro do serviço (câmera que não abre, modelo ausente) não é mais substituída na tela pelo código de saída do processo.
8. Removido `tests/test_zz_med.py`: era um script de medição que não verificava nada e entrou por engano no item 7; a medição da troca de abas continua em `test_desempenho_abas.py`.


## D19. Dois aplicativos: bancada (LibrasVision Lab) e app real (LibrasVision)

**Visão (13.1).** O aplicativo de D16 virou a bancada da equipe ("LibrasVision Lab"; `LibrasVisionLab.bat` ou `python -m libras.app`), sem perder função. O app real ("LibrasVision"; `LibrasVision.bat` ou `python -m libras.app_real`) é só câmera e legenda, para quem usa no dia a dia: não tem coleta, dados, treino nem diagnóstico, nem por linha de comando (o único argumento é `--config`, oculto). Mesmo repositório e mesmo pacote `libras`.

**Pacote de modelo (13.2).** O app real recebe da bancada um pacote, nunca dados. Pasta `publicados/<id>/` (fora do Git), com exatamente três arquivos:
- `modelo.keras`: o modelo final (D13/D15);
- `metadados.json`: os metadados que a inferência usa, filtrados por lista branca (classes, features, janela, normalização, MediaPipe, versões, datas, protocolo, contagens); ficam de fora os IDs de sinalizadores, o hardware de treino e os avisos de seleção, que podem citar IDs;
- `cartao.json`: rótulos das classes (do vocabulário), datas de treino e de publicação, protocolo, versões (inclusive a do MediaPipe), parâmetros de decisão (os do config da bancada no momento da publicação), sha256 do modelo e dos metadados e, se existir, o resumo da avaliação por partições (D13) com a ressalva do protocolo.
- O `id` é a data e hora da publicação mais os 8 primeiros caracteres do hash do modelo. Nada de vídeos, landmarks, manifesto, cadastro, IDs ou caminhos (D11); um teste procura IDs e caminhos nos arquivos do pacote e no registro de auditoria.
- Bancada, aba Resultados: "Publicar modelo" e "Exportar pacote (.zip)" só ficam ativos para modelo final; ENSAIO fica bloqueado com o motivo escrito. A confirmação mostra o que vai e o que não vai. Publicar e exportar ficam em `data/auditoria.log`.
- Métricas no cartão: a avaliação mais recente concluída, não ENSAIO, do protocolo principal e com as mesmas classes do modelo. Sem ela, o cartão sai sem métricas.
- App real: pasta de modelos configurável (padrão `publicados/` ao lado do app). Ao abrir, se houver pacote publicado depois do ativo, oferece trocar; recusado, não pergunta de novo. "Importar pacote (.zip)" extrai numa pasta temporária, confere e só então move; .zip com arquivo a mais, caminho para fora da pasta, hash diferente ou de outro formato é recusado com mensagem clara e não deixa nada na pasta. O anterior fica guardado ("Voltar ao modelo anterior"). O serviço confere o hash de novo antes de carregar o modelo. Nunca há predição simulada: sem pacote válido, a tela explica e oferece importar. Se o pacote escolhido está na pasta mas foi alterado ou está incompleto, o app não troca sozinho: mostra o problema e oferece importar ou usar o pacote válido mais recente (o escolhido vira o anterior).
- O hash detecta pacote corrompido ou alterado depois da publicação, mas não é assinatura digital: quem altera o modelo pode alterar o cartão. Assinatura (chave da equipe) fica como opção se houver distribuição remota.
- Distribuição remota (registrada, não decidida): servir os pacotes de um endereço da equipe (por exemplo, releases de um repositório) com o app conferindo hash e assinatura antes de importar. Exigiria rede, mudaria o texto do selo de privacidade (13.9-B) e traria "perda de pacotes" e falhas de download como novas situações; nesta etapa não há rede.

**Tela (13.3).** Uma tela só: câmera ocupando a janela (imagem inteira, sem cortar, para a pessoa ver as próprias mãos), legenda sobre o vídeo, engrenagem no canto superior direito da janela (44 px; até 2026-10-14 ficava no canto da imagem, que em tela larga é centralizada e deixava o botão no meio da tela) e barra mínima embaixo: iniciar e parar câmera, pausar legenda, tela cheia, Sinais reconhecidos, selo de privacidade e, numa segunda linha, o indicador de mãos e o aviso permanente "Reconhece só os sinais do vocabulário deste modelo e não substitui um intérprete de Libras."
- Estados com orientação: sem modelo (explica e oferece importar), falha de câmera (causa provável e ação a partir das mensagens de abertura de câmera de D9, com "Tentar de novo" e "Escolher outra câmera"), aquecendo e ativo. A tela não mostra barras de probabilidade nem ajustes de decisão (só na bancada).
- A decisão vem do pacote, pelo mesmo `Reconhecedor`/`Transcritor` de D15; o usuário não a ajusta.

**Legenda (13.4).**
- Roll-up de 2 ou 3 linhas: as palavras confirmadas entram na linha de baixo; quando ela enche, tudo sobe em 150 ms e a mais antiga sai. Sem atividade por T segundos (padrão 4 s, ajustável), o texto esmaece em 300 ms; com "Reduzir movimento", troca e some de uma vez.
- Componente único em `ui/legenda.py`: lógica pura (`RollUp`, testada com relógio simulado), estilo e desenho. A bancada usa o mesmo componente.
- Posições: embaixo; em cima da pessoa; faixa fixa no alto.
  - Em cima da pessoa: o serviço envia a caixa da cabeça, estimada pelos pontos 0 a 10 da pose (nariz a orelhas), subindo 0,7 vez a distância entre as orelhas para cobrir testa e cabelo. É só para posicionar: não entra no modelo.
  - A interface suaviza a caixa (exponencial, alfa 0,25), centra a legenda acima da cabeça, respeita as margens e a base da barra técnica, e nunca desce sobre o rosto: sem espaço acima da cabeça, a legenda vai para baixo.
  - Pessoa não detectada por mais de 1 s: faixa fixa no alto.
  - Limitação: o Holistic detecta uma pessoa só; com duas pessoas na imagem, a legenda segue a que o MediaPipe escolher.
- A legenda é desenhada pela interface, nunca no JPEG do serviço, e nunca é espelhada (há teste); só a caixa da cabeça e os lados de mão cortada são convertidos para a prévia espelhada.
- Configurações > Legenda, com prévia ao vivo sobre a câmera: tamanho (5 passos), fonte (3), cores (3 predefinições ou personalizada: cor do texto, cor e opacidade do fundo), contorno ou sombra, largura, linhas, tempo até sumir, posição, "Destacar palavra confirmada", "Reduzir movimento" e "Restaurar padrão". Também "Limpar legenda" (limpa também o texto do serviço) e "Copiar texto". Corrigir o texto reconhecido não entrou.
- Preferências do app real em `configs/app_real_preferencias.json` (fora do Git), separadas do `configs/local.yaml` da bancada. Configuração fixa em `configs/app_real.yaml`: pasta de modelos, câmera, limiares do guia, intervalo da barra técnica.

**Configurações e informações técnicas (13.8).**
- Painel aberto pela engrenagem ou F2 (Esc fecha). É lateral nas faixas média e expandida, com a câmera encolhendo ao lado para a legenda de exemplo continuar visível, e folha que sobe de baixo na faixa compacta. A câmera continua rodando.
- Seções: Legenda, Informações técnicas, Câmera (número da câmera, "Tentar de novo", "Avisos de enquadramento"), Modelo (resumo do cartão com a ressalva, importar, voltar ao anterior) e Sobre (aviso, versões, privacidade, "Ver guia de posicionamento").
- Barra técnica: só existe com pelo menos um grupo ligado (todos desligados por padrão). Fica numa linha no topo da imagem e termina antes do botão.
  - Grupos, na ordem: Taxa (FPS da câmera, processados e de exibição), Quadros (capturados, processados, descartados e a porcentagem de pulados), Tempo (latência de ponta a ponta com p95 dos últimos 10 s, MediaPipe e modelo), Recursos (memória da interface e do subprocesso) e Sistema (modelo, câmera e estado).
  - Atualiza a cada 500 ms, lendo o último quadro: a interface só formata; o serviço mede.
  - Os quadros pulados aparecem em cor neutra, com a explicação "quadros pulados para manter o tempo real" quando cabe. O único alerta é "A câmera parou de enviar imagens" (sem quadro por 3 s).
  - O texto tem no mínimo 14 px, números de largura fixa e o mesmo fundo translúcido da legenda (contraste testado).
  - Na faixa compacta usa até 2 linhas. O que não cabe sai por prioridade (primeiro Sistema, depois Recursos...), e a explicação dos pulados sai antes de qualquer grupo.
  - Na tela cheia, barra técnica, engrenagem e barra mínima somem juntas após 3 s sem movimento e voltam ao mover o ponteiro ou tocar, sem mudar a preferência.
- "Copiar relatório técnico" copia para a área de transferência os valores, a versão, o modelo (id e começo do hash) e o hardware; não tem imagens, IDs nem caminhos e não grava nada em disco.
- "Perda de pacotes" não se aplica: câmera e inferência são locais e não há rede; o equivalente são os quadros descartados. Se houver transmissão remota, entra aqui.

**Guia, selo, destaque e pausa (13.9).**
- Guia de enquadramento (só informa; nunca bloqueia nem muda o que o modelo recebe):
  - O serviço mede ombros (pose 11 e 12 visíveis e dentro da imagem: ok, parcial ou fora), mão cortada (mão detectada com ponto a menos de 3% da borda; mão ausente não conta) e brilho (média em cinza de uma amostra de cerca de 64x36 pontos, cerca de 0,05 ms).
  - As frações de pose e de mãos do último 1 s usam os mínimos do modelo, importados de `libras.decisao`.
  - Os limiares novos ficam em `configs/app_real.yaml` como valores iniciais a calibrar na bancada com os primeiros vídeos reais: borda de 3%, brilho mínimo de 60 (o mesmo valor inicial da qualidade da coleta) e visibilidade do ombro de 0,5.
  - Avisos: um por vez, com ícone, texto e borda, na ordem sem pessoa, ombros, mão cortada, pouca luz. Histerese de 1 s para aparecer e 1 s para sumir. Mão cortada também desenha uma barra fina no lado certo da prévia espelhada; o texto não diz esquerda ou direita (D3).
  - A falha de câmera tem prioridade sobre os avisos.
  - Primeira abertura: contorno de cabeça e ombros, área das mãos e lista ao vivo (ombros, mãos ao sinalizar, luz), com "Começar" ou "Continuar mesmo assim". Mostra-se uma vez; reabre em Sobre. O painel do guia procura uma posição que não cruze legenda, etiqueta, botão nem barra técnica (abaixo da imagem na janela estreita); em janela média ele pode cobrir parte do contorno, que fica visível ao redor.
- Selo de privacidade: "Processamento neste computador. Nenhuma imagem é gravada nem enviada." Na faixa compacta é só o ícone; tocar abre uma folha pequena acima da barra mínima com o texto completo (não é diálogo modal). Em Sobre, mais: "Só o texto reconhecido aparece na tela; ele não é guardado em disco." Verificação feita antes de mostrar o texto:
  - No código: nenhum módulo do app real, da pasta comum ou do serviço usa rede nem grava imagem ou vídeo (teste por busca no código).
  - Numa sessão real: o serviço com MediaPipe e TensorFlow de verdade, no modo do app real, rodou com a criação de sockets e toda conexão bloqueadas: nenhuma tentativa de conexão. A única criação de socket vista é a sonda do urllib3, carregado pelo TensorFlow ao ser importado, que abre um socket IPv6 local só para saber se o sistema tem IPv6, sem conectar em nada; ela foi bloqueada no teste sem efeito no reconhecimento e com as pastas monitoradas (nenhum arquivo criado, de nenhum tipo). A interface também foi monitorada numa sessão completa (câmera, palavras, pausa, copiar texto, relatório técnico): nenhum arquivo além das preferências.
  - O app real grava em disco só o arquivo de preferências e os pacotes importados; o serviço, com `--sem-disco`, não grava log nem medição.
  - Se a transcrição vier a ser salva (item futuro), o texto do selo muda.
- Destaque da palavra confirmada: ver D18. A vibração do aparelho fica registrada como opção do cliente móvel, sem implementação.
- Pausa (P ou botão; Espaço não é usado):
  - Congela o texto (sem rolar nem esmaecer; se já tinha esmaecido, volta o último texto) e mostra "Legenda pausada" com ícone e o contador de palavras novas.
  - Câmera e reconhecimento continuam; as palavras confirmadas na pausa entram na ordem ao retomar (mais de uma entra de uma vez, sem animação).
  - "Copiar texto" na pausa copia exatamente o que está congelado; "Limpar legenda" limpa tela e buffer e retoma; parar a câmera ou fechar encerra a pausa. A pausa não vai para as preferências.
  - A sequência de palavras do Transcritor é a mesma com e sem pausa (testado).

**Arquitetura e medição (13.6).**
- O app real não importa coleta, dataset, treino, ablação, augmentation nem a bancada. Testes:
  - importação real da interface em outro processo;
  - análise dos imports declarados no código, inclusive os de dentro de funções;
  - lista dos módulos carregados pelo serviço real numa sessão.
- A interface do app real nunca importa TensorFlow, MediaPipe nem Keras. O serviço continua em subprocesso, iniciado ao ligar a câmera e encerrado ao parar ou fechar.
- Medido no ambiente de desenvolvimento (Xeon 2,8 GHz, 2 núcleos; não é o hardware alvo), com o vídeo sintético de cores no lugar da câmera:

  | Medida | Valor |
  |---|---|
  | Clique até o primeiro quadro | cerca de 4,5 s (carrega TensorFlow, modelo e MediaPipe) |
  | RAM da interface | cerca de 35 MB antes do Qt; cerca de 99 MB aberta; cerca de 103 MB com a câmera rodando (bancada: 153 MB) |
  | RAM do subprocesso | cerca de 1,0 GB, como em D15 |
  | FPS | câmera 30, processados cerca de 29, exibição cerca de 15 |
  | Latência de ponta a ponta | mediana cerca de 17 ms, p95 cerca de 24 ms |

  O vídeo sintético não tem pessoa, então o MediaPipe trabalha menos (cerca de 15 ms contra cerca de 60 ms em D15) e o modelo não roda (sem mãos). Estes números servem para comparar versões entre si, não para o TCC. A medição com pessoa real e no i5 com 4 GB fica com a equipe.

**Instalador (avaliação, não decidido).** PyInstaller 6.22.3, testado neste ambiente Linux. Instalado só durante o teste, como ferramenta de build fora do `requirements.txt`; `pip check` sem conflitos com ele instalado e depois de removido. O comando está em `empacotamento/LEIA-ME.md`.
- Resultado: pasta única (onedir) de cerca de 2,4 GB com o app real e o serviço. Partes maiores: TensorFlow 1,4 GB; jaxlib 323 MB, trazido pelas dependências do MediaPipe; OpenCV cerca de 170 MB; PySide6 99 MB; MediaPipe 56 MB; SciPy cerca de 80 MB. O pacote do modelo fica fora e é importado pelo app.
- O executável congelado funcionou: a interface abriu (cerca de 110 MB) e iniciou o serviço congelado (cerca de 1,1 GB), que carregou TensorFlow, MediaPipe e o modelo e processou cerca de 29 quadros por segundo do vídeo de teste.
- Ajuste necessário: congelado não existe `python -m`, então `empacotamento/entrada_app_real.py` despacha o serviço quando o executável é chamado com `-m libras.tempo_real_servico`. É preciso incluir `tokens.json` e os SVG como dados.
- Compatibilidade com D1: usa o mesmo Python 3.11 e as mesmas versões; nada muda no ambiente.
- Riscos:
  1. o PyInstaller não gera executável para Windows a partir do Linux; o build precisa ser feito e testado no Windows da equipe;
  2. o tamanho (cerca de 2,4 GB, e o TensorFlow sozinho 1,4 GB) pesa no download e no disco do usuário; converter para TensorFlow Lite (D15, D18) é a alternativa a avaliar para reduzir;
  3. antivírus costumam desconfiar de executáveis do PyInstaller não assinados;
  4. a pasta de instalação pode não permitir escrita (Program Files): preferências e pacotes deveriam ir para a pasta do usuário (%APPDATA%) num instalador;
  5. TensorFlow, MediaPipe (com jax) e Qt no mesmo pacote precisam de teste a cada atualização de versão.

**Mobile.** O app real é o candidato a cliente móvel; nada foi portado. Funções da fachada de D18 que ele usa:
- pacote: `pacote.verificar_pacote`, `pacote.importar_zip`, `pacote.listar_pacotes`, `pacote.rotulo`, `pacote.resumo`; `app_real.modelos.ControleModelos`;
- serviço: `config_real.carregar` e `config_real.config_servico`; o serviço `tempo_real_servico` (protocolo `LVQ1`, `Decodificador`) com `inferencia.Reconhecedor`/`Transcritor`, `decisao` e `enquadramento.MedidorEnquadramento`;
- câmera: `camera.abrir_camera_ou_video`, `Captura` e `camera.explicar_falha`;
- legenda e tela: `ui.legenda.RollUp`, `estilo_de` e `CamadaLegenda`; `ui.sobreposicoes.posicionar`; `app_real.guia.GuiaAvisos`; `app_real.tecnicas`.

Num cliente nativo, legenda, guia e sobreposições são lógica pura e portável. O serviço depende de MediaPipe e TensorFlow (ver as opções para mobile em D18).


**Abertura da janela (2026-10-14).** Os dois aplicativos abrem maximizados, em modo janela (com bordas e barra de título). No app real, F11 continua alternando a tela cheia, e ao sair dela a janela volta ao estado anterior (maximizada ou não).
## Decisões tomadas sem consulta à equipe (quarta rodada, item 13)

1. `LibrasVision.bat` foi renomeado para `LibrasVisionLab.bat` (bancada); o nome `LibrasVision.bat` passou a abrir o app real.
2. Componentes compartilhados movidos para `libras/ui/` e a câmera para `libras/camera.py`; `app/base.py` reexporta os componentes para a bancada não mudar.
3. Metadados do pacote por lista branca (sem IDs, hardware de treino e avisos de seleção); o `id` do pacote é a data e hora mais o começo do hash.
4. Avaliação do cartão escolhida automaticamente (a mais recente válida com as mesmas classes); sem ela, sem métricas.
5. "Exportar pacote (.zip)" não publica em `publicados/`; os dois ficam na auditoria.
6. Hash sha256 como conferência de integridade, sem assinatura (registrado como limite).
7. Pacote recusado no app real e no serviço se for ENSAIO ou se os metadados tiverem IDs de sinalizadores.
8. Imagem inteira na janela (sem cortar), centralizada, com faixas pretas quando a proporção não bate.
9. Configurações empurra a câmera para o lado (não fica por cima dela) nas faixas média e expandida, para a legenda de exemplo continuar visível.
10. "Limpar legenda" e "Copiar texto" em Configurações > Legenda, com atalhos Ctrl+L e Ctrl+C; fora da pausa, copiar pega todo o texto desde a última limpeza.
11. Na posição "em cima da pessoa", sem espaço acima da cabeça a legenda vai para baixo; caixa da cabeça estimada pelos pontos da pose (fator 0,7 da distância entre as orelhas para a testa) e suavizada com alfa 0,25.
12. Brilho por amostragem de cerca de 64x36 pontos em vez de `cv2.resize` (cerca de 0,05 ms contra cerca de 1 ms aqui), para ficar abaixo de 1 ms por quadro.
13. Limiares iniciais do guia: 3% de borda, brilho 60 e visibilidade do ombro 0,5; "sem pessoa" quando a pose aparece em menos de 50% do último segundo (o mínimo do modelo).
14. Etiqueta de enquadramento pode ocupar a faixa preta acima da imagem na janela estreita, para nunca cruzar a legenda; na faixa compacta, o guia da primeira abertura vai para baixo da imagem.
15. Na pausa, mais de uma palavra acumulada entra de uma vez, sem animação nem destaque.
16. Destaque: um pulso só começa 750 ms depois do anterior (no máximo 3 mudanças por segundo).
17. Alerta de quadros só quando a câmera para de entregar imagens por 3 s; descarte normal em cor neutra.
18. Tela cheia: os controles somem após 3 s sem movimento (F11 e Esc também funcionam).
19. O aviso permanente fica na barra mínima e some junto com ela na tela cheia sem movimento.
20. O "item 11" citado no pedido não existe nas rodadas anteriores; a falha de câmera reaproveita as mensagens de abertura de D9, traduzidas em causa provável e ação.
21. PyInstaller instalado só durante a avaliação (ferramenta de build, fora do `requirements.txt`) e removido depois; `empacotamento/entrada_app_real.py` despacha o serviço no executável congelado.
22. Pacote escolhido inválido com outro válido na pasta: o app mostra o problema e oferece a troca, em vez de trocar sozinho (achado da conferência independente).
23. "Tentar de novo" com a câmera ligada encerra o reconhecimento em andamento antes de abrir de novo (achado da conferência: o processo anterior ficava preso à câmera).
24. A câmera do serviço só começa a ser lida depois que o MediaPipe está pronto (a carga inicial inflava a porcentagem de pulados no começo).
25. Selo na faixa compacta abre uma folha pequena, não um diálogo; o teste de privacidade bloqueia também a criação de sockets.
