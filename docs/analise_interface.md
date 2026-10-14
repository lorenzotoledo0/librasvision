# Análise de interfaces de referência (item 12.1)

Pesquisa feita em 06/10/2026 lendo as páginas citadas (documentação oficial, central de ajuda, lojas de aplicativos). Os trechos foram lidos por uma ferramenta que resume a página, então citações podem não estar literais; o que não foi confirmado está marcado **não verificado**. Esta análise orienta o redesenho; não é revisão de literatura do TCC.

## Achados por grupo

**(a) Libras e acessibilidade.** Hand Talk: avatares (Hugo e Maya), velocidade da sinalização ajustável, uso offline, dicionário por temas ([handtalk.me](https://www.handtalk.me/br/aplicativo/)); o usuário avalia as traduções ([Google Play](https://play.google.com/store/apps/details?id=br.com.handtalk)). VLibras: plugin, desktop, Android/iOS e widget; aviso explícito de que "não substitui um intérprete humano" ([gov.br](https://www.gov.br/governodigital/pt-br/acessibilidade-e-usuario/vlibras)). Navegação, modo escuro e onboarding dos dois: não verificado.

**(b) Legendas ao vivo.** Live Transcribe: tamanho do texto ajustável (pinça ou controle), modo escuro, indicador de som no canto superior direito, vibração quando falam um nome escolhido, rótulos visuais de sons ([ajuda Google](https://support.google.com/accessibility/android/answer/9158064); [android.com](https://www.android.com/accessibility/live-transcribe/)). O mesmo app tem sinal para texto (ASL) pela câmera frontal, pede para enquadrar "mãos e rosto" e, em erro, sugere apagar a frase e sinalizar de novo ou digitar (mesma página de ajuda). Legenda ao vivo do Windows: posição acima, abaixo ou sobreposta, estilo que segue o tema, janela redimensionável ([suporte Microsoft](https://support.microsoft.com/en-us/topic/use-live-captions-to-better-understand-audio-b52da59c-14b8-4031-aeeb-f6a47e6055df)). Ava: cor por falante ([ava.me](https://www.ava.me/)); fonte e modo escuro não verificados.

**(c) Ferramentas de ML.** TensorBoard: suavização padrão 0,6 ([código](https://raw.githubusercontent.com/tensorflow/tensorboard/master/tensorboard/webapp/metrics/store/metrics_types.ts)); a documentação manda zerar a suavização para ver os valores reais ([tensorflow.org](https://www.tensorflow.org/tensorboard/scalars_and_keras)); uma cor por execução ([README](https://github.com/tensorflow/tensorboard/blob/master/README.md)). W&B: curva original sem suavização exibida apagada ao fundo por padrão ([docs](https://docs.coreweave.com/products/wandb/app/features/panels/line-plot/smoothing.md)); cor por execução editável ([docs](https://docs.coreweave.com/products/wandb/runs/run-colors)). Roboflow: controle de limiar de confiança que atualiza métricas e gráfico por classe ([docs](https://docs.roboflow.com/models/evaluate/evaluate-trained-models)). Edge Impulse: classificação ao vivo por janela, com confiança por rótulo ([docs](https://docs.edgeimpulse.com/docs/edge-impulse-studio/live-classification)); cores dessa tela não verificadas.

**(d) Editor de vídeo.** CapCut: corte movendo o playhead e usando "Split" na barra de ferramentas; Ctrl+Z desfaz ([tutorial](https://www.capcut.com/resource/capcut-tutorial-for-beginners); [split](https://www.capcut.com/resource/split-video-into-parts)). Alças de corte nas pontas do clipe: não verificado nas páginas oficiais.

**Diretrizes.** Barra de navegação para 3 a 5 destinos em janela compacta ([Android Developers](https://developer.android.com/develop/ui/compose/components/navigation-bar); [Material Components](https://github.com/material-components/material-components-android/blob/master/docs/components/BottomNavigation.md)). Apple: abas servem para navegar, não para ações; no iPhone ficam embaixo; controles de 44x44 pt ([HIG tab bars](https://developer.apple.com/tutorials/data/design/human-interface-guidelines/tab-bars.json); [HIG acessibilidade](https://developer.apple.com/tutorials/data/design/human-interface-guidelines/accessibility.json)). Android: alvo de toque de 48x48 dp ([Android](https://developer.android.com/guide/topics/ui/accessibility/apps)). WCAG: contraste 4,5:1 (texto) e 3:1 (texto grande) ([1.4.3](https://www.w3.org/WAI/WCAG21/Understanding/contrast-minimum.html)); 3:1 para componentes e gráficos (1.4.11, critério já conhecido, não reaberto nesta pesquisa); alvo de 24x24 px no AA ([2.5.8](https://www.w3.org/WAI/WCAG22/Understanding/target-size-minimum.html)) e 44x44 px no AAA ([2.5.5](https://www.w3.org/WAI/WCAG21/Understanding/target-size.html)).

## Hipóteses

- **H1 (visual primeiro) — parcialmente confirmada.** Vibração, rótulos visuais de sons, texto ajustável e modo escuro aparecem nas fontes de (b); "nada depende de som" não está escrito como princípio em nenhuma delas.
- **H2 (texto parcial que se atualiza) — parcialmente confirmada.** As APIs de reconhecimento entregam resultado intermediário que é substituído pelo final ([Google `is_final`](https://docs.cloud.google.com/speech-to-text/docs/v1/transcribe-streaming-audio)); no exemplo da Microsoft o intermediário aparece em cinza ([Azure](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/get-started-speech-to-text)). A tela do Live Transcribe e do Ava com texto parcial: não verificado.
- **H3 (poucas cores saturadas, só nos dados) — não verificada.** Só se confirmou cor por série/execução (TensorBoard, W&B) e a curva original apagada ao fundo (W&B).
- **H4 (barra inferior com até 5 destinos) — confirmada no Material; parcial na Apple** (sem número fixo para iPhone; "menos abas" é mais fácil).

## Padrões e decisão para o LibrasVision

| Padrão | Onde aparece | Adotar ou descartar | Motivo |
|---|---|---|---|
| Sinal "em análise" em cinza, confirmado em cor plena | Azure (intermediário em cinza); Google `is_final` | Adotar | Equivale ao "aguardando N predições iguais" (D15) sem alerta extra |
| Tamanho do texto ajustável | Live Transcribe; Apple (texto ampliável) | Adotar (100% a 150% em Configurações) | Público surdo; leitura à distância |
| Tema escuro com contraste medido | Live Transcribe; Windows | Adotar | Já existe; contraste agora é testado |
| Indicador de captação no canto | Indicador de som do Live Transcribe | Adotar como "mãos detectadas" | Mesma função visual, sem som |
| Orientar enquadramento de mãos e rosto | Sinal para texto do Live Transcribe | Adotar (estado vazio e guia do Gravar) | Mesma tarefa |
| Corrigir apagando e sinalizando de novo | Live Transcribe | Adotar ("Limpar texto") | Erro tratado sem som |
| Vibração ou flash para avisos | Live Transcribe, Android | Descartar no desktop; anotar para mobile | Desktop não vibra; o app não emite som |
| Confiança por classe | Edge Impulse | Adotar (barras de todas as classes) | Explica a baixa confiança |
| Controle de limiar de confiança | Roboflow | Adotar só no Diagnóstico | Ajuste de teste, não do usuário final |
| Curva suavizada por padrão | TensorBoard (0,6) | Descartar | A curva do TCC deve ser o dado real; suavizar fica opcional e rotulado |
| Curva original visível quando há suavização | W&B | Adotar | Nunca esconder o dado real |
| Cor por série, resto em neutros | TensorBoard, W&B | Adotar | Rampa só em dados e assinatura (H3 não verificada) |
| Cor por falante | Ava | Descartar | Uma pessoa sinalizando por vez |
| Barra inferior com até 5 destinos | Material; Apple (iPhone embaixo) | Adotar na faixa compacta | Diretriz oficial |
| Ações fora da barra de navegação | Apple HIG | Adotar | Gravar e parar são ações, não abas |
| Alvos de 44 px | Apple 44 pt; Android 48 dp; WCAG 2.5.5 | Adotar (faixas compacta e média) | Toque futuro |
| Playhead e "dividir" na barra do editor | CapCut | Adotar (já existe, D9) | Modelo mental conhecido |
| Aviso de que não substitui intérprete | VLibras | Adotar no Tempo real | Honestidade sobre um protótipo |
| Avaliar a tradução | Hand Talk | Descartar por ora | O "Testar acertos" (D15) já cobre o teste; coleta de feedback exige outro TCLE |
