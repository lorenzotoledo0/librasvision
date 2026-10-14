# Roadmap

O acompanhamento detalhado fica nas [issues](https://github.com/lorenzotoledo0/librasvision/issues) e nos [milestones](https://github.com/lorenzotoledo0/librasvision/milestones). Esta página resume as etapas.

## v0.1.0: Base (concluída)

- [x] Núcleo: landmarks, normalização única, janelas, splits por sinalizador, augmentation, LSTM, treino, métricas, ablação, robustez, inferência
- [x] LibrasVision Lab: coleta com TCLE, vocabulário configurável, gravação (take e rajada), importação, revisão com corte, dataset, treino, resultados, teste ao vivo
- [x] LibrasVision: câmera e legenda roll-up, tema escuro, três faixas de largura
- [x] Testes com dados sintéticos, CI no Windows e no Ubuntu
- [x] Documentação, diretrizes de privacidade e modelo de TCLE

## v0.2.0: Coleta e dataset

- [ ] Confirmar com a orientação o texto final do TCLE e a necessidade de Comitê de Ética
- [ ] Validar o vocabulário e a forma dos sinais com intérpretes e pessoas surdas
- [ ] Coletar vídeos de mais de 10 sinalizadores (meta: 10 takes por classe por pessoa)
- [ ] Calibrar limiares iniciais com dados reais (seleção de janelas por atividade, guia de enquadramento)
- [ ] Verificar se o rótulo `left_hand` do Holistic corresponde à mão esquerda da pessoa

## v0.3.0: Modelo

- [ ] Treinar o baseline no protocolo principal
- [ ] Rodar a ablação completa de augmentation
- [ ] Avaliar robustez
- [ ] Registrar resultados em `docs/EVALUATION.md`

## v0.4.0: App real

- [ ] Medir latência e memória no hardware alvo (i5, 4 GB)
- [ ] Avaliar TensorFlow Lite se a memória apertar
- [ ] Decidir a distribuição (instalador PyInstaller, pacotes remotos, assinatura digital)
- [ ] Levantar opções para mobile
- [ ] Gravar o GIF de demonstração

## v1.0.0: Entrega do TCC

- [ ] Textos do TCC com os resultados medidos
- [ ] Apresentação à comunidade surda da instituição
- [ ] Release final com notas e citação
