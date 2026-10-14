# Changelog

Todas as mudanças relevantes deste projeto são registradas aqui.

O formato segue o [Keep a Changelog](https://keepachangelog.com/pt-BR/1.1.0/) e o projeto adota o [Versionamento Semântico](https://semver.org/lang/pt-BR/).

## [Não publicado]

## [0.1.0] - 2026-10-14

Primeira versão pública. Base completa do sistema, **sem dados reais e sem resultados medidos**.

### Adicionado

- Núcleo `libras`: extração de landmarks com MediaPipe Holistic, normalização única para treino e tempo real, janelas de 30 frames reamostradas para 30 FPS, splits agrupados por sinalizador com checagem de vazamento, data augmentation em landmarks (famílias espacial, temporal e ruído) e em vídeo (offline), modelo LSTM empilhado, treino por partição, métricas por janela e por vídeo, ablação com teste t corrigido e correção de Holm, testes de robustez e inferência em tempo real.
- LibrasVision Lab (bancada da equipe): cadastro de sinalizadores com TCLE, vocabulário configurável, gravação pela webcam (take único e rajada), importação de vídeos, revisão com editor de corte, dataset, treino com curvas ao vivo, resultados, publicação de pacotes de modelo e teste ao vivo.
- LibrasVision (app do usuário final): câmera e legenda em tempo real estilo roll-up, posição e aparência da legenda configuráveis, tema escuro, três faixas de largura preparadas para mobile.
- Rajada com extração no fim: os takes são gravados em sequência e os landmarks de todos são extraídos depois do último take ou de "Parar rajada", com limite de memória configurável (`coleta.gravacao.memoria_max_rajada_mb`).
- Documentação: README, guias em `docs/`, ADRs, protocolo de coleta, diretrizes de privacidade e modelo de TCLE.
- Engenharia: `pyproject.toml` com extras `dev`, ruff, mypy gradual, pre-commit, CI no GitHub Actions (Windows e Ubuntu), Dependabot e templates de issue e PR.
- Revisão rápida no fim da rajada: cada take passa na tela com o esqueleto e só os marcados são salvos.
- Revisão em lote na aba Revisão: marcar pendentes ou todos e aprovar ou rejeitar de uma vez.
- Prévia de augmentation na aba Revisão: o take e 5 variações animadas, com famílias e intensidade ajustáveis e exportação de figura.
- Trava de escrita no manifesto, entre threads e processos, inclusive na escolha do número do take.
- Visual neutro nas duas interfaces: grafite com um único azul de destaque, sem gradientes.
- Bancada: cabeçalho compacto, barra de ações fixa no rodapé (ações principais sempre visíveis), seções recolhíveis e barra lateral que rola em vez de cortar itens. Na Gravar, só os botões do estado atual, progresso da meta na legenda durante a rajada e "Parar e descartar o take atual".
- O máximo de takes na rajada acompanha a memória disponível para a resolução da câmera.
- Os dois aplicativos abrem maximizados, em modo janela; a engrenagem do LibrasVision fica no canto superior direito da janela.

[Não publicado]: https://github.com/lorenzotoledo0/librasvision/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/lorenzotoledo0/librasvision/releases/tag/v0.1.0
