# Changelog

Todas as mudanças relevantes deste projeto são registradas aqui.

O formato segue o [Keep a Changelog](https://keepachangelog.com/pt-BR/1.1.0/) e o projeto adota o [Versionamento Semântico](https://semver.org/lang/pt-BR/).

## [Não publicado]

### Alterado

- Os dois aplicativos abrem maximizados, em modo janela.
- A engrenagem do LibrasVision fica no canto superior direito da janela, e não mais no canto da imagem.

## [0.1.0] - 2026-10-14

Primeira versão pública. Base completa do sistema, **sem dados reais e sem resultados medidos**.

### Adicionado

- Núcleo `libras`: extração de landmarks com MediaPipe Holistic, normalização única para treino e tempo real, janelas de 30 frames reamostradas para 30 FPS, splits agrupados por sinalizador com checagem de vazamento, data augmentation em landmarks (famílias espacial, temporal e ruído) e em vídeo (offline), modelo LSTM empilhado, treino por partição, métricas por janela e por vídeo, ablação com teste t corrigido e correção de Holm, testes de robustez e inferência em tempo real.
- LibrasVision Lab (bancada da equipe): cadastro de sinalizadores com TCLE, vocabulário configurável, gravação pela webcam (take único e rajada), importação de vídeos, revisão com editor de corte, dataset, treino com curvas ao vivo, resultados, publicação de pacotes de modelo e teste ao vivo.
- LibrasVision (app do usuário final): câmera e legenda em tempo real estilo roll-up, posição e aparência da legenda configuráveis, tema escuro, três faixas de largura preparadas para mobile.
- Rajada com extração no fim: os takes são gravados em sequência e os landmarks de todos são extraídos depois do último take ou de "Parar rajada", com limite de memória configurável (`coleta.gravacao.memoria_max_rajada_mb`).
- Documentação: README, guias em `docs/`, ADRs, protocolo de coleta, diretrizes de privacidade e modelo de TCLE.
- Engenharia: `pyproject.toml` com extras `dev`, ruff, mypy gradual, pre-commit, CI no GitHub Actions (Windows e Ubuntu), Dependabot e templates de issue e PR.

[Não publicado]: https://github.com/lorenzotoledo0/librasvision/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/lorenzotoledo0/librasvision/releases/tag/v0.1.0
