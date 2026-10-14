# Arquitetura

O LibrasVision é um pacote Python (`libras`) com um núcleo sem interface e dois aplicativos PySide6 em cima dele. As decisões estão em [decisoes.md](decisoes.md) e em [adr/](adr/README.md).

## Visão geral

```mermaid
flowchart TB
    subgraph nucleo["Núcleo (libras)"]
        LM["landmarks.py<br/>MediaPipe Holistic"]
        NORM["normalizacao.py<br/>reamostragem, janelas, features"]
        DEC["decisao.py<br/>limiares e confirmação"]
        INF["inferencia.py<br/>buffer, Reconhecedor, Transcritor"]
        AUG["augmentation/"]
        TR["modelo.py, treino.py, metricas.py"]
        AB["ablacao.py, estatistica.py, robustez.py"]
        DS["extracao.py, dataset.py, splits.py"]
        COL["coleta/<br/>gravação, importação, manifesto, TCLE"]
        PAC["pacote.py<br/>pacote de modelo publicado"]
        UI["ui/<br/>tema, tokens, legenda roll-up"]
    end
    subgraph lab["LibrasVision Lab (libras.app)"]
        LABP["Sinalizadores, Vocabulário, Gravar, Revisão,<br/>Dataset, Treino, Resultados, Tempo real"]
    end
    subgraph real["LibrasVision (libras.app_real)"]
        REALP["Câmera + legenda<br/>Configurações"]
    end
    SERV["tempo_real_servico.py<br/>subprocesso com TensorFlow"]

    LABP --> COL & DS & PAC & UI
    LABP -- "subprocesso" --> TR & AB
    LABP -- "subprocesso" --> SERV
    REALP --> UI
    REALP -- "subprocesso" --> SERV
    REALP -- "lê" --> PAC
    SERV --> LM & NORM & INF & DEC
    TR --> DS & AUG & NORM
    DS --> LM & NORM
    COL --> LM
```

## Pipeline em tempo real

```mermaid
flowchart LR
    CAM["Webcam<br/>thread de captura"] --> MP["MediaPipe Holistic<br/>pose, mãos, face"]
    MP --> BUF["Buffer com instantes reais"]
    BUF --> RES["Reamostragem 30 FPS<br/>últimos 30 frames"]
    RES --> NORM["Normalização pelos ombros<br/>vetor de 550 features"]
    NORM --> LSTM["LSTM 64 + 32<br/>softmax"]
    LSTM --> DEC["Decisão<br/>média móvel, limiar 60%,<br/>3 confirmações"]
    DEC --> LEG["Legenda roll-up<br/>desenhada pela interface"]
```

O processamento roda num subprocesso (`libras.tempo_real_servico`): uma thread captura, outra roda MediaPipe e modelo sobre o frame mais recente, e frames que chegam durante o processamento são descartados. O serviço envia à interface, pela saída padrão, mensagens binárias (cabeçalho `LVQ1`, JSON com resultado e medições, e JPEG do frame) e recebe comandos pela entrada padrão.

## Pipeline de treino

```mermaid
flowchart LR
    V["Vídeos<br/>data/raw_videos"] --> X["Extração<br/>.npz por vídeo"]
    G["Gravação<br/>frames sem compressão"] --> X
    X --> J["Janelas normalizadas<br/>float16 em memória"]
    J --> S["Splits por sinalizador<br/>StratifiedGroupKFold"]
    S --> T["Treino por partição<br/>augmentation só no treino"]
    T --> M["Métricas por janela e por vídeo"]
    T --> F["Modelo final<br/>modelo.keras + metadados.json"]
    F --> P["Pacote publicado<br/>publicados/id"]
    P --> A["App LibrasVision"]
```

## Regras de arquitetura

| Regra | Como é garantida |
|---|---|
| Uma única normalização para treino, avaliação e tempo real | módulo único e teste de igualdade entre as trajetórias |
| O processo da interface nunca importa TensorFlow | treino, ablação e tempo real em subprocesso; teste de arquitetura |
| O app real não importa coleta, treino nem augmentation | teste de arquitetura (`tests/test_arquitetura_app_real.py`) |
| Nenhuma tarefa longa na thread da interface | `Tarefa` (QThread) e `Subprocesso` (QProcess) em `ui/componentes.py` |
| Parâmetros do tempo real vêm do modelo, não do config atual | `metadados.json` do modelo |
| Cores e medidas visuais vêm de tokens | `ui/tokens.json`; teste falha com cor hexadecimal solta |
| Câmera e arquivos são fechados antes de reabrir | encerramentos em segundo plano esperados antes da reabertura |

## Organização do código

```text
src/libras/
  config.py, config_real.py   configuração (config.yaml + local.yaml; app_real.yaml)
  landmarks.py                extração com MediaPipe Holistic (módulo único)
  normalizacao.py             reamostragem, janelas, normalização e features (módulo único)
  decisao.py                  constantes e etapas da decisão em tempo real
  inferencia.py               buffer, reconhecimento e transcrição (sem interface)
  tempo_real_servico.py       subprocesso de tempo real
  extracao.py, dataset.py     extração em lote com cache; janelas e tf.data
  splits.py, selecao.py       divisões por sinalizador; seleção de classes do treino
  modelo.py, treino.py        LSTM, treino por partição, modelo final
  metricas.py, estatistica.py métricas e testes estatísticos
  ablacao.py, robustez.py     ablação e robustez
  relatorio_ablacao.py        tabelas LaTeX/CSV e figuras
  pacote.py, publicacao.py    pacote de modelo para o app real
  augmentation/               transformações em landmarks e em vídeo
  coleta/                     gravação, importação, cortes, manifesto, cadastro, qualidade
  app/                        LibrasVision Lab (bancada)
  app_real/                   LibrasVision (usuário final)
  ui/                         sistema de design, componentes, legenda, ícones
scripts/                      linha de comando
tests/                        testes com dados sintéticos
configs/                      config.yaml (documentado) e app_real.yaml
```

## Preparação para mobile

Nada está decidido sobre a plataforma mobile. O desenho atual reduz o custo de uma futura versão:

- três faixas de largura (compacta até 600 px, média e expandida), testadas;
- alvos de toque de 44 px;
- a lógica da legenda (`RollUp`) é pura e testada com relógio simulado, independente do Qt;
- o app real consome um pacote de modelo autocontido (`modelo.keras`, `metadados.json`, `cartao.json`), candidato a conversão para TensorFlow Lite.
