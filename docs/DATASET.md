# Dataset

Situação: **nenhum vídeo real foi coletado ainda.** Esta página descreve a estrutura esperada e o protocolo. O roteiro detalhado para as sessões de gravação está em [protocolo_coleta.md](protocolo_coleta.md).

## Estrutura de pastas

Tudo fica em `data/`, criada automaticamente e **fora do Git**. Os caminhos são configurados na seção `caminhos` de `configs/config.yaml`.

```text
data/
  vocabulario.json              palavras do projeto (criado na primeira execução)
  sinalizadores.csv             cadastro anônimo: ID, mão dominante, data do TCLE
  manifest.csv                  uma linha por vídeo (metadados, qualidade, hashes, status)
  raw_videos/<classe>/          vídeos .mp4 dos takes
    S01_001.mp4
  landmarks/<classe>/           landmarks extraídos, um .npz por vídeo
    S01_prova_001.npz
  landmarks_aug/                landmarks de vídeos degradados (augmentation de vídeo e robustez)
  splits/<protocolo>.json       divisões treino/validação/teste com assinatura do conjunto
  lixeira/                      takes e sinalizadores removidos (reversível)
  backup/                       cópias de tabelas antes de regravar uma leitura com problema
  auditoria.log                 remoções, exclusões e publicações (só IDs)
```

## Convenções de nomes

| Item | Formato | Exemplo |
|---|---|---|
| Sinalizador | `S` + dois dígitos, gerado pela ferramenta | `S03` |
| Classe | minúsculas, sem acento, `_` no lugar de espaço | `sala_de_aula` |
| Vídeo | `data/raw_videos/<classe>/<sinalizador>_<take>.mp4`, take com 3 dígitos | `raw_videos/prova/S03_007.mp4` |
| `video_id` | `<sinalizador>_<classe>_<take>` | `S03_prova_007` |
| Landmarks | `data/landmarks/<classe>/<video_id>.npz` | `landmarks/prova/S03_prova_007.npz` |

Para importar vídeos gravados fora da ferramenta, use `scripts/importar.py` com `--padrao projeto` (pasta por classe, arquivos `S01_7.mp4`) ou `--padrao tcc1` (estrutura do TCC I, com `--mapa`). Sempre simule antes com `--simular`.

## Conteúdo de um `.npz`

Coordenadas cruas do MediaPipe Holistic para **todos** os frames do vídeo: pose (33 pontos com visibilidade), mão esquerda e mão direita (21 pontos cada) e face (468 pontos), flags de presença por parte, instantes de cada frame (segundos), largura e altura do vídeo, e metadados (versão do MediaPipe, parâmetros, hash do vídeo, origem dos landmarks). A seleção de pontos e a normalização são feitas depois, então mudar o vetor de features não exige reextrair.

## Status de um vídeo no manifesto

- `ok`: aprovado; entra no treino.
- `pendente`: aguardando revisão humana (por exemplo, take da rajada com aviso de qualidade).
- `rejeitado`: fora do treino; o arquivo é mantido.

## Protocolo de coleta (resumo)

| Item | Orientação |
|---|---|
| Vocabulário inicial | `biblioteca`, `prova`, `secretaria` e a classe de rejeição `nenhum_sinal` |
| Repetições | meta de 10 takes por classe por sinalizador (`coleta.metas`), configurável por palavra e por pessoa |
| Sinalizadores | meta de mais de 10 pessoas, preferencialmente surdas e fluentes em Libras, todas com TCLE |
| Mínimo para treinar | 2 sinalizadores por classe e 2 classes; com menos, só como ENSAIO |
| Duração do take | 3 s no modo fixo; até 8 s no modo manual |
| Enquadramento | cintura para cima, cabeça inteira, mãos sempre dentro do quadro; guia tracejado na prévia |
| Distância | cerca de 1,0 a 1,5 m; variar entre perto, média e longe |
| Iluminação | de frente ou de lado, nunca só por trás; registrar natural, artificial, mista ou baixa |
| Fundo | neutro na maioria; algumas gravações em sala de aula real |
| Câmera | 640x480 a 30 FPS por padrão; takes abaixo de 24 FPS recebem aviso |
| `nenhum_sinal` | mãos em repouso, coçar o rosto, ajustar óculos, transições, gravado por todos |

Os indicadores de qualidade de cada take (FPS medido, brilho, fração de frames com mão e com pose) são calculados na gravação e registrados no manifesto.

## Modos de gravação

- **Take único:** Espaço grava, Enter aceita, Esc descarta, R regrava. A extração dos landmarks acontece ao parar, antes da revisão.
- **Rajada:** grava N takes seguidos com intervalo curto. A extração de todos acontece só no fim (depois do último take ou de "Parar rajada"), para os takes saírem um atrás do outro. Takes com aviso de qualidade ficam `pendente`. O número de takes é limitado pela memória (`coleta.gravacao.memoria_max_rajada_mb`, padrão 1500 MB).

## Validação

```powershell
.venv\Scripts\python scripts\validar_dados.py      # contagens, detecção de mãos, vídeos problemáticos
.venv\Scripts\python scripts\verificar_manifesto.py # integridade entre manifesto, arquivos e hashes
```
