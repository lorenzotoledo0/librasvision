# Como contribuir

Obrigado pelo interesse. Este é um projeto acadêmico (TCC II, UDF) com regras de privacidade e de método que valem para qualquer contribuição.

## Antes de começar

1. Leia o [README](README.md), a [arquitetura](docs/ARCHITECTURE.md) e as [decisões de projeto](docs/decisoes.md).
2. Leia as [diretrizes de privacidade](docs/DATA_PRIVACY.md). **Nunca** envie vídeos, imagens, landmarks de pessoas reais, nomes ou caminhos pessoais em commits, issues ou PRs.
3. Procure uma issue existente ou abra uma antes de mudanças grandes.

## Ambiente

Windows (PowerShell):

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pip install -e ".[dev]"
.venv\Scripts\pre-commit install
```

Linux/macOS:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/pre-commit install
```

## Fluxo de branches

- `main` é protegida: só recebe mudanças por Pull Request com o CI verde.
- Crie uma branch a partir da `main` com prefixo do tipo da mudança: `feat/rajada-por-tempo`, `fix/camera-dshow`, `docs/protocolo`.
- Um PR resolve um assunto. Prefira PRs pequenos.

## Commits

Use [Conventional Commits](https://www.conventionalcommits.org/pt-br/v1.0.0/), com a mensagem em inglês:

```text
feat(collect): defer landmark extraction to the end of a burst
fix(camera): wait for release before reopening on Windows
docs(dataset): document folder naming convention
test(splits): cover signer leakage across folds
```

Tipos: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `chore`, `ci`, `style`.

## Idioma do código

O código existente usa nomes em português (`normalizacao`, `extracao`, `gravar`), alinhados ao texto do TCC. Mantenha esse padrão nos módulos existentes: misturar idiomas dentro de um módulo dificulta a leitura. Commits, nomes de branch e mensagens de commit ficam em inglês.

## Regras que não podem ser quebradas

Resumo das regras em [CLAUDE.md](CLAUDE.md) e em [docs/decisoes.md](docs/decisoes.md):

1. Um único módulo de normalização para treino, avaliação e tempo real.
2. Sem vazamento de dados: todos os vídeos de um sinalizador ficam do mesmo lado de qualquer divisão. Augmentation só no treino.
3. Nenhum frame é espelhado antes do MediaPipe.
4. Escolhas de modelo são feitas pela validação, nunca pelo teste.
5. Ensaios ficam marcados como ENSAIO e não viram número do TCC.
6. Nunca simular predição.
7. Privacidade: sinalizadores só por ID; sem TCLE, nada é gravado.
8. Nenhuma tarefa longa na thread da interface; o processo da interface nunca importa TensorFlow.
9. Sem emojis na interface, no código e nos logs; nenhum retorno depende de som.
10. Cores e medidas visuais vêm dos tokens do sistema de design.

Toda decisão técnica nova ou alterada é registrada em `docs/decisoes.md`, com justificativa.

## Antes de abrir o PR

```powershell
.venv\Scripts\python -m ruff check src scripts tests
.venv\Scripts\python -m ruff format --check src scripts tests
.venv\Scripts\python -m mypy
.venv\Scripts\python -m pytest
```

- Testes usam apenas dados sintéticos, identificados como tal.
- Não declare resultados de desempenho que não foram medidos. Onde não houver número, escreva "a ser medido".
- Atualize o [CHANGELOG.md](CHANGELOG.md) na seção "Não publicado".

## Contribuição de dados

Dados de pessoas **não** são aceitos por PR. Se você quer contribuir com gravações, abra uma issue com o modelo "Contribuição de dados" para combinar o processo de consentimento (TCLE) com a equipe.
