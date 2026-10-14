# ADR 0006: pip com requirements.txt fixado e pyproject para metadados

- Situação: aceita
- Data: 2026-10-14

## Contexto

A equipe usa Windows e precisa de uma instalação de dois cliques. As versões de MediaPipe, TensorFlow, NumPy e protobuf são travadas entre si. Ferramentas como `uv` e Poetry acrescentariam mais um passo de instalação para a equipe.

## Decisão

- O `requirements.txt` é a fonte da instalação da equipe, com versões exatas e o motivo de cada uma. O `instalar.bat` usa o pip do próprio Python 3.11.
- O `pyproject.toml` descreve o pacote (metadados, scripts `librasvision` e `librasvision-lab`, extra `dev`) e configura ruff, mypy e pytest.
- A verificação de tipos com mypy é gradual. Começa pelos módulos do núcleo sem interface (normalização, splits, decisão, métricas, estatística, vocabulário e configuração), que são os que mais pesam no resultado do TCC.
- O Dependabot ignora as dependências travadas. Atualizá-las exige verificação manual.

## Consequências

- Quem preferir `uv` pode usar `uv venv --python 3.11` e `uv pip install -r requirements.txt`, com o mesmo resultado. Não é obrigatório.
- A lista de dependências fica em dois lugares (requirements e pyproject). Ao mudar uma versão, é preciso mudar nos dois.
