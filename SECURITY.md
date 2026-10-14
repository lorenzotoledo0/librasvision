# Política de segurança

## Versões com suporte

| Versão | Suporte |
|---|---|
| 0.1.x | Sim |

## Como reportar uma vulnerabilidade

Não abra issue pública para falhas de segurança ou de privacidade.

Use o reporte privado do GitHub: aba **Security** do repositório > **Report a vulnerability**. A mensagem chega só aos mantenedores.

Inclua:

- descrição do problema e impacto possível;
- passos para reproduzir;
- versão, sistema operacional e versão do Python.

A equipe responde em até 7 dias corridos durante o período letivo.

## Escopo que merece atenção especial

O projeto lida com vídeos de pessoas. São tratados como vulnerabilidade, mesmo sem exploração técnica:

- qualquer caminho em que vídeos, imagens, nomes ou caminhos pessoais saiam do computador ou entrem no Git;
- pacotes de modelo publicados que contenham IDs de sinalizadores, nomes ou imagens;
- logs que registrem dados pessoais.

Veja também [docs/DATA_PRIVACY.md](docs/DATA_PRIVACY.md).
