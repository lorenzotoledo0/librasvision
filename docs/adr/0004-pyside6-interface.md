# ADR 0004: PySide6 (Qt) para as interfaces

- Situação: aceita
- Data: 2026-10-06
- Detalhes: [decisoes.md, D16, D18 e D19](../decisoes.md#d16-interface-única)

## Contexto

As interfaces precisam exibir o vídeo da webcam com baixa latência, rodar tarefas longas fora da thread da interface, ter tema escuro com contraste WCAG 2.1 AA e ficar preparadas para uma futura versão mobile.

## Decisão

Usar PySide6-Essentials 6.11.2, o Qt oficial para Python (licença LGPL). A preparação para mobile está no desenho, não no framework: três faixas de largura (compacta, média e expandida), alvos de toque de 44 px e cores e medidas vindas de um único arquivo de tokens. O Flet foi considerado. A escolha pelo Qt veio do controle fino de vídeo, threads e subprocessos no desktop, que é a plataforma do TCC.

## Consequências

- Uma versão mobile real vai exigir outra camada de interface. O núcleo (`libras`) e o pacote de modelo publicado podem ser reaproveitados.
- O pacote Essentials evita cerca de 70 MB dos Addons, que o projeto não usa.
