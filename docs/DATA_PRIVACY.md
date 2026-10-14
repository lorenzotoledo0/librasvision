# Privacidade e proteção de dados

O LibrasVision trabalha com vídeos de pessoas: rosto, corpo e mãos. Pela Lei Geral de Proteção de Dados Pessoais (LGPD, Lei nº 13.709/2018), imagens de pessoas identificáveis são dados pessoais. Este documento reúne as regras que o projeto segue e as que a equipe precisa seguir fora do software.

> Este documento é uma diretriz técnica da equipe, **não um parecer jurídico**. O texto final do TCLE e a necessidade de submissão a Comitê de Ética em Pesquisa (CEP) devem ser confirmados com a orientação e com a instituição.

## Princípios

1. **Mínimo necessário.** O modelo usa apenas landmarks (coordenadas de pontos do corpo), não imagens. Os vídeos existem para revisão e reextração, e só no computador da coleta.
2. **Consentimento antes de tudo.** Sem TCLE confirmado na ferramenta, ela não grava nem importa vídeos da pessoa.
3. **Pseudonimização.** Sinalizadores são identificados apenas por ID (S01, S02...). A ligação entre o ID e a pessoa existe só no TCLE físico, guardado pela equipe.
4. **Nada sai do computador.** Vídeos e imagens nunca vão para o Git, para a nuvem nem para pacotes de modelo.
5. **Retirada a qualquer momento.** A pessoa pode retirar o consentimento, e a ferramenta tem um fluxo de remoção e exclusão definitiva.

## O que o software garante

| Garantia | Onde |
|---|---|
| Não grava nem importa sem TCLE confirmado | `coleta/sinalizadores.py` (`exigir_tcle`) |
| Cadastro sem nome: só ID, mão dominante e data do TCLE | `coleta/sinalizadores.py` |
| Manifesto guarda só o nome do arquivo importado, nunca o caminho completo | `coleta/importacao.py` |
| Logs e auditoria usam só IDs | `registro.py`, `data/auditoria.log` |
| Pacote de modelo sem IDs, nomes, caminhos, vídeos ou landmarks (lista branca, com teste) | `pacote.py` |
| A interface do app real não cria arquivos com dados da sessão (com teste) | `app_real/` |
| Remoção em duas etapas: lixeira e exclusão definitiva | aba Sinalizadores, `scripts/sinalizadores.py` |
| `data/`, `runs/`, `publicados/`, vídeos, `.npz` e modelos ficam fora do Git | `.gitignore` e hook do pre-commit |

## O que a equipe precisa fazer

### Antes da coleta

- Confirmar com a orientação se o projeto precisa passar por Comitê de Ética em Pesquisa (CEP/Plataforma Brasil) antes de qualquer gravação com pessoas.
- Imprimir o TCLE ([modelo](templates/TCLE_modelo.md)) e colher a assinatura antes da gravação. Anotar no termo o ID gerado pela ferramenta.
- Explicar à pessoa, em Libras, o que será gravado, para quê, por quanto tempo será guardado e como retirar o consentimento. Sempre que possível, com intérprete de Libras presente.

### Armazenamento

- Guardar `data/` apenas no computador da coleta, em disco com acesso restrito (conta de usuário com senha; criptografia de disco, como o BitLocker, quando disponível).
- Não sincronizar `data/` com serviços de nuvem pessoais (OneDrive, Google Drive, Dropbox). **Atenção:** se a pasta do projeto estiver dentro de uma pasta sincronizada, mova o projeto ou exclua `data/` da sincronização.
- Guardar os TCLEs físicos em local fechado, separado do computador.
- Renomear antes de importar os arquivos de vídeo cujo nome tenha o nome da pessoa.

### Descarte

- Ao fim do TCC, decidir com a orientação o prazo de guarda dos vídeos. Recomendação: manter apenas os landmarks (`.npz`) e excluir os vídeos quando não forem mais necessários para revisão.
- Em caso de retirada de consentimento, usar **Remover** e depois **Excluir definitivamente** na aba Sinalizadores. Execuções antigas em `runs/` que usaram os dados da pessoa devem ser removidas à mão.
- A exclusão definitiva é a única operação da ferramenta que apaga arquivos de fato. Ela é registrada em `data/auditoria.log`, só com o ID.

## Validação com a comunidade surda

O sistema é feito para pessoas surdas, e a Libras é uma língua com gramática própria. Recomendações:

- Validar o vocabulário e a forma de cada sinal com intérpretes de Libras e com pessoas surdas fluentes antes da coleta.
- Priorizar sinalizadores surdos fluentes em Libras, com variação de idade, região e estilo de sinalização.
- Apresentar o protótipo à comunidade surda da instituição e registrar os retornos (issue com o rótulo `data`).
- Deixar claro em toda comunicação que o sistema **não substitui um intérprete de Libras**. O app real mostra esse aviso de forma permanente.

## Incidentes

Se um vídeo, imagem ou dado pessoal for parar no Git, num pacote ou num serviço externo:

1. Interromper a distribuição (remover o pacote, tornar o repositório privado se preciso).
2. Reescrever o histórico do Git para remover o arquivo (por exemplo, com `git filter-repo`) e invalidar cópias conhecidas.
3. Avisar a orientação e a pessoa afetada.
4. Registrar o ocorrido e a correção.

Reporte privado: veja o [SECURITY.md](../SECURITY.md).
