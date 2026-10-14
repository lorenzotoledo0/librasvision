# Protocolo de coleta de dados

Guia para a equipe gravar e importar vídeos de sinais de Libras. Siga este roteiro em todas as sessões para que os dados de pessoas e dias diferentes sejam comparáveis.

## 1. Antes da primeira sessão

1. Instale o ambiente (README, seção "Instalação no Windows") e rode `.venv\Scripts\python scripts\verificar_ambiente.py` (o `instalar.bat` já faz isso). Depois, abra a bancada com `LibrasVisionLab.bat`. A câmera deve abrir e o FPS medido deve ficar perto de 30.
2. Imprima os TCLEs. Cada participante assina o termo antes de qualquer gravação.
3. Cadastre o participante na aba **Sinalizadores**. A ferramenta gera um ID (S01, S02...). **Anote esse ID no TCLE físico.** É a única ligação entre a pessoa e os dados. Nunca escreva nomes na ferramenta.
4. Sem TCLE confirmado na ferramenta, o sinalizador nem aparece na lista de gravação, e a importação de vídeos dele é recusada.

## 2. Montagem

| Item | Orientação |
|---|---|
| Câmera | Na altura do peito ou dos ombros, de frente para o sinalizador, imóvel (tripé ou apoio firme). |
| Enquadramento | Da cintura para cima, com cabeça inteira e espaço acima dela. As mãos não podem sair do quadro em nenhum momento do sinal. Use o guia tracejado da pré-visualização: a linha dos ombros perto da marca. |
| Distância | Cerca de 1,0 a 1,5 m. Varie entre sessões (perto, média, longe) e registre no campo Distância. |
| Iluminação | Luz de frente ou de lado, nunca só atrás da pessoa (janela às costas deixa o rosto e as mãos escuros). Webcams reduzem o FPS no escuro: se o FPS na tela cair abaixo de 24, melhore a luz. |
| Fundo | Neutro na maior parte das gravações. Grave também algumas em sala de aula real e registre no campo Fundo. |
| Roupa | Preferir mangas de cor lisa que contrastem com a pele e o fundo. Varie entre sessões. |

A pré-visualização aparece espelhada (como um espelho), para conforto do sinalizador. O arquivo salvo **não** é espelhado. Isso é proposital: espelhar trocaria mão direita por esquerda nos dados.

## 3. Roteiro de gravação (aba Gravar)

1. Ligue a câmera. Escolha Sinalizador, Palavra (a lista mostra quantos takes a pessoa já tem e a meta) e as condições (iluminação, distância, fundo). A mão dominante vem do cadastro.
2. Ative "Mostrar landmarks na prévia" e confira se as duas mãos aparecem desenhadas ao levantá-las. O indicador "mão: SIM" no rodapé deve acender.
3. Para cada take:
   - O sinalizador começa com as mãos em repouso (apoiadas ou baixas).
   - Pressione **Espaço**. Há uma contagem regressiva de 3 s.
   - Durante os 3 s de gravação (borda vermelha), o sinalizador executa o sinal uma vez, de forma natural, e volta ao repouso.
   - Ao parar, a ferramenta extrai os landmarks de todos os frames (barra de progresso) e mostra os avisos de qualidade.
   - Revise o clipe: **Enter** aceita, **Esc** descarta, **R** regrava. Só ao aceitar são salvos o vídeo e os landmarks.
   - Leia os avisos de qualidade antes de aceitar. "mao_pouco_detectada" quase sempre significa mão fora do quadro ou borrada; nesse caso, regrave.
4. Modo rajada: depois que o enquadramento estiver bom, grave várias takes seguidas da mesma classe sem reconfigurar. Takes com aviso ficam com status "pendente" e devem ser revisadas na aba **Revisão**.
5. Se o sinal não couber em 3 s, use o modo Manual (Espaço inicia e para; limite de 8 s) e registre isso em Observação.
6. Palavra nova: digite no campo "Nova palavra" e clique em **Criar palavra**. Ela entra no vocabulário (aba Vocabulário) e já pode ser gravada.
7. Pode gravar além da meta; ela só orienta.

Os arquivos são salvos automaticamente em `data/raw_videos/<classe>/<ID>_<take>.mp4`, com numeração automática, sem nunca sobrescrever, e os landmarks em `data/landmarks/<classe>/<video_id>.npz`.

## 4. Variação: o que mudar entre takes e sessões

O modelo aprende a reconhecer o que varia nos dados. Se todos os vídeos forem iguais, ele falha na primeira situação diferente. Entre takes e sessões, varie:

- velocidade de execução (mais lenta, natural, mais rápida);
- pequenos deslocamentos de posição e de ângulo em relação à câmera (até uns 15 graus);
- distância da câmera;
- iluminação (natural, artificial, mista; algumas com luz mais fraca);
- roupa e fundo, em sessões diferentes.

Não varie o sinal em si: cada take deve ser o sinal correto, como um sinalizador fluente o faria.

## 5. A classe "nenhum_sinal"

Grave para todos os sinalizadores, como se fosse mais um sinal, cerca de 3 s cada:

- mãos em repouso, paradas;
- coçar o rosto, ajeitar o cabelo ou os óculos;
- gestos de conversa que não são os sinais do vocabulário;
- começo e fim de movimentos, transições;
- pessoa entrando ou saindo levemente do quadro.

Sem essa classe, o sistema em tempo real sempre "acha" um dos sinais, mesmo quando ninguém está sinalizando.

## 6. Metas

A meta de cada palavra fica na aba **Vocabulário** (padrão da seção `coleta.metas` do `config.yaml`) e pode ser diferente para um sinalizador específico. A aba **Início** e a lista de palavras da aba Gravar mostram a contagem.

| Meta | Valor padrão |
|---|---|
| Takes "ok" por classe e por sinalizador | 10 |
| Sinalizadores | 10 ou mais |
| Total com 4 classes e 10 sinalizadores | 400 vídeos ou mais |

Prioridade: **mais sinalizadores vale mais do que mais takes da mesma pessoa.** Se o tempo for curto, é melhor 15 pessoas com 6 takes por classe do que 5 pessoas com 18.

## 7. Por que mais sinalizadores pesa mais do que data augmentation

A avaliação principal testa o modelo em pessoas que ele nunca viu no treino. O que muda de uma pessoa para outra (tamanho e formato das mãos, ritmo, amplitude, estilo de sinalização) é justamente o que o modelo precisa aprender a ignorar. A data augmentation (rotações, escala, ruído, variação de velocidade) simula parte dessas diferenças de forma geométrica, mas não cria estilos de sinalização novos. Ela ajuda a aproveitar melhor os dados existentes; não substitui pessoas diferentes. Por isso a coleta deve priorizar diversidade de sinalizadores e de condições reais.

Sobre o número mínimo: a avaliação independente de sinalizador exige no mínimo 2 sinalizadores e é pouco informativa com menos de 3. Com 5 ou mais ela começa a ser confiável; com a meta de 10 ou mais, permite validação cruzada em 5 partições com 2 ou mais pessoas inéditas em cada teste.

## 8. Importar vídeos já gravados (aba Gravar > Importar vídeos existentes)

- Selecione arquivos soltos ou uma pasta inteira.
- Classe: pelo nome da pasta (`.../biblioteca/arquivo.mp4`), pelo nome do arquivo, ou escolhida no formulário.
- Sinalizador: pelo nome do arquivo (padrão `projeto`: `S01_1.mp4`) ou escolhido no formulário. Para a estrutura antiga do TCC I (`dataset/sinais/<classe>/usuario1_01.mp4`), use o padrão `tcc1` e informe o mapa, por exemplo `usuario1=S03 usuario2=S04`, depois de cadastrar essas pessoas com TCLE.
- Use **Simular** primeiro: mostra o que seria importado, ignorado ou rejeitado, sem copiar nada.
- O arquivo original nunca é alterado. Vídeos repetidos (mesmo conteúdo) são ignorados, então importar a mesma pasta duas vezes não duplica nada.
- Rejeição automática: arquivo que não abre, sem FPS, menor que 0,8 s, maior que 15 s, FPS abaixo de 15, ou pessoa não detectada. Avisos (duração fora de 1,5 a 4 s, resolução baixa, escuro, mão pouco detectada) deixam o vídeo como "pendente" para revisão.
- Os landmarks de cada vídeo importado são extraídos na própria importação.
- Cada importação gera um relatório em `data/relatorios_importacao/`.
- Renomeie arquivos que contenham nomes de pessoas antes de importar.

## 9. Revisão (aba Revisão)

- Filtre por "pendente", assista cada vídeo (com o esqueleto, se quiser) e decida: **Aprovar** (vira "ok"), **Rejeitar** (fica registrado, mas fora do treino) ou **Excluir** (vai para `data/lixeira/`).
- Corte: se o take tem repouso longo no começo ou no fim, arraste as alças da linha do tempo (ou pare no quadro certo e aperte I para início e O para fim). A curva "movimento das mãos" ajuda a ver onde o sinal começa e termina. Se a pessoa fez o sinal duas vezes, pare entre as duas e aperte S (dividir). Ctrl+Z desfaz. Nada é gravado até **Salvar corte**. Cada trecho precisa de pelo menos 30 quadros (a alça para no limite). O original vai para `data/lixeira/cortes/`. Não é possível remover um trecho do meio.
- Só vídeos com status "ok" entram no treino.
- Use **Verificar integridade** (aba Dataset) ao fim de cada sessão; a aba Início também mostra os alertas.
- Não edite o `manifest.csv` no Excel com o aplicativo aberto. Se editar e salvar, a ferramenta ainda lê o arquivo (inclusive com `;`), ignora linhas inválidas apontando o número da linha e guarda uma cópia em `data/backup/` antes de regravar.

## 10. Depois de cada sessão

1. Revise os pendentes (seção 9).
2. Aba **Dataset** > **Extrair landmarks**: só é necessário para vídeos antigos ou importados sem landmarks; os gravados e importados já têm.
3. Aba **Dataset** > **Relatório de validação**: confira a tabela por classe e sinalizador e a lista "Vídeos que merecem revisão". Vídeos com mão detectada em poucos frames ou sem nenhuma janela válida devem ser assistidos e, se for o caso, regravados.
4. Leve o relatório salvo em `data/relatorios/` para a próxima reunião da equipe.

## 11. Retirada de consentimento

Se um participante pedir para sair, na aba **Sinalizadores**:

1. **Remover (lixeira)**: digite o ID para confirmar. Vídeos e landmarks dele vão para `data/lixeira/<ID>_<data>/` e saem do manifesto e do cadastro.
2. **Excluir definitivamente**: digite o ID e a palavra EXCLUIR. Apaga de fato o que estava na lixeira dele. Não há como desfazer.

Pela linha de comando: `scripts\sinalizadores.py remover S03` e depois `scripts\sinalizadores.py excluir-definitivamente S03`.

As duas etapas invalidam os splits (é preciso gerar de novo) e ficam registradas em `data/auditoria.log`, só com o ID. Modelos e execuções já existentes em `runs/` não são apagados automaticamente: devem ser retreinados antes de qualquer nova divulgação de resultados, e as pastas antigas removidas pela equipe se necessário.

## 12. Backup

`data/` não vai para o Git. Faça cópia de `data/` (vídeos, landmarks, `manifest.csv`, `sinalizadores.csv`, `vocabulario.json`) ao fim de cada sessão em um local seguro e restrito à equipe.
