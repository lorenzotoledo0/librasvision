"""LibrasVision (app real, para o usuario final): so camera e legenda (item 13, D19).

Sem coleta, dados, treino nem diagnostico, nem por linha de comando. Recebe da bancada apenas pacotes
de modelo publicados (libras.pacote). Nao importa coleta, dataset, treino, ablacao, augmentation nem a
interface da bancada (libras.app); ha teste que falha se isso for violado. TensorFlow e MediaPipe rodam
so no subprocesso de tempo real (D15).
"""
