"""Constantes e textos da decisao do tempo real (D3, D15), num so lugar.

Compartilhado pela bancada, pelo servico de tempo real e pelo app real. Os VALORES efetivos vem do
modelo (metadados.json, secao normalizacao) e do pacote publicado (cartao.json, secao decisao); os
padroes abaixo so valem quando um config antigo nao tem a chave. Nenhum modulo deve repetir estes
numeros: importa daqui (ha teste que procura numeros duplicados).

Nao importa numpy, TensorFlow, MediaPipe nem Qt.
"""

from __future__ import annotations

# Parametros da decisao por janela (D15): secao tempo_real do config da bancada e secao "decisao" do
# cartao do pacote publicado.
PARAMETROS_PADRAO = {
    "limiar_confianca": 0.6,  # abaixo disso nada vira texto
    "suavizacao": 5,  # media movel das ultimas N predicoes
    "predizer_a_cada": 2,  # roda o modelo a cada N frames processados
    "min_fracao_mao": 0.3,  # janela com mao em menos que isso = "nenhuma mao visivel" (sem rodar o modelo)
    "frames_estaveis": 3,  # predicoes iguais seguidas para confirmar um sinal
    "max_frases": 12,  # sinais guardados no texto
}
CHAVES_DECISAO = tuple(PARAMETROS_PADRAO)

# Janela com pose em menos que isso e descartada (sem referencia de ombros), D3.
FRACAO_MINIMA_POSE_PADRAO = 0.5


def parametros(cfg_tr: dict | None) -> dict:
    """So as chaves de decisao, com os padroes preenchidos."""
    cfg_tr = cfg_tr or {}
    return {k: type(v)(cfg_tr.get(k, v)) for k, v in PARAMETROS_PADRAO.items()}


def fracao_minima_pose(meta_modelo: dict | None) -> float:
    """Fracao minima de frames com pose na janela, a mesma que o modelo usou no treino (D3)."""
    return float(((meta_modelo or {}).get("normalizacao") or {}).get("fracao_minima_pose", FRACAO_MINIMA_POSE_PADRAO))


def min_fracao_mao(cfg_tr: dict | None) -> float:
    """Fracao minima de frames com mao na janela para o modelo rodar (D15)."""
    return float((cfg_tr or {}).get("min_fracao_mao", PARAMETROS_PADRAO["min_fracao_mao"]))


# ----------------------------------------------------------------------
# Etapa da decisao (para a pessoa entender por que nada aparece)
# ----------------------------------------------------------------------
TEXTOS_ETAPA = {
    "aquecendo": "Aguardando 1 s de imagem",
    "sem_maos": "Nenhuma mão visível",
    "sem_pose": "Corpo fora do quadro",
    "baixa_confianca": "Confiança abaixo do limiar",
    "nenhum_sinal": "Nenhum sinal (pessoa sem sinalizar)",
    "aguardando": "Aguardando predições iguais",
    "confirmado": "Sinal confirmado",
}


def etapa_decisao(motivo: str, candidato: str | None, contador: int, necessarias: int, aceito: str | None) -> dict:
    """Etapa em que a decisao parou, com o texto para a tela."""
    if aceito:
        return {"etapa": "confirmado", "texto": f"{TEXTOS_ETAPA['confirmado']}: {aceito}"}
    if motivo:
        return {"etapa": motivo, "texto": TEXTOS_ETAPA.get(motivo, motivo)}
    if candidato and contador < necessarias:
        return {
            "etapa": "aguardando",
            "contador": contador,
            "necessarias": necessarias,
            "texto": f"{TEXTOS_ETAPA['aguardando']} ({contador} de {necessarias}): {candidato}",
        }
    if candidato:
        return {"etapa": "confirmado", "texto": f"{TEXTOS_ETAPA['confirmado']}: {candidato}"}
    return {"etapa": "aquecendo", "texto": TEXTOS_ETAPA["aquecendo"]}


def estado_resumido(etapa: str | None) -> str:
    """Uma palavra para a barra tecnica do app real (13.8): aquecendo, sem mãos ou ativo."""
    if etapa in (None, "", "aquecendo"):
        return "aquecendo"
    if etapa in ("sem_maos", "sem_pose"):
        return "sem mãos"
    return "ativo"
