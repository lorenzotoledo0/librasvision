"""Excecoes do projeto. Toda falha prevista vira ErroLibras com mensagem acionavel em portugues."""


class ErroLibras(Exception):
    """Erro esperado, com mensagem dirigida ao usuario (o que faltou e como resolver)."""


class ErroConfig(ErroLibras):
    pass


class ErroDadosInsuficientes(ErroLibras):
    """Levantado quando uma etapa precisa de dados reais que ainda nao existem."""


class ErroCamera(ErroLibras):
    pass


class ErroManifesto(ErroLibras):
    pass


class ErroConsentimento(ErroLibras):
    """Sinalizador sem TCLE confirmado: nada pode ser gravado ou importado para ele."""
