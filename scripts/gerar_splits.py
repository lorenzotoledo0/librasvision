"""Gera e salva a divisao treino/validacao/teste por sinalizador (data/splits/<protocolo>.json).

Protocolos:
  grupos_sinalizador   principal: k particoes, nenhuma pessoa em treino e teste ao mesmo tempo
  loso                 leave-one-signer-out: uma particao por pessoa (mais caro)
  aleatorio_por_video  SECUNDARIO: a mesma pessoa aparece em treino e teste (so para comparacao)

Uso:
  python scripts/gerar_splits.py
  python scripts/gerar_splits.py --protocolo loso
  python scripts/gerar_splits.py --todos
"""

import argparse

import _bootstrap


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument(
        "--classes", nargs="*", default=None, help="classes do treino (padrao: as classes ativas com videos)"
    )
    ap.add_argument(
        "--ensaio",
        action="store_true",
        help="permite classes abaixo dos minimos; saidas marcadas como ENSAIO (nao usar no TCC)",
    )
    ap.add_argument("--protocolo", default=None)
    ap.add_argument("--todos", action="store_true", help="gera os tres protocolos")
    ap.add_argument("--folds", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()
    from libras.config import carregar_config
    from libras.erros import ErroDadosInsuficientes
    from libras.splits import PROTOCOLOS, gerar_splits, resumo_split, salvar_split, videos_elegiveis

    cfg = carregar_config(a.config)
    if a.classes:
        from libras.selecao import fixar_classes

        fixar_classes(cfg, a.classes)
    if a.ensaio:
        cfg["treino"]["ensaio"] = True
    linhas, excluidos = videos_elegiveis(cfg)
    if not linhas:
        motivo = ""
        if excluidos:
            motivo = f" ({len(excluidos)} video(s) 'ok' sem landmarks atualizados: rode scripts/extrair.py)"
        raise ErroDadosInsuficientes(
            "Nenhum video elegivel para dividir" + motivo + ". "
            "E preciso ter videos 'ok' no manifesto com landmarks extraidos."
        )
    if excluidos:
        print(f"Aviso: {len(excluidos)} video(s) 'ok' ficaram de fora (landmarks ausentes/desatualizados).")
    protocolos = PROTOCOLOS if a.todos else [a.protocolo or cfg["avaliacao"]["protocolo"]]
    por_id = {l["video_id"]: l for l in linhas}
    for p in protocolos:
        sp = gerar_splits(cfg, linhas, protocolo=p, n_folds=a.folds, seed=a.seed)
        caminho = salvar_split(cfg, sp)
        print(resumo_split(sp, por_id))
        print(f"Salvo em {caminho}\n")
    return 0


if __name__ == "__main__":
    _bootstrap.rodar(main)
