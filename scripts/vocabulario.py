"""Gerencia o vocabulario (data/vocabulario.json).

Exemplos:
  python scripts/vocabulario.py listar
  python scripts/vocabulario.py adicionar "Sala de aula" --meta 12
  python scripts/vocabulario.py editar sala_de_aula --rotulo "sala de aula" --meta 15
  python scripts/vocabulario.py meta prova --sinalizador S03 15      (excecao de meta para um sinalizador)
  python scripts/vocabulario.py arquivar prova  |  reativar prova
  python scripts/vocabulario.py remover prova                         (vazia: sai; com dados: vai para a lixeira)
"""

import argparse

import _bootstrap


def main():
    _bootstrap.sem_tensorflow()
    comum = argparse.ArgumentParser(add_help=False)
    comum.add_argument("--config", default=None)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("listar", parents=[comum])
    a_ = sub.add_parser("adicionar", parents=[comum])
    a_.add_argument("rotulo")
    a_.add_argument("--meta", type=int, default=None)
    a_.add_argument("--tipo", default="sinal", choices=["sinal", "nenhum_sinal"])
    e = sub.add_parser("editar", parents=[comum])
    e.add_argument("slug")
    e.add_argument("--rotulo", default=None)
    e.add_argument("--meta", type=int, default=None)
    m = sub.add_parser("meta", parents=[comum])
    m.add_argument("slug")
    m.add_argument("--sinalizador", required=True)
    m.add_argument("valor", help="numero, ou 'padrao' para remover a excecao")
    for nome in ("arquivar", "reativar", "remover"):
        sub.add_parser(nome, parents=[comum]).add_argument("slug")
    a = ap.parse_args()
    from libras.config import carregar_config
    from libras.vocabulario import Vocabulario

    cfg = carregar_config(a.config)
    v = Vocabulario(cfg)
    if a.cmd == "listar":
        for c in v.todas():
            exc = ", ".join(f"{k}={n}" for k, n in c.metas_sinalizador.items())
            print(
                f"{c.slug:20} {c.rotulo:20} {c.tipo:13} meta {c.meta_por_sinalizador:3}  "
                f"{'ativa' if c.ativa else 'ARQUIVADA'}" + (f"  excecoes: {exc}" if exc else "")
            )
    elif a.cmd == "adicionar":
        c = v.adicionar(a.rotulo, a.tipo, a.meta)
        print(f"Classe criada: {c.slug} ({c.rotulo}), meta {c.meta_por_sinalizador}.")
    elif a.cmd == "editar":
        v.editar(a.slug, a.rotulo, a.meta)
        print("Ok.")
    elif a.cmd == "meta":
        v.definir_meta_sinalizador(a.slug, a.sinalizador, None if a.valor == "padrao" else int(a.valor))
        print("Ok.")
    elif a.cmd in ("arquivar", "reativar"):
        v.arquivar(a.slug, ativa=(a.cmd == "reativar"))
        print("Ok.")
    elif a.cmd == "remover":
        from libras.coleta.gestao import remover_classe

        conf = input(f"Remover a classe {a.slug}? Se houver videos, eles vao para a lixeira. Digite {a.slug}: ")
        r = remover_classe(cfg, a.slug, conf)
        print(
            f"Classe removida. Videos movidos: {r.videos}."
            + (f" Lixeira: {r.pasta_lixeira}" if r.pasta_lixeira else "")
        )
        for av in r.avisos:
            print("AVISO:", av)
    return 0


if __name__ == "__main__":
    _bootstrap.rodar(main)
