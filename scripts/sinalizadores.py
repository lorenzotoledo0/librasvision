"""Cadastro anonimo de sinalizadores e registro de TCLE (linha de comando).

Exemplos:
  python scripts/sinalizadores.py listar
  python scripts/sinalizadores.py cadastrar --mao direita --tcle 2026-10-10
  python scripts/sinalizadores.py cadastrar --mao esquerda            (sem TCLE: nao pode gravar ainda)
  python scripts/sinalizadores.py confirmar-tcle S03 --data 2026-10-12
  python scripts/sinalizadores.py revogar-tcle S03
  python scripts/sinalizadores.py remover S03                  (etapa 1: tudo vai para data/lixeira/S03_<data>/)
  python scripts/sinalizadores.py excluir-definitivamente S03  (etapa 2, retirada de consentimento: apaga a lixeira)

Nunca coloque nomes em --obs. A ligacao ID <-> pessoa fica apenas no TCLE fisico.
"""

import argparse

import _bootstrap


def main():
    comum = argparse.ArgumentParser(add_help=False)
    comum.add_argument("--config", default=None)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("listar", parents=[comum])
    c = sub.add_parser("cadastrar", parents=[comum])
    c.add_argument("--mao", required=True, help="mao dominante: direita | esquerda")
    c.add_argument("--tcle", default="", help="data de assinatura do TCLE (AAAA-MM-DD); omita se ainda nao assinou")
    c.add_argument("--obs", default="", help="observacao SEM nomes")
    t = sub.add_parser("confirmar-tcle", parents=[comum])
    t.add_argument("signer_id")
    t.add_argument("--data", required=True)
    r = sub.add_parser("revogar-tcle", parents=[comum])
    r.add_argument("signer_id")
    d = sub.add_parser("remover", parents=[comum])
    d.add_argument("signer_id")
    e = sub.add_parser("excluir-definitivamente", parents=[comum])
    e.add_argument("signer_id")
    a = ap.parse_args()

    from libras.coleta.sinalizadores import CadastroSinalizadores
    from libras.config import carregar_config

    cad = CadastroSinalizadores(carregar_config(a.config))
    if a.cmd == "listar":
        itens = cad.listar()
        if not itens:
            print("Nenhum sinalizador cadastrado ainda.")
        for s in itens:
            print(
                f"{s.signer_id}  TCLE={'sim' if s.tcle_confirmado else 'NAO':3}  data={s.data_tcle or '-':10}  "
                f"mao={s.mao_dominante:8}  {s.observacoes}"
            )
    elif a.cmd == "cadastrar":
        s = cad.cadastrar(a.mao, bool(a.tcle), a.tcle, a.obs)
        print(
            f"Cadastrado: {s.signer_id} (TCLE {'confirmado' if s.tcle_confirmado else 'PENDENTE'}). "
            "Anote este ID no TCLE fisico."
        )
    elif a.cmd == "confirmar-tcle":
        s = cad.confirmar_tcle(a.signer_id, a.data)
        print(f"TCLE de {s.signer_id} confirmado em {s.data_tcle}.")
    elif a.cmd == "remover":
        from libras.coleta.gestao import remover_sinalizador

        conf = input(f"Remover {a.signer_id}: videos e landmarks vao para a lixeira. Digite o ID para confirmar: ")
        r = remover_sinalizador(carregar_config(a.config), a.signer_id, conf)
        print(f"{r.videos} video(s) e {r.landmarks} arquivo(s) de landmarks movidos para {r.pasta_lixeira}.")
        for av in r.avisos:
            print("AVISO:", av)
    elif a.cmd == "excluir-definitivamente":
        from libras.coleta.gestao import PALAVRA_EXCLUSAO, excluir_definitivamente

        print(f"ATENCAO: apaga DE VEZ tudo o que estiver na lixeira em nome de {a.signer_id}. Nao ha como desfazer.")
        c1 = input("Digite o ID: ")
        c2 = input(f"Digite {PALAVRA_EXCLUSAO} para confirmar: ")
        n = excluir_definitivamente(carregar_config(a.config), a.signer_id, c1, c2)
        print(f"{n} arquivo(s) apagado(s) definitivamente.")
    elif a.cmd == "revogar-tcle":
        s = cad.revogar_tcle(a.signer_id)
        print(
            f"TCLE de {s.signer_id} revogado. Nenhuma nova gravacao sera aceita. "
            "Para remover os videos ja coletados, veja docs/protocolo_coleta.md (retirada de consentimento)."
        )


if __name__ == "__main__":
    _bootstrap.rodar(main)
