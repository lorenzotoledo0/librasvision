"""Relatorio de validacao do dataset antes do treino.

Mostra: contagem por classe e sinalizador, desbalanceamento, taxa de deteccao de maos,
videos que nao geram janelas, e confere os splits salvos (atualizacao e vazamento).
Salva o relatorio em data/relatorios/validacao_<data>.md (+ CSV por video).

Uso:  python scripts/validar_dados.py [--rapido]
Codigo de saida 1 se houver problema que impeca o treino.
"""

import argparse

import _bootstrap


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--rapido", action="store_true", help="nao monta as janelas (so contagens e integridade)")
    a = ap.parse_args()
    from libras.config import carregar_config
    from libras.validacao import gerar_relatorio, salvar_relatorio

    cfg = carregar_config(a.config)
    r = gerar_relatorio(cfg, montar_janelas=not a.rapido)
    print(r.texto())
    if r.linhas and any(l.startswith("Registros: 0 ") for l in r.linhas):
        return 0
    md, csv_p = salvar_relatorio(cfg, r)
    print(f"Relatorio salvo em {md}" + (f" e {csv_p}" if csv_p else ""))
    return 1 if r.problemas else 0


if __name__ == "__main__":
    _bootstrap.rodar(main)
