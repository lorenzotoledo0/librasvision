"""Verifica a integridade do dataset: registros sem arquivo, arquivos sem registro,
sinalizadores sem TCLE, classes fora do config, duplicatas. Tambem mostra o progresso.

Uso:  python scripts/verificar_manifesto.py
Saida com codigo 1 se houver problemas (util para conferir antes de treinar).
"""

import argparse

import _bootstrap


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    from libras.coleta.manifesto import Manifesto
    from libras.coleta.sinalizadores import CadastroSinalizadores
    from libras.config import carregar_config

    cfg = carregar_config(a.config)
    m, cad = Manifesto(cfg), CadastroSinalizadores(cfg)
    print(f"Sinalizadores cadastrados: {len(cad.ids())} ({len(cad.ids_com_tcle())} com TCLE)")
    print(f"Registros no manifesto: {len(m)}")
    if len(m) == 0:
        print("Ainda nao ha videos. Use scripts/coletar.py (gravar) ou scripts/importar.py.")
    else:
        from collections import Counter

        st = Counter(l["status"] for l in m.linhas)
        print("Por status: " + ", ".join(f"{k}={v}" for k, v in sorted(st.items())))
        ok = m.contagens(("ok",))
        meta = cfg["coleta"]["metas"]["takes_por_classe_por_sinalizador"]
        print(f"\nTakes 'ok' por classe x sinalizador (meta {meta}):")
        ids = sorted({s for (_, s) in ok} | set(cad.ids()))
        print("classe".ljust(16) + "".join(s.rjust(6) for s in ids))
        for c in cfg.classes:
            print(c.ljust(16) + "".join(str(ok.get((c, s), 0)).rjust(6) for s in ids))
    r = m.verificar_integridade(cad.ids(), cad.ids_com_tcle())
    print("\n" + r.texto())
    return 0 if r.ok else 1


if __name__ == "__main__":
    _bootstrap.rodar(main)
