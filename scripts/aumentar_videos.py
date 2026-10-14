"""Augmentation em nivel de video (familia D): degrada os videos e reextrai os landmarks.

Variantes: escuro, claro, desfoque, ruido_imagem, rotacao_zoom (parametros em
src/libras/augmentation/video.py). Guarda so landmarks em data/landmarks_aug/.
Custa o mesmo que uma extracao completa POR VARIANTE: rode com tempo.

Uso:
  python scripts/aumentar_videos.py                          # variantes do config
  python scripts/aumentar_videos.py --variantes escuro claro
  python scripts/aumentar_videos.py --processos 2
"""

import argparse

import _bootstrap


def main():
    _bootstrap.sem_tensorflow()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--variantes", nargs="*", default=None)
    ap.add_argument("--forcar", action="store_true")
    ap.add_argument("--processos", type=int, default=None)
    a = ap.parse_args()
    from libras.augmentation.video import aumentar_lote
    from libras.config import carregar_config

    cfg = carregar_config(a.config)
    variantes = a.variantes or cfg["augmentation"]["video"]["variantes"]

    def prog(i, n, vid, v, r):
        txt = f"mao {r['taxa_mao']:.0%}, corpo {r['taxa_pose']:.0%}" if r.get("ok") else f"FALHA {r.get('motivo')}"
        print(f"[{i}/{n}] {vid} ({v}): {txt}")

    res = aumentar_lote(
        cfg, variantes, forcar=a.forcar, processos=a.processos or cfg["extracao"].get("processos", 1), progresso=prog
    )
    print(f"\nProcessados: {res['processados']} | em cache: {res['em_cache']} | falhas: {len(res['falhas'])}")
    return 1 if res["falhas"] else 0


if __name__ == "__main__":
    _bootstrap.rodar(main)
