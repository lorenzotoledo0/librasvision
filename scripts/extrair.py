"""Extrai landmarks (MediaPipe Holistic) dos videos do manifesto, de forma incremental.

So processa videos novos ou alterados; os demais vem do cache (data/landmarks/).
Reprocessa automaticamente se o video, a versao do MediaPipe ou a secao 'mediapipe'
do config mudarem. Mudar features/normalizacao NAO exige reextrair.

Uso:
  python scripts/extrair.py                 # videos 'ok'
  python scripts/extrair.py --pendentes     # inclui os pendentes de revisao
  python scripts/extrair.py --forcar        # reprocessa tudo
  python scripts/extrair.py --processos 2   # paralelo (cuidado com a RAM em maquinas de 4 GB)
"""

import argparse

import _bootstrap


def main():
    _bootstrap.sem_tensorflow()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--pendentes", action="store_true")
    ap.add_argument("--forcar", action="store_true")
    ap.add_argument("--processos", type=int, default=None)
    a = ap.parse_args()
    from libras.config import carregar_config
    from libras.extracao import extrair_lote

    cfg = carregar_config(a.config)
    status = list(cfg["extracao"]["status_incluidos"])
    if a.pendentes and "pendente" not in status:
        status.append("pendente")

    def progresso(i, n, vid, r):
        if r.get("ok"):
            print(
                f"[{i}/{n}] {vid}: {r['n_frames']} frames, mao {r['taxa_mao']:.0%}, corpo {r['taxa_pose']:.0%}, "
                f"{r['segundos']:.1f} s"
            )
        else:
            print(f"[{i}/{n}] {vid}: FALHA - {r.get('motivo')}")

    res = extrair_lote(cfg, status=status, forcar=a.forcar, processos=a.processos, progresso=progresso)
    print(
        f"\nProcessados: {len(res.processados)} | ja em cache: {len(res.em_cache)} | falhas: {len(res.falhas)} "
        f"| tempo: {res.segundos:.0f} s"
    )
    for vid, m in res.falhas.items():
        print(f"  FALHA {vid}: {m}")
    print("Proximo passo: scripts/validar_dados.py")
    return 1 if res.falhas else 0


if __name__ == "__main__":
    _bootstrap.rodar(main)
