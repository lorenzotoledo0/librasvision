"""Reconhecimento de sinais em tempo real pela webcam, ou sobre um video gravado.

Uso:
  python scripts/tempo_real.py                         # abre o aplicativo na aba Tempo real (webcam)
  python scripts/tempo_real.py --modelo runs/<pasta>   # escolhe o modelo (padrao: final mais recente)
  python scripts/tempo_real.py --video teste.mp4       # roda o mesmo pipeline sobre um video e mede tempos

Precisa de um modelo treinado com: scripts/treinar.py --nome final --final
Sem modelo, a interface apenas informa o que falta (nenhuma predicao e simulada).
As medicoes de tempo (com o hardware) sao salvas em runs/tempo_real/.
"""

import argparse
from pathlib import Path

import _bootstrap


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--modelo", type=Path, default=None)
    ap.add_argument("--video", type=Path, default=None)
    a = ap.parse_args()
    from libras.config import carregar_config

    cfg = carregar_config(a.config)
    if a.modelo:
        cfg.setdefault("tempo_real", {})["modelo"] = str(a.modelo.resolve())
    if a.video is None:  # webcam: abre o aplicativo na aba Tempo real (inferencia em processo separado)
        from libras.app.janela import main as app_main

        if a.modelo:
            from libras.config import salvar_local

            salvar_local(cfg, {"tempo_real": {"modelo": str(a.modelo.resolve())}})
        return app_main((["--config", a.config] if a.config else []) + ["--aba", "tempo_real"])

    from libras.erros import ErroLibras
    from libras.inferencia import carregar_modelo, localizar_modelo, processar_video, salvar_medicao
    from libras.utils import escrever_csv_atomico

    if not a.video.exists():
        raise ErroLibras(f"Video nao encontrado: {a.video}")
    modelo = carregar_modelo(localizar_modelo(cfg))
    print(f"Modelo: {modelo.pasta}\nProcessando {a.video.name}...")
    linhas, cron = processar_video(cfg, modelo, a.video)
    aceitos = [l["aceito"] for l in linhas if l["aceito"]]
    print("Texto reconhecido: " + (" ".join(aceitos) if aceitos else "(nenhum sinal aceito)"))
    p = salvar_medicao(cfg, cron, modelo, f"video:{a.video.name}")
    if linhas:
        escrever_csv_atomico(p.with_suffix(".csv"), list(linhas[0].keys()), linhas)
    r = cron.resumo()
    for k in ("mediapipe", "preparo", "modelo", "ponta_a_ponta"):
        if k in r:
            print(f"{k:14} mediana {r[k]['mediana_ms']:7.1f} ms | p95 {r[k]['p95_ms']:7.1f} ms")
    print(f"FPS processado: {r['fps_processado']} | memoria max: {r['memoria_mb_max']} MB")
    print(f"Medicao salva em {p}")
    return 0


if __name__ == "__main__":
    _bootstrap.rodar(main)
