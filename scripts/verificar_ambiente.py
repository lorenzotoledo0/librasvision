"""Confere se o computador esta pronto: versao do Python, pacotes, Tkinter, ffmpeg e webcam.

Uso:  python scripts/verificar_ambiente.py [--sem-camera]
"""

import argparse
import shutil
import sys

import _bootstrap


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--sem-camera", action="store_true")
    a = ap.parse_args()
    problemas = 0
    v = sys.version_info
    ok_py = (3, 10) <= v[:2] <= (3, 12)
    print(f"Python {v.major}.{v.minor}.{v.micro}: {'OK' if ok_py else 'INCOMPATIVEL (use 3.11)'}")
    problemas += not ok_py
    for mod, nome in (
        ("cv2", "OpenCV"),
        ("mediapipe", "MediaPipe"),
        ("tensorflow", "TensorFlow"),
        ("yaml", "PyYAML"),
        ("PIL", "Pillow"),
        ("sklearn", "scikit-learn"),
        ("PySide6", "PySide6 (interface)"),
    ):
        try:
            m = __import__(mod)
            print(f"{nome}: {getattr(m, '__version__', getattr(m, 'TkVersion', 'ok'))}")
        except Exception as e:  # noqa: BLE001
            print(f"{nome}: FALTANDO ({type(e).__name__}: {e})")
            problemas += 1
    try:
        import mediapipe as mp

        mp.solutions.holistic  # noqa: B018
        print("MediaPipe Holistic (API legada): OK")
    except Exception:  # noqa: BLE001
        print("MediaPipe Holistic: INDISPONIVEL (instale mediapipe==0.10.21)")
        problemas += 1
    print(f"ffmpeg: {shutil.which('ffmpeg') or 'nao encontrado (opcional; so para padronizar videos importados)'}")
    from libras.config import carregar_config

    cfg = carregar_config(a.config)
    print(f"Config: {cfg.arquivo}  | classes: {', '.join(cfg.classes)}")
    if not a.sem_camera:
        import time

        from libras.camera import Captura, abrir_camera
        from libras.erros import ErroCamera

        try:
            cap, info = abrir_camera(cfg["coleta"]["camera"])
            c = Captura(cap, info).iniciar()
            time.sleep(3)
            fps = c.fps_atual
            c.parar()
            print(
                f"Camera {info.indice} via {info.backend}: {info.largura}x{info.altura}, "
                f"{info.fps_declarado:.0f} FPS declarados, {fps:.1f} FPS medidos"
            )
            if fps < cfg["coleta"]["qualidade"]["fps_minimo"]:
                print("  Aviso: FPS medido abaixo do minimo. Melhore a iluminacao (webcams reduzem o FPS no escuro).")
        except ErroCamera as e:
            print(f"Camera: PROBLEMA\n  {e}")
            problemas += 1
    print("\nTudo certo." if not problemas else f"\n{problemas} problema(s) encontrado(s).")
    return 1 if problemas else 0


if __name__ == "__main__":
    _bootstrap.rodar(main)
