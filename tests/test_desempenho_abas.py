"""Item 7: troca de abas. Mede a troca para cada aba (Qt offscreen) e falha acima de 200 ms.

Inclui os dois casos que travavam a interface: sair do Gravar com a camera e a previa ligadas
(MediaPipe falso com encerramento lento) e sair do Tempo real com o processo rodando (servico
falso que demora 2 s para terminar). Dados e videos SINTETICOS.
"""

import os
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")
pytestmark = pytest.mark.desempenho  # mede tempos: sensivel a maquinas compartilhadas (CI)

from conftest import ExtratorFalso, cadastrar, criar_videos_com_landmarks, gerar_video_sintetico  # noqa: E402

LIMITE_MS = 200  # ambiente de desenvolvimento; a meta do item 7 e 100 ms


class ExtratorLento(ExtratorFalso):
    def fechar(self):
        time.sleep(1.5)  # o fechamento do grafo do MediaPipe e lento


def esperar(app, cond, limite=20.0):
    t = time.time()
    while not cond() and time.time() - t < limite:
        app.processEvents()
        time.sleep(0.02)
    return cond()


def trocar(app, janela, chave) -> float:
    janela.ultima_troca = {}
    t = time.perf_counter()
    janela.ir_para(chave)
    app.processEvents()
    assert esperar(app, lambda: janela.ultima_troca.get("para") == chave, 5)
    return max(janela.ultima_troca["total_ms"], (time.perf_counter() - t) * 1000)


def test_troca_de_abas_rapida(cfg, tmp_path, monkeypatch):
    import libras.app.paginas.tempo_real as tr
    import libras.coleta.previa as previa
    from libras.app.base import aguardar_encerramentos
    from libras.app.janela import PAGINAS, JanelaPrincipal
    from libras.ui import tema

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    app.setStyleSheet(tema.folha_de_estilo())
    monkeypatch.setattr(previa, "ExtratorHolistic", lambda *a, **k: ExtratorLento())
    monkeypatch.setenv("LIBRAS_CAMERA_VIDEO", str(gerar_video_sintetico(tmp_path / "cam.mp4", n_frames=60)))
    monkeypatch.setenv("PYTHONPATH", os.path.dirname(__file__) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    monkeypatch.setattr(tr, "MODULO_SERVICO", "servico_lento")
    criar_videos_com_landmarks(cfg, n_sinalizadores=3, por_classe=2)
    cadastrar(cfg)
    w = JanelaPrincipal(cfg)
    w.resize(1360, 860)
    w.show()
    try:
        assert esperar(app, lambda: len(w.paginas) == len(PAGINAS), 30)  # pre-carregamento
        tempos = {}
        trocar(app, w, "sinalizadores")
        for chave, _, _ in PAGINAS:
            tempos[chave] = trocar(app, w, chave)
            esperar(app, lambda: False, 0.3)
        # caso 1: Gravar com camera e previa ligadas -> Configuracoes
        trocar(app, w, "gravar")
        g = w.paginas["gravar"]
        g.ck_lm.setChecked(True)
        g.ligar()
        assert esperar(app, lambda: g.estado == "previa")
        assert esperar(app, lambda: g.previa._t.is_alive() and g.previa.lm is not None)  # MediaPipe falso em uso
        tempos["gravar(camera) -> configuracoes"] = trocar(app, w, "configuracoes")
        # caso 2: Tempo real com processo rodando -> Configuracoes
        trocar(app, w, "tempo_real")
        t = w.paginas["tempo_real"]
        assert esperar(app, lambda: t.proc is not None and t.proc.rodando())
        esperar(app, lambda: False, 0.5)
        tempos["tempo_real(processo) -> configuracoes"] = trocar(app, w, "configuracoes")
        tempos["configuracoes -> tempo_real"] = trocar(app, w, "tempo_real")
        lentos = {k: round(v) for k, v in tempos.items() if v > LIMITE_MS}
        print("\nTempos de troca (ms):", {k: round(v) for k, v in tempos.items()})
        assert not lentos, f"Trocas acima de {LIMITE_MS} ms: {lentos}"
        # o processo anterior termina sozinho, e o novo so comeca depois
        assert esperar(app, lambda: t.proc is not None and t.proc.rodando(), 10)
    finally:
        w.close()
        assert aguardar_encerramentos(10)
        w.deleteLater()
