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
REPETICOES = 3  # mediana de varias trocas: uma troca isolada oscila com a carga da maquina


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


def aguardar_encerramentos_sem_bloquear() -> bool:
    from libras.app.base import ha_encerramento_pendente

    return not ha_encerramento_pendente()


def mediana(valores: list[float]) -> float:
    v = sorted(valores)
    return v[len(v) // 2]


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
        amostras: dict[str, list[float]] = {}
        for _ in range(REPETICOES):
            trocar(app, w, "sinalizadores")
            for chave, _, _ in PAGINAS:
                amostras.setdefault(chave, []).append(trocar(app, w, chave))
                esperar(app, lambda: False, 0.3)
        g = w.paginas["gravar"]
        t = w.paginas["tempo_real"]
        for _ in range(REPETICOES):
            # caso 1: Gravar com camera e previa ligadas -> Configuracoes
            trocar(app, w, "gravar")
            g.ck_lm.setChecked(True)
            assert esperar(app, aguardar_encerramentos_sem_bloquear, 15)  # camera anterior ja fechou
            g.ligar()
            assert esperar(app, lambda: g.estado == "previa")
            assert esperar(app, lambda: g.previa._t.is_alive() and g.previa.lm is not None)  # MediaPipe falso
            amostras.setdefault("gravar(camera) -> configuracoes", []).append(trocar(app, w, "configuracoes"))
            # caso 2: Tempo real com processo rodando -> Configuracoes
            trocar(app, w, "tempo_real")
            assert esperar(app, lambda: t.proc is not None and t.proc.rodando(), 15)
            esperar(app, lambda: False, 0.5)
            amostras.setdefault("tempo_real(processo) -> configuracoes", []).append(trocar(app, w, "configuracoes"))
            amostras.setdefault("configuracoes -> tempo_real", []).append(trocar(app, w, "tempo_real"))
            trocar(app, w, "configuracoes")
        tempos = {k: mediana(v) for k, v in amostras.items()}
        lentos = {k: round(v) for k, v in tempos.items() if v > LIMITE_MS}
        print(f"\nTempos de troca (ms, mediana de {REPETICOES}):", {k: round(v) for k, v in tempos.items()})
        assert not lentos, f"Trocas acima de {LIMITE_MS} ms (mediana de {REPETICOES}): {lentos}"
        # o processo anterior termina sozinho, e o novo so comeca depois
        trocar(app, w, "tempo_real")
        assert esperar(app, lambda: t.proc is not None and t.proc.rodando(), 10)
    finally:
        w.close()
        assert aguardar_encerramentos(10)
        w.deleteLater()
