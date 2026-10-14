"""13.6, 13.8 e 13.9-B: o servico de tempo real no modo do app real (pacote publicado, sem disco), os
contadores de quadros e o selo de privacidade (sem rede, sem gravar imagem). Modelo minimo treinado com
dados SINTETICOS; videos SINTETICOS. Nada aqui e numero do TCC."""

import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from conftest import ExtratorFalso, gerar_video_sintetico
from test_teste_ao_vivo import ExtratorPlausivel, modelo_minimo  # noqa: F401 - fixture compartilhada

RAIZ = Path(__file__).resolve().parents[1]
EXT_MIDIA = {".png", ".jpg", ".jpeg", ".bmp", ".mp4", ".avi", ".mov", ".mkv", ".webm", ".npz"}


class ExtratorLento(ExtratorPlausivel):
    """SINTETICO: MediaPipe falso e artificialmente lento (90 ms por frame), para haver descarte."""

    def processar(self, frame):
        time.sleep(0.09)
        return super().processar(frame)


class CameraContada:
    """Envolve a camera de arquivo e conta, de forma independente, os frames entregues."""

    def __init__(self, cam):
        self.cam, self.lidos = cam, 0

    def read(self):
        ok, f = self.cam.read()
        self.lidos += int(bool(ok))
        return ok, f

    def release(self):
        self.cam.release()


def _quadros(cfg, video, extrator, duracao=4.0, camera=None, **kw):
    from libras.camera import abrir_arquivo_como_camera
    from libras.tempo_real_servico import Decodificador, executar_servico

    saida = io.BytesIO()

    def entrada():
        time.sleep(duracao)
        yield "sair\n"

    def abrir(_c):
        cam, info = abrir_arquivo_como_camera(video)
        if camera is not None:
            camera.append(CameraContada(cam))
            return camera[-1], info
        return cam, info

    executar_servico(cfg, saida, entrada(), abrir_camera=abrir, extrator=extrator, **kw)
    return [m for m, _ in Decodificador().alimentar(saida.getvalue())]


def test_contadores_de_quadros_com_processamento_lento(modelo_minimo, tmp_path):  # noqa: F811
    cfg, _ = modelo_minimo
    video = gerar_video_sintetico(tmp_path / "v.mp4", n_frames=60)
    cams = []
    msgs = _quadros(cfg, video, ExtratorLento(), duracao=4.0, camera=cams)
    infos = [m for m in msgs if m["tipo"] == "info" and "camera" in m]
    assert infos and infos[0]["camera"]["largura"] == 640
    qs = [m for m in msgs if m["tipo"] == "quadro"]
    assert len(qs) > 10
    for q in qs:
        c = q["quadros"]
        assert c["capturados"] == c["processados"] + c["descartados"]
    assert qs[-1]["quadros"]["descartados"] > 0  # 30 FPS de video, cerca de 11 processados por segundo
    # contagem independente: frames que a camera entregou (menos os que chegaram depois do ultimo processado)
    # (o ultimo quadro enviado pode ser de alguns frames antes do fim: o envio e limitado a 20 por segundo)
    assert 0 <= cams[0].lidos - qs[-1]["quadros"]["capturados"] <= 8
    assert qs[-1]["quadros"]["descartados"] / qs[-1]["quadros"]["capturados"] > 0.4  # 90 ms por frame a 30 FPS
    # no comeco nada e contado como pulado por causa da carga do MediaPipe (a camera so comeca depois)
    assert qs[0]["quadros"]["capturados"] <= 5
    assert "ponta_a_ponta" in qs[-1]["tempos_recentes"] and "p95_ms" in qs[-1]["tempos_recentes"]["mediapipe"]
    enq = qs[-1]["enquadramento"]
    assert set(enq) >= {"pessoa", "ombros", "mao_cortada", "brilho", "pouca_luz", "cabeca", "ms"}
    custos = [q["enquadramento"]["ms"] for q in qs]
    assert np.median(custos) < 1.0, f"brilho, bordas e ombros custam {np.median(custos):.3f} ms por frame"


def test_sem_disco_nao_grava_medicao(modelo_minimo, tmp_path):  # noqa: F811
    cfg, _ = modelo_minimo
    video = gerar_video_sintetico(tmp_path / "v.mp4", n_frames=60)
    pasta = cfg.caminho("runs") / "tempo_real"
    antes = set(pasta.glob("*")) if pasta.exists() else set()
    _quadros(cfg, video, ExtratorFalso(), duracao=2.5, gravar_em_disco=False)
    depois = set(pasta.glob("*")) if pasta.exists() else set()
    assert depois == antes


def _arquivos(raiz: Path) -> set:
    return {p for p in raiz.rglob("*") if p.is_file() and ".git" not in p.parts and "__pycache__" not in p.parts}


def test_servico_real_sem_rede_e_sem_gravar_imagem(modelo_minimo, tmp_path):  # noqa: F811
    """Selo de privacidade (13.9-B): o servico REAL (MediaPipe e TensorFlow de verdade), no modo do app
    real, com toda conexao de rede bloqueada e as pastas monitoradas, numa sessao com video SINTETICO."""
    from libras.publicacao import publicar

    cfg, pasta_modelo = modelo_minimo
    pacote = publicar(cfg, pasta_modelo)
    app = tmp_path / "app"
    (app / "configs").mkdir(parents=True)
    (app / "configs" / "app_real.yaml").write_text(f"pasta_modelos: {pacote.parent.as_posix()}\n", encoding="utf-8")
    video = gerar_video_sintetico(tmp_path / "entrada" / "v.mp4", n_frames=60)
    antes_app, antes_repo = _arquivos(tmp_path), _arquivos(RAIZ)
    rede, mods, sondas = tmp_path / "rede.txt", tmp_path / "modulos.json", tmp_path / "sondas.txt"
    env = {
        **os.environ,
        "PYTHONPATH": str(RAIZ / "src"),
        "LIBRAS_TESTE_REDE": str(rede),
        "LIBRAS_TESTE_SONDAS": str(sondas),
        "LIBRAS_TESTE_MODULOS": str(mods),
        "TF_CPP_MIN_LOG_LEVEL": "3",
    }
    p = subprocess.Popen(
        [
            sys.executable,
            str(RAIZ / "tests" / "servico_sem_rede.py"),
            "--pacote",
            str(pacote),
            "--sem-disco",
            "--config-real",
            str(app / "configs" / "app_real.yaml"),
            "--video",
            str(video),
        ],
        cwd=app,
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    time.sleep(12)
    out, err = p.communicate(b"sair\n", timeout=120)
    from libras.tempo_real_servico import Decodificador

    msgs = [m for m, _ in Decodificador().alimentar(out)]
    assert any(m["tipo"] == "quadro" for m in msgs), err.decode("utf-8", "replace")[-2000:]
    assert not rede.exists(), rede.read_text()
    novos = (_arquivos(tmp_path) - antes_app - {rede, mods, sondas}) | (_arquivos(RAIZ) - antes_repo)
    assert not [f for f in novos if f.suffix.lower() in EXT_MIDIA], novos
    assert not novos, f"arquivos criados na sessao: {sorted(map(str, novos))}"
    assert not (app / "runs").exists() and not (app / "data").exists()
    # o servico do app real tambem nao importa a bancada
    importados = json.loads(mods.read_text())
    from test_arquitetura_app_real import proibido

    assert not [m for m in importados if proibido(m)]


def test_pacote_alterado_nao_carrega(modelo_minimo, tmp_path):  # noqa: F811
    from libras.publicacao import publicar

    cfg, pasta_modelo = modelo_minimo
    pacote = publicar(cfg, pasta_modelo)
    with open(pacote / "modelo.keras", "ab") as f:
        f.write(b"x")
    env = {**os.environ, "PYTHONPATH": str(RAIZ / "src")}
    r = subprocess.run(
        [sys.executable, "-m", "libras.tempo_real_servico", "--pacote", str(pacote), "--sem-disco"],
        input=b"sair\n",
        capture_output=True,
        env=env,
        timeout=60,
    )
    from libras.tempo_real_servico import Decodificador

    msgs = [m for m, _ in Decodificador().alimentar(r.stdout)]
    assert r.returncode == 2 and msgs[0]["tipo"] == "erro" and "hash diferente" in msgs[0]["mensagem"]
    assert "tensorflow" not in r.stderr.decode(errors="replace").lower()  # recusado antes de carregar o modelo


def test_nenhum_modulo_do_app_real_usa_rede():
    raiz = RAIZ / "src" / "libras"
    arquivos = (
        list((raiz / "app_real").glob("*.py"))
        + list((raiz / "ui").glob("*.py"))
        + [
            raiz / n
            for n in (
                "tempo_real_servico.py",
                "pacote.py",
                "config_real.py",
                "camera.py",
                "inferencia.py",
                "enquadramento.py",
                "decisao.py",
            )
        ]
    )
    for arq in arquivos:
        txt = arq.read_text(encoding="utf-8")
        for termo in (
            "import socket",
            "urllib",
            "http.client",
            "requests",
            "QtNetwork",
            "QNetwork",
            "imwrite(",
            "VideoWriter(",
            "QImageWriter",
        ):
            assert termo not in txt, f"{arq.name} usa {termo}"


@pytest.mark.parametrize("texto", ["Processamento neste computador. Nenhuma imagem é gravada nem enviada."])
def test_texto_do_selo(texto):
    from libras.app_real.janela import SELO, SELO_DETALHE

    assert SELO == texto and SELO_DETALHE.startswith(texto) and "não é guardado em disco" in SELO_DETALHE
