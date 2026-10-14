"""13.8: textos da barra de informacoes tecnicas e o relatorio para a equipe (valores SINTETICOS)."""

from libras.app_real.preferencias import GRUPOS_TECNICOS
from libras.app_real.tecnicas import EXPLICACAO_PULADOS, grupos, relatorio, texto_barra

META = {
    "tempos": {"fps_camera": 30.0, "fps_processado": 14.2, "memoria_mb": 1126.4},
    "tempos_recentes": {
        "ponta_a_ponta": {"mediana_ms": 82.0, "p95_ms": 131.0},
        "mediapipe": {"mediana_ms": 55.0, "p95_ms": 70.0},
        "modelo": {"mediana_ms": 5.0},
    },
    "quadros": {"capturados": 300, "processados": 264, "descartados": 36},
    "etapa": {"etapa": "sem_maos"},
}
EXTRA = {
    "fps_exibicao": 14.0,
    "memoria_interface_mb": 160.0,
    "modelo": "20261007_101010_abcd1234",
    "camera": {"indice": 0, "largura": 640, "altura": 480},
}


def test_grupos_e_rotulos_curtos():
    g = grupos(META, EXTRA)
    assert g["taxa"] == ["FPS cam 30", "Proc 14", "Exib 14"]
    assert g["quadros"][:3] == ["Capturados 300", "Processados 264", "Descartados 36"] and g["quadros"][3].startswith(
        "Pulados 12%"
    )
    assert EXPLICACAO_PULADOS in g["quadros"][3]
    assert g["tempo"][0] == "Lat 82 ms (p95 131)" and "MediaPipe 55 ms" in g["tempo"]
    assert "RAM reconhecimento 1,1 GB" in g["recursos"] and "RAM interface 160 MB" in g["recursos"]
    assert "Estado sem mãos" in g["sistema"] and "Câmera 0 640x480" in g["sistema"]


def test_so_os_grupos_ligados_na_ordem_e_omissao_por_prioridade():
    ligados = {"sistema": True, "taxa": True}
    linhas = texto_barra(META, EXTRA, ligados, "media", lambda s: True, 1)
    assert len(linhas) == 1 and linhas[0].index("FPS cam") < linhas[0].index("Modelo")
    assert texto_barra(META, EXTRA, {}, "media", lambda s: True) == []  # tudo desligado: nada
    todos = {g: True for g in GRUPOS_TECNICOS}
    curtas = texto_barra(META, EXTRA, todos, "compacta", lambda s: len(s) <= 80, 2)  # compacta: ate 2 linhas
    assert 1 <= len(curtas) <= 2 and all(len(l) <= 80 for l in curtas)
    juntas = " ".join(curtas)
    assert "FPS cam" in juntas and "Modelo 2026" not in juntas  # Sistema (ultimo) sai antes de Taxa


def test_explicacao_sai_antes_de_um_grupo():
    todos = {"taxa": True, "quadros": True, "tempo": True}
    larg = len("  |  ".join("  ".join(grupos(META, EXTRA, False)[g]) for g in todos))
    linhas = texto_barra(META, EXTRA, todos, "expandida", lambda s: len(s) <= larg, 1)
    assert len(linhas) == 1 and "Lat" in linhas[0] and EXPLICACAO_PULADOS not in linhas[0]


def test_relatorio_sem_caminhos_nem_ids():
    r = relatorio(
        META,
        EXTRA,
        {"id": "20261007_101010_abcd1234", "hash": {"modelo.keras": "f" * 64}},
        "0.3.0",
        {"sistema": "Windows 10", "processador": "i5", "nucleos_logicos": 4, "ram_gb": 4},
    )
    assert "0.3.0" in r and "Windows 10" in r and "Pulados 12%" in r
    assert "/" not in r.replace("Lat 82 ms", "") or "S0" not in r
    assert "S01" not in r and "\\\\" not in r and "/home" not in r and "C:" not in r
