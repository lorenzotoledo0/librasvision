"""Item 13.2: pacote de modelo publicado para o app real. Modelos e avaliacoes SINTETICOS (bytes e
JSON escritos pelo teste; nenhum modelo treinado de verdade nem numero do TCC)."""

import json
import os
import zipfile

import pytest

from libras import pacote as pk
from libras.publicacao import exportar_zip, preparar, publicado_atual, publicar


def _modelo_final(cfg, nome="20261006_120000_final", ensaio=False, classes=("biblioteca", "prova", "nenhum_sinal")):
    p = cfg.caminho("runs") / nome
    p.mkdir(parents=True)
    (p / "modelo.keras").write_bytes(b"SINTETICO-modelo-" + nome.encode())
    meta = {
        "classes": list(classes),
        "nome_nenhum_sinal": "nenhum_sinal",
        "features": {"dim": 3},
        "janela": {"comprimento": 30, "fps_alvo": 30},
        "normalizacao": {"fracao_minima_pose": 0.5},
        "mediapipe": {"model_complexity": 1},
        "ensaio": ensaio,
        "criado_em": "2026-10-06T12:00:00",
        "protocolo": "final: todos os sinalizadores",
        "videos": 12,
        "sinalizadores": ["S01", "S02", "S03"],
        "hardware_treino": {"processador": "x"},
        "avisos_selecao": ["S02 tem poucos vídeos"],
        "versoes": {
            "python": "3.11.15",
            "mediapipe": "0.10.21",
            "tensorflow": "2.19.1",
            "sistema": "Linux",
            "processador": "x86_64",
        },
    }
    (p / "metadados.json").write_text(json.dumps(meta), encoding="utf-8")
    return p


def _avaliacao(cfg, nome, classes, ensaio=False, secundario=False, f1=0.81):
    p = cfg.caminho("runs") / nome
    p.mkdir(parents=True)
    (p / "ambiente.json").write_text("{}")
    (p / "resumo.json").write_text(
        json.dumps(
            {
                "classes": list(classes),
                "ensaio": ensaio,
                "secundario": secundario,
                "protocolo": "grupos_sinalizador",
                "n_resultados": 5,
                "seeds": [42],
                "n_sinalizadores": 6,
                "n_videos": 60,
                "f1_macro_video_media": f1,
                "f1_macro_video_dp": 0.05,
                "acuracia_video_media": 0.84,
                "acuracia_video_dp": 0.04,
                "acuracia_janela_media": 0.7,
                "acuracia_janela_dp": 0.06,
                "f1_macro_janela_media": 0.68,
                "f1_macro_janela_dp": 0.07,
            }
        ),
        encoding="utf-8",
    )
    return p


CLASSES = ["biblioteca", "prova", "nenhum_sinal"]


def test_publicar_conteudo_hash_e_auditoria(cfg, tmp_path):
    from libras.vocabulario import vocabulario_em_cache

    voc = vocabulario_em_cache(cfg)
    voc.editar("biblioteca", rotulo="Biblioteca")
    m = _modelo_final(cfg)
    _avaliacao(cfg, "20261005_100000_avaliacao", CLASSES, f1=0.81)
    _avaliacao(cfg, "20261005_110000_ensaio", CLASSES, ensaio=True)
    _avaliacao(cfg, "20261005_120000_secundario", CLASSES, secundario=True)
    _avaliacao(cfg, "20261005_130000_outras", ["biblioteca", "prova"])
    destino = publicar(cfg, m)
    assert destino.parent == cfg.caminho("publicados")
    assert sorted(x.name for x in destino.iterdir()) == sorted(pk.ARQUIVOS)
    c = pk.verificar_pacote(destino)
    assert c["id"] == destino.name and c["classes"] == CLASSES and c["rotulos"]["biblioteca"] == "Biblioteca"
    assert c["decisao"]["limiar_confianca"] == cfg["tempo_real"]["limiar_confianca"]
    assert c["decisao"]["frames_estaveis"] == cfg["tempo_real"]["frames_estaveis"]
    assert c["hash"]["modelo.keras"] == pk.sha256_arquivo(destino / "modelo.keras")
    assert c["avaliacao"]["execucao"] == "20261005_100000_avaliacao"  # ignora ENSAIO, secundario e outras classes
    assert c["avaliacao"]["metricas"]["f1_macro_video"]["media"] == 0.81 and "partições" in c["avaliacao"]["ressalva"]
    meta = json.loads((destino / "metadados.json").read_text(encoding="utf-8"))
    for proibido in ("sinalizadores", "hardware_treino", "avisos_selecao"):
        assert proibido not in meta
    assert meta["n_sinalizadores"] == 3 and "sistema" not in meta["versoes"]
    # nada de IDs de sinalizadores nem caminhos em nenhum arquivo de texto do pacote
    for nome in ("metadados.json", "cartao.json"):
        txt = (destino / nome).read_text(encoding="utf-8")
        assert "S01" not in txt and "S02" not in txt and str(cfg.raiz) not in txt and "runs/" not in txt
    reg = [json.loads(x) for x in cfg.caminho("auditoria").read_text(encoding="utf-8").splitlines()]
    assert reg[-1]["acao"] == "publicar_modelo" and reg[-1]["alvo"] == c["id"]
    assert str(cfg.raiz) not in json.dumps(reg[-1])
    assert publicado_atual(cfg)["id"] == c["id"]


def test_ensaio_e_modelo_nao_final_bloqueados(cfg):
    m = _modelo_final(cfg, ensaio=True)
    with pytest.raises(pk.ErroPacote, match="ENSAIO"):
        publicar(cfg, m)
    aval = _avaliacao(cfg, "20261005_100000_avaliacao", CLASSES)
    with pytest.raises(pk.ErroPacote, match="modelos finais"):
        publicar(cfg, aval)
    assert not cfg.caminho("publicados").exists() or not list(cfg.caminho("publicados").iterdir())


def test_sem_avaliacao_o_pacote_sai_sem_metricas(cfg):
    c = preparar(cfg, _modelo_final(cfg))["cartao"]
    assert c["avaliacao"] is None


@pytest.mark.parametrize("arquivo", ["modelo.keras", "metadados.json"])
def test_adulteracao_recusada(cfg, arquivo):
    destino = publicar(cfg, _modelo_final(cfg))
    with open(destino / arquivo, "ab") as f:
        f.write(b" ")
    with pytest.raises(pk.ErroPacote, match="hash diferente"):
        pk.verificar_pacote(destino)
    it = pk.listar_pacotes(cfg.caminho("publicados"))[0]
    assert it["erro"] and it["cartao"]["id"] == destino.name


def test_exportar_e_importar_zip_sem_data_nem_runs(cfg, tmp_path):
    zp = exportar_zip(cfg, _modelo_final(cfg), tmp_path / "saida" / "pacote.zip")
    with zipfile.ZipFile(zp) as z:
        assert sorted(z.namelist()) == sorted(pk.ARQUIVOS)
    assert not cfg.caminho("publicados").exists()  # exportar nao publica
    reg = cfg.caminho("auditoria").read_text(encoding="utf-8")
    assert "exportar_pacote" in reg
    # app real: outra pasta, sem data/ nem runs/
    base = tmp_path / "outro_pc" / "publicados"
    p1 = pk.importar_zip(zp, base)
    assert pk.verificar_pacote(p1)["id"] == p1.name
    assert pk.importar_zip(zp, base) == p1  # importar de novo: o mesmo pacote
    assert not (tmp_path / "outro_pc" / "data").exists() and not (tmp_path / "outro_pc" / "runs").exists()
    assert [x.name for x in base.iterdir()] == [p1.name]


def test_zip_invalido_ou_adulterado_nao_deixa_nada(cfg, tmp_path):
    zp = exportar_zip(cfg, _modelo_final(cfg), tmp_path / "pacote.zip")
    base = tmp_path / "publicados"
    # adulterado
    ruim = tmp_path / "adulterado.zip"
    with zipfile.ZipFile(zp) as z, zipfile.ZipFile(ruim, "w") as w:
        for n in z.namelist():
            dados = z.read(n)
            w.writestr(n, dados + b"x" if n == "modelo.keras" else dados)
    with pytest.raises(pk.ErroPacote, match="hash diferente"):
        pk.importar_zip(ruim, base)
    # arquivo a mais e caminho para fora da pasta
    for extra in ("../fora.txt", "videos/S01_001.mp4"):
        z2 = tmp_path / "extra.zip"
        with zipfile.ZipFile(zp) as z, zipfile.ZipFile(z2, "w") as w:
            for n in z.namelist():
                w.writestr(n, z.read(n))
            w.writestr(extra, b"x")
        with pytest.raises(pk.ErroPacote, match="formato"):
            pk.importar_zip(z2, base)
    (tmp_path / "nao_zip.zip").write_text("texto")
    with pytest.raises(pk.ErroPacote, match="não é um pacote"):
        pk.importar_zip(tmp_path / "nao_zip.zip", base)
    assert not base.exists() or not [x for x in base.iterdir()]
    assert not (tmp_path / "fora.txt").exists()


def test_pacote_com_ids_ou_ensaio_recusado(cfg):
    destino = publicar(cfg, _modelo_final(cfg))
    c = json.loads((destino / "cartao.json").read_text(encoding="utf-8"))
    meta = json.loads((destino / "metadados.json").read_text(encoding="utf-8"))
    meta["sinalizadores"] = ["S01"]
    (destino / "metadados.json").write_text(json.dumps(meta), encoding="utf-8")
    c["hash"]["metadados.json"] = pk.sha256_arquivo(destino / "metadados.json")
    (destino / "cartao.json").write_text(json.dumps(c), encoding="utf-8")
    with pytest.raises(pk.ErroPacote, match="não deveriam sair"):
        pk.verificar_pacote(destino)


# ---------------------------------------------------------------------- interface (bancada)
@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_botoes_de_publicar_na_aba_resultados(qapp, cfg, monkeypatch, tmp_path):
    import libras.app.paginas.resultados as res
    from libras.app.janela import JanelaPrincipal

    _modelo_final(cfg, "20261006_120000_final")
    _modelo_final(cfg, "20261006_130000_final", ensaio=True)
    _avaliacao(cfg, "20261005_100000_avaliacao", CLASSES)
    monkeypatch.setattr(res, "perguntar", lambda *a, **k: True)
    w = JanelaPrincipal(cfg, transicao_ms=0)
    try:
        w.ir_para("resultados")
        p = w.paginas["resultados"]
        nomes = [it["nome"] for it in p.itens]
        p.tabela.selectRow(nomes.index("20261006_130000_final"))
        assert not p.bt_publicar.isEnabled() and "ENSAIO" in p.lb_publicar.text()
        p.tabela.selectRow(nomes.index("20261005_100000_avaliacao"))
        assert not p.bt_publicar.isEnabled() and "modelo final" in p.lb_publicar.text()
        p.tabela.selectRow(nomes.index("20261006_120000_final"))
        assert p.bt_publicar.isEnabled() and p.bt_exportar.isEnabled()
        destino = p.publicar()
        assert destino is not None and pk.verificar_pacote(destino)["modelo_origem"] == "20261006_120000_final"
        zp = p.exportar(str(tmp_path / "x.zip"))
        assert zp is not None and zipfile.is_zipfile(zp)
    finally:
        w.close()
        w.deleteLater()
