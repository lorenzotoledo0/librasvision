"""Regressao do bug 'quebra ao reabrir depois de gravar' e tolerancia das tabelas.
Clipes SINTETICOS (quadros coloridos, sem pessoas)."""

import csv
import os
import time

import pytest

from conftest import cadastrar, frame_sintetico_assimetrico
from libras.coleta.gravacao import MetaTake, registrar_take
from libras.coleta.manifesto import COLUNAS, Manifesto
from libras.coleta.qualidade import avaliar_clipe
from libras.coleta.sinalizadores import CadastroSinalizadores
from libras.erros import ErroManifesto


def _gravar(cfg, sid, classe, desloc):
    frames = [frame_sintetico_assimetrico(i=i + desloc) for i in range(45)]
    info = avaliar_clipe(frames, [i / 30 for i in range(45)], 30, cfg["coleta"], None)
    meta = MetaTake(classe, sid, "natural", "media", "neutro", "direita", 'observação, com "aspas"')
    return registrar_take(cfg, Manifesto(cfg), CadastroSinalizadores(cfg), frames, info, meta)


def test_gravar_fechar_reabrir_estado_consistente(cfg):
    (sid,) = cadastrar(cfg)
    _gravar(cfg, sid, "biblioteca", 0)
    _gravar(cfg, sid, "biblioteca", 7)
    _gravar(cfg, sid, "prova", 13)
    # "reabrir": novas instancias de toda a camada de logica
    m, c = Manifesto(cfg), CadastroSinalizadores(cfg)
    assert not m.avisos and not c.avisos
    assert len(m) == 3 and c.ids_com_tcle() == [sid]
    assert m.contagens(("ok",)) == {("biblioteca", sid): 2, ("prova", sid): 1}
    assert m.proximo_take(sid, "biblioteca") == 3 and m.proximo_take(sid, "secretaria") == 1
    assert m.linhas[0]["observacoes"] == 'observação, com "aspas"'
    assert m.verificar_integridade(c.ids(), c.ids_com_tcle()).ok
    _gravar(cfg, sid, "biblioteca", 21)  # gravar de novo depois de reabrir
    assert Manifesto(cfg).proximo_take(sid, "biblioteca") == 4


def _regravar_como_excel(caminho):
    linhas = list(csv.reader(open(caminho, encoding="utf-8")))
    with open(caminho, "w", encoding="cp1252", newline="") as f:
        csv.writer(f, delimiter=";").writerows(linhas)


def test_manifesto_salvo_pelo_excel_abre_e_e_normalizado(cfg):
    (sid,) = cadastrar(cfg)
    _gravar(cfg, sid, "prova", 0)
    _regravar_como_excel(cfg.caminho("manifesto"))
    _regravar_como_excel(cfg.caminho("sinalizadores"))
    m, c = Manifesto(cfg), CadastroSinalizadores(cfg)
    assert len(m) == 1 and m.linhas[0]["observacoes"].startswith("observação")
    assert any("Excel" in a for a in m.avisos) and any("Excel" in a for a in c.avisos)
    assert m.backup is not None and m.backup.exists()
    assert c.ids_com_tcle() == [sid]
    m.definir_status(m.linhas[0]["video_id"], "pendente")  # proxima escrita normaliza o formato
    texto = cfg.caminho("manifesto").read_text(encoding="utf-8")
    assert texto.splitlines()[0].split(",")[0] == "video_id" and not Manifesto(cfg).avisos


def test_linha_ruim_e_ignorada_com_aviso_e_preservada(cfg):
    (sid,) = cadastrar(cfg)
    _gravar(cfg, sid, "prova", 0)
    arq = cfg.caminho("manifesto")
    with open(arq, "a", encoding="utf-8", newline="") as f:
        f.write("\n\nS01_x,abc,def\n")  # linha em branco + linha truncada
        f.write("," * (len(COLUNAS) - 1) + "\n")  # linha so com separadores (vazia)
    m = Manifesto(cfg)
    assert len(m) == 1
    assert any("linha 5 ignorada" in a for a in m.avisos), m.avisos
    ign = list(cfg.caminho("backup").glob("*_linhas_ignoradas.txt"))
    assert ign and "S01_x,abc,def" in ign[0].read_text(encoding="utf-8")


def test_manifesto_de_versao_anterior_ganha_colunas_novas(cfg):
    (sid,) = cadastrar(cfg)
    _gravar(cfg, sid, "prova", 0)
    arq = cfg.caminho("manifesto")
    linhas = list(csv.DictReader(open(arq, encoding="utf-8")))
    antigas = [c for c in COLUNAS if c not in ("sha256_npz", "frames_originais", "corte")]
    with open(arq, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=antigas, extrasaction="ignore")
        w.writeheader()
        w.writerows(linhas)
    m = Manifesto(cfg)
    assert len(m) == 1 and m.linhas[0]["corte"] == "" and not m.avisos


def test_coluna_obrigatoria_faltando_mensagem_clara(cfg):
    cfg.caminho("manifesto").parent.mkdir(parents=True, exist_ok=True)
    cfg.caminho("manifesto").write_text("foo,bar\n1,2\n", encoding="utf-8")
    with pytest.raises(ErroManifesto, match="backup"):
        Manifesto(cfg)


def test_temporarios_orfaos_vao_para_a_lixeira(cfg):
    from libras.coleta.manutencao import limpar_temporarios

    d = cfg.caminho("videos") / "prova"
    d.mkdir(parents=True)
    velho = d / "S01_003.parcial.mp4"
    velho.write_bytes(b"x")
    os.utime(velho, (time.time() - 3600, time.time() - 3600))
    novo = d / "S01_004.parcial.mp4"
    novo.write_bytes(b"x")
    movidos = limpar_temporarios(cfg)
    assert len(movidos) == 1 and not velho.exists() and novo.exists()
    assert "orfaos" in movidos[0].parts


def test_log_e_handler_global(tmp_path):
    import logging
    import threading

    from libras import registro

    registro._estado.update(configurado=False, notificar=None)
    arq = registro.configurar_log(tmp_path / "logs", "teste")
    vistos = []
    registro.definir_notificador(lambda t, m: vistos.append(m))
    t = threading.Thread(target=lambda: 1 / 0, name="falha")
    t.start()
    t.join()
    for h in logging.getLogger().handlers:
        h.flush()
    texto = arq.read_text(encoding="utf-8")
    assert "ZeroDivisionError" in texto and "falha" in texto
    assert vistos and "inesperado" in vistos[0]
    registro._estado.update(notificar=None)
