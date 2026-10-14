"""Importa videos ja gravados para o dataset (copia, valida, deduplica e registra no manifesto).

O arquivo original nunca e alterado. Reimportar a mesma pasta nao duplica nada (hash SHA-256).

Exemplos:
  # pasta organizada por classe, nomes no padrao do projeto (S01_1.mp4):
  python scripts/importar.py D:/videos --padrao projeto

  # estrutura antiga do TCC I (dataset/sinais/<classe>/usuario1_01.mp4):
  python scripts/importar.py dataset/sinais --padrao tcc1 --mapa usuario1=S01 usuario2=S02

  # arquivos soltos de um mesmo sinalizador e classe:
  python scripts/importar.py a.mp4 b.mp4 --classe prova --sinalizador S04 --iluminacao natural

  # so validar, sem copiar nada:
  python scripts/importar.py D:/videos --padrao projeto --simular
"""

import argparse
from pathlib import Path

import _bootstrap


def main():
    _bootstrap.sem_tensorflow()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("entradas", nargs="+", type=Path, help="arquivos e/ou pastas")
    ap.add_argument("--config", default=None)
    ap.add_argument("--classe", default=None, help="classe para arquivos sem classe na pasta/nome")
    ap.add_argument("--sinalizador", default=None, help="ID (S01...) para arquivos sem sinalizador no nome")
    ap.add_argument(
        "--padrao",
        default=None,
        help="padrao de nome do config (projeto, tcc1) ou uma regex com grupos signer/take/classe",
    )
    ap.add_argument("--mapa", nargs="*", default=[], help="rotulos antigos para IDs: usuario1=S01 ...")
    ap.add_argument("--sem-pasta-classe", action="store_true", help="nao usar o nome da pasta como classe")
    ap.add_argument("--iluminacao", default="")
    ap.add_argument("--distancia", default="")
    ap.add_argument("--fundo", default="")
    ap.add_argument("--mao", default=None, help="mao dominante (padrao: a do cadastro)")
    ap.add_argument("--obs", default="")
    ap.add_argument("--padronizar", action="store_true", help="re-codificar para o formato do projeto")
    ap.add_argument("--sem-validar-pessoa", action="store_true", help="pula a checagem com MediaPipe (mais rapido)")
    ap.add_argument("--simular", action="store_true", help="so validar; nao copia nem registra")
    ap.add_argument(
        "--sem-landmarks", action="store_true", help="nao gera o .npz agora (use scripts/extrair.py depois)"
    )
    a = ap.parse_args()

    from collections import Counter

    from libras.coleta.importacao import MetaPadrao, importar, parse_mapa
    from libras.config import carregar_config

    cfg = carregar_config(a.config)
    padrao = MetaPadrao(
        classe=a.classe,
        signer_id=a.sinalizador,
        iluminacao=a.iluminacao,
        distancia=a.distancia,
        fundo=a.fundo,
        mao_dominante=a.mao,
        observacoes=a.obs,
    )

    def progresso(i, n, it):
        extra = it.motivo or (", ".join(it.avisos) if it.avisos else "")
        print(f"[{i}/{n}] {it.situacao:9} {it.origem.name}  {it.video_id}  {extra}")

    itens, rel = importar(
        cfg,
        a.entradas,
        padrao,
        usar_pasta_classe=not a.sem_pasta_classe,
        padrao_nome=a.padrao,
        mapa=parse_mapa(a.mapa),
        padronizar=a.padronizar or None,
        validar_pessoa=not a.sem_validar_pessoa,
        simular=a.simular,
        progresso=progresso,
        extrair_landmarks=not a.sem_landmarks,
    )
    c = Counter(i.situacao for i in itens)
    print("\nResumo: " + ", ".join(f"{k}={v}" for k, v in sorted(c.items())))
    if rel:
        print(f"Relatorio: {rel}")
    else:
        print("Simulacao: nada foi copiado nem registrado.")
    return 0


if __name__ == "__main__":
    _bootstrap.rodar(main)
