from pathlib import Path
import re

import pandas as pd


FAMILIAS_TECNICAS = ["RNA", "SVC", "RandomForest"]


def chave_ordenacao_algoritmo(nome):
    match = re.fullmatch(r"(.+)_(\d+)", nome)

    if not match:
        return nome, 0

    return match.group(1), int(match.group(2))


def extrair_familia(nome_arquivo):
    for familia in FAMILIAS_TECNICAS:
        if nome_arquivo.startswith(f"{familia}_target_"):
            return familia

    raise ValueError(f"Familia de tecnica nao identificada em: {nome_arquivo}")


def extrair_target(nome_arquivo):
    match = re.search(r"target_(\d+[_\.]\d+)", nome_arquivo)

    if not match:
        raise ValueError(f"Target nao encontrado em: {nome_arquivo}")

    return match.group(1).replace(".", "_")


def identificar_algoritmos(df, familia):
    algoritmos = set()

    for coluna in df.columns:
        if "_rend_decisao_" in coluna or "_rend_venda_" in coluna:
            continue

        match = re.fullmatch(rf".+_({re.escape(familia)}_\d+)", coluna)
        if not match:
            continue

        algoritmo = match.group(1)
        ativo = coluna[: -(len(algoritmo) + 1)]

        if (
            f"{ativo}_rend_decisao_{algoritmo}" in df.columns
            and f"{ativo}_rend_venda_{algoritmo}" in df.columns
        ):
            algoritmos.add(algoritmo)

    algoritmos = sorted(algoritmos, key=chave_ordenacao_algoritmo)

    if not algoritmos:
        raise ValueError(
            f"Nenhuma tecnica+janela encontrada para {familia}. "
            f"Colunas disponiveis: {df.columns.tolist()}"
        )

    return algoritmos


def identificar_ativos(df, algoritmo):
    ativos = []

    for coluna in df.columns:
        if "_rend_decisao_" in coluna or "_rend_venda_" in coluna:
            continue

        if not coluna.endswith(f"_{algoritmo}"):
            continue

        ativo = coluna[: -(len(algoritmo) + 1)]

        if (
            f"{ativo}_rend_decisao_{algoritmo}" in df.columns
            and f"{ativo}_rend_venda_{algoritmo}" in df.columns
        ):
            ativos.append(ativo)

    return sorted(set(ativos))


def selecionar_colunas_algoritmo(df, algoritmo, ativos):
    colunas = ["data"]
    colunas_binarias = []

    for ativo in ativos:
        col_bin = f"{ativo}_{algoritmo}"
        col_rend_decisao = f"{ativo}_rend_decisao_{algoritmo}"
        col_rend_venda = f"{ativo}_rend_venda_{algoritmo}"

        if col_bin in df.columns:
            colunas.append(col_bin)
            colunas_binarias.append(col_bin)

        if col_rend_decisao in df.columns:
            colunas.append(col_rend_decisao)

        if col_rend_venda in df.columns:
            colunas.append(col_rend_venda)

    novo_df = df[colunas].copy()

    if colunas_binarias:
        novo_df = novo_df[novo_df[colunas_binarias].notna().any(axis=1)]

    return novo_df


def separar_targets_por_tecnica_tecnicas(
    input_folder="./data/tecnicas/targets_alinhados_tecnicas_5/",
    output_folder="./data/tecnicas/targets_por_tecnica_tecnicas_6/",
):
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    arquivos = sorted(input_folder.glob("*.csv"))

    if not arquivos:
        raise FileNotFoundError(f"Nenhum arquivo encontrado em {input_folder}")

    total_gerados = 0

    for caminho in arquivos:
        print(f"\nArquivo: {caminho.name}")

        familia = extrair_familia(caminho.name)
        target = extrair_target(caminho.name)
        df = pd.read_csv(caminho)

        algoritmos = identificar_algoritmos(df, familia)
        print(f"Tecnicas+janelas encontradas: {', '.join(algoritmos)}")

        for algoritmo in algoritmos:
            ativos = identificar_ativos(df, algoritmo)
            novo_df = selecionar_colunas_algoritmo(df, algoritmo, ativos)

            nome_saida = f"{algoritmo}_target_{target}.csv"
            caminho_saida = output_folder / nome_saida
            novo_df.to_csv(caminho_saida, index=False)
            total_gerados += 1

            print(
                f"  Salvo: {nome_saida} "
                f"({len(novo_df)} linhas, {len(novo_df.columns)} colunas, "
                f"{len(ativos)} ativos)"
            )

    print("\nResumo")
    print(f"Arquivos de entrada: {len(arquivos)}")
    print(f"Arquivos gerados: {total_gerados}")


def run_separar_targets_por_tecnica_tecnicas():
    separar_targets_por_tecnica_tecnicas()


if __name__ == "__main__":
    run_separar_targets_por_tecnica_tecnicas()
