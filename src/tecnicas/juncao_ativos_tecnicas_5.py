from pathlib import Path
import re

import pandas as pd


FAMILIAS_TECNICAS = ["RNA", "SVC", "RandomForest"]


def extrair_familia(nome_arquivo):
    for familia in FAMILIAS_TECNICAS:
        if f"_{familia}_monetario_capital.csv" in nome_arquivo:
            return familia

    raise ValueError(f"Familia de tecnica nao identificada no arquivo: {nome_arquivo}")


def chave_ordenacao_algoritmo(coluna):
    match = re.fullmatch(r"(.+)_(\d+)", coluna)

    if not match:
        return coluna, 0

    return match.group(1), int(match.group(2))


def identificar_algoritmos(df, familia):
    algoritmos = [
        col
        for col in df.columns
        if re.fullmatch(rf"{familia}_\d+", col)
        and f"rend_decisao_{col}" in df.columns
        and f"rend_venda_{col}" in df.columns
    ]

    algoritmos = sorted(algoritmos, key=chave_ordenacao_algoritmo)

    if not algoritmos:
        raise ValueError(
            f"Nenhuma janela encontrada para {familia}. "
            f"Colunas disponiveis: {df.columns.tolist()}"
        )

    return algoritmos


def validar_colunas_base(df, caminho):
    colunas = ["ativo", "target", "data"]
    faltantes = [col for col in colunas if col not in df.columns]

    if faltantes:
        raise ValueError(
            f"Colunas faltando em {caminho}: {faltantes}. "
            f"Colunas disponiveis: {df.columns.tolist()}"
        )


def nome_target(target):
    return str(target).replace(".", "_")


def selecionar_colunas_por_target(df, ativo, algoritmos, target):
    df_target = df[df["target"] == target].copy()

    colunas = ["data"]
    rename = {}

    for algoritmo in algoritmos:
        colunas.extend([
            algoritmo,
            f"rend_decisao_{algoritmo}",
            f"rend_venda_{algoritmo}",
        ])

        rename[algoritmo] = f"{ativo}_{algoritmo}"
        rename[f"rend_decisao_{algoritmo}"] = f"{ativo}_rend_decisao_{algoritmo}"
        rename[f"rend_venda_{algoritmo}"] = f"{ativo}_rend_venda_{algoritmo}"

    df_target = df_target[colunas].rename(columns=rename)

    return df_target


def run_juncao_ativos_tecnicas(
    input_folder="./data/tecnicas/capital_tecnicas_4/",
    output_folder="./data/tecnicas/targets_alinhados_tecnicas_5/",
):
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    arquivos = sorted(input_folder.glob("*_monetario_capital.csv"))

    if not arquivos:
        raise FileNotFoundError(f"Nenhum arquivo de capital encontrado em {input_folder}")

    dfs_por_familia_target = {}

    for caminho in arquivos:
        print(f"\nArquivo: {caminho.name}")

        familia = extrair_familia(caminho.name)
        df = pd.read_csv(caminho, sep="|")
        validar_colunas_base(df, caminho)

        df["data"] = pd.to_datetime(df["data"], errors="coerce")
        df["target"] = pd.to_numeric(df["target"], errors="coerce")
        ativo = str(df["ativo"].dropna().iloc[0]).strip()
        algoritmos = identificar_algoritmos(df, familia)

        print(f"Familia: {familia} | Ativo: {ativo} | Janelas: {', '.join(algoritmos)}")

        for target in sorted(df["target"].dropna().unique()):
            df_target = selecionar_colunas_por_target(df, ativo, algoritmos, target)
            chave = (familia, target)

            if chave not in dfs_por_familia_target:
                dfs_por_familia_target[chave] = df_target
            else:
                dfs_por_familia_target[chave] = pd.merge(
                    dfs_por_familia_target[chave],
                    df_target,
                    on="data",
                    how="outer",
                )

    for (familia, target), df_final in sorted(
        dfs_por_familia_target.items(),
        key=lambda item: (item[0][0], item[0][1]),
    ):
        df_final = df_final.sort_values("data").reset_index(drop=True)

        nome_arquivo = f"{familia}_target_{nome_target(target)}.csv"
        caminho_saida = output_folder / nome_arquivo
        df_final.to_csv(caminho_saida, index=False)

        print(
            f"Salvo: {caminho_saida} "
            f"({len(df_final)} linhas, {len(df_final.columns)} colunas)"
        )

    print("\nResumo")
    print(f"Arquivos de entrada: {len(arquivos)}")
    print(f"Arquivos gerados: {len(dfs_por_familia_target)}")


if __name__ == "__main__":
    run_juncao_ativos_tecnicas()
