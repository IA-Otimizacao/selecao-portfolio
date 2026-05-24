from pathlib import Path
import re

import pandas as pd


FAMILIAS_TECNICAS = ["RNA", "SVC", "RandomForest"]
PREFIXOS_MONETARIOS = ["compra", "rend_decisao", "rend_venda", "dias", "venda"]


def pertence_a_familia(coluna, familia):
    if re.fullmatch(rf"{familia}_\d+", coluna):
        return True

    return any(
        re.fullmatch(rf"{prefixo}_{familia}_\d+", coluna)
        for prefixo in PREFIXOS_MONETARIOS
    )


def pertence_a_alguma_familia(coluna):
    return any(pertence_a_familia(coluna, familia) for familia in FAMILIAS_TECNICAS)


def extrair_ativo(df, caminho):
    if "ativo" in df.columns and not df["ativo"].dropna().empty:
        return str(df["ativo"].dropna().iloc[0]).strip()

    return caminho.name.split("_")[0]


def separar_arquivo_por_familia(caminho, output_folder):
    print(f"\nProcessando: {caminho.name}")

    df = pd.read_csv(caminho, sep="|")
    ativo = extrair_ativo(df, caminho)
    arquivos_gerados = 0

    colunas_base = [col for col in df.columns if not pertence_a_alguma_familia(col)]

    for familia in FAMILIAS_TECNICAS:
        colunas_familia = [col for col in df.columns if pertence_a_familia(col, familia)]

        if not colunas_familia:
            print(f"  Sem colunas para {familia}; pulando.")
            continue

        df_saida = df[colunas_base + colunas_familia].copy()
        caminho_saida = output_folder / f"{ativo}_{familia}_monetario.csv"
        df_saida.to_csv(caminho_saida, sep="|", index=False)

        print(
            f"  Salvo: {caminho_saida.name} "
            f"({len(df_saida.columns)} colunas)"
        )
        arquivos_gerados += 1

    return arquivos_gerados


def run_separacao_tecnicas(
    input_folder="./data/tecnicas/monetario_tecnicas_2/",
    output_folder="./data/tecnicas/separacao_tecnicas_3/",
):
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    arquivos = sorted(input_folder.glob("*_monetario.csv"))

    if not arquivos:
        raise FileNotFoundError(f"Nenhum arquivo monetario encontrado em {input_folder}")

    total_arquivos = 0

    for caminho in arquivos:
        total_arquivos += separar_arquivo_por_familia(caminho, output_folder)

    print("\nResumo")
    print(f"Arquivos de entrada: {len(arquivos)}")
    print(f"Arquivos gerados: {total_arquivos}")


if __name__ == "__main__":
    run_separacao_tecnicas()
