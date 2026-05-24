from pathlib import Path
import re

import pandas as pd


CHAVES = ["ativo", "target", "data"]
COLUNAS_REMOVER_INTRADAY = ["esmble_jan_tot", "esmble_jan_par", "in_precision"]
COLUNAS_IGNORAR_TECNICAS = {
    "ativo",
    "target",
    "data",
    "target_real",
    "resultado_real",
    "esmble_jan_tot",
    "esmble_jan_par",
}

SUFIXO_INTRADAY = "_comparacao_completa_intraday.csv"
SUFIXO_TECNICAS = "_ensemble_jan_tot_e_parcial.csv"
PADRAO_COLUNA_TECNICA = re.compile(r".+_\d+$")


def extrair_ativo(nome_arquivo, sufixo):
    if nome_arquivo.endswith(sufixo):
        return nome_arquivo[: -len(sufixo)]

    return Path(nome_arquivo).stem.split("_")[0]


def mapear_arquivos_por_ativo(input_folder, sufixo):
    input_folder = Path(input_folder)
    arquivos = {}

    for caminho in sorted(input_folder.glob("*.csv")):
        ativo = extrair_ativo(caminho.name, sufixo)

        if ativo in arquivos:
            raise ValueError(
                f"Mais de um arquivo encontrado para o ativo {ativo}: "
                f"{arquivos[ativo]} e {caminho}"
            )

        arquivos[ativo] = caminho

    return arquivos


def validar_colunas(df, colunas, caminho):
    faltantes = [col for col in colunas if col not in df.columns]

    if faltantes:
        raise ValueError(
            f"Colunas faltando em {caminho}: {faltantes}. "
            f"Colunas disponiveis: {df.columns.tolist()}"
        )


def normalizar_chaves(df, caminho):
    validar_colunas(df, CHAVES, caminho)

    df = df.copy()
    df["ativo"] = df["ativo"].astype(str).str.strip()
    df["target"] = pd.to_numeric(df["target"], errors="coerce").round(6)
    df["data"] = pd.to_datetime(df["data"], errors="coerce").dt.strftime("%Y-%m-%d")

    if df[CHAVES].isna().any().any():
        linhas_invalidas = df[df[CHAVES].isna().any(axis=1)].index.tolist()[:10]
        raise ValueError(
            f"Chaves invalidas em {caminho}. "
            f"Primeiras linhas com problema: {linhas_invalidas}"
        )

    return df


def identificar_colunas_tecnicas(df):
    colunas = [
        col
        for col in df.columns
        if col not in COLUNAS_IGNORAR_TECNICAS and PADRAO_COLUNA_TECNICA.fullmatch(col)
    ]

    if not colunas:
        raise ValueError(
            "Nenhuma coluna de tecnica foi encontrada. "
            f"Colunas disponiveis: {df.columns.tolist()}"
        )

    return colunas


def converter_colunas_binarias(df, colunas):
    for col in colunas:
        serie = pd.to_numeric(df[col], errors="coerce")
        valores_validos = serie.dropna().unique()

        if set(valores_validos).issubset({0, 1, 0.0, 1.0}):
            df[col] = serie.astype("Int64")
        else:
            df[col] = serie

    return df


def juntar_ativo(caminho_intraday, caminho_tecnicas):
    df_intraday = pd.read_csv(caminho_intraday, sep="|")
    df_tecnicas = pd.read_csv(caminho_tecnicas)

    df_intraday = normalizar_chaves(df_intraday, caminho_intraday)
    df_tecnicas = normalizar_chaves(df_tecnicas, caminho_tecnicas)

    colunas_tecnicas = identificar_colunas_tecnicas(df_tecnicas)

    df_intraday = df_intraday.drop(columns=COLUNAS_REMOVER_INTRADAY, errors="ignore")
    df_tecnicas = df_tecnicas[CHAVES + colunas_tecnicas].copy()

    duplicadas = df_tecnicas.duplicated(CHAVES).sum()
    if duplicadas:
        print(
            f"AVISO: {caminho_tecnicas.name}: {duplicadas} linhas duplicadas nas chaves; "
            "mantendo a ultima ocorrencia."
        )
        df_tecnicas = df_tecnicas.drop_duplicates(CHAVES, keep="last")

    df_final = pd.merge(
        df_intraday,
        df_tecnicas,
        on=CHAVES,
        how="left",
        validate="m:1",
    )

    linhas_sem_tecnica = df_final[colunas_tecnicas].isna().all(axis=1).sum()
    df_final = converter_colunas_binarias(df_final, colunas_tecnicas)

    return df_final, linhas_sem_tecnica, colunas_tecnicas


def run_juncao_tecnicas_intraday(
    input_intraday="./data/ensemble/7_intraday_join/",
    input_tecnicas="./data/ensemble/2_tot_par/",
    output_folder="./data/tecnicas/juncao_tecnicas_1/",
):
    input_intraday = Path(input_intraday)
    input_tecnicas = Path(input_tecnicas)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    arquivos_intraday = mapear_arquivos_por_ativo(input_intraday, SUFIXO_INTRADAY)
    arquivos_tecnicas = mapear_arquivos_por_ativo(input_tecnicas, SUFIXO_TECNICAS)

    ativos_sem_tecnica = sorted(set(arquivos_intraday) - set(arquivos_tecnicas))
    ativos_sem_intraday = sorted(set(arquivos_tecnicas) - set(arquivos_intraday))

    if ativos_sem_tecnica:
        print(f"AVISO: Ativos sem arquivo de tecnicas: {ativos_sem_tecnica}")

    if ativos_sem_intraday:
        print(f"AVISO: Ativos sem arquivo intraday: {ativos_sem_intraday}")

    total_processados = 0
    total_linhas_sem_tecnica = 0

    for ativo in sorted(set(arquivos_intraday) & set(arquivos_tecnicas)):
        print(f"Processando {ativo}...")

        df_final, linhas_sem_tecnica, colunas_tecnicas = juntar_ativo(
            arquivos_intraday[ativo],
            arquivos_tecnicas[ativo],
        )

        caminho_saida = output_folder / f"{ativo}_tecnicas_intraday.csv"
        df_final.to_csv(caminho_saida, sep="|", index=False)

        total_processados += 1
        total_linhas_sem_tecnica += linhas_sem_tecnica

        print(
            f"Salvo: {caminho_saida} "
            f"({len(df_final)} linhas, {len(colunas_tecnicas)} colunas de tecnicas, "
            f"{linhas_sem_tecnica} linhas sem match)"
        )

    print("\nResumo")
    print(f"Arquivos processados: {total_processados}")
    print(f"Linhas sem match nas tecnicas: {total_linhas_sem_tecnica}")


if __name__ == "__main__":
    run_juncao_tecnicas_intraday()
