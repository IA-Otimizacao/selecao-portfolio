from pathlib import Path
import re

import numpy as np
import pandas as pd


COLUNAS_BASE = {
    "ativo",
    "target",
    "data",
    "Close",
    "Open",
    "Low",
    "High",
    "ln_open",
    "ln_high",
    "log_return",
    "log_return_binario",
}

PADRAO_TECNICA_JANELA = re.compile(r".+_\d+$")


def identificar_tecnicas(df):
    tecnicas = []

    for col in df.columns:
        if col in COLUNAS_BASE:
            continue

        if not PADRAO_TECNICA_JANELA.fullmatch(col):
            continue

        serie = pd.to_numeric(df[col], errors="coerce")
        valores_validos = set(serie.dropna().unique())

        if valores_validos.issubset({0, 1, 0.0, 1.0}):
            tecnicas.append(col)

    if not tecnicas:
        raise ValueError(
            "Nenhuma coluna binaria de tecnica+janela foi encontrada. "
            f"Colunas disponiveis: {df.columns.tolist()}"
        )

    return tecnicas


def validar_colunas_necessarias(df, caminho):
    colunas = ["ativo", "target", "data", "Open", "High", "Close"]
    faltantes = [col for col in colunas if col not in df.columns]

    if faltantes:
        raise ValueError(
            f"Colunas faltando em {caminho}: {faltantes}. "
            f"Colunas disponiveis: {df.columns.tolist()}"
        )


def calcular_monetario_por_tecnica(g, tecnica, target):
    compra = np.zeros(len(g))
    rend = np.zeros(len(g))
    rend_venda = np.zeros(len(g))
    dias = np.zeros(len(g))

    preco_compra = 0
    contador = 0
    sinal_shift = pd.to_numeric(g[tecnica], errors="coerce").shift(1)

    for i in range(len(g)):
        if contador == 0:
            if sinal_shift.iloc[i] == 1:
                preco_compra = g["Open"].iloc[i]
                compra[i] = preco_compra
                contador = 1
            else:
                compra[i] = 0
                dias[i] = 0
                continue
        else:
            compra[i] = preco_compra

        rend[i] = (g["High"].iloc[i] - preco_compra) / preco_compra
        rend_venda[i] = (g["Close"].iloc[i] - preco_compra) / preco_compra

        atingiu = rend[i] >= (target - 1)

        if atingiu:
            dias[i] = 0
            contador = 0
            preco_compra = 0
        else:
            dias[i] = contador
            contador += 1

            if contador > 4:
                contador = 0
                preco_compra = 0

    g[f"compra_{tecnica}"] = compra
    g[f"rend_decisao_{tecnica}"] = rend
    g[f"rend_venda_{tecnica}"] = rend_venda
    g[f"dias_{tecnica}"] = dias

    g[f"venda_{tecnica}"] = np.where(
        g[f"dias_{tecnica}"] == 0,
        g[f"compra_{tecnica}"] * target,
        np.where(
            g[f"dias_{tecnica}"] == 4,
            g["Close"],
            0,
        ),
    )

    return g


def calcula_investimento_tecnicas(
    input_folder="./data/tecnicas/juncao_tecnicas_1/",
    output_folder="./data/tecnicas/monetario_tecnicas_2/",
):
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    arquivos = [
        caminho
        for caminho in sorted(input_folder.glob("*.csv"))
        if "_intraday" in caminho.name.lower()
    ]

    if not arquivos:
        raise FileNotFoundError(f"Nenhum arquivo intraday encontrado em {input_folder}")

    for caminho in arquivos:
        print(f"\nProcessando arquivo: {caminho.name}")

        df = pd.read_csv(caminho, sep="|")
        validar_colunas_necessarias(df, caminho)

        tecnicas = identificar_tecnicas(df)
        print(f"Tecnicas encontradas: {', '.join(tecnicas)}")

        df["data"] = pd.to_datetime(df["data"], errors="coerce")
        df["target"] = pd.to_numeric(df["target"], errors="coerce")
        df = df.sort_values(["ativo", "target", "data"])

        results = []

        for (ativo, target), g in df.groupby(["ativo", "target"], sort=False):
            print(f"  ativo={ativo} | target={target}")

            g = g.reset_index(drop=True)
            target_float = float(target)

            for tecnica in tecnicas:
                g = calcular_monetario_por_tecnica(g, tecnica, target_float)

            results.append(g)

        if not results:
            print("Nenhum dado processado.")
            continue

        final_df = pd.concat(results, ignore_index=True)
        out_path = output_folder / caminho.name.replace(".csv", "_monetario.csv")
        final_df.to_csv(out_path, index=False, sep="|")

        print(f"Salvo em: {out_path}")


def run_invest_tecnicas():
    calcula_investimento_tecnicas()


if __name__ == "__main__":
    run_invest_tecnicas()
