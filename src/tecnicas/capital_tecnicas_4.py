from pathlib import Path
import re

import numpy as np
import pandas as pd


def chave_ordenacao_algoritmo(nome):
    match = re.fullmatch(r"(.+)_(\d+)", nome)

    if not match:
        return nome, 0

    return match.group(1), int(match.group(2))


def identificar_algoritmos(df):
    algoritmos = []

    for col in df.columns:
        if not col.startswith("compra_"):
            continue

        algoritmo = col.replace("compra_", "", 1)

        if f"venda_{algoritmo}" in df.columns:
            algoritmos.append(algoritmo)

    algoritmos = sorted(set(algoritmos), key=chave_ordenacao_algoritmo)

    if not algoritmos:
        raise ValueError(
            "Nenhum par compra_/venda_ foi encontrado. "
            f"Colunas disponiveis: {df.columns.tolist()}"
        )

    return algoritmos


def validar_colunas_necessarias(df, caminho):
    colunas = ["ativo", "target", "data"]
    faltantes = [col for col in colunas if col not in df.columns]

    if faltantes:
        raise ValueError(
            f"Colunas faltando em {caminho}: {faltantes}. "
            f"Colunas disponiveis: {df.columns.tolist()}"
        )


def calcular_capital_algoritmo(g, algoritmo, capital_inicial):
    dinheiro = np.zeros(len(g))
    investido = np.zeros(len(g))
    qtde = np.zeros(len(g))

    capital = capital_inicial
    posicao_aberta = False
    qtde_ativos = 0

    compras = pd.to_numeric(g[f"compra_{algoritmo}"], errors="coerce").fillna(0)
    vendas = pd.to_numeric(g[f"venda_{algoritmo}"], errors="coerce").fillna(0)

    for i in range(len(g)):
        compra = compras.iloc[i]
        venda = vendas.iloc[i]

        if not posicao_aberta:
            if compra > 0:
                investido[i] = capital
                qtde_ativos = capital / compra if compra > 0 else 0
                qtde[i] = qtde_ativos
                dinheiro[i] = 0
                posicao_aberta = True
            else:
                dinheiro[i] = capital
                investido[i] = 0
                qtde[i] = 0
        else:
            investido[i] = capital
            qtde[i] = qtde_ativos
            dinheiro[i] = 0

        if posicao_aberta and venda > 0:
            capital = qtde_ativos * venda
            posicao_aberta = False
            qtde_ativos = 0

    g[f"dinheiro_{algoritmo}"] = dinheiro
    g[f"investido_{algoritmo}"] = investido
    g[f"qtde_ativos_{algoritmo}"] = qtde
    g[f"capital_teorico_{algoritmo}"] = (
        g[f"dinheiro_{algoritmo}"] + g[f"investido_{algoritmo}"]
    )

    return g


def calcula_capital_tecnicas(
    input_folder="./data/tecnicas/separacao_tecnicas_3/",
    output_folder="./data/tecnicas/capital_tecnicas_4/",
    capital_inicial=1000,
):
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    arquivos = sorted(input_folder.glob("*_monetario.csv"))

    if not arquivos:
        raise FileNotFoundError(f"Nenhum arquivo monetario encontrado em {input_folder}")

    for caminho in arquivos:
        print(f"\nProcessando arquivo: {caminho.name}")

        df = pd.read_csv(caminho, sep="|")
        validar_colunas_necessarias(df, caminho)

        algoritmos = identificar_algoritmos(df)
        print(f"Algoritmos encontrados: {', '.join(algoritmos)}")

        df["data"] = pd.to_datetime(df["data"], errors="coerce")
        df["target"] = pd.to_numeric(df["target"], errors="coerce")
        df = df.sort_values(["ativo", "target", "data"])

        results = []

        for (ativo, target), g in df.groupby(["ativo", "target"], sort=False):
            print(f"  ativo={ativo} | target={target}")

            g = g.reset_index(drop=True)

            for algoritmo in algoritmos:
                g = calcular_capital_algoritmo(g, algoritmo, capital_inicial)

            results.append(g)

        if not results:
            print("Nenhum dado processado.")
            continue

        final_df = pd.concat(results, ignore_index=True)
        out_path = output_folder / caminho.name.replace(".csv", "_capital.csv")
        final_df.to_csv(out_path, index=False, sep="|")

        print(f"Salvo em: {out_path}")


def run_calculo_capital_tecnicas():
    calcula_capital_tecnicas()


if __name__ == "__main__":
    run_calculo_capital_tecnicas()
