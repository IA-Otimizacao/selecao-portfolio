import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "./visualization/.matplotlib")

import matplotlib
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt


TECNICAS = ["esmble_jan_tot", "esmble_jan_par", "in_precision"]
ALVOS = ["1_01", "1_015", "1_02"]

CORES = {
    "1_01": "tab:blue",
    "1_015": "tab:green",
    "1_02": "tab:red",
}

ESTILOS_LINHA = {
    "esmble_jan_tot": "-",
    "esmble_jan_par": "--",
    "in_precision": ":",
}

NOMES_TECNICAS = {
    "esmble_jan_tot": "ensemble total",
    "esmble_jan_par": "ensemble parcial",
    "in_precision": "precision",
}

NOMES_ALVOS = {
    "1_01": "1.01",
    "1_015": "1.015",
    "1_02": "1.02",
}

GRAFICOS = [
    {
        "titulo": "Capital - estrategia 1/n",
        "pasta": Path("./data/ensemble/11_otimi"),
        "coluna_capital": "total_verdadeiro",
        "arquivo_saida": "capital_1n.png",
        "eixo_y": "Total verdadeiro",
    },
    {
        "titulo": "Capital - MV Sharpe",
        "pasta": Path("./data/ensemble/16_capital_mv_sharpe"),
        "coluna_capital": "capital_total_mv_sharpe",
        "arquivo_saida": "capital_mv_sharpe.png",
        "eixo_y": "Capital total MV Sharpe",
    },
    {
        "titulo": "Capital - MV minima variancia",
        "pasta": Path("./data/ensemble/17_capital_mv_min_variancia"),
        "coluna_capital": "capital_total_min_variancia",
        "arquivo_saida": "capital_mv_min_variancia.png",
        "eixo_y": "Capital total MV minima variancia",
    },
]


def carregar_dados(pasta, coluna_capital):
    dados = []

    for tecnica in TECNICAS:
        for alvo in ALVOS:
            arquivo = pasta / f"{tecnica}_target_{alvo}.csv"

            if not arquivo.exists():
                print(f"Aviso: arquivo nao encontrado - {arquivo}")
                continue

            try:
                df = pd.read_csv(arquivo, usecols=["data", coluna_capital])
            except ValueError:
                print(f"Aviso: coluna '{coluna_capital}' nao encontrada em {arquivo}")
                continue

            df["data"] = pd.to_datetime(df["data"], errors="coerce")
            df[coluna_capital] = pd.to_numeric(df[coluna_capital], errors="coerce")
            df = df.dropna(subset=["data", coluna_capital])
            df["tecnica"] = tecnica
            df["alvo"] = alvo

            dados.append(df)

    if not dados:
        return pd.DataFrame(columns=["data", coluna_capital, "tecnica", "alvo"])

    df_all = pd.concat(dados, ignore_index=True)
    return df_all.sort_values("data").reset_index(drop=True)


def plotar_capital(config, output_folder):
    coluna_capital = config["coluna_capital"]
    df_all = carregar_dados(config["pasta"], coluna_capital)

    if df_all.empty:
        print(f"Aviso: nenhum dado encontrado para {config['titulo']}")
        return

    plt.figure(figsize=(13, 7))

    for (tecnica, alvo), grupo in df_all.groupby(["tecnica", "alvo"]):
        label = f"{NOMES_ALVOS[alvo]} - {NOMES_TECNICAS[tecnica]}"

        plt.plot(
            grupo["data"],
            grupo[coluna_capital],
            color=CORES[alvo],
            linestyle=ESTILOS_LINHA[tecnica],
            linewidth=1.8,
            label=label,
        )

    plt.xlabel("Data")
    plt.ylabel(config["eixo_y"])
    plt.title(config["titulo"])
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.tight_layout()

    imagem_saida = output_folder / config["arquivo_saida"]
    plt.savefig(imagem_saida, dpi=160)
    plt.close()

    print(f"Grafico salvo em: {imagem_saida}")


def main():
    output_folder = Path("./visualization")
    output_folder.mkdir(parents=True, exist_ok=True)

    for config in GRAFICOS:
        plotar_capital(config, output_folder)


if __name__ == "__main__":
    main()
