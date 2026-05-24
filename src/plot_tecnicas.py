import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "./visualization/.matplotlib")

import matplotlib
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt


FAMILIAS = ["RNA", "SVC", "RandomForest"]
JANELAS = ["60", "75", "90"]
ALVOS = ["1_01", "1_015", "1_02"]

NOMES_FAMILIAS = {
    "RNA": "RNA",
    "SVC": "SVC",
    "RandomForest": "Random Forest",
}

NOMES_ALVOS = {
    "1_01": "1.01",
    "1_015": "1.015",
    "1_02": "1.02",
}

CORES_ALVOS = {
    "1_01": "#1f77b4",
    "1_015": "#2ca02c",
    "1_02": "#d62728",
}

GRAFICOS = [
    {
        "titulo": "Capital - estrategia 1/n",
        "pasta": Path("./data/tecnicas/otimi_tecnicas_6"),
        "coluna_capital": "total_verdadeiro",
        "arquivo_saida": "capital_1n_{familia}.png",
        "eixo_y": "Total verdadeiro",
    },
    {
        "titulo": "Capital - MV Sharpe",
        "pasta": Path("./data/tecnicas/capital_mv_sharpe_tecnicas_10"),
        "coluna_capital": "capital_total_mv_sharpe",
        "arquivo_saida": "capital_mv_sharpe_{familia}.png",
        "eixo_y": "Capital total MV Sharpe",
    },
    {
        "titulo": "Capital - MV minima variancia",
        "pasta": Path("./data/tecnicas/capital_mv_min_variancia_tecnicas_11"),
        "coluna_capital": "capital_total_min_variancia",
        "arquivo_saida": "capital_mv_min_variancia_{familia}.png",
        "eixo_y": "Capital total MV minima variancia",
    },
]


def carregar_dados(config, familia):
    coluna_capital = config["coluna_capital"]
    pasta = config["pasta"]
    dados = []

    for janela in JANELAS:
        algoritmo = f"{familia}_{janela}"

        for alvo in ALVOS:
            arquivo = pasta / f"{algoritmo}_target_{alvo}.csv"

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
            df["familia"] = familia
            df["janela"] = janela
            df["alvo"] = alvo

            dados.append(df)

    if not dados:
        return pd.DataFrame(
            columns=["data", coluna_capital, "familia", "janela", "alvo"]
        )

    df_all = pd.concat(dados, ignore_index=True)
    return df_all.sort_values("data").reset_index(drop=True)


def plotar_capital_tecnica(config, familia, output_folder):
    coluna_capital = config["coluna_capital"]
    df_all = carregar_dados(config, familia)

    if df_all.empty:
        print(f"Aviso: nenhum dado encontrado para {config['titulo']} - {familia}")
        return

    fig, axes = plt.subplots(
        nrows=len(JANELAS),
        ncols=1,
        figsize=(14, 10.8),
        sharex=True,
    )
    fig.subplots_adjust(
        top=0.92,
        bottom=0.16,
        left=0.08,
        right=0.98,
        hspace=0.28,
    )

    handles_por_label = {}

    for ax, janela in zip(axes, JANELAS):
        df_janela = df_all[df_all["janela"] == janela]

        for alvo in ALVOS:
            grupo = df_janela[df_janela["alvo"] == alvo]

            if grupo.empty:
                continue

            label = f"Target {NOMES_ALVOS[alvo]}"
            linha, = ax.plot(
                grupo["data"],
                grupo[coluna_capital],
                color=CORES_ALVOS[alvo],
                linestyle="-",
                linewidth=2.2,
                alpha=0.92,
                label=label,
            )
            handles_por_label[label] = linha

        ax.set_title(
            f"{NOMES_FAMILIAS[familia]} - janela {janela}",
            loc="left",
            fontsize=11,
            fontweight="bold",
        )
        ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.65)
        ax.margins(x=0)

    fig.suptitle(
        f"{config['titulo']} - {NOMES_FAMILIAS[familia]}",
        fontsize=15,
        fontweight="bold",
    )
    fig.supxlabel("Data")
    fig.supylabel(config["eixo_y"])

    labels = list(handles_por_label.keys())
    handles = [handles_por_label[label] for label in labels]
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.035),
        ncol=len(labels),
        frameon=False,
        fontsize=10,
    )

    nome_saida = config["arquivo_saida"].format(familia=familia)
    imagem_saida = output_folder / nome_saida
    fig.savefig(imagem_saida, dpi=170)
    plt.close()

    print(f"Grafico salvo em: {imagem_saida}")


def main():
    output_folder = Path("./visualization/tecnicas")
    output_folder.mkdir(parents=True, exist_ok=True)

    for config in GRAFICOS:
        for familia in FAMILIAS:
            plotar_capital_tecnica(config, familia, output_folder)


if __name__ == "__main__":
    main()
