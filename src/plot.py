import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "./visualization/.matplotlib")

import matplotlib
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt


TECNICAS = ["esmble_jan_tot", "esmble_jan_par", "in_precision", "aleatorio"]
ALVOS = ["1_01", "1_015", "1_02"]

CORES_TECNICAS = {
    "esmble_jan_tot": "#1f77b4",
    "esmble_jan_par": "#2ca02c",
    "in_precision": "#d62728",
    "aleatorio": "#111111",
}

ESTILOS_LINHA = {
    "esmble_jan_tot": "-",
    "esmble_jan_par": "-",
    "in_precision": "-",
    "aleatorio": "-",
}

NOMES_TECNICAS = {
    "esmble_jan_tot": "ensemble total",
    "esmble_jan_par": "ensemble parcial",
    "in_precision": "precision",
    "aleatorio": "aleatorio",
}

NOMES_ALVOS = {
    "1_01": "1.01",
    "1_015": "1.015",
    "1_02": "1.02",
}

GRAFICOS = [
    {
        "titulo": "Capital - estrategia 1/n",
        "fontes": [
            {
                "pasta": Path("./data/ensemble/11_otimi"),
                "tecnicas": ["esmble_jan_tot", "esmble_jan_par", "in_precision"],
            },
            {
                "pasta": Path("./data/ensemble/18_1n_aleatorio"),
                "tecnicas": ["aleatorio"],
            },
        ],
        "coluna_capital": "total_verdadeiro",
        "arquivo_saida": "capital_1n.png",
        "eixo_y": "Total verdadeiro",
    },
    {
        "titulo": "Capital - MV Sharpe",
        "fontes": [
            {
                "pasta": Path("./data/ensemble/16_capital_mv_sharpe"),
                "tecnicas": ["esmble_jan_tot", "esmble_jan_par", "in_precision"],
            },
            {
                "pasta": Path("./data/ensemble/21_capital_mv_sharpe_aleatorio"),
                "tecnicas": ["aleatorio"],
            },
        ],
        "coluna_capital": "capital_total_mv_sharpe",
        "arquivo_saida": "capital_mv_sharpe.png",
        "eixo_y": "Capital total MV Sharpe",
    },
    {
        "titulo": "Capital - MV minima variancia",
        "fontes": [
            {
                "pasta": Path("./data/ensemble/17_capital_mv_min_variancia"),
                "tecnicas": ["esmble_jan_tot", "esmble_jan_par", "in_precision"],
            },
            {
                "pasta": Path("./data/ensemble/22_capital_mv_min_variancia_aleatorio"),
                "tecnicas": ["aleatorio"],
            },
        ],
        "coluna_capital": "capital_total_min_variancia",
        "arquivo_saida": "capital_mv_min_variancia.png",
        "eixo_y": "Capital total MV minima variancia",
    },
]


def carregar_dados(config, coluna_capital):
    dados = []
    fontes = config.get("fontes")

    if fontes is None:
        fontes = [
            {
                "pasta": config["pasta"],
                "tecnicas": TECNICAS,
            }
        ]

    for fonte in fontes:
        pasta = Path(fonte["pasta"])
        tecnicas = fonte.get("tecnicas", TECNICAS)

        for tecnica in tecnicas:
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
    df_all = carregar_dados(config, coluna_capital)

    if df_all.empty:
        print(f"Aviso: nenhum dado encontrado para {config['titulo']}")
        return

    fig, axes = plt.subplots(
        nrows=len(ALVOS),
        ncols=1,
        figsize=(14, 10.8),
        sharex=True,
    )
    fig.subplots_adjust(
        top=0.93,
        bottom=0.16,
        left=0.08,
        right=0.98,
        hspace=0.28,
    )
    handles_por_label = {}

    for ax, alvo in zip(axes, ALVOS):
        df_alvo = df_all[df_all["alvo"] == alvo]

        for tecnica in TECNICAS:
            grupo = df_alvo[df_alvo["tecnica"] == tecnica]

            if grupo.empty:
                continue

            label = NOMES_TECNICAS[tecnica]
            linha, = ax.plot(
                grupo["data"],
                grupo[coluna_capital],
                color=CORES_TECNICAS[tecnica],
                linestyle=ESTILOS_LINHA[tecnica],
                linewidth=2.2 if tecnica != "aleatorio" else 2.6,
                alpha=0.9 if tecnica != "aleatorio" else 0.95,
                label=label,
            )
            handles_por_label[label] = linha

        ax.set_title(
            f"Target {NOMES_ALVOS[alvo]}",
            loc="left",
            fontsize=11,
            fontweight="bold",
        )
        ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.65)
        ax.margins(x=0)

    fig.suptitle(config["titulo"], fontsize=15, fontweight="bold")
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

    imagem_saida = output_folder / config["arquivo_saida"]
    fig.savefig(imagem_saida, dpi=170)
    plt.close()

    print(f"Grafico salvo em: {imagem_saida}")


def main():
    output_folder = Path("./visualization")
    output_folder.mkdir(parents=True, exist_ok=True)

    for config in GRAFICOS:
        plotar_capital(config, output_folder)


if __name__ == "__main__":
    main()
