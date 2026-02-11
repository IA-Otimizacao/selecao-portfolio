import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm


def gerar_graficos_capital(
    input_path="./data/monetario_real",
    output_path="./src/visualization",
    data_limite="2021-06-04"
):
    """
    Gera gráficos de evolução do capital por target e técnica (TOT, PAR, PRECISION),
    incluindo versão completa e versão filtrada até uma data limite.
    """

    os.makedirs(output_path, exist_ok=True)

    arquivos = [f for f in os.listdir(input_path) if f.endswith(".csv")]

    for file in tqdm(arquivos, desc="Gerando gráficos", unit="arquivo"):

        ativo = file.replace(".csv", "")
        df = pd.read_csv(os.path.join(input_path, file), sep="|")

        # Garantir datetime
        df["data"] = pd.to_datetime(df["data"])

        # Targets únicos
        targets = sorted(df["target"].unique())

        # Cores fixas por target
        cores = ["blue", "red", "green"]

        # Estilos por técnica
        estilos = {
            "tot": "-",
            "par": "--",
            "precision": ":"
        }

        # =========================
        # 1) GRÁFICO COMPLETO
        # =========================
        plt.figure(figsize=(14, 7))

        for idx, target in enumerate(targets):
            df_t = df[df["target"] == target]
            cor = cores[idx]

            plt.plot(
                df_t["data"], df_t["capital_tot"],
                label=f"TOT - Target {target}",
                color=cor, linestyle=estilos["tot"], linewidth=2
            )

            plt.plot(
                df_t["data"], df_t["capital_par"],
                label=f"PAR - Target {target}",
                color=cor, linestyle=estilos["par"], linewidth=2
            )

            plt.plot(
                df_t["data"], df_t["capital_precision"],
                label=f"PRECISION - Target {target}",
                color=cor, linestyle=estilos["precision"], linewidth=2
            )

        plt.title(f"Evolução do Capital por Target e Técnica — {ativo}", fontsize=14)
        plt.xlabel("Data")
        plt.ylabel("Capital")
        plt.legend(ncol=3)
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.ticklabel_format(style="plain", axis="y")
        plt.tight_layout()

        out_file = os.path.join(output_path, f"{ativo}.png")
        plt.savefig(out_file, dpi=300)
        plt.close()

        # =========================
        # 2) GRÁFICO FILTRADO
        # =========================
        df_filtrado = df[df["data"] <= data_limite]

        plt.figure(figsize=(14, 7))

        for idx, target in enumerate(targets):
            df_t = df_filtrado[df_filtrado["target"] == target]
            cor = cores[idx]

            plt.plot(
                df_t["data"], df_t["capital_tot"],
                label=f"TOT - Target {target}",
                color=cor, linestyle=estilos["tot"], linewidth=2
            )

            plt.plot(
                df_t["data"], df_t["capital_par"],
                label=f"PAR - Target {target}",
                color=cor, linestyle=estilos["par"], linewidth=2
            )

            plt.plot(
                df_t["data"], df_t["capital_precision"],
                label=f"PRECISION - Target {target}",
                color=cor, linestyle=estilos["precision"], linewidth=2
            )

        plt.title(f"Evolução do Capital até {data_limite} — {ativo}", fontsize=14)
        plt.xlabel("Data")
        plt.ylabel("Capital")
        plt.legend(ncol=3)
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.ticklabel_format(style="plain", axis="y")
        plt.tight_layout()

        out_file_filtrado = os.path.join(
            output_path,
            f"{ativo}_ate_{data_limite}.png"
        )
        plt.savefig(out_file_filtrado, dpi=300)
        plt.close()

    print(f"✅ Gráficos gerados em {output_path}")
