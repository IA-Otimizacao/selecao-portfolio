import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# Caminhos
input_path = "./data/monetario_completo"
output_path = "./src/visualization"
os.makedirs(output_path, exist_ok=True)

# Lista de arquivos
arquivos = [f for f in os.listdir(input_path) if f.endswith(".csv")]

for file in tqdm(arquivos, desc="Gerando gráficos", unit="arquivo"):

    ativo = file.replace(".csv", "")
    df = pd.read_csv(os.path.join(input_path, file))

    # Garantir datetime
    df["data"] = pd.to_datetime(df["data"])

    # Targets únicos
    targets = sorted(df["target"].unique())

    # Mesmas cores para cada target
    cores = ["blue", "red", "green"]

    # Estilos para cada técnica
    estilos = {
        "tot": "-",
        "par": "--",
        "precision": ":"
    }

    plt.figure(figsize=(14, 7))

    # Loop por target
    for idx, target in enumerate(targets):
        df_t = df[df["target"] == target]

        cor = cores[idx]  # cor fixa para o target

        # TOT
        plt.plot(df_t["data"], df_t["capital_tot"],
                 label=f"TOT - Target {target}",
                 color=cor,
                 linestyle=estilos["tot"],
                 linewidth=2)

        # PAR
        plt.plot(df_t["data"], df_t["capital_par"],
                 label=f"PAR - Target {target}",
                 color=cor,
                 linestyle=estilos["par"],
                 linewidth=2)

        # PRECISION
        plt.plot(df_t["data"], df_t["capital_precision"],
                 label=f"PRECISION - Target {target}",
                 color=cor,
                 linestyle=estilos["precision"],
                 linewidth=2)

    # Layout
    plt.title(f"Evolução do Capital por Target e Técnica — {ativo}", fontsize=14)
    plt.xlabel("Data")
    plt.ylabel("Capital")
    plt.legend(ncol=3)
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.ticklabel_format(style='plain', axis='y')


    plt.tight_layout()

    # Salvar
    out_file = os.path.join(output_path, f"{ativo}.png")
    plt.savefig(out_file, dpi=300)
    plt.close()

    # ================================
    # 2) GRÁFICO FILTRADO ATÉ 2021-06-04
    # ================================
    df_filtrado = df[df["data"] <= "2021-06-04"]

    plt.figure(figsize=(14, 7))

    for idx, target in enumerate(targets):
        df_t = df_filtrado[df_filtrado["target"] == target]

        cor = cores[idx]

        plt.plot(df_t["data"], df_t["capital_tot"],
                label=f"TOT - Target {target}",
                color=cor, linestyle=estilos["tot"], linewidth=2)

        plt.plot(df_t["data"], df_t["capital_par"],
                label=f"PAR - Target {target}",
                color=cor, linestyle=estilos["par"], linewidth=2)

        plt.plot(df_t["data"], df_t["capital_precision"],
                label=f"PRECISION - Target {target}",
                color=cor, linestyle=estilos["precision"], linewidth=2)

    plt.title(f"Evolução do Capital até 2021-06-04 — {ativo}", fontsize=14)
    plt.xlabel("Data")
    plt.ylabel("Capital")
    plt.legend(ncol=3)
    plt.grid(True)
    plt.xticks(rotation=45)

    plt.ticklabel_format(style='plain', axis='y')  # mantém dinheiro sem notação científica
    plt.tight_layout()

    out_file_filtrado = os.path.join(output_path, f"{ativo}_até_2021-06-04.png")
    plt.savefig(out_file_filtrado, dpi=300)
    plt.close()


print("Gráficos gerados em src/visualization/")



