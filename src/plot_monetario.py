import os
import pandas as pd
import matplotlib.pyplot as plt

def load_simulated_csvs(folder, file_pattern="_monetario.csv"):
    dfs = []
    for fname in os.listdir(folder):
        if fname.endswith(file_pattern):
            df = pd.read_csv(os.path.join(folder, fname), parse_dates=['data'])
            dfs.append(df)
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    else:
        return pd.DataFrame()


def plotar_valor_obtido(df: pd.DataFrame, pasta_saida: str = "plots"):
    os.makedirs(pasta_saida, exist_ok=True)

    # Garante que a coluna de data está no formato datetime
    df["data"] = pd.to_datetime(df["data"])

    # Agrupa por ativo e target
    for (ativo, target), df_sub in df.groupby(["ativo", "target"]):
        plt.figure(figsize=(12, 6))

        # Cada combinação janela+técnica vai ser uma linha
        for (janela, tecnica), df_plot in df_sub.groupby(["janela", "tecnica"]):
            df_plot = df_plot.sort_values("data")
            label = f"Janela={janela}, Tec={tecnica}"
            plt.plot(df_plot["data"], df_plot["valor_obtido"], label=label, linewidth=1.8)  # sem bolinhas

        plt.title(f"{ativo} - Target {target}")
        plt.xlabel("Data")
        plt.ylabel("Valor Obtido")
        plt.legend(fontsize=9)
        plt.grid(True, linestyle="--", alpha=0.6)

        # Salva imagem
        nome_arquivo = f"{ativo}_target{target}.png"
        caminho = os.path.join(pasta_saida, nome_arquivo)
        plt.savefig(caminho, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Gráfico salvo: {caminho}")


folder = "./data/monetario/"
df_all = load_simulated_csvs(folder)

# Gera e salva os gráficos
plotar_valor_obtido(df_all, pasta_saida="./src/visualization/")
