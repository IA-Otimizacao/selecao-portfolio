import pandas as pd
import os

# Lista de ativos
todos = ['PETR4', 'ITUB4', 'VALE3']

# Dicionário para armazenar os DataFrames finais
dfs = {}

for file in todos:
    caminho_in = f"./data/train_out/inputs/target_in_{file}.csv"

    if not os.path.exists(caminho_in):
        print(f"⚠️ Arquivo não encontrado: {caminho_in}")
        continue

    # Lê o CSV
    df = pd.read_csv(caminho_in)

    # Converte a coluna de precisão para numérico
    df["precision"] = pd.to_numeric(df["precision"], errors="coerce")

    # Cria a coluna técnica_janela (ex: RNA_15)
    df["tecnica_janela"] = df["tecnica"].str.replace(" ", "") + "_" + df["janela"].astype(str)

    # Mantém apenas as colunas relevantes
    df = df[["ativo", "target", "data_inicio_janela", "tecnica_janela", "precision"]]

    # Faz o pivot
    df_pivot = df.pivot_table(
        index=["ativo", "target", "data_inicio_janela"],
        columns="tecnica_janela",
        values="precision"
    ).reset_index()

    # Renomeia coluna de data
    df_pivot = df_pivot.rename(columns={"data_inicio_janela": "data"})

    # Reorganiza as colunas
    cols_tecnicas = [c for c in df_pivot.columns if c not in ["ativo", "target", "data"]]
    df_pivot = df_pivot[["ativo", "target", "data"] + cols_tecnicas]

    # 🔹 Função para identificar todas as técnicas com o maior precision
    def encontrar_melhores(row):
        max_val = row[cols_tecnicas].max()
        melhores = [col for col in cols_tecnicas if pd.notna(row[col]) and row[col] == max_val]
        return " | ".join(melhores) if melhores else None

    # Aplica a função linha a linha
    df_pivot["melhor_precision"] = df_pivot.apply(encontrar_melhores, axis=1)

    # Ordena por data
    df_pivot = df_pivot.sort_values(by="data").reset_index(drop=True)

    # Salva o resultado individual
    saida = f"./data/comparison/precision/target_in_{file}_pivot.csv"
    df_pivot.to_csv(saida, index=False)

    # Armazena o resultado no dicionário
    dfs[file] = df_pivot

# Exemplo de acesso:
# dfs["VALE3"][["data", "melhor_precision"]].head()
