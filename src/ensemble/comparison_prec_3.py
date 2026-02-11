import pandas as pd
import os

def run_comparison_precision():

    todos = ['PETR4', 'ITUB4', 'VALE3']
    os.makedirs("./data/ensemble/3_precision", exist_ok=True)

    for file in todos:
        path = f"./data/train/inputs/target_in_{file}.csv"
        if not os.path.exists(path):
            continue

        df = pd.read_csv(path)
        df["precision"] = pd.to_numeric(df["precision"], errors="coerce")
        df["tecnica_janela"] = df["tecnica"].str.replace(" ", "") + "_" + df["janela"].astype(str)

        df = df[["ativo","target","data_inicio_janela","tecnica_janela","precision"]]

        df_pivot = df.pivot_table(
            index=["ativo","target","data_inicio_janela"],
            columns="tecnica_janela",
            values="precision"
        ).reset_index().rename(columns={"data_inicio_janela":"data"})

        cols_tecnicas = [c for c in df_pivot.columns if c not in ["ativo","target","data"]]

        def melhores(row):
            maxv = row[cols_tecnicas].max()
            return " | ".join([c for c in cols_tecnicas if row[c] == maxv])

        df_pivot["melhor_precision"] = df_pivot.apply(melhores, axis=1)
        df_pivot.to_csv(f"./data/ensemble/3_precision/target_in_{file}_pivot.csv", index=False)

    print("✅ comparison_precision concluído")
