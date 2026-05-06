import pandas as pd
import os
import numpy as np
import re


def run_comparison_precision():
    
    os.makedirs("./data/ensemble/3_precision", exist_ok=True)

    # =========================
    # PEGA TODOS OS ATIVOS DA PASTA
    # =========================
    pasta_input = "./data/train/inputs/"

    arquivos = [
        f for f in os.listdir(pasta_input)
        if f.startswith("target_in_") and f.endswith(".csv")
    ]

    todos = [
        re.search(r"target_in_(.*)\.csv", f).group(1)
        for f in arquivos
    ]

    # =========================
    # LOOP PRINCIPAL
    # =========================
    for file in todos:

        path = f"{pasta_input}target_in_{file}.csv"

        if not os.path.exists(path):
            continue

        df = pd.read_csv(path)

        # garante tipo numérico
        df["precision"] = pd.to_numeric(df["precision"], errors="coerce")

        # técnica + janela
        df["tecnica_janela"] = (
            df["tecnica"].str.replace(" ", "", regex=False)
            + "_"
            + df["janela"].astype(str)
        )

        df = df[[
            "ativo",
            "target",
            "data_inicio_janela",
            "tecnica_janela",
            "precision"
        ]]

        # pivot
        df_pivot = df.pivot_table(
            index=["ativo", "target", "data_inicio_janela"],
            columns="tecnica_janela",
            values="precision",
            aggfunc="max"
        ).reset_index()

        df_pivot = df_pivot.rename(columns={"data_inicio_janela": "data"})

        # colunas de técnicas
        cols_tecnicas = [
            c for c in df_pivot.columns
            if c not in ["ativo", "target", "data"]
        ]

        # =========================
        # MELHOR PRECISION (ROBUSTO)
        # =========================
        valores = df_pivot[cols_tecnicas].values

        # evita erro quando linha é toda NaN
        max_vals = np.nanmax(valores, axis=1)

        mask = valores == max_vals[:, None]

        melhores = [
            " | ".join(np.array(cols_tecnicas)[linha]) if linha.any() else None
            for linha in mask
        ]

        df_pivot["melhor_precision"] = melhores

        df_pivot.to_csv(
            f"./data/ensemble/3_precision/target_in_{file}_pivot.csv",
            index=False
        )

    print("✅ comparison_precision concluído")