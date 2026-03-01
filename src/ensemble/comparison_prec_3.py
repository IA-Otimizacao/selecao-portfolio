import pandas as pd
import os
import numpy as np

def run_comparison_precision():
    
    todos = ['PETR4', 'ITUB4', 'VALE3']
    os.makedirs("./data/ensemble/3_precision", exist_ok=True)

    # Loop por cada ativo
    for file in todos:

        path = f"./data/train/inputs/target_in_{file}.csv"

        # Caso o arquivo não exista, pula para o próximo ativo
        if not os.path.exists(path):
            continue

        # Leitura do dataset
        df = pd.read_csv(path)

        # Garante que precision seja numérico
        df["precision"] = pd.to_numeric(df["precision"], errors="coerce")

        # Cria a coluna que identifica técnica + janela. Ex: RNA_60, SVC_90 etc
        df["tecnica_janela"] = (
            df["tecnica"].str.replace(" ", "", regex=False)
            + "_"
            + df["janela"].astype(str)
        )

        # Mantém apenas colunas necessárias
        df = df[[
            "ativo",
            "target",
            "data_inicio_janela",
            "tecnica_janela",
            "precision"
        ]]

        # Pivot da tabela
        df_pivot = df.pivot_table(
            index=["ativo", "target", "data_inicio_janela"],
            columns="tecnica_janela",
            values="precision",
            aggfunc="max"
        ).reset_index()

        # Renomeia coluna de data
        df_pivot = df_pivot.rename(columns={"data_inicio_janela": "data"})

        # Identificar técnicas
        cols_tecnicas = [
            c for c in df_pivot.columns
            if c not in ["ativo", "target", "data"]
        ]

        # Encontrar maior precision
        valores = df_pivot[cols_tecnicas].values

        # maior valor por linha
        max_vals = np.nanmax(valores, axis=1)

        # máscara indicando quais técnicas atingiram o máximo
        mask = valores == max_vals[:, None]

        # gerar lista das melhores técnicas
        melhores = [
            " | ".join(np.array(cols_tecnicas)[linha])
            for linha in mask
        ]

        # adiciona coluna final
        df_pivot["melhor_precision"] = melhores

        df_pivot.to_csv(
            f"./data/ensemble/3_precision/target_in_{file}_pivot.csv",
            index=False
        )

    print("✅ comparison_precision concluído")