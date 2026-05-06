import os
import pandas as pd
import numpy as np
import re


def join_intr(
    input_folder="./data/ensemble/5_6_completo/",
    curated_folder="./data/pre_process/curated/",
    output_folder="./data/ensemble/7_intraday_join/"
):

    os.makedirs(output_folder, exist_ok=True)

    # =========================
    # PEGA APENAS ARQUIVOS VÁLIDOS
    # =========================
    arquivos = [
        f for f in os.listdir(input_folder)
        if f.lower().endswith(".csv") and "_comparacao_completa" in f
    ]

    for fname in arquivos:

        path = os.path.join(input_folder, fname)

        print(f"\n📂 Processando arquivo: {fname}")

        df = pd.read_csv(path, sep="|")

        # limpeza colunas
        df.columns = (
            df.columns
            .str.strip()
            .str.replace("\ufeff", "", regex=True)
        )

        # garante datetime
        df['data'] = pd.to_datetime(df['data'], errors='coerce')

        df = df.sort_values(['ativo', 'target', 'data'])

        results = []

        # =========================
        # LOOP POR ATIVO + TARGET
        # =========================
        for (ativo, target), g in df.groupby(['ativo', 'target'], sort=False):

            print(f"  ▶ Join | ativo={ativo} | target={target}")

            g = g.reset_index(drop=True)

            curated_path = os.path.join(
                curated_folder,
                f"{ativo}_target_{target}.csv"
            )

            if not os.path.exists(curated_path):
                print(f"  ⚠️ Não encontrado: {curated_path}")
                continue

            df_cur = pd.read_csv(curated_path)

            df_cur.columns = df_cur.columns.str.strip()

            df_cur['Exchange Date'] = pd.to_datetime(
                df_cur['Exchange Date'],
                errors='coerce'
            )

            # =========================
            # MERGE
            # =========================
            merged = pd.merge(
                g,
                df_cur[['Exchange Date', 'Close', 'Open', 'Low', 'High']],
                left_on='data',
                right_on='Exchange Date',
                how='left'
            )

            merged = merged.drop(columns=['Exchange Date'])

            # =========================
            # FEATURE ENGINEERING
            # =========================
            merged['ln_open'] = np.log(merged['Open'])
            merged['ln_high'] = np.log(merged['High'])

            merged['log_return'] = merged['ln_high'] - merged['ln_open']
            merged['target_log'] = np.log(merged['target'])

            merged['log_return_binario'] = (
                merged['log_return'] >= merged['target_log']
            ).astype(int)

            # =========================
            # COLUNAS FINAIS
            # =========================
            merged = merged[
                [
                    'ativo',
                    'target',
                    'data',
                    'Close',
                    'Open',
                    'Low',
                    'High',
                    'ln_open',
                    'ln_high',
                    'log_return',
                    'log_return_binario',
                    'esmble_jan_tot',
                    'esmble_jan_par',
                    'in_precision'
                ]
            ]

            results.append(merged)

        # =========================
        # SALVAMENTO
        # =========================
        if results:

            final_df = pd.concat(results, ignore_index=True)

            out_path = os.path.join(
                output_folder,
                fname.replace('.csv', '_intraday.csv')
            )

            final_df.to_csv(out_path, index=False, sep="|")

            print(f"✅ Salvo em: {out_path}")

        else:
            print("⚠️ Nenhum dado processado.")


# runner
def run_join_intr():
    join_intr()


if __name__ == "__main__":
    run_join_intr()