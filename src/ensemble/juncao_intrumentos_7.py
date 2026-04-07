import os
import pandas as pd
import numpy as np


def join_intr(
    input_folder="./data/ensemble/5_6_completo/",   # 📥 Pasta com os CSVs de entrada (ensemble)
    curated_folder="./data/pre_process/curated/",   # 📊 Pasta com dados OHLC (curated)
    output_folder="./data/ensemble/7_intraday_join/" # 📤 Pasta de saída final
):

    # Garante que a pasta de saída existe
    os.makedirs(output_folder, exist_ok=True)

    # 🔁 Loop em todos os arquivos da pasta de entrada
    for fname in os.listdir(input_folder):

        # Ignora arquivos que não são CSV
        if not fname.lower().endswith(".csv"):
            continue

        path = os.path.join(input_folder, fname)

        print(f"\n📂 Processando arquivo: {fname}")

        # 📥 Leitura do arquivo principal (ensemble)
        df = pd.read_csv(path, sep="|")

        # 🧹 Limpeza de nomes de colunas (remove espaços e caracteres invisíveis)
        df.columns = (
            df.columns
            .str.strip()
            .str.replace("\ufeff", "", regex=True)
        )

        # 🔄 Conversão de tipos
        df['data'] = pd.to_datetime(df['data'])

        # 📊 Ordenação importante para manter consistência temporal
        df = df.sort_values(['ativo', 'target', 'data'])

        results = []  # Lista para acumular resultados por ativo/target

        # 🔁 Loop principal: separa por ativo e target
        for (ativo, target), g in df.groupby(['ativo', 'target'], sort=False):

            print(f"  ▶ Join | ativo={ativo} | target={target}")

            # Reset de índice após groupby
            g = g.reset_index(drop=True)

            # 📂 Monta caminho do arquivo curated correspondente
            curated_path = os.path.join(
                curated_folder,
                f"{ativo}_target_{target}.csv"
            )

            # ⚠️ Se não existir o arquivo curated, pula
            if not os.path.exists(curated_path):
                print(f"  ⚠️ Arquivo não encontrado: {curated_path}")
                continue

            # 📥 Leitura do curated (dados OHLC)
            df_cur = pd.read_csv(curated_path)

            # Limpeza de colunas
            df_cur.columns = df_cur.columns.str.strip()

            # 🔄 Conversão de data (nome diferente no curated)
            df_cur['Exchange Date'] = pd.to_datetime(df_cur['Exchange Date'])

            # 🔗 MERGE PRINCIPAL
            # Junta dados do modelo com preços OHLC do mesmo dia
            merged = pd.merge(
                g,
                df_cur[['Exchange Date', 'Close', 'Open', 'Low', 'High']],
                left_on='data',              # data do modelo
                right_on='Exchange Date',    # data do mercado
                how='left'                  # mantém tudo do modelo (mesmo sem preço)
            )

            # Remove coluna duplicada de data
            merged = merged.drop(columns=['Exchange Date'])

            # 🔥 FEATURE ENGINEERING (LOG)
            # Log dos preços → usado para retorno logarítmico
            merged['ln_open'] = np.log(merged['Open'])
            merged['ln_high'] = np.log(merged['High'])

            # 📈 Retorno intraday: high vs open
            merged['log_return'] = merged['ln_high'] - merged['ln_open']

            # 🎯 Target em log (pra comparar com retorno log)
            merged['target_log'] = np.log(merged['target'])

            # 🔥 LABEL FINAL (muito importante pro seu modelo)
            # 1 = atingiu target intraday
            # 0 = não atingiu
            merged['log_return_binario'] = (
                merged['log_return'] >= merged['target_log']
            ).astype(int)

            # 📦 Seleção final de colunas (define seu dataset final)
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

            # Guarda resultado desse ativo/target
            results.append(merged)

        # 🔗 Concatena todos os ativos/targets do arquivo
        if results:
            final_df = pd.concat(results, ignore_index=True)

            # 📤 Define nome do arquivo de saída
            out_path = os.path.join(
                output_folder,
                fname.replace('.csv', '_intraday.csv')
            )

            # 💾 Salva resultado final
            final_df.to_csv(out_path, index=False, sep="|")

            print(f"✅ Salvo em: {out_path}")
        else:
            print("⚠️ Nenhum dado processado para este arquivo.")


# 🚀 Runner (ponto de entrada)
def run_join_intr():
    join_intr()


# ▶ Execução direta do script
if __name__ == "__main__":
    run_join_intr()