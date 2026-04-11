import os
import pandas as pd


def separar_por_tecnica(
    input_folder="./data/ensemble/10_targets_alinhados/",
    output_folder="./data/ensemble/11_targets_por_tecnica/"
):

    os.makedirs(output_folder, exist_ok=True)

    tecnicas = ['esmble_jan_tot', 'esmble_jan_par', 'in_precision']
    ativos = ['PETR4', 'VALE3', 'ITUB4']

    for fname in os.listdir(input_folder):

        if not fname.endswith(".csv"):
            continue

        caminho = os.path.join(input_folder, fname)
        df = pd.read_csv(caminho)

        # 🔥 pega o target direto do nome
        target = fname.replace("target_", "").replace(".csv", "")

        for tecnica in tecnicas:

            colunas = ['data']

            for ativo in ativos:
                colunas.append(f"{ativo}_{tecnica}")
                colunas.append(f"{ativo}_rend_decisao_{tecnica}")

            novo_df = df[colunas].copy()

            output_name = f"{tecnica}_target_{target}.csv"
            output_path = os.path.join(output_folder, output_name)

            novo_df.to_csv(output_path, index=False)

            print(f"Salvo: {output_name}")


separar_por_tecnica()