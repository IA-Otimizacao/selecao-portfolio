import os
import pandas as pd
import re


def separar_por_tecnica(
    input_folder="./data/ensemble/10_targets_alinhados/",
    output_folder="./data/ensemble/11_targets_por_tecnica/"
):

    os.makedirs(output_folder, exist_ok=True)

    tecnicas = ['esmble_jan_tot', 'esmble_jan_par', 'in_precision']

    # =========================
    # LOOP NOS ARQUIVOS
    # =========================
    for fname in os.listdir(input_folder):

        if not fname.endswith(".csv"):
            continue

        caminho = os.path.join(input_folder, fname)

        print(f"\n📂 Processando: {fname}")

        df = pd.read_csv(caminho)

        # =========================
        # IDENTIFICA ATIVOS DINAMICAMENTE
        # =========================
        # padrão: ATIVO_tecnica
        ativos = sorted({
            re.match(r"(.*?)_", col).group(1)
            for col in df.columns
            if "_" in col and not col.startswith("data")
        })

        # pega target do nome
        target = fname.replace("target_", "").replace(".csv", "")

        # =========================
        # LOOP POR TÉCNICA
        # =========================
        for tecnica in tecnicas:

            colunas = ['data']
            colunas_binarias = []

            for ativo in ativos:

                col_bin = f"{ativo}_{tecnica}"
                col_rend_dec = f"{ativo}_rend_decisao_{tecnica}"
                col_rend_venda = f"{ativo}_rend_venda_{tecnica}"

                # só adiciona se existir (robustez)
                if col_bin in df.columns:
                    colunas.append(col_bin)
                    colunas_binarias.append(col_bin)

                if col_rend_dec in df.columns:
                    colunas.append(col_rend_dec)

                if col_rend_venda in df.columns:
                    colunas.append(col_rend_venda)

            novo_df = df[colunas].copy()

            # =========================
            # FILTRO (mantém linhas com pelo menos 1 sinal)
            # =========================
            if colunas_binarias:
                novo_df = novo_df[
                    novo_df[colunas_binarias].notna().any(axis=1)
                ]

            output_name = f"{tecnica}_target_{target}.csv"
            output_path = os.path.join(output_folder, output_name)

            novo_df.to_csv(output_path, index=False)

            print(f"✅ Salvo: {output_name}")