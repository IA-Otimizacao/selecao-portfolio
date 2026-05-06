import pandas as pd
import os
from tqdm import tqdm
import re


def run_comparison_esmbs():

    os.makedirs("./data/ensemble/4_melhor_precision_valor", exist_ok=True)

    # =========================
    # PEGA ATIVOS DAS DUAS PASTAS
    # =========================
    pasta_ens = "./data/ensemble/2_tot_par/"
    pasta_prec = "./data/ensemble/3_precision/"

    arquivos_ens = [
        f for f in os.listdir(pasta_ens)
        if f.endswith("_ensemble_jan_tot_e_parcial.csv")
    ]

    arquivos_prec = [
        f for f in os.listdir(pasta_prec)
        if f.startswith("target_in_") and f.endswith("_pivot.csv")
    ]

    ativos_ens = {
        re.search(r"(.*)_ensemble_jan_tot_e_parcial\.csv", f).group(1)
        for f in arquivos_ens
    }

    ativos_prec = {
        re.search(r"target_in_(.*)_pivot\.csv", f).group(1)
        for f in arquivos_prec
    }

    # interseção → só roda onde existe nos dois lados
    todos = list(ativos_ens & ativos_prec)

    # =========================
    # LOOP PRINCIPAL
    # =========================
    for file in tqdm(todos, desc="Ativos"):

        df_ens = pd.read_csv(
            f"{pasta_ens}{file}_ensemble_jan_tot_e_parcial.csv"
        )

        df_prec = pd.read_csv(
            f"{pasta_prec}target_in_{file}_pivot.csv"
        )

        res = []

        for _, r in df_prec.iterrows():

            tecnicas = [
                t.strip()
                for t in str(r['melhor_precision']).split('|')
                if t.strip() != ""
            ]

            linha = df_ens[
                (df_ens['ativo'] == r['ativo']) &
                (df_ens['target'] == r['target']) &
                (df_ens['data'] == r['data'])
            ]

            valores = []

            for t in tecnicas:

                t = t.replace(" ", "")

                if not linha.empty and t in linha.columns:
                    valores.append(str(linha.iloc[0][t]))
                else:
                    valores.append("NA")

            res.append({
                'ativo': r['ativo'],
                'target': r['target'],
                'data': r['data'],
                'valor_melhor_precision': ','.join(valores)
            })

        df = pd.DataFrame(res)

        # =========================
        # FUNÇÃO DE CONCORDÂNCIA
        # =========================
        def conc(v):
            vs = [
                int(float(x))
                for x in v.split(',')
                if x not in ['NA', '']
            ]

            if vs and all(x == vs[0] for x in vs):
                return vs[0]

            return 0

        df['concordancia_valor'] = df['valor_melhor_precision'].apply(conc)

        # remove linhas com NA
        df = df[~df['valor_melhor_precision'].str.contains('NA')]

        df.to_csv(
            f"./data/ensemble/4_melhor_precision_valor/{file}_melhor_precision_valor.csv",
            index=False
        )

    print("✅ comparison_esmbs concluído")