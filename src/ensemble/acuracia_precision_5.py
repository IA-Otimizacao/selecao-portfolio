import pandas as pd
import os
import re


def run_acuracia_precision():

    os.makedirs("./data/train/analytics", exist_ok=True)

    # =========================
    # PEGA ATIVOS DAS DUAS PASTAS
    # =========================
    pasta_m = "./data/ensemble/4_melhor_precision_valor/"
    pasta_e = "./data/ensemble/2_tot_par/"

    arquivos_m = [
        f for f in os.listdir(pasta_m)
        if f.endswith("_melhor_precision_valor.csv")
    ]

    arquivos_e = [
        f for f in os.listdir(pasta_e)
        if f.endswith("_ensemble_jan_tot_e_parcial.csv")
    ]

    ativos_m = {
        re.search(r"(.*)_melhor_precision_valor\.csv", f).group(1)
        for f in arquivos_m
    }

    ativos_e = {
        re.search(r"(.*)_ensemble_jan_tot_e_parcial\.csv", f).group(1)
        for f in arquivos_e
    }

    # interseção → só roda onde existe nos dois
    todos = list(ativos_m & ativos_e)

    res = []

    # =========================
    # LOOP PRINCIPAL
    # =========================
    for file in todos:

        df_m = pd.read_csv(
            f"{pasta_m}{file}_melhor_precision_valor.csv"
        )

        df_e = pd.read_csv(
            f"{pasta_e}{file}_ensemble_jan_tot_e_parcial.csv"
        )

        # garante consistência de data (evita erro silencioso no merge)
        if 'data' in df_m.columns:
            df_m['data'] = pd.to_datetime(df_m['data'], errors='coerce')

        if 'data' in df_e.columns:
            df_e['data'] = pd.to_datetime(df_e['data'], errors='coerce')

        df = pd.merge(

            df_m[['ativo','target','data','concordancia_valor']]
            .rename(columns={'concordancia_valor':'in_precision'}),

            df_e[['ativo','target','data','target_real','resultado_real',
                  'esmble_jan_tot','esmble_jan_par']],

            on=['ativo','target','data']
        )

        for col in ['in_precision','esmble_jan_tot','esmble_jan_par']:

            acc = (df[col] == df['target_real']).mean() * 100

            res.append({
                'ativo': file,
                'modelo': col,
                'acc': round(acc, 2)
            })

    pd.DataFrame(res).to_csv(
        "./data/train/analytics/acuracia_precision_ensembles1.csv",
        index=False
    )

    print("✅ acuracia_precision concluído")