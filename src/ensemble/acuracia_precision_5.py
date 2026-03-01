import pandas as pd
import os


def run_acuracia_precision():

    todos = ['PETR4','ITUB4','VALE3']

    os.makedirs("./data/train/analytics", exist_ok=True)

    res = []

    # Loop pelos ativos
    for file in todos:

        # Dataset que contém: valor previsto pelas técnicas com melhor precision e concordância entre essas técnicas
        df_m = pd.read_csv(
            f"./data/ensemble/4_melhor_precision_valor/{file}_melhor_precision_valor.csv"
        )

        # Dataset que contém: resultados dos ensembles de janelas (total e parcial) e target real e resultado real
        df_e = pd.read_csv(
            f"./data/ensemble/2_tot_par/{file}_ensemble_jan_tot_e_parcial.csv"
        )

        # Merge para juntar as previsões com o target real, usando como chave: ativo + target + data
        df = pd.merge(

            # Dataset com o valor previsto pelas técnicas de melhor precision
            df_m[['ativo','target','data','concordancia_valor']]
            .rename(columns={'concordancia_valor':'in_precision'}),

            # Dataset com os resultados dos ensembles
            df_e[['ativo','target','data','target_real','resultado_real',
                  'esmble_jan_tot','esmble_jan_par']],

            # Chaves usadas para unir os dados
            on=['ativo','target','data']
        )

        # Cálculo da acurácia: Para cada modelo de decisão, calcula a acurácia comparando a previsão com o target real
        for col in ['in_precision','esmble_jan_tot','esmble_jan_par']:

            acc = (df[col] == df['target_real']).mean() * 100

            res.append({
                'ativo': file,     # ativo analisado
                'modelo': col,     # modelo/estratégia
                'acc': round(acc,2) # acurácia arredondada
            })

    pd.DataFrame(res).to_csv(
        "./data/train/analytics/acuracia_precision_ensembles1.csv",
        index=False
    )

    print("✅ acuracia_precision concluído")