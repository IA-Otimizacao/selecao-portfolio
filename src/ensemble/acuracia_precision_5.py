import pandas as pd
import os

def run_acuracia_precision():

    todos = ['PETR4','ITUB4','VALE3']
    os.makedirs("./data/train/analytics", exist_ok=True)

    res = []

    for file in todos:
        df_m = pd.read_csv(f"./data/ensemble/4_melhor_precision_valor/{file}_melhor_precision_valor.csv")
        df_e = pd.read_csv(f"./data/ensemble/2_tot_par/{file}_ensemble_jan_tot_e_parcial.csv")

        df = pd.merge(
            df_m[['ativo','target','data','concordancia_valor']].rename(columns={'concordancia_valor':'in_precision'}),
            df_e[['ativo','target','data','target_real','resultado_real','esmble_jan_tot','esmble_jan_par']],
            on=['ativo','target','data']
        )

        for col in ['in_precision','esmble_jan_tot','esmble_jan_par']:
            acc = (df[col]==df['target_real']).mean()*100
            res.append({'ativo':file,'modelo':col,'acc':round(acc,2)})

    pd.DataFrame(res).to_csv("./data/train/analytics/acuracia_precision_ensembles1.csv", index=False)
    print("✅ acuracia_precision concluído")
