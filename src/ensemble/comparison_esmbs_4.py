import pandas as pd
import os
from tqdm import tqdm

def run_comparison_esmbs():

    todos = ['PETR4','ITUB4','VALE3']
    os.makedirs("./data/ensemble/4_melhor_precision_valor", exist_ok=True)

    for file in tqdm(todos):
        df_ens = pd.read_csv(f"./data/ensemble/2_tot_par/{file}_ensemble_jan_tot_e_parcial.csv")
        df_prec = pd.read_csv(f"./data/ensemble/3_precision/target_in_{file}_pivot.csv")

        res = []
        for _, r in df_prec.iterrows():
            tecnicas = [t.strip() for t in str(r['melhor_precision']).split('|')]
            linha = df_ens[
                (df_ens['ativo']==r['ativo']) &
                (df_ens['target']==r['target']) &
                (df_ens['data']==r['data'])
            ]

            valores = []
            for t in tecnicas:
                t = t.replace(" ","")
                valores.append(str(linha.iloc[0][t]) if not linha.empty and t in linha.columns else "NA")

            res.append({
                'ativo': r['ativo'],
                'target': r['target'],
                'data': r['data'],
                'valor_melhor_precision': ','.join(valores)
            })

        df = pd.DataFrame(res)

        def conc(v):
            vs = [int(float(x)) for x in v.split(',') if x not in ['NA','']]
            return vs[0] if vs and all(x==vs[0] for x in vs) else 0

        df['concordancia_valor'] = df['valor_melhor_precision'].apply(conc)
        df = df[~df['valor_melhor_precision'].str.contains('NA')]

        df.to_csv(f"./data/ensemble/4_melhor_precision_valor/{file}_melhor_precision_valor.csv", index=False)

    print("✅ comparison_esmbs concluído")
