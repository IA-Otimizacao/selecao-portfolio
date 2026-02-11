import pandas as pd
from tqdm import tqdm
import os

def run_comparison_jan():
    todos = ['PETR4', 'ITUB4', 'VALE3']
    os.makedirs("./data/ensemble/2_tot_par", exist_ok=True)

    def carregar_e_criar_ensembles(file):
        base = pd.read_csv(f'./data/ensemble/1_comparison/{file}_comparison.csv')

        tabela = (
            base
            .pivot_table(
                index=['ativo', 'target', 'data', 'target_real', 'resultado_real'],
                columns='janela',
                values=['RNA', 'Random Forest', 'SVC'],
                aggfunc='first'
            )
        )

        tabela.columns = [f"{c[0]}_{c[1]}" for c in tabela.columns]
        tabela = tabela.reset_index()
        tabela.columns = [c.replace("Random Forest", "RandomForest") for c in tabela.columns]

        tecnicas_cols = [c for c in tabela.columns if any(t in c for t in ['RNA','RandomForest','SVC'])]
        tabela = tabela.dropna(subset=tecnicas_cols)

        tabela['esmble_jan_tot'] = (tabela[tecnicas_cols].sum(axis=1) == 9).astype(int)

        tabela['esmble_jan_par'] = (
            ((tabela[tecnicas_cols].sum(axis=1) >= 5) &
             (tabela[['RNA_60','RNA_75','RNA_90']].sum(axis=1) >= 1) &
             (tabela[['RandomForest_60','RandomForest_75','RandomForest_90']].sum(axis=1) >= 1) &
             (tabela[['SVC_60','SVC_75','SVC_90']].sum(axis=1) >= 1))
        ).astype(int)

        return tabela, tecnicas_cols

    for file in tqdm(todos):
        df, cols = carregar_e_criar_ensembles(file)
        df[['ativo','target','data','target_real','resultado_real'] + cols + ['esmble_jan_tot','esmble_jan_par']] \
            .to_csv(f'./data/ensemble/2_tot_par/{file}_ensemble_jan_tot_e_parcial.csv', index=False)

    print("✅ comparison_jan concluído")
