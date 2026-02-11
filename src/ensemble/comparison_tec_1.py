import pandas as pd
from tqdm import tqdm
import os

def run_comparison_tec():
    
    os.makedirs("./data/ensemble/comparison", exist_ok=True)
    todos = ['PETR4','ITUB4','VALE3']

    def carregar_dados_comparacao(file):
        base_dados = pd.read_csv(f'./data/train/outputs/target_previsto_{file}.csv')

        if 'data' in base_dados.columns:
            base_dados['data'] = pd.to_datetime(base_dados['data'], errors='coerce')

        index_cols = ['ativo', 'target', 'janela', 'data', 'target_real']
        if 'resultado_real' in base_dados.columns:
            index_cols.append('resultado_real')

        tabela = (
            base_dados
            .pivot_table(
                index=index_cols,
                columns='tecnica',
                values='target_pred',
                aggfunc='first'
            )
            .reset_index()
        )

        tabela.columns.name = None  

        if all(col in tabela.columns for col in ['RNA', 'Random Forest', 'SVC']):
            soma = tabela['RNA'] + tabela['Random Forest'] + tabela['SVC']
            tabela['ensemble_tecnicas'] = soma.apply(lambda x: 1 if x in [0, 3] else 0)
        else:
            tabela['ensemble_tecnicas'] = None

        return tabela   

    for file in tqdm(todos, desc="Ativos"):
        df = carregar_dados_comparacao(file)
        df.to_csv(f'./data/ensemble/1_comparison/{file}_comparison.csv', index=False)

    print("✅ comparison_tec concluído")
