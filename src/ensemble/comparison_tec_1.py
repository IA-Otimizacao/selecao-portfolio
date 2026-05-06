import pandas as pd
from tqdm import tqdm
import os
import re


def run_comparison_tec():
    os.makedirs("./data/ensemble/comparison", exist_ok=True)

    # =========================
    # PEGA TODOS OS ATIVOS DA PASTA
    # =========================
    pasta_input = "./data/train/outputs/"

    arquivos = [
        f for f in os.listdir(pasta_input)
        if f.startswith("target_previsto_") and f.endswith(".csv")
    ]

    # extrai nome do ativo do arquivo
    todos = [
        re.search(r"target_previsto_(.*)\.csv", f).group(1)
        for f in arquivos
    ]

    # =========================
    # FUNÇÃO DE PROCESSAMENTO
    # =========================
    def carregar_dados_comparacao(file):

        base_dados = pd.read_csv(f'{pasta_input}target_previsto_{file}.csv')

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
            tabela['ensemble_tecnicas'] = soma.apply(
                lambda x: 1 if x in [0, 3] else 0
            )
        else:
            tabela['ensemble_tecnicas'] = None

        # 🔽 ORDENAÇÃO: primeiro pelo target (1.01, 1.015, 1.02...), depois pela data
        tabela = tabela.sort_values(['target', 'data'])

        return tabela

    # =========================
    # LOOP PRINCIPAL
    # =========================
    for file in tqdm(todos, desc="Ativos"):
        df = carregar_dados_comparacao(file)
        df.to_csv(
            f'./data/ensemble/1_comparison/{file}_comparison.csv',
            index=False
        )

    print("✅ comparison_tec concluído")


if __name__ == "__main__":
    run_comparison_tec()