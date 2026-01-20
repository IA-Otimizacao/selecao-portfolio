import pandas as pd
from tqdm import tqdm
import os

# Garante que a pasta existe
os.makedirs("./data/comparison", exist_ok=True)

todos = ['PETR4','ITUB4','VALE3']

def carregar_dados_comparacao(file):
    # Carrega os dados
    base_dados = pd.read_csv(f'./data/train_out/outputs/target_previsto_{file}.csv')

    # Padroniza a coluna de data
    if 'data' in base_dados.columns:
        base_dados['data'] = pd.to_datetime(base_dados['data'], errors='coerce')

    # Lista das colunas que serão usadas no pivot
    index_cols = ['ativo', 'target', 'janela', 'data', 'target_real']

    # Se a coluna resultado_real existir, adiciona
    if 'resultado_real' in base_dados.columns:
        index_cols.append('resultado_real')

    # Faz o pivot
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

    # Remove o nome do eixo das colunas
    tabela.columns.name = None  

    # Cria coluna de ensemble binária
    if all(col in tabela.columns for col in ['RNA', 'Random Forest', 'SVC']):
        soma = tabela['RNA'] + tabela['Random Forest'] + tabela['SVC']
        tabela['ensemble_tecnicas'] = soma.apply(lambda x: 1 if x in [0, 3] else 0)
    else:
        tabela['ensemble_tecnicas'] = None

    return tabela   


# Loop pelos ativos
for file in tqdm(todos, desc="Ativos", unit="ativo"):
    df = carregar_dados_comparacao(file)

    # Caminho de saída CSV
    output_path = f'./data/comparison/{file}_comparison.csv'

    # Salva apenas em CSV simples
    df.to_csv(output_path, index=False, sep=",")

    print(f"✅ CSV salvo: {output_path}")
