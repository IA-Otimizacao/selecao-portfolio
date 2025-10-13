import pandas as pd
from tqdm import tqdm
import os

# Garante que a pasta existe
os.makedirs("./data/comparison", exist_ok=True)

todos = ['PETR4', 'ITUB4', 'VALE3']

def carregar_dados_comparacao(file):
    # Carrega os dados
    base_dados = pd.read_csv(f'./data/train_out/target_previsto_{file}.csv')

    # Padroniza a coluna de data
    if 'data' in base_dados.columns:
        base_dados['data'] = pd.to_datetime(base_dados['data'], errors='coerce')

    # Faz o pivot: cada técnica vira uma coluna com seu target_pred
    tabela = (
        base_dados
        .pivot_table(
            index=['ativo', 'target', 'janela', 'data', 'target_real'],  # chaves únicas
            columns='tecnica',                                          # vira coluna
            values='target_pred',                                       # valor preenchido
            aggfunc='first'                                             # caso tenha duplicados
        )
        .reset_index()
    )

    # Garante que as colunas tenham nomes simples (RNA, RF, SVM...)
    tabela.columns.name = None  

    return tabela


resultados = {}

for file in tqdm(todos, desc="Ativos", unit="ativo"):
    resultados[file] = carregar_dados_comparacao(file)
    # Caminho de saída
    output_path = f'./data/comparison/{file}_comparison.xlsx'
    # Salva em Excel
    resultados[file].to_excel(output_path, index=False)
