import pandas as pd
import numpy as np
import warnings
import os
from tqdm import tqdm
from src.utils import *
warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)


# file = 'ITUB4'
# target = [1.01]


todos = ['PETR4', 'ITUB4', 'VALE3']
targets = [1.01, 1.015, 1.02]
ibov = 'ibov_filtrado'
janela_temporal = 4

def main():
    ibov_status = carregar_ibov(ibov)
    base_petr4 = carregar_dados('PETR4')

    os.makedirs("./data/raw/ibov_status", exist_ok=True)
    os.makedirs("./data/curated", exist_ok=True)

    print("\n Iniciando processamento e salvamento em curated...\n")

    for file in tqdm(todos, desc="Ativos", unit="ativo"):
        base_dados_original = carregar_dados(file)
        base_dados_original = padronizar_colunas(base_dados_original)
        base_dados_original = remover_linhas_invalidas(base_dados_original)

        base_dados_original = pd.merge(base_petr4[['Exchange Date']], base_dados_original, on='Exchange Date', how='left')
        base_dados_original['data_adicional'] = np.where(base_dados_original['Open'].isnull(), 1, 0)
        base_dados_original = base_dados_original.ffill()

        base_dados_original = pd.merge(base_dados_original, ibov_status[['Exchange Date', file]], on='Exchange Date', how='left')
        base_dados_original.rename(columns={file: 'ibov_status'}, inplace=True)
        base_dados_original.to_csv(f"./data/raw/ibov_status/{file}.csv", index=False)

        for target in tqdm(targets, desc=f"Targets de {file}", leave=False):
            base_dados = base_dados_original.copy()
            base_dados = base_dados[base_dados['ibov_status'] == 1].copy()
            base_dados = garantir_numerico(base_dados, ['Open', 'High', 'Close'])
            base_dados = calcular_variaveis(base_dados)
            base_dados = calcular_target(base_dados, target, janela_temporal)
            base_dados = base_dados.iloc[14:]

            curated_path = f"./data/curated/{file}_target_{target}.csv"
            base_dados.to_csv(curated_path, index=False)

    print("\nProcessamento inicial finalizado! Dados salvos em ./data/curated.")

if __name__ == "__main__":
    main()
