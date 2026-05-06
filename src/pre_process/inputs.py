import pandas as pd
import numpy as np
import warnings
import os
from tqdm import tqdm
from src.utils import *

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)      

targets = [1.01, 1.015, 1.02]
ibov = 'ibov_completo_tratado'
janela_temporal = 4

# Pasta onde estão os CSVs dos ativos
input_folder = "./data/pre_process/raw/refinitiv"

# Lista automática de ativos (nome do arquivo sem .csv)
todos = [
    os.path.splitext(f)[0] 
    for f in os.listdir(input_folder) 
    if f.endswith(".csv")
]


def main():

    ibov_status = carregar_ibov(ibov)
    base_petr4 = carregar_dados(os.path.join(input_folder, "PETR4.csv"))

    os.makedirs("./data/pre_process/raw/ibov_status", exist_ok=True)
    os.makedirs("./data/pre_process/curated", exist_ok=True)

    print("\n Iniciando processamento e salvamento em curated...\n")

    # Loop principal que percorre cada ativo
    for file in tqdm(todos, desc="Ativos", unit="ativo"):

        file_path = os.path.join(input_folder, f"{file}.csv")

        base_dados_original = carregar_dados(file_path)
        base_dados_original = padronizar_colunas(base_dados_original)
        base_dados_original = remover_linhas_invalidas(base_dados_original)

        # Alinha os dados do ativo com o calendário da PETR4
        base_dados_original = pd.merge(
            base_petr4[['Exchange Date']],
            base_dados_original,
            on='Exchange Date',
            how='left'
        )

        # Marca linhas adicionadas
        base_dados_original['data_adicional'] = np.where(
            base_dados_original['Open'].isnull(), 1, 0
        )

        # Preenche valores faltantes
        base_dados_original = base_dados_original.ffill()
        base_dados_original = base_dados_original.dropna()

        # Junta com IBOV
        base_dados_original = pd.merge(
            base_dados_original,
            ibov_status[['Exchange Date', file]],
            on='Exchange Date',
            how='left'
        )

        # Renomeia coluna de status
        base_dados_original.rename(columns={file: 'ibov_status'}, inplace=True)

        # Salva base intermediária
        base_dados_original.to_csv(
            f"./data/pre_process/raw/ibov_status/{file}.csv",
            index=False
        )

        # Loop de targets
        for target in tqdm(targets, desc=f"Targets de {file}", leave=False):

            base_dados = base_dados_original.copy()

            # Filtra período dentro do IBOV
            base_dados = base_dados[base_dados['ibov_status'] == 1].copy()

            # Garante tipo numérico
            base_dados = garantir_numerico(
                base_dados,
                ['Open', 'High', 'Close']
            )

            # Variáveis derivadas
            base_dados = calcular_variaveis(base_dados)

            # Target
            base_dados = calcular_target(
                base_dados,
                target,
                janela_temporal
            )

            base_dados = base_dados.iloc[14:]

            # Salva
            curated_path = f"./data/pre_process/curated/{file}_target_{target}.csv"

            base_dados.to_csv(
                curated_path,
                index=False
            )

    print("\nProcessamento inicial finalizado! Dados salvos em ./data/pre_process/curated.")
