import pandas as pd
import numpy as np
import warnings
import os
from tqdm import tqdm
from src.utils import *

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)      

todos = ['PETR4', 'ITUB4', 'VALE3']
targets = [1.01, 1.015, 1.02]

ibov = 'ibov_filtrado'

janela_temporal = 4


def main():

    ibov_status = carregar_ibov(ibov)
    base_petr4 = carregar_dados('PETR4')

    os.makedirs("./data/pre_process/raw/ibov_status", exist_ok=True)
    os.makedirs("./data/pre_process/curated", exist_ok=True)

    print("\n Iniciando processamento e salvamento em curated...\n")

 
    # Loop principal que percorre cada ativo
    for file in tqdm(todos, desc="Ativos", unit="ativo"):

        base_dados_original = carregar_dados(file)
        base_dados_original = padronizar_colunas(base_dados_original)
        base_dados_original = remover_linhas_invalidas(base_dados_original)

        # Alinha os dados do ativo com o calendário da PETR4
        base_dados_original = pd.merge(
            base_petr4[['Exchange Date']],
            base_dados_original,
            on='Exchange Date',
            how='left'
        )

        # Cria uma coluna indicando se aquela linha foi adicionada
        base_dados_original['data_adicional'] = np.where(
            base_dados_original['Open'].isnull(), 1, 0
        )

        # Preenche valores faltantes com o último valor válido anterior
        base_dados_original = base_dados_original.ffill()

        # Adiciona informação do IBOV (se o ativo fazia parte do índice)
        base_dados_original = pd.merge(
            base_dados_original,
            ibov_status[['Exchange Date', file]],
            on='Exchange Date',
            how='left'
        )

        # Renomeia a coluna do ativo para "ibov_status"
        base_dados_original.rename(columns={file: 'ibov_status'}, inplace=True)

        # Salva base intermediária contendo o status do IBOV
        base_dados_original.to_csv(
            f"./data/raw/ibov_status/{file}.csv",
            index=False
        )

        # Loop para calcular diferentes targets
        for target in tqdm(targets, desc=f"Targets de {file}", leave=False):

            # Cria uma cópia da base original para não sobrescrever os dados
            base_dados = base_dados_original.copy()

            # Mantém apenas períodos em que o ativo fazia parte do IBOV
            base_dados = base_dados[base_dados['ibov_status'] == 1].copy()

            # Garante que as colunas numéricas estejam no formato correto
            base_dados = garantir_numerico(
                base_dados,
                ['Open', 'High', 'Close']
            )

            # Calcula variáveis derivadas
            base_dados = calcular_variaveis(base_dados)

            # Calcula o target (se o preço atingiu o retorno desejado dentro da janela temporal definida)
            base_dados = calcular_target(
                base_dados,
                target,
                janela_temporal
            )
            base_dados = base_dados.iloc[14:]

            curated_path = f"./data/pre_process/curated/{file}_target_{target}.csv"

            base_dados.to_csv(
                curated_path,
                index=False
            )

    print("\nProcessamento inicial finalizado! Dados salvos em ./data/pre_process/curated.")

if __name__ == "__main__":
    main()