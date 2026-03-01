import pandas as pd
from tqdm import tqdm
import os


def gerar_comparacao_completa(
    path_ensemble="./data/ensemble/2_tot_par",
    path_precision="./data/ensemble/4_melhor_precision_valor",
    path_saida="./data/ensemble/5_6_completo"
):
    """
    Função responsável por gerar um dataset final de comparação
    entre diferentes estratégias de decisão.

    Ela une:
    - os resultados dos ensembles de janelas
    - o resultado baseado na melhor precision

    O objetivo é centralizar todas as estratégias em um único dataset
    para facilitar análises e comparação de performance.

    Parâmetros:
    path_ensemble -> pasta onde estão os arquivos de ensemble
    path_precision -> pasta onde estão os arquivos de melhor precision
    path_saida -> pasta onde será salvo o dataset final
    """

    ativos = ['PETR4', 'ITUB4', 'VALE3']
    os.makedirs(path_saida, exist_ok=True)

    # Loop pelos ativos
    for ativo in tqdm(ativos):

        # Dataset contendo os resultados dos ensembles (ensemble total e ensemble parcial)
        df_ens = pd.read_csv(
            f"{path_ensemble}/{ativo}_ensemble_jan_tot_e_parcial.csv"
        )

        # Dataset contendo os resultados da estratégia baseada nas técnicas com maior precision
        df_prec = pd.read_csv(
            f"{path_precision}/{ativo}_melhor_precision_valor.csv"
        )

        # Garantia de consistência nos tipos das colunas
        for col in ['ativo', 'target', 'data']:

            df_ens[col] = df_ens[col].astype(str)
            df_prec[col] = df_prec[col].astype(str)

        # Seleção e renomeação da coluna de interesse
        # Do dataset de precision, mantemos apenas: ativo, target, data e concordancia_valor
        df_prec = (
            df_prec[['ativo', 'target', 'data', 'concordancia_valor']]
            .rename(columns={'concordancia_valor': 'in_precision'})
        )


        # Merge dos datasets. Realiza a junção dos dois datasets usando como chave: ativo + target + data
        df_final = df_ens.merge(
            df_prec,
            on=['ativo', 'target', 'data'],
            how='left',
            validate='one_to_one'
        )


        # Seleção das colunas finais
        df_final = df_final[
            [
                'ativo',          # ativo financeiro
                'target',         # target analisado
                'data',           # data da observação
                'target_real',    # resultado real observado
                'resultado_real', # resultado financeiro real
                'esmble_jan_tot', # ensemble com concordância total
                'esmble_jan_par', # ensemble com concordância parcial
                'in_precision'    # decisão baseada na melhor precision
            ]
        ]

        df_final.to_csv(
            f"{path_saida}/{ativo}_comparacao_completa.csv",
            index=False,
            sep="|"
        )