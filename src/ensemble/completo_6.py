import pandas as pd
from tqdm import tqdm
import os


def gerar_comparacao_completa(
    path_ensemble="./data/ensemble/2_tot_par",
    path_precision="./data/ensemble/4_melhor_precision_valor",
    path_saida="./data/ensemble/5_6_completo"
):
    """
    Gera arquivos de comparação completa unindo ensemble e melhor precision.
    Ativos definidos internamente.
    """

    ativos = ['PETR4', 'ITUB4', 'VALE3']

    # Garante que a pasta de saída existe
    os.makedirs(path_saida, exist_ok=True)

    for ativo in tqdm(ativos):
        # Leitura dos dados
        df_ens = pd.read_csv(
            f"{path_ensemble}/{ativo}_ensemble_jan_tot_e_parcial.csv"
        )

        df_prec = pd.read_csv(
            f"{path_precision}/{ativo}_melhor_precision_valor.csv"
        )

        # Garantia de alinhamento de tipos
        for col in ['ativo', 'target', 'data']:
            df_ens[col] = df_ens[col].astype(str)
            df_prec[col] = df_prec[col].astype(str)

        # Seleciona e renomeia coluna de interesse
        df_prec = (
            df_prec[['ativo', 'target', 'data', 'concordancia_valor']]
            .rename(columns={'concordancia_valor': 'in_precision'})
        )

        # Merge
        df_final = df_ens.merge(
            df_prec,
            on=['ativo', 'target', 'data'],
            how='left',
            validate='one_to_one'
        )

        # Colunas finais
        df_final = df_final[
            [
                'ativo',
                'target',
                'data',
                'target_real',
                'resultado_real',
                'esmble_jan_tot',
                'esmble_jan_par',
                'in_precision'
            ]
        ]

        # Salva
        df_final.to_csv(
            f"{path_saida}/{ativo}_comparacao_completa.csv",
            index=False,
            sep="|"
        )
