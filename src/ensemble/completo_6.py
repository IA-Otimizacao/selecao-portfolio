import pandas as pd
from tqdm import tqdm
import os
import re


def gerar_comparacao_completa(
    path_ensemble="./data/ensemble/2_tot_par",
    path_precision="./data/ensemble/4_melhor_precision_valor",
    path_saida="./data/ensemble/5_6_completo"
):
    """
    Gera dataset final consolidando:
    - ensemble total
    - ensemble parcial
    - estratégia baseada em precision
    """

    os.makedirs(path_saida, exist_ok=True)

    # =========================
    # PEGA ATIVOS DAS DUAS PASTAS
    # =========================
    arquivos_ens = [
        f for f in os.listdir(path_ensemble)
        if f.endswith("_ensemble_jan_tot_e_parcial.csv")
    ]

    arquivos_prec = [
        f for f in os.listdir(path_precision)
        if f.endswith("_melhor_precision_valor.csv")
    ]

    ativos_ens = {
        re.search(r"(.*)_ensemble_jan_tot_e_parcial\.csv", f).group(1)
        for f in arquivos_ens
    }

    ativos_prec = {
        re.search(r"(.*)_melhor_precision_valor\.csv", f).group(1)
        for f in arquivos_prec
    }

    # interseção → só roda onde existe nos dois
    ativos = list(ativos_ens & ativos_prec)

    # =========================
    # LOOP PRINCIPAL
    # =========================
    for ativo in tqdm(ativos, desc="Ativos"):

        df_ens = pd.read_csv(
            f"{path_ensemble}/{ativo}_ensemble_jan_tot_e_parcial.csv"
        )

        df_prec = pd.read_csv(
            f"{path_precision}/{ativo}_melhor_precision_valor.csv"
        )

        # =========================
        # GARANTE CONSISTÊNCIA
        # =========================
        for col in ['ativo', 'target', 'data']:
            df_ens[col] = df_ens[col].astype(str)
            df_prec[col] = df_prec[col].astype(str)

        # =========================
        # PREPARA DATASET DE PRECISION
        # =========================
        df_prec = (
            df_prec[['ativo', 'target', 'data', 'concordancia_valor']]
            .rename(columns={'concordancia_valor': 'in_precision'})
        )

        # =========================
        # MERGE
        # =========================
        df_final = df_ens.merge(
            df_prec,
            on=['ativo', 'target', 'data'],
            how='left',
            validate='one_to_one'
        )

        # =========================
        # COLUNAS FINAIS
        # =========================
        df_final = df_final[
            [
                'ativo',
                'target',
                'data',
                'esmble_jan_tot',
                'esmble_jan_par',
                'in_precision'
            ]
        ]

        df_final.to_csv(
            f"{path_saida}/{ativo}_comparacao_completa.csv",
            index=False,
            sep="|"
        )

    print("✅ comparacao_completa concluído")