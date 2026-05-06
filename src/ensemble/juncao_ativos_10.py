import os
import pandas as pd


def ler_csv_corrigido(caminho):
    df = pd.read_csv(caminho, header=None)

    linhas = df.apply(lambda x: '|'.join(x.astype(str)), axis=1)

    df_corrigido = linhas.str.split('|', expand=True)

    df_corrigido.columns = df_corrigido.iloc[0]
    df_corrigido = df_corrigido[1:].reset_index(drop=True)

    df_corrigido.columns = df_corrigido.columns.str.strip().str.lower()

    return df_corrigido


def run_df_por_target(
    input_folder="./data/ensemble/9_capital/",
    output_folder="./data/ensemble/10_targets_alinhados/"
):

    os.makedirs(output_folder, exist_ok=True)

    arquivos = [f for f in os.listdir(input_folder) if f.endswith(".csv")]

    dfs_targets = {}

    for arquivo in arquivos:

        caminho = os.path.join(input_folder, arquivo)

        df = ler_csv_corrigido(caminho)

        print(f"\nArquivo: {arquivo}")
        print("Colunas corrigidas:", df.columns.tolist())

        # =========================
        # COLUNAS NECESSÁRIAS (ATUALIZADO)
        # =========================
        colunas_necessarias = [
            'ativo', 'target', 'data',
            'esmble_jan_tot', 'rend_decisao_esmble_jan_tot', 'rend_venda_esmble_jan_tot',
            'esmble_jan_par', 'rend_decisao_esmble_jan_par', 'rend_venda_esmble_jan_par',
            'in_precision', 'rend_decisao_in_precision', 'rend_venda_in_precision'
        ]

        for col in colunas_necessarias:
            if col not in df.columns:
                raise ValueError(
                    f"\n❌ Coluna '{col}' não encontrada no arquivo {arquivo}.\n"
                    f"Colunas disponíveis: {df.columns.tolist()}"
                )

        df['data'] = pd.to_datetime(df['data'])

        ativo = df['ativo'].iloc[0]

        for target in df['target'].unique():

            df_target = df[df['target'] == target].copy()

            # =========================
            # SELEÇÃO DE COLUNAS (ATUALIZADO)
            # =========================
            df_target = df_target[[
                'data',
                'esmble_jan_tot',
                'rend_decisao_esmble_jan_tot',
                'rend_venda_esmble_jan_tot',
                'esmble_jan_par',
                'rend_decisao_esmble_jan_par',
                'rend_venda_esmble_jan_par',
                'in_precision',
                'rend_decisao_in_precision',
                'rend_venda_in_precision'
            ]]

            # =========================
            # RENAME (ATUALIZADO)
            # =========================
            df_target = df_target.rename(columns={
                'esmble_jan_tot': f'{ativo}_esmble_jan_tot',
                'rend_decisao_esmble_jan_tot': f'{ativo}_rend_decisao_esmble_jan_tot',
                'rend_venda_esmble_jan_tot': f'{ativo}_rend_venda_esmble_jan_tot',

                'esmble_jan_par': f'{ativo}_esmble_jan_par',
                'rend_decisao_esmble_jan_par': f'{ativo}_rend_decisao_esmble_jan_par',
                'rend_venda_esmble_jan_par': f'{ativo}_rend_venda_esmble_jan_par',

                'in_precision': f'{ativo}_in_precision',
                'rend_decisao_in_precision': f'{ativo}_rend_decisao_in_precision',
                'rend_venda_in_precision': f'{ativo}_rend_venda_in_precision'
            })

            if target not in dfs_targets:
                dfs_targets[target] = df_target
            else:
                dfs_targets[target] = pd.merge(
                    dfs_targets[target],
                    df_target,
                    on='data',
                    how='outer'
                )

    for target, df_final in dfs_targets.items():

        df_final = df_final.sort_values('data').reset_index(drop=True)

        nome_arquivo = f"target_{str(target).replace('.', '_')}.csv"
        caminho_saida = os.path.join(output_folder, nome_arquivo)

        df_final.to_csv(caminho_saida, index=False)

        print(f"✅ Arquivo salvo: {caminho_saida}")


if __name__ == "__main__":
    run_df_por_target()