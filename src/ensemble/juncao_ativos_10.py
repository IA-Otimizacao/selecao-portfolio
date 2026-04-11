import os
import pandas as pd


def ler_csv_corrigido(caminho):
    """
    Lê CSV corrigindo separação quebrada por '|'
    """

    # Lê o arquivo sem considerar separador (tudo vira uma linha "quebrada")
    df = pd.read_csv(caminho, header=None)

    # Junta todas as colunas em uma única string por linha (reconstrói a linha original)
    linhas = df.apply(lambda x: '|'.join(x.astype(str)), axis=1)

    # Divide corretamente cada linha usando o separador '|'
    df_corrigido = linhas.str.split('|', expand=True)

    # Define a primeira linha como nome das colunas
    df_corrigido.columns = df_corrigido.iloc[0]
    df_corrigido = df_corrigido[1:].reset_index(drop=True)

    # Padroniza nomes das colunas (remove espaços e deixa minúsculo)
    df_corrigido.columns = df_corrigido.columns.str.strip().str.lower()

    return df_corrigido


def run_df_por_target(
    input_folder="./data/ensemble/9_capital/",
    output_folder="./data/ensemble/10_targets_alinhados/"
):

    # Cria pasta de saída se não existir
    os.makedirs(output_folder, exist_ok=True)

    # Lista todos os arquivos CSV da pasta
    arquivos = [f for f in os.listdir(input_folder) if f.endswith(".csv")]

    # Dicionário para armazenar um dataframe por target
    dfs_targets = {}

    # Loop por cada arquivo (cada ativo)
    for arquivo in arquivos:

        caminho = os.path.join(input_folder, arquivo)

        # Lê o CSV corrigindo problema de separação
        df = ler_csv_corrigido(caminho)

        # Debug: mostra colunas após correção
        print(f"\nArquivo: {arquivo}")
        print("Colunas corrigidas:", df.columns.tolist())

        # Lista de colunas obrigatórias
        colunas_necessarias = [
            'ativo', 'target', 'data',
            'esmble_jan_tot', 'rend_decisao_esmble_jan_tot',
            'esmble_jan_par', 'rend_decisao_esmble_jan_par',
            'in_precision', 'rend_decisao_in_precision'
        ]

        # Valida se todas as colunas existem
        for col in colunas_necessarias:
            if col not in df.columns:
                raise ValueError(
                    f"\n❌ Coluna '{col}' não encontrada no arquivo {arquivo}.\n"
                    f"Colunas disponíveis: {df.columns.tolist()}"
                )

        # Converte coluna de data para datetime
        df['data'] = pd.to_datetime(df['data'])

        # Identifica o ativo (assume que o arquivo é de um único ativo)
        ativo = df['ativo'].iloc[0]

        # Loop para cada target dentro do arquivo
        for target in df['target'].unique():

            # Filtra apenas linhas daquele target
            df_target = df[df['target'] == target].copy()

            # Seleciona apenas colunas relevantes
            df_target = df_target[[
                'data',
                'esmble_jan_tot',
                'rend_decisao_esmble_jan_tot',
                'esmble_jan_par',
                'rend_decisao_esmble_jan_par',
                'in_precision',
                'rend_decisao_in_precision'
            ]]

            # Renomeia colunas adicionando prefixo do ativo
            df_target = df_target.rename(columns={
                'esmble_jan_tot': f'{ativo}_esmble_jan_tot',
                'rend_decisao_esmble_jan_tot': f'{ativo}_rend_decisao_esmble_jan_tot',
                'esmble_jan_par': f'{ativo}_esmble_jan_par',
                'rend_decisao_esmble_jan_par': f'{ativo}_rend_decisao_esmble_jan_par',
                'in_precision': f'{ativo}_in_precision',
                'rend_decisao_in_precision': f'{ativo}_rend_decisao_in_precision'
            })

            # Se ainda não existe dataframe para esse target → cria
            if target not in dfs_targets:
                dfs_targets[target] = df_target
            else:
                # Caso já exista → faz merge pela data (mantendo todas as datas)
                dfs_targets[target] = pd.merge(
                    dfs_targets[target],
                    df_target,
                    on='data',
                    how='outer'
                )

    # Loop final para salvar um arquivo por target
    for target, df_final in dfs_targets.items():

        # Ordena por data
        df_final = df_final.sort_values('data').reset_index(drop=True)

        # Nome do arquivo (troca ponto por underline)
        nome_arquivo = f"target_{str(target).replace('.', '_')}.csv"
        caminho_saida = os.path.join(output_folder, nome_arquivo)

        # Salva CSV final
        df_final.to_csv(caminho_saida, index=False)

        print(f"✅ Arquivo salvo: {caminho_saida}")


# Executa a função principal
if __name__ == "__main__":
    run_df_por_target()