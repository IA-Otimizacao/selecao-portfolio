import pandas as pd
import os
from tqdm import tqdm


def run_comparison_esmbs():

    todos = ['PETR4','ITUB4','VALE3']

    os.makedirs("./data/ensemble/4_melhor_precision_valor", exist_ok=True)

    # Loop pelos ativos com barra de progresso
    for file in tqdm(todos):

        # Dataset contendo os valores previstos por técnica
        df_ens = pd.read_csv(
            f"./data/ensemble/2_tot_par/{file}_ensemble_jan_tot_e_parcial.csv"
        )

        # Dataset contendo qual técnica teve melhor precision
        df_prec = pd.read_csv(
            f"./data/ensemble/3_precision/target_in_{file}_pivot.csv"
        )

        # Lista onde serão armazenados os resultados finais
        res = []

        # Loop pelas linhas do dataframe de precision
        for _, r in df_prec.iterrows():

            # A coluna melhor_precision pode conter várias técnicas separadas por "|", então precisamos quebrar em lista
            tecnicas = [
                t.strip() 
                for t in str(r['melhor_precision']).split('|')
            ]

            # Filtra no dataframe de ensemble a linha correspondente ao mesmo ativo, target e data
            linha = df_ens[
                (df_ens['ativo'] == r['ativo']) &
                (df_ens['target'] == r['target']) &
                (df_ens['data'] == r['data'])
            ]

            # Lista que armazenará os valores previstos pelas técnicas
            valores = []

            # Buscar valor de cada técnica vencedora

            for t in tecnicas:

                # Remove possíveis espaços no nome da técnica
                t = t.replace(" ","")

                # Verifica se: a linha existe e a técnica está presente nas colunas
                if not linha.empty and t in linha.columns:

                    # Pega o valor da técnica
                    valores.append(str(linha.iloc[0][t]))

                else:
                    # Caso não exista valor disponível
                    valores.append("NA")

            # Salvar resultado da linha

            res.append({
                'ativo': r['ativo'],
                'target': r['target'],
                'data': r['data'],
                'valor_melhor_precision': ','.join(valores)
            })

        # Converte lista de resultados em dataframe
        df = pd.DataFrame(res)

        # Função para verificar concordância dos valores

        def conc(v):

            # Converte valores da string em números
            vs = [
                int(float(x)) 
                for x in v.split(',')
                if x not in ['NA','']
            ]

            # Se todos os valores forem iguais → retorna esse valor
            if vs and all(x == vs[0] for x in vs):
                return vs[0]

            # Caso contrário → não há concordância
            return 0


        # Aplica a função de concordância
        df['concordancia_valor'] = df['valor_melhor_precision'].apply(conc)

        # Remove linhas que possuem valores NA
        df = df[~df['valor_melhor_precision'].str.contains('NA')]


        df.to_csv(
            f"./data/ensemble/4_melhor_precision_valor/{file}_melhor_precision_valor.csv",
            index=False
        )

    print("✅ comparison_esmbs concluído")