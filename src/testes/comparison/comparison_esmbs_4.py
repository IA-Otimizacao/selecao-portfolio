import pandas as pd
import os
from tqdm import tqdm

# Lista de ativos
todos = ['PETR4', 'ITUB4', 'VALE3']

# Caminhos
path_ensemble = "./data/comparison/ensemble"
path_precision = "./data/comparison/precision"
path_saida = "./data/comparison/melhor_precision_valor"

# Garante que a pasta de saída existe
os.makedirs(path_saida, exist_ok=True)

for file in tqdm(todos, desc="Processando ativos"):
    # Carrega os dois dataframes
    df_ensemble = pd.read_csv(f"{path_ensemble}/{file}_ensemble_jan_tot_e_parcial.csv")
    df_precision = pd.read_csv(f"{path_precision}/target_in_{file}_pivot.csv", sep=None, engine="python")

    # Normaliza os nomes de colunas
    df_ensemble.columns = df_ensemble.columns.str.strip()
    df_precision.columns = df_precision.columns.str.strip()

    resultados = []

    for _, linha in df_precision.iterrows():
        ativo = linha['ativo']
        target = linha['target']
        data = linha['data']
        tecnicas = str(linha['melhor_precision']).split('|')
        tecnicas = [t.strip() for t in tecnicas if t.strip() != '']

        linha_ensemble = df_ensemble[
            (df_ensemble['ativo'] == ativo) &
            (df_ensemble['target'] == target) &
            (df_ensemble['data'] == data)
        ]

        valores = []
        if not linha_ensemble.empty:
            for tecnica in tecnicas:
                tecnica_corrigida = tecnica.replace(" ", "")
                if tecnica_corrigida in linha_ensemble.columns:
                    valores.append(str(linha_ensemble.iloc[0][tecnica_corrigida]))
                else:
                    valores.append("NA")
        else:
            valores.append("NA")

        resultados.append({
            'ativo': ativo,
            'target': target,
            'data': data,
            'melhor_precision': ', '.join(tecnicas),
            'valor_melhor_precision': ', '.join(valores)
        })

    df_final = pd.DataFrame(resultados)

    # --- NOVO TRECHO: cálculo da concordância ---
    def calcular_concordancia(valor_str):
        # separa e remove NA
        valores = [v.strip() for v in valor_str.split(',') if v.strip() != '' and v.strip() != 'NA']
        if not valores:
            return None

        # converte para 0 ou 1, mesmo que esteja como 1.0, 0.0, etc.
        try:
            valores_num = [int(float(v)) for v in valores]
        except:
            return 0  # qualquer valor estranho -> 0

        # Se todos forem iguais, retorna o valor (0 ou 1)
        if all(v == valores_num[0] for v in valores_num):
            return valores_num[0]

        # Se forem diferentes, retorna 0
        return 0


    # Cria a coluna concordancia_valor
    df_final['concordancia_valor'] = df_final['valor_melhor_precision'].apply(calcular_concordancia)

    # Remove linhas com valor_melhor_precision vazio ou contendo apenas NA
    df_final = df_final[
        df_final['valor_melhor_precision'].str.strip().ne('') &
        (~df_final['valor_melhor_precision'].str.contains('NA', na=False))
    ]

    # Salva em CSV
    df_final.to_csv(f"{path_saida}/{file}_melhor_precision_valor.csv", index=False)

print("✅ Todos os arquivos foram processados e salvos com sucesso!")
