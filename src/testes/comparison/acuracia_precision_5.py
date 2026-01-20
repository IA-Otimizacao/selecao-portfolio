import pandas as pd
import os
from tqdm import tqdm

# Lista de ativos
todos = ['PETR4', 'ITUB4', 'VALE3']

# Pastas de saída
os.makedirs("./data/analytics", exist_ok=True)
os.makedirs("./data/comparison/completo", exist_ok=True)

# Lista que vai armazenar os resultados finais
resultados_analytics = []

for file in tqdm(todos, desc="Ativos", unit="ativo"):
    # Caminhos
    path_melhor = f"./data/comparison/melhor_precision_valor/{file}_melhor_precision_valor.csv"
    path_ensemble = f"./data/comparison/ensemble/{file}_ensemble_jan_tot_e_parcial.csv"

    # --- Carregamento dos dados ---
    df_melhor = pd.read_csv(path_melhor, sep=",")[
        ['ativo', 'target', 'data', 'concordancia_valor']
    ].rename(columns={'concordancia_valor': 'in_precision'})

    df_ensemble = pd.read_csv(path_ensemble)[
        ['ativo', 'target', 'data','target_real','resultado_real', 'esmble_jan_tot', 'esmble_jan_par']
    ]

    # --- Padroniza o tipo de data ---
    df_melhor['data'] = pd.to_datetime(df_melhor['data'])
    df_ensemble['data'] = pd.to_datetime(df_ensemble['data'])

    # --- Merge garantindo que as chaves coincidam ---
    df = pd.merge(df_melhor, df_ensemble, on=['ativo', 'target', 'data'], how='inner')

    # --- Salva o dataframe completo ---
    df_completo = df[['ativo', 'target', 'data', 'target_real', 'resultado_real', 'esmble_jan_tot', 'esmble_jan_par', 'in_precision']]

    # Caminhos de saída
    output_csv = f"./data/comparison/completo/{file}_completo.csv"
    output_xlsx = f"./data/comparison/completo/{file}_completo.xlsx"

    # Salva em CSV
    df_completo.to_csv(output_csv, index=False, sep="|")

    # Salva em Excel
    df_completo.to_excel(output_xlsx, index=False)

    total_linhas = len(df)
    resultados_ativos = {'ativo': file, 'target': 'GERAL', 'total_linhas': total_linhas}

    # --- Cálculo geral de acurácia e precisão ---
    for col in ['in_precision', 'esmble_jan_tot', 'esmble_jan_par']:
        acertos = (df[col] == df['target_real']).sum()
        acc = round(acertos / total_linhas * 100, 2) if total_linhas > 0 else 0

        vp = ((df[col] == 1) & (df['target_real'] == 1)).sum()
        fp = ((df[col] == 1) & (df['target_real'] == 0)).sum()
        precision = round(vp / (vp + fp) * 100, 2) if (vp + fp) > 0 else 0

        resultados_ativos[f'acertos_{col}'] = acertos
        resultados_ativos[f'acc_{col} (%)'] = acc
        resultados_ativos[f'precision_{col}'] = precision

    resultados_analytics.append(resultados_ativos)

    # --- Resultados por target ---
    for target, grupo in df.groupby('target'):
        resultados_target = {'ativo': file, 'target': target, 'total_linhas': len(grupo)}

        for col in ['in_precision', 'esmble_jan_tot', 'esmble_jan_par']:
            acertos = (grupo[col] == grupo['target_real']).sum()
            acc = round(acertos / len(grupo) * 100, 2) if len(grupo) > 0 else 0

            vp = ((grupo[col] == 1) & (grupo['target_real'] == 1)).sum()
            fp = ((grupo[col] == 1) & (grupo['target_real'] == 0)).sum()
            precision = round(vp / (vp + fp) * 100, 2) if (vp + fp) > 0 else 0

            resultados_target[f'acertos_{col}'] = acertos
            resultados_target[f'acc_{col} (%)'] = acc
            resultados_target[f'precision_{col}'] = precision

        resultados_analytics.append(resultados_target)

# --- Salva resultado final ---
df_result = pd.DataFrame(resultados_analytics)
output_csv = "./data/analytics/acuracia_precision_ensembles1.csv"
output_xlsx = "./data/analytics/acuracia_precision_ensembles1.xlsx"

df_result.to_csv(output_csv, index=False, sep="|")
df_result.to_excel(output_xlsx, index=False)

print(f"✅ Arquivos salvos em:")
print(f"   ├── {output_csv}")
print(f"   ├── {output_xlsx}")
print(f"   └── ./data/comparison/completo/")
