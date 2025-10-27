import pandas as pd
from tqdm import tqdm
import os

# Lista de ativos
todos = ['PETR4', 'ITUB4', 'VALE3']

# Pasta de saída
os.makedirs("./data/comparison/ensemble", exist_ok=True)

def carregar_e_criar_ensembles(file):
    # Carrega o Excel de comparação
    base = pd.read_excel(f'./data/comparison/{file}_comparison.xlsx')

    # Pivot: cada janela vira sufixo nas colunas de técnicas
    tabela = (
        base
        .pivot_table(
            index=['ativo', 'target', 'data', 'target_real'],
            columns='janela',
            values=['RNA', 'Random Forest', 'SVC'],
            aggfunc='first'
        )
    )

    # Renomeia colunas (ex: RNA_60, RF_75, SVC_90)
    tabela.columns = [f"{col[0]}_{col[1]}" for col in tabela.columns]
    tabela = tabela.reset_index()

    # Colunas de técnicas
    tecnicas_cols = [col for col in tabela.columns if any(t in col for t in ['RNA', 'Random Forest', 'SVC'])]

    # Remove linhas incompletas (faltando alguma técnica/janela)
    tabela = tabela.dropna(subset=tecnicas_cols)

    # Ensemble total: todas as 9 concordam
    tabela['esmble_jan_tot'] = (tabela[tecnicas_cols].sum(axis=1) == 9).astype(int)

    # Ensemble parcial:
    # 1) maioria concorda (≥5)
    # 2) pelo menos 1 acerto de cada técnica
    tabela['esmble_jan_par'] = (
        ((tabela[tecnicas_cols].sum(axis=1) >= 5) &
         ((tabela[['RNA_60','RNA_75','RNA_90']].sum(axis=1) >= 1) &
          (tabela[['Random Forest_60','Random Forest_75','Random Forest_90']].sum(axis=1) >= 1) &
          (tabela[['SVC_60','SVC_75','SVC_90']].sum(axis=1) >= 1)))
    ).astype(int)

    return tabela, tecnicas_cols

# Loop para processar cada ativo
for file in tqdm(todos, desc="Ativos", unit="ativo"):
    df_ensemble, tecnicas_cols = carregar_e_criar_ensembles(file)

    # Colunas finais: ativo, target, data, target_real, técnicas, esmble_jan_tot, esmble_jan_par
    colunas_final = ['ativo', 'target', 'data', 'target_real'] + tecnicas_cols + ['esmble_jan_tot','esmble_jan_par']
    df_ensemble = df_ensemble[colunas_final]

    # Caminho de saída
    output_path = f'./data/comparison/ensemble/{file}_ensemble_jan_tot_e_parcial.xlsx'

    # Salva com cores: apenas esmble_jan_tot e esmble_jan_par comparando com target_real
    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
        df_ensemble.to_excel(writer, index=False, sheet_name='Ensemble')

        workbook  = writer.book
        worksheet = writer.sheets['Ensemble']

        verde = workbook.add_format({'bg_color': '#C6EFCE', 'font_color': '#006100'})
        vermelho = workbook.add_format({'bg_color': '#FFC7CE', 'font_color': '#9C0006'})

        n_linhas = len(df_ensemble)
        col_target_real = df_ensemble.columns.get_loc("target_real")

        for col_nome in ['esmble_jan_tot','esmble_jan_par']:
            col_excel = chr(ord('A') + df_ensemble.columns.get_loc(col_nome))
            # Verde se igual
            worksheet.conditional_format(
                f'{col_excel}2:{col_excel}{n_linhas+1}',
                {'type':'formula','criteria':f'=${col_excel}2=${chr(ord("A")+col_target_real)}2','format':verde}
            )
            # Vermelho se diferente
            worksheet.conditional_format(
                f'{col_excel}2:{col_excel}{n_linhas+1}',
                {'type':'formula','criteria':f'=${col_excel}2<>${chr(ord("A")+col_target_real)}2','format':vermelho}
            )

    print(f"✅ Arquivo final salvo: {output_path}")

resultados_analytics = []

for file in tqdm(todos, desc="Ativos", unit="ativo"):
    df = pd.read_excel(f"./data/comparison/ensemble/{file}_ensemble_jan_tot_e_parcial.xlsx")
    total_linhas = len(df)

    # --- Resultados gerais por ativo ---
    resultados_ativos = {'ativo': file, 'target': 'GERAL', 'total_linhas': total_linhas}

    for col in ['esmble_jan_tot','esmble_jan_par']:
        acertos = (df[col] == df['target_real']).sum()
        acc = round(acertos / total_linhas * 100, 2)

        # Precision
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

        for col in ['esmble_jan_tot','esmble_jan_par']:
            acertos = (grupo[col] == grupo['target_real']).sum()
            acc = round(acertos / len(grupo) * 100, 2)

            vp = ((grupo[col] == 1) & (grupo['target_real'] == 1)).sum()
            fp = ((grupo[col] == 1) & (grupo['target_real'] == 0)).sum()
            precision = round(vp / (vp + fp) * 100, 2) if (vp + fp) > 0 else 0

            resultados_target[f'acertos_{col}'] = acertos
            resultados_target[f'acc_{col} (%)'] = acc
            resultados_target[f'precision_{col}'] = precision

        resultados_analytics.append(resultados_target)

# Cria DataFrame final e salva CSV
df_result = pd.DataFrame(resultados_analytics)
os.makedirs("./data/analytics", exist_ok=True)
output_csv = "./data/analytics/acuracia_precision_ensembles.csv"
df_result.to_csv(output_csv, index=False)

print(f"✅ Acurácias e precision salvas em: {output_csv}")
print(df_result)
