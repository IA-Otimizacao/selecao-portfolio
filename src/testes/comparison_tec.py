import pandas as pd
from tqdm import tqdm
import os

# Garante que a pasta existe
os.makedirs("./data/comparison", exist_ok=True)

todos = ['PETR4','ITUB4','VALE3']

def carregar_dados_comparacao(file):
    # Carrega os dados
    base_dados = pd.read_csv(f'./data/train_out/target_previsto_{file}.csv')

    # Padroniza a coluna de data
    if 'data' in base_dados.columns:
        base_dados['data'] = pd.to_datetime(base_dados['data'], errors='coerce')

    # Faz o pivot: cada técnica vira uma coluna com seu target_pred
    tabela = (
        base_dados
        .pivot_table(
            index=['ativo', 'target', 'janela', 'data', 'target_real'],  # chaves únicas
            columns='tecnica',                                          # vira coluna
            values='target_pred',                                       # valor preenchido
            aggfunc='first'                                             # caso tenha duplicados
        )
        .reset_index()
    )

    # Garante que as colunas tenham nomes simples (RNA, RF, SVM...)
    tabela.columns.name = None  

    # Cria coluna de concordância binária (1 se todas iguais, 0 caso contrário)
    if all(col in tabela.columns for col in ['RNA', 'Random Forest', 'SVC']):
        soma = tabela['RNA'] + tabela['Random Forest'] + tabela['SVC']
        tabela['ensemble_tecnicas'] = soma.apply(lambda x: 1 if x in [0, 3] else 0)
    else:
        tabela['ensemble_tecnicas'] = None  # caso falte alguma técnica

    return tabela   


resultados = {}

# Loop pelos ativos
for file in tqdm(todos, desc="Ativos", unit="ativo"):
    df = carregar_dados_comparacao(file)

    # Caminho de saída
    output_path = f'./data/comparison/{file}_comparison.xlsx'

    # Escreve no Excel com o motor xlsxwriter
    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Comparacao')

        workbook = writer.book
        worksheet = writer.sheets['Comparacao']

        # Formatos de cor
        verde = workbook.add_format({'bg_color': '#C6EFCE', 'font_color': '#006100'})   # Acertou
        vermelho = workbook.add_format({'bg_color': '#FFC7CE', 'font_color': '#9C0006'})  # Errou

        # Colunas que queremos comparar
        tecnicas = ['RNA', 'Random Forest', 'SVC', 'ensemble_tecnicas']

        # Localiza as posições das colunas
        col_target_real = df.columns.get_loc("target_real")
        col_target_excel = chr(ord('A') + col_target_real)
        n_linhas = len(df)

        # Aplica a formatação para cada técnica existente
        for tecnica in tecnicas:
            if tecnica in df.columns:
                col_idx = df.columns.get_loc(tecnica)
                col_excel = chr(ord('A') + col_idx)

                # Formatação verde (acerto)
                worksheet.conditional_format(
                    f'{col_excel}2:{col_excel}{n_linhas+1}',
                    {
                        'type': 'formula',
                        'criteria': f'=${col_excel}2=${col_target_excel}2',
                        'format': verde
                    }
                )

                # Formatação vermelha (erro)
                worksheet.conditional_format(
                    f'{col_excel}2:{col_excel}{n_linhas+1}',
                    {
                        'type': 'formula',
                        'criteria': f'=${col_excel}2<>${col_target_excel}2',
                        'format': vermelho
                    }
                )

    print(f"✅ Arquivo salvo com formatação condicional: {output_path}")