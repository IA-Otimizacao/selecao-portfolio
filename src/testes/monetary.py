import os
import pandas as pd


# Função principal de simulação monetária
def simulate_windows(df, initial_capital=100000.0, wind=4, date_col='data'):
    """
    Simula o capital investido e disponível em janelas de até 'wind' dias,
    seguindo as regras de target_pred e target.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    # Inicializa colunas
    df['valor_obtido'] = pd.NA
    df['investido'] = pd.NA
    df['disponivel'] = pd.NA
    df['inicio_janela'] = pd.NA
    df['fim_janela'] = pd.NA

    group_cols = ['ativo', 'target', 'janela', 'tecnica']
    results = []

    for key, g in df.groupby(group_cols, sort=False):
        g = g.sort_values(date_col).reset_index(drop=True)
        n = len(g)
        capital = float(initial_capital)

        start = 0
        while start < n:
            end = min(start + wind, n)
            window = g.loc[start:end-1].copy()

            # Inicialmente marca o início da janela
            inicio_data = g.loc[start, date_col]

            # Primeiro dia com target_pred==1
            mask_one = window['target_pred'].fillna(0).astype(float) == 1

            if mask_one.any():
                first_idx = window.index[mask_one][0]
                target_val = float(window.loc[first_idx, 'target'])
                novo_capital = capital * target_val

                # Ajusta o fim da janela para o dia do target
                fim_data = g.loc[first_idx, date_col]

                # Dias antes do target -> capital disponível
                if first_idx > start:
                    g.loc[start:first_idx-1, 'valor_obtido'] = capital
                    g.loc[start:first_idx-1, 'investido'] = 0
                    g.loc[start:first_idx-1, 'disponivel'] = capital
                    g.loc[start:first_idx-1, 'inicio_janela'] = inicio_data
                    g.loc[start:first_idx-1, 'fim_janela'] = fim_data

                # Dia do target -> valor investido
                g.loc[first_idx, 'valor_obtido'] = novo_capital
                g.loc[first_idx, 'investido'] = novo_capital
                g.loc[first_idx, 'disponivel'] = 0
                g.loc[first_idx, 'inicio_janela'] = inicio_data
                g.loc[first_idx, 'fim_janela'] = fim_data

                capital = novo_capital
                start = first_idx + 1  # próxima janela começa no dia seguinte

            else:
                # Nenhum target -> multiplica pelo resultado_real no último dia da janela
                last_idx = window.index[-1]
                fim_data = g.loc[last_idx, date_col]
                realized = float(window.loc[last_idx, 'resultado_real'])
                novo_capital = capital * (1.0 + realized)

                # Dias antes do último -> capital disponível
                if end - start > 1:
                    g.loc[start:end-2, 'valor_obtido'] = capital
                    g.loc[start:end-2, 'investido'] = 0
                    g.loc[start:end-2, 'disponivel'] = capital
                    g.loc[start:end-2, 'inicio_janela'] = inicio_data
                    g.loc[start:end-2, 'fim_janela'] = fim_data

                # Último dia da janela
                g.loc[last_idx, 'valor_obtido'] = novo_capital
                g.loc[last_idx, 'investido'] = novo_capital
                g.loc[last_idx, 'disponivel'] = 0
                g.loc[last_idx, 'inicio_janela'] = inicio_data
                g.loc[last_idx, 'fim_janela'] = fim_data

                capital = novo_capital
                start = end  # próxima janela começa após o fim da janela

        results.append(g)

    out = pd.concat(results, ignore_index=True).sort_values(group_cols + [date_col])
    return out


# Função para processar todos os arquivos CSV
def process_folder(input_folder, output_folder, initial_capital=100000.0, wind=4, file_pattern='.csv'):
    """
    Processa todos os arquivos CSV em uma pasta de entrada, aplica a simulação monetária
    e salva os resultados em uma pasta de saída.
    """
    os.makedirs(output_folder, exist_ok=True)

    for fname in os.listdir(input_folder):
        if not fname.lower().endswith(file_pattern):
            continue

        path = os.path.join(input_folder, fname)
        df = pd.read_csv(path)

        out = simulate_windows(df, initial_capital=initial_capital, wind=wind, date_col='data')

        out_path = os.path.join(output_folder, fname.replace('.csv', '_monetario.csv'))
        out.to_csv(out_path, index=False)
        print(f'Processado {fname} -> {out_path}')


# Chamando a função para processar os CSVs
process_folder(
    input_folder="./data/train_out/",
    output_folder="./data/monetario/",
    initial_capital=100000,
    wind=4
)
