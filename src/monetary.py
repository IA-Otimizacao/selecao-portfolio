import os
import pandas as pd

def simulate_windows(df, initial_capital=100000.0, wind=4, date_col='data'):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df['valor_obtido'] = pd.NA

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

            # procura o primeiro dia com target_pred == 1
            mask_one = window['target_pred'].fillna(0).astype(float) == 1

            if mask_one.any():
                first_idx = window.index[mask_one][0]
                realized = float(window.loc[first_idx, 'resultado_real'])
                novo_capital = capital * (1.0 + realized)

                # dias antes -> mantêm capital inicial
                g.loc[start:first_idx-1, 'valor_obtido'] = capital
                # do primeiro target_pred==1 até o fim da janela -> valor ajustado
                g.loc[first_idx:end-1, 'valor_obtido'] = novo_capital

            else:
                # nenhum target_pred == 1 -> só multiplica no último dia
                last_idx = window.index[-1]
                realized = float(window.loc[last_idx, 'resultado_real'])
                novo_capital = capital * (1.0 + realized)

                # todos os dias menos o último = capital inicial
                if end - start > 1:
                    g.loc[start:end-2, 'valor_obtido'] = capital
                # último dia recebe valor ajustado
                g.loc[last_idx, 'valor_obtido'] = novo_capital
                

            capital = novo_capital
            start += wind

        results.append(g)

    out = pd.concat(results, ignore_index=True).sort_values(group_cols + [date_col])
    return out


def process_folder(input_folder, output_folder, initial_capital=100000.0, wind=4, file_pattern='.csv'):
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


process_folder(
    input_folder="./data/train_out/",  
    output_folder="./data/monetario/",     
    initial_capital=100000,
    wind=4
)

