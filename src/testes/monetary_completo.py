import os
import pandas as pd

# ------------------------------------------------
# Função que simula uma estratégia
# ------------------------------------------------
def simulate_strategy(g, signal_col, initial_capital=100000.0, wind=4):
    """
    Retorna um DataFrame com:
    - capital
    - inicio
    - fim
    para uma estratégia específica.
    """

    g = g.copy().reset_index(drop=True)
    n = len(g)
    capital = float(initial_capital)

    capital_out = [pd.NA] * n
    inicio_out = [pd.NA] * n
    fim_out = [pd.NA] * n

    start = 0
    while start < n:
        end = min(start + wind, n)
        window = g.loc[start:end - 1]

        inicio_data = g.loc[start, 'data']
        mask_signal = window[signal_col].fillna(0).astype(float) == 1

        if mask_signal.any():
            first_idx = window.index[mask_signal][0]
            fim_data = g.loc[first_idx, 'data']

            target_val = float(g.loc[first_idx, 'target'])
            novo_capital = capital * target_val

            for i in range(start, first_idx):
                capital_out[i] = capital
                inicio_out[i] = inicio_data
                fim_out[i] = fim_data

            capital_out[first_idx] = novo_capital
            inicio_out[first_idx] = inicio_data
            fim_out[first_idx] = fim_data

            capital = novo_capital
            start = first_idx + 1

        else:
            last_idx = window.index[-1]
            fim_data = g.loc[last_idx, 'data']

            realized = float(g.loc[last_idx, 'resultado_real'])
            novo_capital = capital * (1 + realized)

            for i in range(start, end - 1):
                capital_out[i] = capital
                inicio_out[i] = inicio_data
                fim_out[i] = fim_data

            capital_out[last_idx] = novo_capital
            inicio_out[last_idx] = inicio_data
            fim_out[last_idx] = fim_data

            capital = novo_capital
            start = end

    return pd.DataFrame({
        'capital': capital_out,
        'inicio': inicio_out,
        'fim': fim_out
    })


# ------------------------------------------------------
# Função principal
# ------------------------------------------------------
def process_folder(input_folder, output_folder, initial_capital=100000.0, wind=4):

    os.makedirs(output_folder, exist_ok=True)

    for fname in os.listdir(input_folder):
        if not fname.lower().endswith(".csv"):
            continue

        path = os.path.join(input_folder, fname)
        df = pd.read_csv(path, sep="|")

        df.columns = df.columns.str.strip().str.replace("\ufeff", "", regex=True)
        df['data'] = pd.to_datetime(df['data'])

        df = df.sort_values(['ativo', 'target', 'data'])

        results = []
        for _, g in df.groupby(['ativo', 'target'], sort=False):
            g = g.reset_index(drop=True)

            # TOT
            sim_tot = simulate_strategy(g, 'target_real', initial_capital, wind)
            g['capital_tot'] = sim_tot['capital']
            g['inicio_tot'] = sim_tot['inicio']
            g['fim_tot'] = sim_tot['fim']


            results.append(g)

        out = pd.concat(results, ignore_index=True)

        out_path = os.path.join(
            output_folder,
            fname.replace('.csv', '_monetario.csv')
        )
        out.to_csv(out_path, index=False)


# Execução
process_folder(
    input_folder="./data/comparison/completo/",
    output_folder="./data/monetario_real/",
    initial_capital=100000,
    wind=4
)
