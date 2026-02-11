import os
import pandas as pd


# ------------------------------------------------
# Função que simula uma estratégia monetária
# ------------------------------------------------
def simulate_strategy(g, signal_col, initial_capital=100000.0, wind=4):
    """
    Retorna um DataFrame com:
    - capital
    - inicio
    - fim
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

        # ------------------------------------------------
        # Caso 1: sinal aparece dentro da janela
        # ------------------------------------------------
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

        # ------------------------------------------------
        # Caso 2: nenhum sinal na janela
        # ------------------------------------------------
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
        print(f"\n📂 Processando arquivo: {fname}")

        # Leitura (padrão do projeto: |)
        df = pd.read_csv(path, sep="|")

        # Limpeza de nomes de coluna
        df.columns = (
            df.columns
            .str.strip()
            .str.replace("\ufeff", "", regex=True)
        )

        # Validação de colunas obrigatórias
        colunas_obrigatorias = {
            'ativo',
            'target',
            'data',
            'target_real',
            'resultado_real',
            'esmble_jan_tot',
            'esmble_jan_par',
            'in_precision'
        }

        faltantes = colunas_obrigatorias - set(df.columns)
        if faltantes:
            raise ValueError(
                f"Arquivo {fname} não possui colunas obrigatórias: {faltantes}\n"
                f"Colunas encontradas: {df.columns.tolist()}"
            )

        # Conversões
        df['data'] = pd.to_datetime(df['data'])

        # Ordenação
        df = df.sort_values(['ativo', 'target', 'data'])

        results = []

        # ------------------------------------------------
        # Processamento por ativo e target
        # ------------------------------------------------
        for (ativo, target), g in df.groupby(['ativo', 'target'], sort=False):
            print(f"  ▶ Simulando | ativo={ativo} | target={target}")

            g = g.reset_index(drop=True)

            # -----------------------------
            # Ensemble Total
            # -----------------------------
            sim_tot = simulate_strategy(
                g=g,
                signal_col='esmble_jan_tot',
                initial_capital=initial_capital,
                wind=wind
            )

            g['capital_tot'] = sim_tot['capital']
            g['inicio_tot'] = sim_tot['inicio']
            g['fim_tot'] = sim_tot['fim']

            # -----------------------------
            # Ensemble Parcial
            # -----------------------------
            sim_par = simulate_strategy(
                g=g,
                signal_col='esmble_jan_par',
                initial_capital=initial_capital,
                wind=wind
            )

            g['capital_par'] = sim_par['capital']
            g['inicio_par'] = sim_par['inicio']
            g['fim_par'] = sim_par['fim']

            # -----------------------------
            # Precision
            # -----------------------------
            sim_prec = simulate_strategy(
                g=g,
                signal_col='in_precision',
                initial_capital=initial_capital,
                wind=wind
            )

            g['capital_precision'] = sim_prec['capital']
            g['inicio_precision'] = sim_prec['inicio']
            g['fim_precision'] = sim_prec['fim']

            results.append(g)

        out = pd.concat(results, ignore_index=True)

        out_path = os.path.join(
            output_folder,
            fname.replace('.csv', '_monetario.csv')
        )

        out.to_csv(out_path, index=False, sep="|")
        print(f"✅ Arquivo salvo em: {out_path}")


# ------------------------------------------------------
# Runner
# ------------------------------------------------------
def run_monetary():
    process_folder(
        input_folder="./data/ensemble/5_6_completo/",
        output_folder="./data/ensemble/7_monetario/",
        initial_capital=100000,
        wind=4
    )


# Execução direta
if __name__ == "__main__":
    run_monetary()
