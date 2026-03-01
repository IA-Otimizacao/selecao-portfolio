import os
import pandas as pd


# Função que simula uma estratégia monetária
def simulate_strategy(g, signal_col, initial_capital=100000.0, wind=4):

    """
    Esta função simula a evolução de capital de uma estratégia de trading.

    A lógica funciona da seguinte forma:

    • A cada janela temporal (wind), verifica-se se houve um sinal de entrada.
    • Caso o sinal ocorra, assume-se que a operação atingiu o target.
    • Caso não ocorra sinal, considera-se o retorno real do ativo.

    Retorno
    ----------
    DataFrame contendo:
        - capital : capital após cada operação
        - inicio  : data de início da operação
        - fim     : data de término da operação
    """

    # Cria cópia do dataset para evitar alterar o original
    g = g.copy().reset_index(drop=True)

    # Número total de observações
    n = len(g)

    # Capital inicial da estratégia
    capital = float(initial_capital)

    # Vetores que armazenarão os resultados da simulação
    capital_out = [pd.NA] * n
    inicio_out = [pd.NA] * n
    fim_out = [pd.NA] * n

    # Índice que controla o início da janela de análise
    start = 0

    # ------------------------------------------------
    # Loop principal da simulação
    # ------------------------------------------------
    while start < n:

        # Define o final da janela de análise
        end = min(start + wind, n)

        # Subconjunto da janela atual
        window = g.loc[start:end - 1]

        # Data de início da operação
        inicio_data = g.loc[start, 'data']

        # Verifica se existe sinal de compra dentro da janela
        mask_signal = window[signal_col].fillna(0).astype(float) == 1

        # Caso 1: sinal aparece dentro da janela
        if mask_signal.any():

            # Primeiro índice onde o sinal ocorre
            first_idx = window.index[mask_signal][0]

            # Data de término da operação
            fim_data = g.loc[first_idx, 'data']

            # Valor do target
            target_val = float(g.loc[first_idx, 'target'])

            # Novo capital após atingir o target
            novo_capital = capital * target_val

            # Preenche os registros intermediários
            for i in range(start, first_idx):

                capital_out[i] = capital
                inicio_out[i] = inicio_data
                fim_out[i] = fim_data

            # Atualiza o capital na linha onde ocorreu o target
            capital_out[first_idx] = novo_capital
            inicio_out[first_idx] = inicio_data
            fim_out[first_idx] = fim_data

            # Atualiza capital da estratégia
            capital = novo_capital

            # Próxima janela começa após o target
            start = first_idx + 1


        # Caso 2: nenhum sinal na janela
        else:

            # Último índice da janela
            last_idx = window.index[-1]

            # Data de encerramento da operação
            fim_data = g.loc[last_idx, 'data']

            # Retorno real do ativo
            realized = float(g.loc[last_idx, 'resultado_real'])

            # Atualização do capital com retorno real
            novo_capital = capital * (1 + realized)

            # Preenche registros intermediários
            for i in range(start, end - 1):

                capital_out[i] = capital
                inicio_out[i] = inicio_data
                fim_out[i] = fim_data

            # Atualiza capital no último dia da janela
            capital_out[last_idx] = novo_capital
            inicio_out[last_idx] = inicio_data
            fim_out[last_idx] = fim_data

            # Atualiza capital da estratégia
            capital = novo_capital

            # Próxima janela começa após o final da atual
            start = end

    # Retorna DataFrame com os resultados da simulação
    return pd.DataFrame({
        'capital': capital_out,
        'inicio': inicio_out,
        'fim': fim_out
    })


# Função principal que processa os arquivos
def process_folder(input_folder, output_folder, initial_capital=100000.0, wind=4):

    """
    Processa todos os arquivos de comparação completa e executa
    a simulação monetária para diferentes estratégias.

    Estratégias simuladas:
    - Ensemble Total
    - Ensemble Parcial
    - Melhor Precision

    Para cada arquivo de ativo é gerado um novo arquivo contendo
    a evolução do capital ao longo do tempo.
    """

    os.makedirs(output_folder, exist_ok=True)

    # Loop pelos arquivos da pasta
    for fname in os.listdir(input_folder):
        if not fname.lower().endswith(".csv"):
            continue

        path = os.path.join(input_folder, fname)

        print(f"\n📂 Processando arquivo: {fname}")

        # Leitura do arquivo (separador padrão do projeto)
        df = pd.read_csv(path, sep="|")

        # Limpeza dos nomes das colunas
        df.columns = (
            df.columns
            .str.strip()
            .str.replace("\ufeff", "", regex=True)
        )

        # Verificação de colunas obrigatórias
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

        # Conversão de tipos
        df['data'] = pd.to_datetime(df['data'])

        # Ordenação dos dados
        df = df.sort_values(['ativo', 'target', 'data'])

        results = []

        # Processamento por ativo e target
        for (ativo, target), g in df.groupby(['ativo', 'target'], sort=False):

            print(f"  ▶ Simulando | ativo={ativo} | target={target}")

            g = g.reset_index(drop=True)

            # Estratégia 1: Ensemble Total
            sim_tot = simulate_strategy(
                g=g,
                signal_col='esmble_jan_tot',
                initial_capital=initial_capital,
                wind=wind
            )

            g['capital_tot'] = sim_tot['capital']
            g['inicio_tot'] = sim_tot['inicio']
            g['fim_tot'] = sim_tot['fim']

            # Estratégia 2: Ensemble Parcial
            sim_par = simulate_strategy(
                g=g,
                signal_col='esmble_jan_par',
                initial_capital=initial_capital,
                wind=wind
            )

            g['capital_par'] = sim_par['capital']
            g['inicio_par'] = sim_par['inicio']
            g['fim_par'] = sim_par['fim']

            # Estratégia 3: Melhor Precision
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

        # Consolidação dos resultados
        out = pd.concat(results, ignore_index=True)

        out_path = os.path.join(
            output_folder,
            fname.replace('.csv', '_monetario.csv')
        )

        out.to_csv(out_path, index=False, sep="|")

        print(f"✅ Arquivo salvo em: {out_path}")


# Função runner (executa todo o processo)
def run_monetary():

    """
    Executa todo o pipeline de simulação monetária
    utilizando os dados de comparação completa.
    """

    process_folder(
        input_folder="./data/ensemble/5_6_completo/",
        output_folder="./data/ensemble/7_monetario/",
        initial_capital=100000,
        wind=4
    )


if __name__ == "__main__":
    run_monetary()