import os
import pandas as pd
import numpy as np


def calcula_capital(
    input_folder="./data/ensemble/8_monetario/",
    output_folder="./data/ensemble/9_capital/"
):

    os.makedirs(output_folder, exist_ok=True)

    algoritmos = ['esmble_jan_tot', 'esmble_jan_par', 'in_precision']

    # =========================
    # PEGA APENAS ARQUIVOS VÁLIDOS
    # =========================
    arquivos = [
        f for f in os.listdir(input_folder)
        if f.lower().endswith(".csv") and "_monetario" in f
    ]

    # =========================
    # LOOP PRINCIPAL
    # =========================
    for fname in arquivos:

        path = os.path.join(input_folder, fname)

        print(f"\n📂 Processando arquivo: {fname}")

        df = pd.read_csv(path, sep="|")

        # =========================
        # TRATAMENTO INICIAL
        # =========================
        df['data'] = pd.to_datetime(df['data'], errors='coerce')
        df = df.sort_values(['ativo', 'target', 'data'])

        results = []

        # =========================
        # LOOP ATIVO + TARGET
        # =========================
        for (ativo, target), g in df.groupby(['ativo', 'target'], sort=False):

            print(f"  ▶ ativo={ativo} | target={target}")

            g = g.reset_index(drop=True)

            for alg in algoritmos:

                dinheiro = np.zeros(len(g))
                investido = np.zeros(len(g))
                qtde = np.zeros(len(g))

                capital = 1000
                posicao_aberta = False
                qtde_ativos = 0

                for i in range(len(g)):

                    compra = g[f'compra_{alg}'].iloc[i]
                    venda = g[f'venda_{alg}'].iloc[i]

                    # =========================
                    # SEM POSIÇÃO
                    # =========================
                    if not posicao_aberta:

                        if compra > 0:
                            investido[i] = capital

                            # proteção contra divisão por zero
                            if compra > 0:
                                qtde_ativos = capital / compra
                            else:
                                qtde_ativos = 0

                            qtde[i] = qtde_ativos
                            dinheiro[i] = 0
                            posicao_aberta = True

                        else:
                            dinheiro[i] = capital
                            investido[i] = 0
                            qtde[i] = 0

                    # =========================
                    # COM POSIÇÃO
                    # =========================
                    else:
                        investido[i] = capital
                        qtde[i] = qtde_ativos
                        dinheiro[i] = 0

                    # =========================
                    # VENDA
                    # =========================
                    if posicao_aberta and venda > 0:

                        capital = qtde_ativos * venda

                        posicao_aberta = False
                        qtde_ativos = 0

                # =========================
                # SALVA COLUNAS
                # =========================
                g[f'dinheiro_{alg}'] = dinheiro
                g[f'investido_{alg}'] = investido
                g[f'qtde_ativos_{alg}'] = qtde

                g[f'capital_teorico_{alg}'] = (
                    g[f'dinheiro_{alg}'] + g[f'investido_{alg}']
                )

            results.append(g)

        # =========================
        # SALVAMENTO
        # =========================
        if results:

            final_df = pd.concat(results, ignore_index=True)

            out_path = os.path.join(
                output_folder,
                fname.replace('.csv', '_capital.csv')
            )

            final_df.to_csv(out_path, index=False, sep="|")

            print(f"✅ Salvo em: {out_path}")

        else:
            print("⚠️ Nenhum dado processado.")


def run_calculo_capital():
    calcula_capital()


if __name__ == "__main__":
    run_calculo_capital()