import os
import pandas as pd
import numpy as np


def calcula_investimento(
    input_folder="./data/ensemble/7_intraday_join/",
    output_folder="./data/ensemble/8_monetario/"
):

    os.makedirs(output_folder, exist_ok=True)

    algoritmos = ['esmble_jan_tot', 'esmble_jan_par', 'in_precision']

    # =========================
    # PEGA APENAS ARQUIVOS VÁLIDOS
    # =========================
    arquivos = [
        f for f in os.listdir(input_folder)
        if f.lower().endswith(".csv") and "_intraday" in f
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

            print(f"  ▶ Processando | ativo={ativo} | target={target}")

            g = g.reset_index(drop=True)

            for alg in algoritmos:

                compra = np.zeros(len(g))
                rend = np.zeros(len(g))
                rend_venda = np.zeros(len(g))
                dias = np.zeros(len(g))

                preco_compra = 0
                contador = 0

                sinal_shift = g[alg].shift(1)

                for i in range(len(g)):

                    # =========================
                    # ENTRADA
                    # =========================
                    if contador == 0:

                        if sinal_shift.iloc[i] == 1:
                            preco_compra = g['Open'].iloc[i]
                            compra[i] = preco_compra
                            contador = 1
                        else:
                            compra[i] = 0
                            dias[i] = 0
                            continue

                    else:
                        compra[i] = preco_compra

                    # =========================
                    # RETORNOS
                    # =========================
                    rend[i] = (g['High'].iloc[i] - preco_compra) / preco_compra
                    rend_venda[i] = (g['Close'].iloc[i] - preco_compra) / preco_compra

                    # =========================
                    # SAÍDA
                    # =========================
                    atingiu = rend[i] >= (target - 1)

                    if atingiu:
                        dias[i] = 0
                        contador = 0
                        preco_compra = 0
                    else:
                        dias[i] = contador
                        contador += 1

                        if contador > 4:
                            contador = 0
                            preco_compra = 0

                # =========================
                # SALVA RESULTADOS
                # =========================
                g[f'compra_{alg}'] = compra
                g[f'rend_decisao_{alg}'] = rend
                g[f'rend_venda_{alg}'] = rend_venda
                g[f'dias_{alg}'] = dias

                venda = np.where(
                    g[f'dias_{alg}'] == 0,
                    g[f'compra_{alg}'] * target,
                    np.where(
                        g[f'dias_{alg}'] == 4,
                        g['Close'],
                        0
                    )
                )

                g[f'venda_{alg}'] = venda

            results.append(g)

        # =========================
        # SALVAMENTO
        # =========================
        if results:

            final_df = pd.concat(results, ignore_index=True)

            out_path = os.path.join(
                output_folder,
                fname.replace('.csv', '_monetario.csv')
            )

            final_df.to_csv(out_path, index=False, sep="|")

            print(f"✅ Salvo em: {out_path}")

        else:
            print("⚠️ Nenhum dado processado.")


# runner
def run_invest():
    calcula_investimento()


if __name__ == "__main__":
    run_invest()